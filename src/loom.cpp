#include "loom/loom.h"
#include "loom/loom_profiler_shim.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <climits>
#include <cstddef>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <deque>
#include <mutex>
#include <thread>
#include <vector>

#if defined(__linux__)
#include <linux/futex.h>
#include <pthread.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace loom {

// The internals in brief (docs/design.md has the long form):
//   - Task slots live in fixed chunks that never move. Free slots sit in a
//     small per-thread cache and on a lock-free (tagged Treiber) stack of
//     batches the caches trade with 15 at a time. Submitting and finishing a
//     task takes no lock.
//   - AnyWorker work sits in one Chase-Lev deque per worker plus one for the
//     main thread; the owner pushes and pops the bottom, every other thread
//     steals the top. Threads with no deque (dedicated threads, outside
//     threads) submit into one shared injection queue.
//   - Submit(span) chains its tasks into groups handed over as one entry;
//     whoever takes a group spreads the rest from its own deque.
//   - A counter is an atomic; only its final decrement takes its lock (to
//     release the tasks parked on it and wake its waiters). A worker applies
//     its finished tasks' signals to one counter in one go.
//   - Every sleep in loom goes through one Waiter per thread: a futex word
//     the waker sets and notifies. Idle workers and sleeping helpers register
//     in the idle list, so new work wakes them; a waiter also registers on its
//     counter (and on the main or dedicated queue it may run from).

static constexpr uint32_t kNoSlot = ~0u;

static inline void CpuRelax()
{
#if defined(__x86_64__) || defined(__i386__)
	__builtin_ia32_pause();
#elif defined(__aarch64__)
	asm volatile("yield");
#endif
}

// ---- Task slots ---------------------------------------------------------------
// One cache line pair per slot: adjacent slots are run by different threads.

struct alignas(64) TaskSlot
{
	// Everything a submit writes and a run reads is in the first 48 bytes
	// (the first cache line), so a task that does not use inline storage
	// moves one line between cores, not two.
	TaskFn        fn       = nullptr;                 // 0
	void*         userData = nullptr;                 // 8
	Affinity      affinity = Affinity::AnyWorker;     // 16
	uint8_t       window   = 0;                       // 17
	DedicatedId   thread   = 0;                       // 20
	CounterHandle signals  = kInvalidCounter;         // 24
	CounterHandle waitsFor = kInvalidCounter;         // 28
	uint32_t      nameId   = 0;                       // 32
	// The next task of a group (a queue entry with kGroupBit), written by the
	// submitter before the push and read by whoever takes the group.
	uint32_t      groupNext = kNoSlot;                // 36
	// Free-list fields, on a free batch's head slot: how many slots the batch
	// holds (the head, plus the rest listed in the bytes a free slot does not
	// use, see SetBatch), and the next batch on the shared stack. nextBatch is
	// atomic: a popper may read it from a batch another thread has just
	// claimed (its CAS then fails).
	uint32_t      batchCount = 0;                      // 40
	std::atomic<uint32_t> nextBatch{ kNoSlot };        // 44
	alignas(16) uint8_t inlineData[48]; // 48: the Submit(F) convenience
};
static_assert(offsetof(TaskSlot, nextBatch) + sizeof(uint32_t) <= 48 && offsetof(TaskSlot, inlineData) == 48,
	"the submit/run fields stay in the first cache line");
// A free batch head lists its slots in bytes [0, 40) and [48, 64): plain
// fields only (no atomic), and clear of batchCount and nextBatch.
static_assert(offsetof(TaskSlot, batchCount) == 40 && offsetof(TaskSlot, nextBatch) == 44,
	"the free-batch list bytes [0, 40) and [48, 64) hold no free-list field");

static constexpr uint32_t kSlotChunkBits = 12;
static constexpr uint32_t kSlotChunkSize = 1u << kSlotChunkBits;
static constexpr uint32_t kMaxSlotChunks = 1024; // 4M slots
// Free slots move between a thread's cache and the shared stack this many at
// a time (a head slot listing up to 14 more); a cache holds at most twice that.
// The list lives in the head's FIRST cache line, in bytes a free slot does not
// use: [0, 40) (the task fields) and [48, 64) (the start of inlineData).
static constexpr uint32_t kSlotBatchLow = 40 / sizeof(uint32_t);
static constexpr uint32_t kSlotBatchListed = kSlotBatchLow + 16 / sizeof(uint32_t);
static constexpr uint32_t kSlotBatch = 1 + kSlotBatchListed;
static constexpr uint32_t kSlotCacheMax = 2 * kSlotBatch;

// A queue entry is a slot index, or, with this bit, the first slot of a
// group: tasks chained through groupNext and handed over as one entry. (Slot
// indices stay below 2^22, so the bit is free; kNoSlot is never an entry.)
static constexpr uint32_t kGroupBit = 1u << 31;
// The most tasks Submit(span) chains into one entry.
static constexpr uint32_t kMaxGroup = 32;
// A thread applies its deferred task signals at least this often, so a
// counter's Load never lags far behind.
static constexpr int kMaxDeferredSignals = 64;

// ---- Waiting ------------------------------------------------------------------

enum : uint32_t
{
	kArmed         = 0, // asleep (or about to be)
	kWokeSearching = 1, // an idle worker woken to look for work; the waker counted it as searching
	kWokeRecheck   = 2, // anything else: re-check whatever you were waiting for
};

struct Waiter
{
	std::atomic<uint32_t> word{ kArmed };
	bool isWorker = false; // a worker's idle sleep (not a helper's)
};

// One per thread: a thread sleeps in at most one place at a time.
static thread_local Waiter t_Waiter;

// The sleep itself. On Linux a raw futex on the word: the kernel re-checks
// the word under its own lock, so a store-then-wake can never be missed
// (std::atomic::wait/notify keep their own waiter bookkeeping in the library,
// which loom cannot see into or test against).
#if defined(__linux__)
static_assert(sizeof(std::atomic<uint32_t>) == sizeof(uint32_t), "futex word layout");
static void FutexWait(std::atomic<uint32_t>& word, uint32_t expected)
{
	syscall(SYS_futex, reinterpret_cast<uint32_t*>(&word), FUTEX_WAIT_PRIVATE, expected, nullptr, nullptr, 0);
}
static void FutexWakeOne(std::atomic<uint32_t>& word)
{
	syscall(SYS_futex, reinterpret_cast<uint32_t*>(&word), FUTEX_WAKE_PRIVATE, 1, nullptr, nullptr, 0);
}
#else
static void FutexWait(std::atomic<uint32_t>& word, uint32_t expected) { word.wait(expected, std::memory_order_acquire); }
static void FutexWakeOne(std::atomic<uint32_t>& word) { word.notify_one(); }
#endif

// Wakers call this while holding the lock of the list they found the waiter
// in; the waiter takes that lock to unregister before it returns, so the
// Waiter can never be notified after it has moved on.
static void Wake(Waiter& w, uint32_t why)
{
	w.word.store(why, std::memory_order_release);
	FutexWakeOne(w.word);
}

static void SleepOn(Waiter& w)
{
	while (w.word.load(std::memory_order_acquire) == kArmed)
		FutexWait(w.word, kArmed);
}

// ---- The work-stealing deque --------------------------------------------------------
// Chase-Lev, after Le et al., "Correct and Efficient Work-Stealing for Weak
// Memory Models" (PPoPP 2013), with seq_cst operations in place of the fences
// (ThreadSanitizer understands those). The buffer grows; old buffers are kept
// until the deque dies, so a thief reading a stale buffer still reads valid
// values. Elements are task slot indices.

class WorkDeque
{
public:
	WorkDeque()
	{
		m_Buffers.push_back(std::make_unique<Buffer>(256));
		m_Buffer.store(m_Buffers.back().get(), std::memory_order_relaxed);
	}

	// Owner only.
	void Push(uint32_t v)
	{
		const int64_t b = m_Bottom.load(std::memory_order_relaxed);
		Buffer* a = m_Buffer.load(std::memory_order_relaxed);
		// The owner's last view of top. Top only grows, so the cached value
		// can only make the buffer look fuller than it is: re-read the real
		// top (the line every thief writes) only when it looks full.
		if (b - m_CachedTop >= a->capacity)
			m_CachedTop = m_Top.load(std::memory_order_acquire);
		if (b - m_CachedTop >= a->capacity)
		{
			const int64_t t = m_CachedTop;
			auto bigger = std::make_unique<Buffer>(a->capacity * 2);
			for (int64_t i = t; i < b; i++)
				bigger->Put(i, a->Get(i));
			a = bigger.get();
			m_Buffers.push_back(std::move(bigger));
			m_Buffer.store(a, std::memory_order_release);
		}
		a->Put(b, v);
		// Release only: the lost-wakeup ordering against sleepers comes from
		// the pusher's read-modify-write of the idle count afterwards
		// (WakeForWork), once per batch rather than a full fence per push.
		m_Bottom.store(b + 1, std::memory_order_release);
	}

	// Owner only.
	uint32_t Pop()
	{
		const int64_t b = m_Bottom.load(std::memory_order_relaxed) - 1;
		Buffer* a = m_Buffer.load(std::memory_order_relaxed);
		m_Bottom.store(b, std::memory_order_seq_cst);
		int64_t t = m_Top.load(std::memory_order_seq_cst);
		if (t > b)
		{
			m_Bottom.store(b + 1, std::memory_order_release);
			return kNoSlot;
		}
		uint32_t v = a->Get(b);
		if (t == b)
		{
			// The last element: race the thieves for it.
			if (!m_Top.compare_exchange_strong(t, t + 1, std::memory_order_seq_cst, std::memory_order_relaxed))
				v = kNoSlot;
			m_Bottom.store(b + 1, std::memory_order_release);
		}
		return v;
	}

	// Any thread. kNoSlot when empty or when another thief won the element.
	uint32_t Steal()
	{
		int64_t t = m_Top.load(std::memory_order_seq_cst);
		const int64_t b = m_Bottom.load(std::memory_order_seq_cst);
		if (t >= b)
			return kNoSlot;
		Buffer* a = m_Buffer.load(std::memory_order_acquire);
		const uint32_t v = a->Get(t);
		if (!m_Top.compare_exchange_strong(t, t + 1, std::memory_order_seq_cst, std::memory_order_relaxed))
			return kNoSlot;
		return v;
	}

	bool Empty() const
	{
		return m_Bottom.load(std::memory_order_seq_cst) <= m_Top.load(std::memory_order_seq_cst);
	}

	int Size() const
	{
		const int64_t n = m_Bottom.load(std::memory_order_acquire) - m_Top.load(std::memory_order_acquire);
		return n > 0 ? (int)n : 0;
	}

private:
	struct Buffer
	{
		explicit Buffer(int64_t c) : capacity(c), mask(c - 1), data(new std::atomic<uint32_t>[(size_t)c]) {}
		uint32_t Get(int64_t i) const { return data[(size_t)(i & mask)].load(std::memory_order_relaxed); }
		void Put(int64_t i, uint32_t v) { data[(size_t)(i & mask)].store(v, std::memory_order_relaxed); }
		int64_t capacity;
		int64_t mask;
		std::unique_ptr<std::atomic<uint32_t>[]> data;
	};

	alignas(64) std::atomic<int64_t> m_Top{ 0 };
	alignas(64) std::atomic<int64_t> m_Bottom{ 0 };
	std::atomic<Buffer*> m_Buffer{ nullptr };
	int64_t m_CachedTop = 0;                        // owner only
	std::vector<std::unique_ptr<Buffer>> m_Buffers; // owner only
};

// ---- Counters ------------------------------------------------------------------
// `count` holds the user's count PLUS ONE while it is nonzero: the decrement
// that takes the user's count to zero leaves 1 behind, releases the parked
// tasks and wakes the waiters under the lock, and only then stores 0. So
// nobody sees zero (and destroys or reuses the counter) before that final
// signaller is done with it. 0 means "zero and settled".

struct alignas(64) CounterState
{
	std::atomic<int>      count{ 0 };
	std::mutex            mutex;
	std::vector<uint32_t> pending; // slots parked until zero
	std::vector<Waiter*>  waiters; // threads asleep in WaitFor
	// Read lock-free by ResolveCounter on every thread while Create/Destroy
	// write them (the old plain fields were a data race the TSAN suite caught).
	std::atomic<bool>     alive{ false };
	std::atomic<uint32_t> generation{ 0 };
};

// ---- Dedicated threads --------------------------------------------------------

struct DedicatedState
{
	std::thread            thread;
	std::string            name;
	DedicatedConfig::Idle  idle = DedicatedConfig::Idle::Sleep;
	std::deque<uint32_t>   queue;  // MPSC: anyone submits, only it pops
	std::mutex             mutex;  // guards queue + waiter
	Waiter*                waiter = nullptr; // set while the thread sleeps
	std::atomic<bool>      stop{ false };
	DedicatedId            id = 0;
};

// ---- Per-thread scheduler state (the main thread and each worker) ------------------

struct alignas(64) WorkerCtx
{
	WorkDeque deque;
	// This thread's share of the live-task count: +1 per task it submits, -1
	// per task it finishes. Only the owner writes it; LiveTaskCount sums them.
	alignas(64) std::atomic<int64_t> live{ 0 };
	uint32_t rng = 0x9E3779B9u;
	bool     isMain = false;
	// Free task slots this thread may reuse without touching the shared
	// stack: it allocates from here, finished tasks land here, and it trades
	// with the shared stack a batch at a time. Owner only.
	uint32_t slotCache[kSlotCacheMax] = {};
	uint32_t slotCacheCount = 0;
	// Task-completion signals not yet applied: `deferredCount` decrements of
	// `deferredSignal`, from consecutive tasks that signal the same counter,
	// applied as one Signal (see FlushDeferredSignals). Owner only.
	CounterHandle deferredSignal = kInvalidCounter;
	int           deferredCount = 0;
};

struct alignas(64) PaddedStats
{
	ThreadStats s;
	// s.name points here: the row outlives the thread (a dedicated thread's
	// own name string dies when it is unregistered).
	std::string name;
};

// ---- The pool state --------------------------------------------------------------
// Everything a pool owns. The workers receive theirs at spawn; the public
// free functions resolve the active pool.

struct PoolState
{
	// Task slots.
	std::atomic<TaskSlot*>  slotChunks[kMaxSlotChunks] = {};
	std::vector<std::unique_ptr<TaskSlot[]>> ownedChunks; // under growMutex
	std::mutex              growMutex;
	std::atomic<uint32_t>   slotCapacity{ 0 };
	alignas(64) std::atomic<uint64_t> freeHead{ kNoSlot }; // (tag << 32) | first slot of the top batch

	// Counters: a fixed table and a free list.
	std::mutex              counterMutex;
	std::unique_ptr<CounterState[]> counters;
	uint32_t                counterCount = 0;
	std::vector<uint32_t>   freeCounters;

	// false from the first instruction of Shutdown (TryHelp, PumpMainThreadTasks
	// and IsInitialised then report the pool as down while it drains).
	std::atomic<bool>       running{ false };
	std::atomic<bool>       stop{ false };
	std::thread::id         mainThreadId;

	// The deques: the main thread's first, then the workers'. Fixed once the
	// workers spawn, so thieves walk it without a lock.
	std::unique_ptr<WorkerCtx> mainCtx;
	std::vector<std::unique_ptr<WorkerCtx>> workerCtx;
	std::vector<WorkerCtx*> deques;

	// The injection queue: AnyWorker work from threads without a deque.
	alignas(64) std::mutex  injectMutex;
	std::deque<uint32_t>    injectQueue;
	std::atomic<int>        injectSize{ 0 };

	// Live tasks submitted or finished by threads without a deque.
	alignas(64) std::atomic<int64_t> externalLive{ 0 };

	// Sleep and wake. numSearching counts workers looking for work (a pusher
	// wakes nobody while one is); numIdle mirrors idle.size() for the
	// pusher's lock-free check.
	alignas(64) std::atomic<int> numSearching{ 0 };
	alignas(64) std::atomic<int> numIdle{ 0 };
	std::mutex              idleMutex;
	std::vector<Waiter*>    idle;

	// The main-affinity queue.
	std::deque<uint32_t>    mainQueue;  // MPSC
	std::mutex              mainMutex;  // guards mainQueue + mainWaiter
	Waiter*                 mainWaiter = nullptr;

	std::mutex              dedicatedMutex;
	std::vector<std::unique_ptr<DedicatedState>> dedicated;
	std::atomic<int>        lanesHeld{ 0 };

	// Thread stats (the pool visualizer).
	std::mutex              statsMutex;
	std::vector<std::unique_ptr<PaddedStats>> threadStats;
	std::vector<std::thread> workers;
	int                     workerCount = 0;

	// Runs first on every pool thread.
	void (*onThreadStart)() = nullptr;

	// The park. parkRequested is written under parkMutex (a thread checks it
	// again there before it counts itself).
	std::mutex              parkMutex;
	std::condition_variable parkCV;    // the parked threads wait for the release
	std::condition_variable parkedCV;  // ParkAll waits for the last arrival
	std::atomic<bool>       parkRequested{ false };
	// Bumped by every UnparkAll. A parked thread waits for a release of ITS
	// round, not for the request to clear: a ParkAll right after an UnparkAll
	// sets the request again before a slow sleeper wakes, which left it asleep
	// in the old round and uncounted by the new one (the park-round timeout).
	uint64_t                parkReleases = 0;
	std::vector<ParkedThread> parked;
	struct ParkParticipant { int id; ParkWakeFn wake; void* user; };
	std::vector<ParkParticipant> participants;
	int nextParticipant = 1;

	// The profiler shim's hooks. Atomic: the owner unhooks while the workers
	// are still alive.
	std::atomic<ProfilerStateFn> profilerStateHook{ nullptr };
	std::atomic<ProfilerZoneFn>  profilerZoneHook{ nullptr };
};

// The two process-scope values:
// - the active pool, a NON-owning access root for the free API: installed by
//   the Pool constructor, cleared at the end of its Shutdown. No lazy
//   auto-init - a call with no active pool is a bug (asserted).
// - the thread-identity allocator behind CurrentThreadIndex(): the index is a
//   thread_local that outlives any one pool, so its allocator must too (a
//   per-pool counter restarting at 0 would hand a fresh worker the main
//   thread's 0 after a re-init).
static std::atomic<PoolState*> s_ActivePool{ nullptr };
static std::atomic<int>         s_NextThreadIndex{ 0 };

// A worker's own state. Worker threads die with their pool, so a stale value
// can never point at a later pool.
static thread_local PoolState*      t_WorkerPool = nullptr;
static thread_local WorkerCtx*      t_WorkerCtx = nullptr;
static thread_local DedicatedState* t_CurrentDedicated = nullptr;
// A pool thread's stats row (workers and dedicated threads; null elsewhere).
static thread_local ThreadStats*    t_ThreadStats = nullptr;

static PoolState& ActivePool()
{
	PoolState* pool = s_ActivePool.load(std::memory_order_acquire);
	assert(pool && "loom: no active pool - the engine's LoomPool (or an explicit loom::Pool) is the only init");
	return *pool;
}

// The calling thread's deque owner state: a worker's, the main thread's, or
// null for every other thread.
static WorkerCtx* CurrentCtx(PoolState& P)
{
	if (t_WorkerPool == &P)
		return t_WorkerCtx;
	if (std::this_thread::get_id() == P.mainThreadId)
		return P.mainCtx.get();
	return nullptr;
}

static DedicatedState* CurrentDedicated() { return t_CurrentDedicated; }

static std::atomic<int64_t>& LiveShard(PoolState& P, WorkerCtx* ctx)
{
	return ctx ? ctx->live : P.externalLive;
}

static int64_t LiveTasksIn(PoolState& P)
{
	int64_t n = P.externalLive.load(std::memory_order_acquire);
	for (WorkerCtx* c : P.deques)
		n += c->live.load(std::memory_order_acquire);
	return n;
}

static void MarkProfilerState(PoolState& P, int threadIndex, uint8_t state)
{
	if (ProfilerStateFn fn = P.profilerStateHook.load(std::memory_order_acquire))
		fn(threadIndex, state);
}

static uint64_t MarkProfilerZone(PoolState& P, int threadIndex, uint32_t nameId, uint8_t window, const char* name, bool begin, uint64_t handle)
{
	if (ProfilerZoneFn fn = P.profilerZoneHook.load(std::memory_order_acquire))
		return fn(threadIndex, nameId, window, name, begin, handle);
	return 0;
}

// Counter handle packing: generation in the high bits, index low.
static constexpr uint32_t kCounterGenBits = 20;
static constexpr uint32_t kCounterIndexMask = (1u << kCounterGenBits) - 1;

static CounterHandle PackCounter(uint32_t index, uint32_t generation)
{
	return (generation << kCounterGenBits) | index;
}
static uint32_t CounterIndex(CounterHandle h) { return h & kCounterIndexMask; }
static uint32_t CounterGeneration(CounterHandle h) { return h >> kCounterGenBits; }

static CounterState* ResolveCounter(PoolState& P, CounterHandle h)
{
	if (h == kInvalidCounter) return nullptr;
	uint32_t idx = CounterIndex(h);
	if (idx >= P.counterCount) return nullptr;
	CounterState& c = P.counters[idx];
	if (!c.alive.load(std::memory_order_relaxed) || CounterGeneration(h) != c.generation.load(std::memory_order_relaxed)) return nullptr;
	return &c;
}

// ---- Forward decls --------------------------------------------------------------

static uint32_t AllocateSlot(PoolState& P, WorkerCtx* ctx);
static bool     EnqueueSlot(PoolState& P, uint32_t slot, WorkerCtx* ctx);
static void     RunSlot(PoolState& P, uint32_t slot, WorkerCtx* ctx);
static void     RunEntry(PoolState& P, uint32_t entry, WorkerCtx* ctx);
static void     SignalCounter(PoolState& P, CounterHandle h, int n);
static void     FlushDeferredSignals(PoolState& P, WorkerCtx* ctx);
static void     WakeForWork(PoolState& P, size_t n);
static bool     TryHelpIn(PoolState& P);
static void     ParkHere(PoolState& P, int index, const char* name);
static void     WorkerLoop(PoolState& P, WorkerCtx& ctx);
static void     DedicatedLoop(PoolState& P, DedicatedState& d);
static ThreadStats* RegisterThreadStats(PoolState& P, int index, const char* name);

// ---- Slot storage -------------------------------------------------------------------

static TaskSlot& Slot(PoolState& P, uint32_t i)
{
	return P.slotChunks[i >> kSlotChunkBits].load(std::memory_order_acquire)[i & (kSlotChunkSize - 1)];
}

// The shared free stack holds BATCHES: a head slot that lists up to
// kSlotBatchListed more free slots in its first cache line, the heads linked
// through nextBatch. A list, not a chain through the slots: taking a batch
// costs one cache miss (the head's first line, which the submit writes
// anyway), where walking a chain was one DEPENDENT miss per slot, on lines
// another core (often on the other die) wrote last. A thread with a cache
// trades a whole batch per CAS; others trade one slot. The stack's head
// carries a 32-bit tag against ABA.

static uint8_t* SlotBytes(TaskSlot& s) { return reinterpret_cast<uint8_t*>(&s); }

static void SetBatch(TaskSlot& head, const uint32_t* rest, uint32_t restCount)
{
	head.batchCount = 1 + restCount;
	const uint32_t low = std::min(restCount, kSlotBatchLow);
	if (low > 0)
		std::memcpy(SlotBytes(head), rest, low * sizeof(uint32_t));
	if (restCount > low)
		std::memcpy(SlotBytes(head) + 48, rest + low, (restCount - low) * sizeof(uint32_t));
}

// Pushes the batches firstHead -> ... -> lastHead (already linked through
// nextBatch) in one CAS.
static void PushBatches(PoolState& P, uint32_t firstHead, uint32_t lastHead)
{
	TaskSlot& tail = Slot(P, lastHead);
	uint64_t head = P.freeHead.load(std::memory_order_relaxed);
	for (;;)
	{
		tail.nextBatch.store((uint32_t)head, std::memory_order_relaxed);
		const uint64_t want = (((head >> 32) + 1) << 32) | firstHead;
		if (P.freeHead.compare_exchange_weak(head, want, std::memory_order_release, std::memory_order_relaxed))
			return;
	}
}

// Makes `slots` (1..kSlotBatch of them) one batch and pushes it.
static void PushBatch(PoolState& P, const uint32_t* slots, uint32_t n)
{
	assert(n >= 1 && n <= kSlotBatch);
	SetBatch(Slot(P, slots[0]), slots + 1, n - 1);
	PushBatches(P, slots[0], slots[0]);
}

// The head of a batch, or kNoSlot when the stack is empty.
static uint32_t PopBatch(PoolState& P)
{
	uint64_t head = P.freeHead.load(std::memory_order_acquire);
	for (;;)
	{
		const uint32_t idx = (uint32_t)head;
		if (idx == kNoSlot)
			return kNoSlot;
		const uint32_t next = Slot(P, idx).nextBatch.load(std::memory_order_relaxed);
		const uint64_t want = (((head >> 32) + 1) << 32) | next;
		if (P.freeHead.compare_exchange_weak(head, want, std::memory_order_acquire, std::memory_order_acquire))
			return idx;
	}
}

// Copies a popped batch's slots into `out` (head last, so it is handed out
// last). Returns how many.
static uint32_t TakeBatch(PoolState& P, uint32_t head, uint32_t* out)
{
	TaskSlot& h = Slot(P, head);
	const uint32_t rest = h.batchCount - 1;
	assert(h.batchCount >= 1 && h.batchCount <= kSlotBatch);
	const uint32_t low = std::min(rest, kSlotBatchLow);
	if (low > 0)
		std::memcpy(out, SlotBytes(h), low * sizeof(uint32_t));
	if (rest > low)
		std::memcpy(out + low, SlotBytes(h) + 48, (rest - low) * sizeof(uint32_t));
	out[rest] = head;
	return rest + 1;
}

// growMutex held. Adds one chunk as kSlotBatch-sized batches (the last one
// takes the remainder).
static void AddSlotChunkLocked(PoolState& P)
{
	const uint32_t base = P.slotCapacity.load(std::memory_order_relaxed);
	const uint32_t chunk = base >> kSlotChunkBits;
	assert(chunk < kMaxSlotChunks && "loom: task slot pool exhausted");
	auto slots = std::make_unique<TaskSlot[]>(kSlotChunkSize);
	uint32_t members[kSlotBatchListed];
	uint32_t lastHead = base;
	for (uint32_t i = 0; i < kSlotChunkSize; i += kSlotBatch)
	{
		const uint32_t count = std::min(kSlotBatch, kSlotChunkSize - i);
		for (uint32_t m = 1; m < count; m++)
			members[m - 1] = base + i + m;
		SetBatch(slots[i], members, count - 1);
		const uint32_t nextHead = i + kSlotBatch < kSlotChunkSize ? base + i + kSlotBatch : kNoSlot;
		slots[i].nextBatch.store(nextHead, std::memory_order_relaxed);
		lastHead = base + i;
	}
	P.slotChunks[chunk].store(slots.get(), std::memory_order_release);
	P.ownedChunks.push_back(std::move(slots));
	P.slotCapacity.store(base + kSlotChunkSize, std::memory_order_relaxed);
	PushBatches(P, base, lastHead);
}

// A batch off the shared stack, growing the pool (with a warning) when it is empty.
static uint32_t PopBatchOrGrow(PoolState& P)
{
	for (;;)
	{
		const uint32_t first = PopBatch(P);
		if (first != kNoSlot)
			return first;
		std::lock_guard<std::mutex> lock(P.growMutex);
		if ((uint32_t)P.freeHead.load(std::memory_order_acquire) == kNoSlot)
		{
			AddSlotChunkLocked(P);
			std::fprintf(stderr, "loom: task pool grown to %u slots\n", P.slotCapacity.load(std::memory_order_relaxed));
		}
	}
}

static uint32_t AllocateSlot(PoolState& P, WorkerCtx* ctx)
{
	if (ctx)
	{
		if (ctx->slotCacheCount == 0)
			ctx->slotCacheCount = TakeBatch(P, PopBatchOrGrow(P), ctx->slotCache);
		return ctx->slotCache[--ctx->slotCacheCount];
	}
	// No cache: take one slot, push the rest back as a batch.
	uint32_t slots[kSlotBatch];
	const uint32_t n = TakeBatch(P, PopBatchOrGrow(P), slots);
	if (n > 1)
		PushBatch(P, slots, n - 1);
	return slots[n - 1];
}

// Returns the newest `n` cached slots to the shared stack as one batch.
static void SpillSlotCache(PoolState& P, WorkerCtx& ctx, uint32_t n)
{
	n = std::min(n, ctx.slotCacheCount);
	if (n == 0)
		return;
	ctx.slotCacheCount -= n;
	PushBatch(P, ctx.slotCache + ctx.slotCacheCount, n);
}

// Every cached slot back to the shared stack (a worker going to sleep: an
// idle thread must not sit on slots another thread may need).
static void FlushSlotCache(PoolState& P, WorkerCtx& ctx)
{
	while (ctx.slotCacheCount > 0)
		SpillSlotCache(P, ctx, kSlotBatch);
}

static void ReleaseSlot(PoolState& P, uint32_t slot, WorkerCtx* ctx)
{
	if (ctx)
	{
		ctx->slotCache[ctx->slotCacheCount++] = slot;
		if (ctx->slotCacheCount == kSlotCacheMax)
			SpillSlotCache(P, *ctx, kSlotBatch);
		return;
	}
	PushBatch(P, &slot, 1);
}

// ---- Lifecycle -------------------------------------------------------------------

Pool::Pool(const LoomConfig& config)
	: m_State(std::make_unique<PoolState>())
{
	// One pool at a time: the free API resolves the single active pool.
	PoolState* expected = nullptr;
	const bool installed = s_ActivePool.compare_exchange_strong(expected, m_State.get(), std::memory_order_acq_rel);
	assert(installed && "loom: a pool is already active (one loom::Pool at a time)");
	(void)installed;

	PoolState& P = *m_State;
	P.running = true;
	P.stop = false;
	P.mainThreadId = std::this_thread::get_id(); // the initializing thread is "main"
	// The initializing thread claims index 0 BEFORE the workers spawn; the
	// allocator is monotonic across pools (a dead worker's index is never
	// reused - restarting at 0 would hand a fresh worker the main thread's 0).
	(void)CurrentThreadIndex();

	{
		std::lock_guard<std::mutex> lock(P.growMutex);
		const uint32_t want = std::max(1u, config.maxTasks);
		while (P.slotCapacity.load(std::memory_order_relaxed) < want)
			AddSlotChunkLocked(P);
	}
	{
		std::lock_guard<std::mutex> lock(P.counterMutex);
		P.counterCount = std::min(std::max(1u, config.maxCounters), kCounterIndexMask);
		P.counters = std::make_unique<CounterState[]>(P.counterCount);
		P.freeCounters.reserve(P.counterCount);
		for (uint32_t i = P.counterCount; i-- > 0;)
			P.freeCounters.push_back(i); // index 0 is handed out first
	}

	P.onThreadStart = config.onThreadStart;
	P.workerCount = config.workerCount >= 0
		? config.workerCount
		: std::max(1, (int)std::thread::hardware_concurrency() - 1);

	P.mainCtx = std::make_unique<WorkerCtx>();
	P.mainCtx->isMain = true;
	P.deques.push_back(P.mainCtx.get());
	for (int i = 0; i < P.workerCount; i++)
	{
		P.workerCtx.push_back(std::make_unique<WorkerCtx>());
		P.workerCtx.back()->rng = 0x9E3779B9u * (uint32_t)(i + 2);
		P.deques.push_back(P.workerCtx.back().get());
	}
	P.idle.reserve((size_t)P.workerCount + 16);
	for (int i = 0; i < P.workerCount; i++)
		P.workers.emplace_back(WorkerLoop, std::ref(P), std::ref(*P.workerCtx[(size_t)i]));
}

Pool::~Pool()
{
	Shutdown();
}

static void WakeAllIdle(PoolState& P)
{
	std::lock_guard<std::mutex> lock(P.idleMutex);
	for (Waiter* w : P.idle)
		Wake(*w, kWokeRecheck);
	P.idle.clear();
	P.numIdle.store(0, std::memory_order_seq_cst);
}

void Pool::Shutdown()
{
	PoolState& P = *m_State;
	if (!P.running) return;
	P.running = false;

	// Every sleeper registers, then re-checks `stop`; the wake below takes the
	// lists' locks, so a sleeper either sees the flag or is in a list.
	P.stop.store(true, std::memory_order_seq_cst);
	WakeAllIdle(P);

	// The workers drain: each exits only after a search finds no work.
	for (auto& w : P.workers)
		if (w.joinable()) w.join();
	P.workers.clear();

	// Dedicated threads last.
	{
		std::lock_guard<std::mutex> lock(P.dedicatedMutex);
		for (auto& d : P.dedicated)
		{
			std::lock_guard<std::mutex> dlock(d->mutex);
			d->stop = true;
			if (d->waiter)
			{
				Wake(*d->waiter, kWokeRecheck);
				d->waiter = nullptr;
			}
		}
		for (auto& d : P.dedicated)
			if (d->thread.joinable()) d->thread.join();
		P.dedicated.clear();
	}

	// A thread still asleep in WaitFor(BlockOnly) tests `stop` too.
	for (uint32_t i = 0; i < P.counterCount; i++)
	{
		CounterState& c = P.counters[i];
		std::lock_guard<std::mutex> lock(c.mutex);
		for (Waiter* w : c.waiters)
			Wake(*w, kWokeRecheck);
		c.waiters.clear();
	}

	// Lanes join like every dedicated thread; a still-held lane at
	// shutdown is a leak (the drain-or-assert policy).
	assert(P.lanesHeld.load() == 0 && "loom lanes still held at shutdown");

	// Drain-or-assert: the queues must be empty at shutdown.
	{
		std::lock_guard<std::mutex> lock(P.injectMutex);
		assert(P.injectQueue.empty() && "AnyWorker tasks abandoned at shutdown");
	}
	for (WorkerCtx* c : P.deques)
	{
		assert(c->deque.Empty() && "AnyWorker tasks abandoned at shutdown");
		(void)c;
	}
	{
		std::lock_guard<std::mutex> lock(P.mainMutex);
		assert(P.mainQueue.empty() && "MainThread tasks abandoned at shutdown");
	}
	assert(LiveTasksIn(P) == 0 && "tasks still live at shutdown");

	{
		std::lock_guard<std::mutex> slock(P.statsMutex);
		P.threadStats.clear();
	}

	// Uninstalled only now: a task still running on a draining worker
	// resolves this pool, not none. The slots and counters live until the
	// Pool is destroyed.
	PoolState* expected = &P;
	s_ActivePool.compare_exchange_strong(expected, nullptr, std::memory_order_acq_rel);
}

bool IsInitialised()
{
	PoolState* pool = s_ActivePool.load(std::memory_order_acquire);
	return pool && pool->running;
}

int WorkerCount()
{
	PoolState* pool = s_ActivePool.load(std::memory_order_acquire);
	return pool ? pool->workerCount : 0;
}

int CurrentThreadIndex()
{
	static thread_local int t_index = -1;
	if (t_index >= 0)
		return t_index;
	t_index = s_NextThreadIndex.fetch_add(1);
	return t_index;
}

int LiveTaskCount()
{
	PoolState* pool = s_ActivePool.load(std::memory_order_acquire);
	return pool ? (int)LiveTasksIn(*pool) : 0;
}

static int PoolQueueDepth(PoolState& P)
{
	int n = P.injectSize.load(std::memory_order_acquire);
	for (WorkerCtx* c : P.deques)
		n += c->deque.Size();
	return n;
}

static int PendingTaskCountIn(PoolState& P)
{
	int n = PoolQueueDepth(P);
	{
		std::lock_guard<std::mutex> lock(P.mainMutex);
		n += (int)P.mainQueue.size();
	}
	std::lock_guard<std::mutex> lock(P.dedicatedMutex);
	for (auto& d : P.dedicated)
	{
		std::lock_guard<std::mutex> dl(d->mutex);
		n += (int)d->queue.size();
	}
	return n;
}

int PendingTaskCount()
{
	PoolState* pool = s_ActivePool.load(std::memory_order_acquire);
	return pool ? PendingTaskCountIn(*pool) : 0;
}

// ---- Finding work and the idle list ------------------------------------------------------

static bool HasPoolWork(PoolState& P)
{
	if (P.injectSize.load(std::memory_order_seq_cst) > 0)
		return true;
	for (WorkerCtx* c : P.deques)
		if (!c->deque.Empty())
			return true;
	return false;
}

static uint32_t PopInjected(PoolState& P)
{
	if (P.injectSize.load(std::memory_order_relaxed) <= 0)
		return kNoSlot;
	std::lock_guard<std::mutex> lock(P.injectMutex);
	if (P.injectQueue.empty())
		return kNoSlot;
	const uint32_t slot = P.injectQueue.front();
	P.injectQueue.pop_front();
	P.injectSize.fetch_sub(1, std::memory_order_relaxed);
	return slot;
}

// One pass over the injection queue and every other deque, from a random victim.
static uint32_t StealOnce(PoolState& P, WorkerCtx* self)
{
	const uint32_t injected = PopInjected(P);
	if (injected != kNoSlot)
		return injected;
	const size_t n = P.deques.size();
	static thread_local uint32_t t_rng = 0x2545F491u;
	uint32_t& rng = self ? self->rng : t_rng;
	rng ^= rng << 13;
	rng ^= rng >> 17;
	rng ^= rng << 5;
	const size_t start = rng % n;
	for (size_t k = 0; k < n; k++)
	{
		WorkerCtx* victim = P.deques[(start + k) % n];
		if (victim == self)
			continue;
		const uint32_t slot = victim->deque.Steal();
		if (slot != kNoSlot)
			return slot;
	}
	return kNoSlot;
}

static void RegisterIdle(PoolState& P, Waiter& w)
{
	std::lock_guard<std::mutex> lock(P.idleMutex);
	P.idle.push_back(&w);
	P.numIdle.fetch_add(1, std::memory_order_seq_cst);
}

// False when a waker already took `w` off the list (and set its word).
static bool UnregisterIdle(PoolState& P, Waiter& w)
{
	std::lock_guard<std::mutex> lock(P.idleMutex);
	auto it = std::find(P.idle.begin(), P.idle.end(), &w);
	if (it == P.idle.end())
		return false;
	P.idle.erase(it);
	P.numIdle.fetch_sub(1, std::memory_order_seq_cst);
	return true;
}

// Wakes up to `max` idle workers (each counted as searching before it wakes),
// or one sleeping helper when no worker is idle.
static void WakeIdle(PoolState& P, int max, bool onlyIfNoSearcher)
{
	std::lock_guard<std::mutex> lock(P.idleMutex);
	if (onlyIfNoSearcher && P.numSearching.load(std::memory_order_seq_cst) > 0)
		return;
	int woken = 0;
	for (size_t i = P.idle.size(); i-- > 0 && woken < max;)
	{
		Waiter* w = P.idle[i];
		if (!w->isWorker)
			continue;
		P.idle.erase(P.idle.begin() + (ptrdiff_t)i);
		P.numIdle.fetch_sub(1, std::memory_order_seq_cst);
		P.numSearching.fetch_add(1, std::memory_order_seq_cst);
		Wake(*w, kWokeSearching);
		woken++;
	}
	if (woken == 0 && !P.idle.empty())
	{
		Waiter* w = P.idle.back();
		P.idle.pop_back();
		P.numIdle.fetch_sub(1, std::memory_order_seq_cst);
		Wake(*w, kWokeRecheck);
	}
}

// After `n` AnyWorker tasks became runnable. One task wakes one sleeper, and
// only when no worker is already searching (that worker will find it; when
// it does, it wakes the next). A batch wakes enough workers to run it.
static void WakeForWork(PoolState& P, size_t n)
{
	// A read-modify-write, not a load: the lost-wakeup pair. A sleeper bumps
	// numIdle (an RMW) and then re-checks the queues. Both RMWs sit in
	// numIdle's modification order: if ours comes first, the push before it
	// happens-before the sleeper's re-check, which sees the work; if the
	// sleeper's comes first, we read its count and wake it. So pushes need no
	// fence of their own.
	if (n == 0 || P.numIdle.fetch_add(0, std::memory_order_seq_cst) == 0)
		return;
	const int searching = P.numSearching.load(std::memory_order_seq_cst);
	if (n == 1)
	{
		if (searching == 0)
			WakeIdle(P, 1, true);
		return;
	}
	const int want = (int)std::min<size_t>(n, (size_t)std::max(1, P.workerCount)) - searching;
	if (want > 0)
		WakeIdle(P, want, false);
}

// ---- Submission -------------------------------------------------------------------

static void PushPool(PoolState& P, uint32_t slot, WorkerCtx* ctx)
{
	if (ctx)
	{
		ctx->deque.Push(slot);
		return;
	}
	std::lock_guard<std::mutex> lock(P.injectMutex);
	P.injectQueue.push_back(slot);
	P.injectSize.fetch_add(1, std::memory_order_seq_cst);
}

// Makes a slot runnable. True for AnyWorker work (the caller then wakes a
// worker for it); main and dedicated work wakes its one thread here.
static bool EnqueueSlot(PoolState& P, uint32_t slot, WorkerCtx* ctx)
{
	TaskSlot& t = Slot(P, slot);
	switch (t.affinity)
	{
	case Affinity::AnyWorker:
		PushPool(P, slot, ctx);
		return true;
	case Affinity::MainThread:
	{
		std::lock_guard<std::mutex> lock(P.mainMutex);
		P.mainQueue.push_back(slot);
		if (P.mainWaiter)
		{
			Wake(*P.mainWaiter, kWokeRecheck);
			P.mainWaiter = nullptr;
		}
		return false;
	}
	case Affinity::Dedicated:
	{
		DedicatedState* d = nullptr;
		{
			std::lock_guard<std::mutex> lock(P.dedicatedMutex);
			for (auto& dd : P.dedicated)
				if (dd->id == t.thread) { d = dd.get(); break; }
		}
		assert(d && "Dedicated task submitted to an unregistered thread");
		if (!d)
			return false;
		std::lock_guard<std::mutex> lock(d->mutex);
		d->queue.push_back(slot);
		if (d->waiter)
		{
			Wake(*d->waiter, kWokeRecheck);
			d->waiter = nullptr;
		}
		return false;
	}
	}
	return false;
}

// Parks the slot on its waitsFor counter, or enqueues it. True when it went
// into the pool (AnyWorker, runnable now).
static bool DispatchSlot(PoolState& P, uint32_t slot, CounterHandle waitsFor, WorkerCtx* ctx)
{
	if (waitsFor != kInvalidCounter)
	{
		CounterState* c = ResolveCounter(P, waitsFor);
		assert(c && "task waits on an invalid/destroyed counter");
		if (c)
		{
			// Under the lock the count is either settled (0) or the final
			// signaller has yet to take this lock and will release the list.
			std::lock_guard<std::mutex> lock(c->mutex);
			if (c->count.load(std::memory_order_acquire) != 0)
			{
				c->pending.push_back(slot);
				return false;
			}
		}
	}
	return EnqueueSlot(P, slot, ctx);
}

static uint32_t PrepareSlot(PoolState& P, const TaskDesc& desc, WorkerCtx* ctx)
{
	assert((desc.affinity != Affinity::Dedicated) || desc.thread != kInvalidDedicated);
	const uint32_t slot = AllocateSlot(P, ctx);
	TaskSlot& t = Slot(P, slot);
	t.fn = desc.fn;
	t.userData = desc.userData;
	t.affinity = desc.affinity;
	t.thread = desc.thread;
	t.signals = desc.signals;
	t.waitsFor = desc.waitsFor;
	t.nameId = desc.nameId;
	t.window = desc.window;
	LiveShard(P, ctx).fetch_add(1, std::memory_order_relaxed);
	return slot;
}

static bool SubmitOne(PoolState& P, const TaskDesc& desc, WorkerCtx* ctx)
{
	assert(desc.fn && "task without a function");
	const uint32_t slot = PrepareSlot(P, desc, ctx);
	return DispatchSlot(P, slot, desc.waitsFor, ctx);
}

void Submit(const TaskDesc& desc)
{
	PoolState& P = ActivePool();
	WorkerCtx* ctx = CurrentCtx(P);
	if (SubmitOne(P, desc, ctx))
		WakeForWork(P, 1);
}

// How many tasks of an n-task batch go into one queue entry: n / 8, so a
// batch always leaves at least 8 entries for thieves to take at once (the
// takers spread the rest), up to kMaxGroup. Tuned on loom_bench (a 9950X,
// batches of 256): groups of 32 beat 8 (one per worker) by about 15% at 31
// workers and matched 64.
static uint32_t GroupSizeFor(size_t n)
{
	return (uint32_t)std::clamp<size_t>(n / 8, 1, kMaxGroup);
}

void Submit(std::span<const TaskDesc> descs)
{
	PoolState& P = ActivePool();
	WorkerCtx* ctx = CurrentCtx(P);
	const uint32_t groupSize = GroupSizeFor(descs.size());

	// Runnable AnyWorker tasks are chained into groups of groupSize, each
	// pushed as ONE entry: a thief takes a whole group, so the submitter's
	// deque sees one steal per group, and the taker spreads the rest from its
	// own deque. The others go one by one.
	size_t entries = 0;
	uint32_t head = kNoSlot, tail = kNoSlot, length = 0;
	auto pushGroup = [&] {
		if (length == 0)
			return;
		Slot(P, tail).groupNext = kNoSlot;
		PushPool(P, length > 1 ? (head | kGroupBit) : head, ctx);
		entries++;
		length = 0;
	};
	for (const TaskDesc& d : descs)
	{
		if (groupSize > 1 && d.affinity == Affinity::AnyWorker && d.waitsFor == kInvalidCounter)
		{
			assert(d.fn && "task without a function");
			const uint32_t slot = PrepareSlot(P, d, ctx);
			if (length == 0)
				head = slot;
			else
				Slot(P, tail).groupNext = slot;
			tail = slot;
			if (++length == groupSize)
				pushGroup();
		}
		else
			entries += SubmitOne(P, d, ctx) ? 1 : 0;
	}
	pushGroup();
	WakeForWork(P, entries);
}

namespace detail {

void SubmitInlineRaw(const void* data, size_t size, void (*trampoline)(void*), const TaskDesc& opts)
{
	PoolState& P = ActivePool();
	assert(opts.fn == nullptr && "Submit(F) and TaskDesc::fn are mutually exclusive");
	assert(size <= 48);
	WorkerCtx* ctx = CurrentCtx(P);
	const uint32_t slot = PrepareSlot(P, opts, ctx);
	TaskSlot& t = Slot(P, slot);
	std::memcpy(t.inlineData, data, size);
	t.fn = trampoline;
	t.userData = t.inlineData;
	if (DispatchSlot(P, slot, opts.waitsFor, ctx))
		WakeForWork(P, 1);
}

} // namespace detail

// ---- ParallelFor ---------------------------------------------------------------

namespace {

	struct ParallelRange { size_t begin = 0, end = 0; void* userData = nullptr; ParallelForFn fn = nullptr; };

	void ParallelRangeTrampoline(void* p)
	{
		ParallelRange& range = *static_cast<ParallelRange*>(p);
		range.fn(range.begin, range.end, range.userData);
	}

} // namespace

void ParallelFor(size_t begin, size_t end, size_t chunkSize, ParallelForFn fn, void* userData)
{
	PoolState& P = ActivePool(); // the no-pool assert even for an empty range
	if (begin >= end || chunkSize == 0 || fn == nullptr)
		return;

	const size_t chunkCount = (end - begin - 1) / chunkSize + 1;
	assert(chunkCount < (size_t)INT_MAX && "ParallelFor: too many chunks for one counter");
	std::vector<ParallelRange> ranges(chunkCount);
	const CounterHandle counter = CreateCounter((int)chunkCount);

	// One batch: every chunk is pushed, then the workers are woken once.
	WorkerCtx* ctx = CurrentCtx(P);
	size_t cursor = begin;
	for (size_t i = 0; i < chunkCount; i++)
	{
		const size_t chunkEnd = end - cursor > chunkSize ? cursor + chunkSize : end;
		ranges[i] = { cursor, chunkEnd, userData, fn };
		const uint32_t slot = PrepareSlot(P, TaskDesc{ .fn = &ParallelRangeTrampoline, .userData = &ranges[i],
			.affinity = Affinity::AnyWorker, .signals = counter }, ctx);
		PushPool(P, slot, ctx);
		cursor = chunkEnd;
	}
	WakeForWork(P, chunkCount);

	WaitFor(counter);
	DestroyCounter(counter);
}

// ---- Lanes ---------------------------------------------------------------------

LaneId AcquireLane(const char* name)
{
	ActivePool().lanesHeld.fetch_add(1, std::memory_order_relaxed);
	return RegisterDedicatedThread(DedicatedConfig{ .name = name, .idle = DedicatedConfig::Idle::Sleep });
}

void ReleaseLane(LaneId lane)
{
	PoolState& P = ActivePool();
	// Only decrement when the lane actually existed - a double release or
	// an invalid id must not corrupt the shutdown-held-lanes counter.
	{
		std::lock_guard<std::mutex> lock(P.dedicatedMutex);
		bool found = false;
		for (const auto& d : P.dedicated)
		{
			if (d->id == lane)
			{
				found = true;
				break;
			}
		}
		if (!found)
		{
			assert(false && "releasing an invalid/already-released lane");
			return;
		}
	}
	UnregisterDedicatedThread(lane);
	P.lanesHeld.fetch_sub(1, std::memory_order_relaxed);
}

// ---- Counters -----------------------------------------------------------------------

CounterHandle CreateCounter(int initial)
{
	PoolState& P = ActivePool();
	assert(initial >= 0 && initial < INT_MAX && "counter initial value out of range");
	std::lock_guard<std::mutex> lock(P.counterMutex);
	if (P.freeCounters.empty())
	{
		assert(false && "counter pool exhausted");
		return kInvalidCounter;
	}
	const uint32_t i = P.freeCounters.back();
	P.freeCounters.pop_back();
	CounterState& c = P.counters[i];
	c.count.store(initial > 0 ? initial + 1 : 0, std::memory_order_relaxed);
	c.alive.store(true, std::memory_order_release);
	// No generation bump here: the DESTROY path invalidates stale handles.
	// Bumping on create too doubled the churn and overflowed the packed
	// generation field under per-frame create/destroy cycles.
	return PackCounter(i, c.generation.load(std::memory_order_relaxed));
}

void DestroyCounter(CounterHandle h)
{
	PoolState& P = ActivePool();
	CounterState* c = ResolveCounter(P, h);
	assert(c && "destroying an invalid/destroyed counter");
	if (!c)
		return;
	{
		// The lock orders this after a final signaller still releasing the list.
		std::lock_guard<std::mutex> lock(c->mutex);
		assert(c->count.load(std::memory_order_relaxed) == 0 && "destroying a counter with nonzero count");
		assert(c->pending.empty() && "destroying a counter with parked tasks");
		assert(c->waiters.empty() && "destroying a counter with waiters");
	}
	// Under the lock CreateCounter holds: the handle is invalidated before the
	// index can be handed out again.
	std::lock_guard<std::mutex> lock(P.counterMutex);
	if (!c->alive.load(std::memory_order_relaxed) || c->generation.load(std::memory_order_relaxed) != CounterGeneration(h))
	{
		assert(false && "counter destroyed twice");
		return;
	}
	c->alive.store(false, std::memory_order_relaxed);
	// Wrap the packed generation explicitly: the 12-bit field must never
	// exceed 0xFFF or the (gen << 20) pack overflows into the index bits. A
	// wrapped generation is safe: no stale handle survives 4096 lifecycles of
	// its slot.
	const uint32_t next = (c->generation.load(std::memory_order_relaxed) + 1) & 0xFFF;
	c->generation.store(next == 0 ? 1 : next, std::memory_order_relaxed);
	P.freeCounters.push_back(CounterIndex(h));
}

void Signal(CounterHandle h, int n)
{
	SignalCounter(ActivePool(), h, n);
}

static void SignalCounter(PoolState& P, CounterHandle h, int n)
{
	CounterState* c = ResolveCounter(P, h);
	assert(c && "signalling an invalid/destroyed counter");
	if (!c || n <= 0)
		return;
	// The stored value is the count + 1, so `before - n == 1` is the user's zero.
	const int before = c->count.fetch_sub(n, std::memory_order_acq_rel);
	assert(before - n >= 1 && "counter underflow");
	if (before - n != 1)
		return;

	// The final signal: release the parked tasks and the waiters, then settle.
	WorkerCtx* ctx = CurrentCtx(P);
	size_t pooled = 0;
	{
		std::lock_guard<std::mutex> lock(c->mutex);
		for (uint32_t slot : c->pending)
			pooled += EnqueueSlot(P, slot, ctx) ? 1 : 0;
		c->pending.clear();
		c->count.store(0, std::memory_order_release);
		for (Waiter* w : c->waiters)
			Wake(*w, kWokeRecheck);
		c->waiters.clear();
	}
	WakeForWork(P, pooled);
}

int Load(CounterHandle h)
{
	PoolState& P = ActivePool();
	FlushDeferredSignals(P, CurrentCtx(P)); // the caller's own finished tasks count
	CounterState* c = ResolveCounter(P, h);
	assert(c && "loading an invalid/destroyed counter");
	if (!c)
		return 0;
	// 1 stored is "reached zero, still settling": report it as 1 until settled.
	const int v = c->count.load(std::memory_order_acquire);
	return v > 1 ? v - 1 : v;
}

// One sleep of a WaitFor: registers on the counter, and (when helping) on
// whatever else this thread may run, re-checks, sleeps, and unregisters.
static void SleepOnCounter(PoolState& P, CounterState& c, bool help)
{
	Waiter& w = t_Waiter;
	w.isWorker = false;
	w.word.store(kArmed, std::memory_order_relaxed);
	{
		std::lock_guard<std::mutex> lock(c.mutex);
		if (c.count.load(std::memory_order_acquire) == 0)
			return;
		c.waiters.push_back(&w);
	}

	bool ready = P.stop.load(std::memory_order_seq_cst);
	bool inMain = false, inIdle = false;
	DedicatedState* d = help ? CurrentDedicated() : nullptr;
	bool inDedicated = false;
	if (!ready && help && std::this_thread::get_id() == P.mainThreadId)
	{
		std::lock_guard<std::mutex> lock(P.mainMutex);
		if (!P.mainQueue.empty())
			ready = true;
		else
		{
			P.mainWaiter = &w;
			inMain = true;
		}
	}
	if (!ready && d)
	{
		std::lock_guard<std::mutex> lock(d->mutex);
		if (!d->queue.empty())
			ready = true;
		else
		{
			d->waiter = &w;
			inDedicated = true;
		}
	}
	const bool poolEligible = help && (!d || d->idle == DedicatedConfig::Idle::StealFromPool);
	if (!ready && poolEligible)
	{
		RegisterIdle(P, w);
		inIdle = true;
		ready = HasPoolWork(P);
	}
	if (!ready)
		ready = c.count.load(std::memory_order_acquire) == 0 || P.stop.load(std::memory_order_seq_cst);
	if (!ready)
		SleepOn(w);

	{
		std::lock_guard<std::mutex> lock(c.mutex);
		auto it = std::find(c.waiters.begin(), c.waiters.end(), &w);
		if (it != c.waiters.end())
			c.waiters.erase(it);
	}
	if (inMain)
	{
		std::lock_guard<std::mutex> lock(P.mainMutex);
		if (P.mainWaiter == &w)
			P.mainWaiter = nullptr;
	}
	if (inDedicated)
	{
		std::lock_guard<std::mutex> lock(d->mutex);
		if (d->waiter == &w)
			d->waiter = nullptr;
	}
	if (inIdle)
		UnregisterIdle(P, w);
}

void WaitFor(CounterHandle h, Help help)
{
	PoolState& P = ActivePool();
	CounterState* c = ResolveCounter(P, h);
	assert(c && "waiting on an invalid/destroyed counter");
	if (!c)
		return;

	const int threadIndex = CurrentThreadIndex();
	WorkerCtx* ctx = CurrentCtx(P);
	FlushDeferredSignals(P, ctx); // this thread's own finished tasks may be what we wait for
	MarkProfilerState(P, threadIndex, 0); // waiting
	while (c->count.load(std::memory_order_acquire) != 0 && !P.stop.load(std::memory_order_relaxed))
	{
		if (help == Help::WhileWaiting && TryHelpIn(P))
		{
			MarkProfilerState(P, threadIndex, 1); // helping
			continue;
		}
		// Nothing to run: apply what helping deferred, then re-check before
		// sleeping (it may have been the last of this counter's signals).
		if (ctx && ctx->deferredCount > 0)
		{
			FlushDeferredSignals(P, ctx);
			continue;
		}
		SleepOnCounter(P, *c, help == Help::WhileWaiting);
	}
	// Never return to the caller holding signals: it may block on anything.
	FlushDeferredSignals(P, ctx);
	MarkProfilerState(P, threadIndex, 2); // wait over
}

// Run at most one eligible task on the CALLING thread.
bool TryHelp()
{
	PoolState* pool = s_ActivePool.load(std::memory_order_acquire);
	if (!pool)
		return false;
	const bool ran = TryHelpIn(*pool);
	FlushDeferredSignals(*pool, CurrentCtx(*pool)); // never return holding signals
	return ran;
}

static uint32_t PopFront(std::mutex& m, std::deque<uint32_t>& q)
{
	std::lock_guard<std::mutex> lock(m);
	if (q.empty())
		return kNoSlot;
	const uint32_t slot = q.front();
	q.pop_front();
	return slot;
}

static bool TryHelpIn(PoolState& P)
{
	if (!P.running) return false;

	WorkerCtx* ctx = CurrentCtx(P);
	uint32_t slot = kNoSlot;
	if (ctx && ctx->isMain)
	{
		// Main-affinity first: it keeps main-only work deadlock-free while the
		// main thread waits.
		slot = PopFront(P.mainMutex, P.mainQueue);
	}
	else if (DedicatedState* d = CurrentDedicated())
	{
		slot = PopFront(d->mutex, d->queue);
		if (slot == kNoSlot && d->idle != DedicatedConfig::Idle::StealFromPool)
			return false;
	}
	if (slot == kNoSlot && ctx)
		slot = ctx->deque.Pop();
	if (slot == kNoSlot)
		slot = StealOnce(P, ctx);
	if (slot == kNoSlot)
		return false;
	// Run OUTSIDE every lock: a task may submit more work (nested ParallelFor).
	// A pool thread shows as Helping while it runs a task inside a wait.
	ThreadStats* stats = t_ThreadStats;
	const ThreadState before = stats ? stats->state.load(std::memory_order_relaxed) : ThreadState::Idle;
	if (stats)
		stats->state.store(ThreadState::Helping, std::memory_order_relaxed);
	RunEntry(P, slot, ctx);
	if (stats)
	{
		stats->tasksRun.fetch_add(1, std::memory_order_relaxed);
		stats->helpedRun.fetch_add(1, std::memory_order_relaxed);
		stats->state.store(before, std::memory_order_relaxed);
	}
	return true;
}

// Runs a queue entry. A group's first task runs here; the rest are pushed
// first, one entry each, onto this thread's deque (where idle workers steal
// them), or back to the injection queue as one group for a thread without one.
static void RunEntry(PoolState& P, uint32_t entry, WorkerCtx* ctx)
{
	const uint32_t first = entry & ~kGroupBit;
	if (entry & kGroupBit)
	{
		const uint32_t rest = Slot(P, first).groupNext;
		if (rest != kNoSlot)
		{
			size_t pushed = 0;
			if (ctx)
			{
				// Read each link BEFORE pushing that task: once pushed, a thief
				// may run and free it, and its slot be reused.
				for (uint32_t s = rest; s != kNoSlot;)
				{
					const uint32_t next = Slot(P, s).groupNext;
					ctx->deque.Push(s);
					pushed++;
					s = next;
				}
			}
			else
			{
				PushPool(P, Slot(P, rest).groupNext != kNoSlot ? (rest | kGroupBit) : rest, nullptr);
				pushed = 1;
			}
			WakeForWork(P, pushed);
		}
	}
	RunSlot(P, first, ctx);
}

static void RunSlot(PoolState& P, uint32_t slot, WorkerCtx* ctx)
{
	// The runner owns the slot until it releases it: no copy, no lock.
	TaskSlot& t = Slot(P, slot);
	const TaskFn fn = t.fn;
	void* const userData = t.userData;
	const CounterHandle signals = t.signals;
	const uint32_t nameId = t.nameId;
	const uint8_t window = t.window;

	// Deferred signals only ever belong to the counter the RUNNING task
	// signals: that counter cannot reach zero before this task finishes, so
	// holding them back delays nothing. Any other task (another counter, or
	// none, maybe a long one) runs only after they are applied.
	if (ctx && ctx->deferredCount > 0 && ctx->deferredSignal != signals)
		FlushDeferredSignals(P, ctx);

	// A per-task profiler zone (null hook = free no-op). The begin hook
	// returns the handle the end mark pairs with.
	const int threadIndex = CurrentThreadIndex();
	const uint64_t zone = MarkProfilerZone(P, threadIndex, nameId, window, "Task", true, 0);
	if (fn)
		fn(userData); // an inline callable's userData is the slot: release after
	MarkProfilerZone(P, threadIndex, nameId, window, "Task", false, zone);

	LiveShard(P, ctx).fetch_sub(1, std::memory_order_relaxed);
	ReleaseSlot(P, slot, ctx);
	if (signals == kInvalidCounter)
		return;
	if (!ctx)
	{
		SignalCounter(P, signals, 1);
		return;
	}
	// One shared-counter RMW per run of same-counter tasks, not per task:
	// every task signalling one counter made it the hottest line in the pool.
	ctx->deferredSignal = signals;
	if (++ctx->deferredCount >= kMaxDeferredSignals)
		FlushDeferredSignals(P, ctx);
}

// Applies this thread's deferred task signals. Called before anything that
// may block, look for other work or hand control back to a caller.
static void FlushDeferredSignals(PoolState& P, WorkerCtx* ctx)
{
	if (!ctx || ctx->deferredCount == 0)
		return;
	const CounterHandle h = ctx->deferredSignal;
	const int n = ctx->deferredCount;
	ctx->deferredSignal = kInvalidCounter;
	ctx->deferredCount = 0;
	SignalCounter(P, h, n); // may release parked tasks onto this deque
}

// ---- Main thread ---------------------------------------------------------------------

void PumpMainThreadTasks()
{
	PoolState* pool = s_ActivePool.load(std::memory_order_acquire);
	if (!pool || !pool->running) return;
	PoolState& P = *pool;
	WorkerCtx* ctx = CurrentCtx(P);
	for (int guard = 0; guard < 1000000; guard++)
	{
		const uint32_t slot = PopFront(P.mainMutex, P.mainQueue);
		if (slot == kNoSlot)
			break;
		RunSlot(P, slot, ctx);
	}
	FlushDeferredSignals(P, ctx);
}

// ---- Workers --------------------------------------------------------------------------

// Rounds of stealing before a worker sleeps: the first spin, the rest yield.
static constexpr int kSearchRounds = 32;

// Names the calling OS thread. Linux takes at most 15 characters and rejects
// a longer name outright, so the name is cut to fit (loom's own records keep
// the full name).
static void SetOsThreadName(const char* name)
{
#if defined(__linux__)
	if (!name || !*name)
		return;
	char shortName[16];
	std::snprintf(shortName, sizeof(shortName), "%s", name);
	pthread_setname_np(pthread_self(), shortName);
#else
	(void)name;
#endif
}
static constexpr int kSpinRounds = 8;

static uint32_t Search(PoolState& P, WorkerCtx& ctx)
{
	for (int round = 0; round < kSearchRounds; round++)
	{
		const uint32_t slot = StealOnce(P, &ctx);
		if (slot != kNoSlot)
			return slot;
		if (round == 0 && ctx.deferredCount > 0)
		{
			// No work at hand: apply the signals held back so far before
			// spinning (a waiter may need only these), and run what they
			// release onto this deque.
			FlushDeferredSignals(P, &ctx);
			const uint32_t released = ctx.deque.Pop();
			if (released != kNoSlot)
				return released;
		}
		if (P.parkRequested.load(std::memory_order_relaxed))
			return kNoSlot;
		if (round < kSpinRounds)
		{
			for (int i = 0; i < 32; i++)
				CpuRelax();
		}
		else
			std::this_thread::yield();
	}
	return kNoSlot;
}

static void WorkerLoop(PoolState& P, WorkerCtx& ctx)
{
	if (P.onThreadStart)
		P.onThreadStart();
	t_WorkerPool = &P;
	t_WorkerCtx = &ctx;
	const int index = CurrentThreadIndex(); // reserve this worker's index
	// A unique per-worker name (the pool visualizer keys its per-thread
	// history on it, and the profiler's lane names benefit too).
	static thread_local char t_name[32];
	std::snprintf(t_name, sizeof(t_name), "loom-worker-%d", index);
	ThreadStats* stats = RegisterThreadStats(P, index, t_name);
	t_ThreadStats = stats;
	if (stats)
		SetOsThreadName(stats->name);

	Waiter& w = t_Waiter;
	// Whether this worker counts in numSearching. Every path out of the
	// search (work found, nothing found, park) drops it: a leaked count would
	// stop pushers from ever waking anyone.
	bool searching = false;
	auto stopSearching = [&]() -> bool {
		if (!searching)
			return false;
		searching = false;
		return P.numSearching.fetch_sub(1, std::memory_order_seq_cst) == 1;
	};
	// After leaving the idle list: a waker that took us off it may have
	// counted us as searching.
	auto leaveIdle = [&]() {
		if (!UnregisterIdle(P, w) && w.word.load(std::memory_order_acquire) == kWokeSearching)
			searching = true;
	};

	for (;;)
	{
		if (P.parkRequested.load(std::memory_order_acquire) && !P.stop.load(std::memory_order_acquire))
		{
			stopSearching();
			FlushDeferredSignals(P, &ctx);
			ParkHere(P, index, stats ? stats->name : "loom-worker");
			continue;
		}

		uint32_t slot = ctx.deque.Pop();
		if (slot == kNoSlot)
		{
			if (!searching)
			{
				searching = true;
				P.numSearching.fetch_add(1, std::memory_order_seq_cst);
			}
			slot = Search(P, ctx);
		}
		if (slot != kNoSlot)
		{
			// The last searcher to find work wakes another: there may be more.
			if (stopSearching() && P.numIdle.load(std::memory_order_seq_cst) > 0)
				WakeIdle(P, 1, true);
			if (stats) stats->state = ThreadState::Running;
			RunEntry(P, slot, &ctx);
			if (stats) stats->tasksRun++;
			continue;
		}

		stopSearching();
		FlushDeferredSignals(P, &ctx); // Search applied them; never sleep or exit holding any
		if (P.stop.load(std::memory_order_acquire))
			break; // drained: a full search found nothing

		// Sleep: register, re-check, sleep.
		FlushSlotCache(P, ctx);
		w.isWorker = true;
		w.word.store(kArmed, std::memory_order_relaxed);
		RegisterIdle(P, w);
		if (HasPoolWork(P) || P.stop.load(std::memory_order_seq_cst) || P.parkRequested.load(std::memory_order_seq_cst))
		{
			leaveIdle();
			continue;
		}
		if (stats) stats->state = ThreadState::Idle;
		MarkProfilerState(P, index, 0); // waiting (the idle sleep)
		SleepOn(w);
		MarkProfilerState(P, index, 2); // wait over
		leaveIdle();
	}
	if (stats) stats->state = ThreadState::Idle;
}

// ---- Thread stats (the pool visualizer) ------------------------------------------------

static ThreadStats* RegisterThreadStats(PoolState& P, int index, const char* name)
{
	std::lock_guard<std::mutex> lock(P.statsMutex);
	for (auto& t : P.threadStats)
		if (t->s.index == index)
			return &t->s;
	auto stats = std::make_unique<PaddedStats>();
	stats->s.index = index;
	stats->name = name ? name : "";
	stats->s.name = stats->name.c_str();
	ThreadStats* raw = &stats->s;
	P.threadStats.push_back(std::move(stats));
	return raw;
}

PoolSnapshot SnapshotThreadStats()
{
	PoolSnapshot snap;
	PoolState* pool = s_ActivePool.load(std::memory_order_acquire);
	if (!pool)
		return snap;
	PoolState& P = *pool;
	{
		std::lock_guard<std::mutex> lock(P.statsMutex);
		for (auto& padded : P.threadStats)
		{
			const ThreadStats& t = padded->s;
			ThreadStatsView view;
			view.index = t.index;
			view.name = t.name;
			view.state = t.state.load();
			view.tasksRun = t.tasksRun.load();
			view.helpedRun = t.helpedRun.load();
			view.queueDepth = 0;
			snap.threads.push_back(std::move(view));
		}
	}
	snap.anyQueueDepth = PoolQueueDepth(P);
	{
		std::lock_guard<std::mutex> lock(P.mainMutex);
		snap.mainQueueDepth = (int)P.mainQueue.size();
	}
	{
		std::lock_guard<std::mutex> lock(P.dedicatedMutex);
		for (auto& d : P.dedicated)
		{
			int depth = 0;
			{
				std::lock_guard<std::mutex> dl(d->mutex);
				depth = (int)d->queue.size();
			}
			// Attach the dedicated queue depth to its stats row.
			for (auto& t : snap.threads)
				if (!t.name.empty() && t.name == (d->name.empty() ? "loom-dedicated" : d->name))
					t.queueDepth = depth;
		}
	}
	snap.liveTasks = (int)LiveTasksIn(P);
	snap.pendingTasks = PendingTaskCountIn(P);
	{
		std::lock_guard<std::mutex> lock(P.counterMutex);
		snap.counterCount = (int)(P.counterCount - P.freeCounters.size());
	}
	return snap;
}

// ---- Dedicated threads --------------------------------------------------------------------

static void DedicatedLoop(PoolState& P, DedicatedState& d)
{
	if (P.onThreadStart)
		P.onThreadStart();
	t_CurrentDedicated = &d;
	const int index = CurrentThreadIndex(); // reserve this thread's index
	const char* name = d.name.empty() ? "loom-dedicated" : d.name.c_str();
	ThreadStats* stats = RegisterThreadStats(P, index, name);
	t_ThreadStats = stats;
	SetOsThreadName(d.name.c_str());
	const bool steals = d.idle == DedicatedConfig::Idle::StealFromPool;
	Waiter& w = t_Waiter;
	for (;;)
	{
		if (P.parkRequested.load(std::memory_order_acquire) && !d.stop.load(std::memory_order_acquire))
		{
			ParkHere(P, index, name);
			continue;
		}
		uint32_t slot = PopFront(d.mutex, d.queue);
		if (slot == kNoSlot && steals)
			slot = StealOnce(P, nullptr);
		if (slot != kNoSlot)
		{
			if (stats) stats->state = ThreadState::Running;
			RunEntry(P, slot, nullptr);
			if (stats) { stats->tasksRun++; stats->state = ThreadState::Idle; }
			continue;
		}
		if (d.stop.load(std::memory_order_acquire))
			break; // its own queue is drained

		// Sleep until its own queue (or, when it steals, the pool) has work.
		w.isWorker = false;
		w.word.store(kArmed, std::memory_order_relaxed);
		bool ready = false;
		{
			std::lock_guard<std::mutex> lock(d.mutex);
			ready = !d.queue.empty() || d.stop.load() || P.parkRequested.load(std::memory_order_seq_cst);
			if (!ready)
				d.waiter = &w;
		}
		bool inIdle = false;
		if (!ready && steals)
		{
			RegisterIdle(P, w);
			inIdle = true;
			ready = HasPoolWork(P) || P.parkRequested.load(std::memory_order_seq_cst);
		}
		if (!ready)
		{
			MarkProfilerState(P, index, 0); // waiting (the idle sleep)
			SleepOn(w);
			MarkProfilerState(P, index, 2); // wait over
		}
		{
			std::lock_guard<std::mutex> lock(d.mutex);
			if (d.waiter == &w)
				d.waiter = nullptr;
		}
		if (inIdle)
			UnregisterIdle(P, w);
	}
	t_CurrentDedicated = nullptr;
	t_ThreadStats = nullptr;
}

DedicatedId RegisterDedicatedThread(const DedicatedConfig& config)
{
	PoolState& P = ActivePool();
	std::lock_guard<std::mutex> lock(P.dedicatedMutex);
	uint32_t id = 0;
	for (;;)
	{
		bool taken = false;
		for (auto& d : P.dedicated)
			if (d->id == id) { taken = true; break; }
		if (!taken) break;
		id++;
	}
	auto d = std::make_unique<DedicatedState>();
	d->id = id;
	d->name = config.name ? config.name : "";
	d->idle = config.idle;
	d->stop = false;
	DedicatedId handle = id;
	d->thread = std::thread(DedicatedLoop, std::ref(P), std::ref(*d));
	P.dedicated.push_back(std::move(d));
	return handle;
}

void UnregisterDedicatedThread(DedicatedId id)
{
	PoolState& P = ActivePool();
	std::unique_ptr<DedicatedState> d;
	{
		std::lock_guard<std::mutex> lock(P.dedicatedMutex);
		for (auto it = P.dedicated.begin(); it != P.dedicated.end(); ++it)
		{
			if ((*it)->id == id)
			{
				d = std::move(*it);
				P.dedicated.erase(it);
				break;
			}
		}
	}
	if (!d) return;
	// The stop flag and the wake under the waiter's own lock (a bare
	// flag-set-then-notify loses the wakeup).
	{
		std::lock_guard<std::mutex> lock(d->mutex);
		d->stop = true;
		if (d->waiter)
		{
			Wake(*d->waiter, kWokeRecheck);
			d->waiter = nullptr;
		}
	}
	if (d->thread.joinable()) d->thread.join();
}

// ---- Parking -------------------------------------------------------------------

long CurrentOsThreadId()
{
#if defined(__linux__)
	return (long)syscall(SYS_gettid);
#else
	return 0;
#endif
}

void CaptureReturnAddresses(std::vector<uintptr_t>& out, size_t max)
{
#if defined(__linux__)
	pthread_attr_t attr;
	if (pthread_getattr_np(pthread_self(), &attr) != 0)
		return;
	void* stackAddr = nullptr;
	size_t stackSize = 0;
	pthread_attr_getstack(&attr, &stackAddr, &stackSize);
	pthread_attr_destroy(&attr);
	const uintptr_t lo = (uintptr_t)stackAddr, hi = lo + stackSize;
	const uintptr_t* fp = (const uintptr_t*)__builtin_frame_address(0);
	while (fp && (uintptr_t)fp >= lo && (uintptr_t)fp + 2 * sizeof(uintptr_t) <= hi && out.size() < max)
	{
		const uintptr_t ret = fp[1];
		if (ret == 0)
			break;
		out.push_back(ret);
		const uintptr_t* next = (const uintptr_t*)fp[0];
		if (next <= fp)
			break; // a frame without a frame pointer ends the walk
		fp = next;
	}
#else
	(void)out;
	(void)max;
#endif
}

static void ParkHere(PoolState& P, int index, const char* name)
{
	ParkedThread self;
	self.index = index;
	self.name = name ? name : "";
	self.osThreadId = CurrentOsThreadId();
	CaptureReturnAddresses(self.returnAddresses);
	std::unique_lock<std::mutex> lock(P.parkMutex);
	if (!P.parkRequested.load())
		return; // released before this thread arrived
	const uint64_t round = P.parkReleases;
	P.parked.push_back(std::move(self));
	P.parkedCV.notify_all();
	P.parkCV.wait(lock, [&] { return P.parkReleases != round; });
}

bool ParkAll(uint32_t timeoutMs)
{
	PoolState& P = ActivePool();
	size_t expected = 0;
	{
		std::lock_guard<std::mutex> lock(P.dedicatedMutex);
		expected = P.workers.size() + P.dedicated.size();
	}
	std::vector<PoolState::ParkParticipant> participants;
	{
		std::lock_guard<std::mutex> lock(P.parkMutex);
		P.parked.clear();
		P.parkRequested = true;
		participants = P.participants;
	}
	expected += participants.size();
	for (auto& p : participants)
		p.wake(p.user);
	// Wake the sleepers under their lists' locks: each re-checks the request
	// after registering, so it either sees it or is woken here.
	WakeAllIdle(P);
	{
		std::lock_guard<std::mutex> lock(P.dedicatedMutex);
		for (auto& d : P.dedicated)
		{
			std::lock_guard<std::mutex> dlock(d->mutex);
			if (d->waiter)
			{
				Wake(*d->waiter, kWokeRecheck);
				d->waiter = nullptr;
			}
		}
	}
	bool all = false;
	{
		std::unique_lock<std::mutex> lock(P.parkMutex);
		all = P.parkedCV.wait_for(lock, std::chrono::milliseconds(timeoutMs), [&] { return P.parked.size() >= expected; });
	}
	if (!all)
		UnparkAll();
	return all;
}

void UnparkAll()
{
	PoolState& P = ActivePool();
	std::lock_guard<std::mutex> lock(P.parkMutex);
	P.parkRequested = false;
	P.parkReleases++;
	P.parked.clear();
	P.parkCV.notify_all();
}

int RegisterParkParticipant(ParkWakeFn wake, void* user)
{
	PoolState* pool = s_ActivePool.load(std::memory_order_acquire);
	if (!pool)
		return 0;
	std::lock_guard<std::mutex> lock(pool->parkMutex);
	const int id = pool->nextParticipant++;
	pool->participants.push_back({ id, wake, user });
	return id;
}

void UnregisterParkParticipant(int id)
{
	PoolState* pool = s_ActivePool.load(std::memory_order_acquire);
	if (!pool || id == 0)
		return;
	std::lock_guard<std::mutex> lock(pool->parkMutex);
	std::erase_if(pool->participants, [id](const PoolState::ParkParticipant& p) { return p.id == id; });
}

bool ParkRequested()
{
	PoolState* pool = s_ActivePool.load(std::memory_order_acquire);
	return pool && pool->parkRequested.load();
}

void ParkPoint(const char* name)
{
	PoolState* pool = s_ActivePool.load(std::memory_order_acquire);
	if (pool && pool->parkRequested.load())
		ParkHere(*pool, CurrentThreadIndex(), name);
}

std::vector<ParkedThread> ParkedThreads()
{
	PoolState& P = ActivePool();
	std::lock_guard<std::mutex> lock(P.parkMutex);
	return P.parked;
}

// ---- The profiler shim's hooks (loom_profiler_shim.h) ----------------------------

void SetProfilerStateHook(ProfilerStateFn fn)
{
	PoolState* pool = s_ActivePool.load(std::memory_order_acquire);
	assert((pool || fn == nullptr) && "loom: a profiler hook installed with no active pool");
	if (pool)
		pool->profilerStateHook.store(fn, std::memory_order_release);
}

void SetProfilerZoneHook(ProfilerZoneFn fn)
{
	PoolState* pool = s_ActivePool.load(std::memory_order_acquire);
	assert((pool || fn == nullptr) && "loom: a profiler hook installed with no active pool");
	if (pool)
		pool->profilerZoneHook.store(fn, std::memory_order_release);
}

} // namespace loom
