#include "loom/loom.h"
#include "loom/loom_profiler_shim.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <climits>
#include <cstddef>
#include <condition_variable>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <deque>
#include <exception>
#include <map>
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
//     release the tasks waiting for it and wake its waiters). A worker applies
//     its finished tasks' signals to one counter in one go.
//   - Every sleep in loom goes through one Waiter per thread: a futex word
//     the waker sets and notifies. Idle workers and sleeping helpers register
//     in the idle list, so new work wakes them; a waiter also registers on its
//     counter (and on the main or dedicated queue it may run from).

// "No slot": an empty queue, the end of a list, an empty free stack.
static constexpr uint32_t kNoSlot = ~0u;

// A spin-wait hint: PAUSE on x86 (backs off the pipeline and frees the core
// for its hyperthread sibling), YIELD on ARM.
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
	uint8_t       profilerTag   = 0;                       // 17
	uint16_t      limiter  = 0;                       // 18: its limiter's index + 1 (0 none)
	DedicatedId   thread   = {};                      // 20
	CounterHandle signals  = kInvalidCounter;         // 24
	Priority      priority = Priority::Normal;        // 28
	// It owns a callable (Submit(F) with a non-trivial or large one): the
	// destroy function sits at the end of inlineData (kOwnedDestroyOffset).
	bool          owned    = false;                   // 29
	uint16_t      isolation = 0;                      // 30: the Isolated region it belongs to (0 none)
	uint32_t      nameId   = 0;                       // 32
	// The next task of a group (a queue entry with kGroupBit), written by the
	// submitter before the push and read by whoever takes the group. While a
	// task waits on counters, its count of unmet ones (see PendingDeps).
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
// An owned callable takes inlineData's first kOwnedInlineSize bytes; its
// destroy function the rest.
static constexpr size_t kOwnedDestroyOffset = detail::kOwnedInlineSize;
static_assert(kOwnedDestroyOffset + sizeof(TaskFn) <= sizeof(TaskSlot::inlineData), "the destroy pointer fits after the callable");

// A slot index is (chunk << 12) | offset. Chunks are allocated whole and never
// freed or moved, so a slot's address is stable for the pool's lifetime and
// a thread can hold a TaskSlot& without a lock.
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
// counter's Count never lags far behind.
static constexpr int kMaxDeferredSignals = 64;

// ---- Waiting ------------------------------------------------------------------

enum : uint32_t
{
	kArmed         = 0, // asleep (or about to be)
	kWokeSearching = 1, // an idle worker woken to look for work; the waker counted it as searching
	kWokeRecheck   = 2, // anything else: re-check whatever you were waiting for
};

// The sleep primitive: a raw futex on Linux, a mutex and condition variable
// elsewhere (LOOM_NO_FUTEX forces the portable one, so it can be tested on
// Linux too).
#if defined(__linux__) && !defined(LOOM_NO_FUTEX)
#define LOOM_FUTEX 1
#else
#define LOOM_FUTEX 0
#endif

struct Waiter
{
	std::atomic<uint32_t> word{ kArmed };
	bool isWorker = false; // a worker's idle sleep (not a helper's)
	int threadIndex = -1;  // who sleeps here, for the debug dump (set before registering)
#if !LOOM_FUTEX
	std::mutex              mutex;
	std::condition_variable cv;
#endif
};

// One per thread: a thread sleeps in at most one place at a time.
static thread_local Waiter t_Waiter;

// When a timed sleep gives up; kNoDeadline for an untimed one.
using Deadline = std::chrono::steady_clock::time_point;
static constexpr Deadline kNoDeadline = Deadline::max();

// On Linux a raw futex on the word: the kernel re-checks the word under its
// own lock, so a store-then-wake can never be missed. (std::atomic::wait and
// notify keep their own waiter bookkeeping in the library, which loom cannot
// see into or test against, and have no timeout.)
#if LOOM_FUTEX
static_assert(sizeof(std::atomic<uint32_t>) == sizeof(uint32_t), "futex word layout");
static void FutexWait(std::atomic<uint32_t>& word, uint32_t expected, const timespec* timeout)
{
	// FUTEX_WAIT's timeout is relative.
	syscall(SYS_futex, reinterpret_cast<uint32_t*>(&word), FUTEX_WAIT_PRIVATE, expected, timeout, nullptr, 0);
}
static void FutexWakeOne(std::atomic<uint32_t>& word)
{
	syscall(SYS_futex, reinterpret_cast<uint32_t*>(&word), FUTEX_WAKE_PRIVATE, 1, nullptr, nullptr, 0);
}
#endif

// Wakers call this while holding the lock of the list they found the waiter
// in; the waiter takes that lock to unregister before it returns, so the
// Waiter can never be notified after it has moved on.
static void Wake(Waiter& w, uint32_t why)
{
#if LOOM_FUTEX
	w.word.store(why, std::memory_order_release);
	FutexWakeOne(w.word);
#else
	// The store under the waiter's mutex: it cannot fall between the
	// sleeper's check of the word and its wait.
	std::lock_guard<std::mutex> lock(w.mutex);
	w.word.store(why, std::memory_order_release);
	w.cv.notify_one();
#endif
}

// The caller arms the word (kArmed), registers wherever a waker will look,
// re-checks its condition, and only then sleeps here. A wake that lands
// between the re-check and the sleep is not lost: FUTEX_WAIT returns at once
// when the word is no longer kArmed (the portable path checks it under the
// mutex the waker stores under). The loops absorb spurious wakeups. Returns
// false when the deadline passed first.
static bool SleepOn(Waiter& w, Deadline deadline = kNoDeadline)
{
#if LOOM_FUTEX
	while (w.word.load(std::memory_order_acquire) == kArmed)
	{
		if (deadline == kNoDeadline)
		{
			FutexWait(w.word, kArmed, nullptr);
			continue;
		}
		const auto left = deadline - std::chrono::steady_clock::now();
		if (left <= std::chrono::steady_clock::duration::zero())
			return false;
		const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(left).count();
		const timespec rel{ (time_t)(ns / 1000000000), (long)(ns % 1000000000) };
		FutexWait(w.word, kArmed, &rel);
	}
	return true;
#else
	std::unique_lock<std::mutex> lock(w.mutex);
	auto woken = [&] { return w.word.load(std::memory_order_acquire) != kArmed; };
	if (deadline == kNoDeadline)
	{
		w.cv.wait(lock, woken);
		return true;
	}
	return w.cv.wait_until(lock, deadline, woken);
#endif
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

	// Owner only. LIFO: the newest entry, still warm in this core's cache.
	uint32_t Pop()
	{
		const int64_t b = m_Bottom.load(std::memory_order_relaxed) - 1;
		Buffer* a = m_Buffer.load(std::memory_order_relaxed);
		// Claim the bottom element BEFORE reading top. With both seq_cst, this
		// store and a thief's top CAS are totally ordered: either the thief
		// sees the lowered bottom and backs off, or we see its raised top.
		// Only for the very last element can both get past that, hence the CAS.
		m_Bottom.store(b, std::memory_order_seq_cst);
		int64_t t = m_Top.load(std::memory_order_seq_cst);
		if (t > b)
		{
			// Empty: undo the claim.
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

	// Any thread. FIFO: the oldest entry, which in a recursive split is the
	// biggest piece of work. kNoSlot when empty or when another thief (or the
	// owner's Pop of the last element) won the element.
	uint32_t Steal()
	{
		// Top before bottom: the mirror of Pop's bottom-then-top.
		int64_t t = m_Top.load(std::memory_order_seq_cst);
		const int64_t b = m_Bottom.load(std::memory_order_seq_cst);
		if (t >= b)
			return kNoSlot;
		// Read the value before the CAS: once top moves past it, the owner may
		// overwrite that cell. A stale buffer is fine (old buffers stay alive),
		// and a value read from it is used only if the CAS proves it current.
		Buffer* a = m_Buffer.load(std::memory_order_acquire);
		const uint32_t v = a->Get(t);
		if (!m_Top.compare_exchange_strong(t, t + 1, std::memory_order_seq_cst, std::memory_order_relaxed))
			return kNoSlot;
		return v;
	}

	// Empty and Size are snapshots: another thread may push or take between
	// the two loads. Fine for "is there work worth looking for" and for stats.
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
	// A ring indexed by the ever-growing top/bottom positions; the capacity is
	// a power of two, so a position maps to its cell with a mask. Cells are
	// relaxed atomics only so that a thief's racy read is not a data race; the
	// top/bottom operations above order them.
	struct Buffer
	{
		explicit Buffer(int64_t c) : capacity(c), mask(c - 1), data(new std::atomic<uint32_t>[(size_t)c]) {}
		uint32_t Get(int64_t i) const { return data[(size_t)(i & mask)].load(std::memory_order_relaxed); }
		void Put(int64_t i, uint32_t v) { data[(size_t)(i & mask)].store(v, std::memory_order_relaxed); }
		int64_t capacity;
		int64_t mask;
		std::unique_ptr<std::atomic<uint32_t>[]> data;
	};

	// Separate lines: thieves write top, the owner writes bottom on every
	// push and pop.
	alignas(64) std::atomic<int64_t> m_Top{ 0 };
	alignas(64) std::atomic<int64_t> m_Bottom{ 0 };
	std::atomic<Buffer*> m_Buffer{ nullptr };
	int64_t m_CachedTop = 0;                        // owner only
	std::vector<std::unique_ptr<Buffer>> m_Buffers; // owner only
};

// ---- Counters ------------------------------------------------------------------
// `count` holds the user's count PLUS ONE while it is nonzero: the decrement
// that takes the user's count to zero leaves 1 behind, then, under the lock,
// stores 0, releases the waiting tasks and wakes the waiters. A task released
// that way, or a waiter, may see zero before the final signaller has let go
// of the lock, so DestroyCounter takes the lock first: the counter outlives
// that signaller. 0 means "zero and settled".

struct alignas(64) CounterState
{
	std::atomic<int>      count{ 0 };
	std::mutex            mutex;
	std::vector<uint32_t> pending; // slots waiting for zero
	std::vector<Waiter*>  waiters; // threads asleep in WaitFor
	// The first exception a task signalling this counter threw (under the
	// lock); `failed` lets WaitFor check without it.
	std::exception_ptr    error;
	// Read lock-free by ResolveCounter on every thread while Create/Destroy
	// write them (the old plain fields were a data race the TSAN suite caught).
	// Their own cache line, away from the count every signal writes: a task
	// reads `cancelled` before it runs.
	alignas(64) std::atomic<bool> alive{ false };
	std::atomic<uint32_t> generation{ 0 };
	std::atomic<bool>     failed{ false };
	std::atomic<bool>     cancelled{ false }; // Cancel, or a task's exception
};

// ---- Limiters ------------------------------------------------------------------
// `count` is the free units less the tasks that wait for one. A task takes a
// unit with one decrement while it is positive; at zero or below it waits,
// queueing itself under the lock. A finishing task gives its unit back with
// one increment, and only when that finds someone waiting (count was
// negative) does it take the lock and hand the unit over: to the queue's
// first task, or, when the waiter has decremented but not yet queued itself,
// as a handoff that the waiter takes instead of queueing.

//
// Lowering the limit takes free units out at once and, for units that are
// held, leaves "retiring" debts: a finishing task that finds one pays it with
// its unit instead of handing it on. Raising it gives the new units back as
// finishing tasks would. Unit indices (CurrentLimiterUnit) are handed out lazily from
// a stack of free ones below the limit; a task over a lowered limit may get
// one above it, which is dropped again when it finishes.

struct alignas(64) LimiterState
{
	std::atomic<int>     count{ 0 };
	std::mutex           mutex;
	std::deque<uint32_t> waiting;      // slots, High first, under mutex
	uint32_t             handoffs = 0; // units left for a waiter not yet queued, under mutex
	std::atomic<uint32_t> limit{ 0 };    // written under mutex
	std::atomic<uint32_t> retiring{ 0 }; // held units to drop when they come back, written under mutex
	std::vector<uint32_t> freeUnits;     // unit indices below the limit nobody holds, under mutex
	std::vector<uint8_t>  unitInUse;     // by index, under mutex
	// Read lock-free by ResolveLimiter, as a counter's are.
	alignas(64) std::atomic<bool> alive{ false };
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
	DedicatedId            id = {};
	// Its CurrentThreadIndex, once the thread has started: the key of its
	// stats row (names need not be unique).
	std::atomic<int>       threadIndex{ -1 };
};

// ---- Per-thread scheduler state (the main thread and each worker) ------------------

struct alignas(64) WorkerCtx
{
	WorkDeque deque; // Normal priority (and every group)
	WorkDeque high;
	WorkDeque low;
	// This thread's share of the live-task count: +1 per task it submits, -1
	// per task it finishes. Only the owner writes it; LiveTaskCount sums them.
	alignas(64) std::atomic<int64_t> live{ 0 };
	// xorshift state for picking a first steal victim. Seeded per worker so
	// the thieves start their scans at different deques.
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

// What a thread is doing, for CaptureDebugState: written only by its own
// thread with relaxed stores, on a line of its own, and read by whoever
// captures. A task run inside a helping wait saves the outer values and
// puts them back, so the innermost task and wait are the ones shown.
struct alignas(64) ThreadDebug
{
	std::atomic<uint64_t>      tasksRun{ 0 };
	std::atomic<int>           running{ 0 };
	std::atomic<uint32_t>      runningName{ 0 };
	std::atomic<CounterHandle> runningSignals{ kInvalidCounter };
	std::atomic<uint16_t>      runningLimiter{ 0 }; // the limiter's index + 1, as in the slot
	std::atomic<CounterHandle> waitingOn{ kInvalidCounter };
	std::atomic<int64_t>       waitingSince{ 0 };   // steady_clock ticks
};

struct alignas(64) PaddedStats
{
	ThreadStats s;
	// s.name points here: the row outlives the thread (a dedicated thread's
	// own name string dies when it is unregistered).
	std::string name;
	ThreadDebug debug;
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

	// Counters: a fixed table and a free list. Their own cache line: every
	// task reads `counters` (to check its counter is not cancelled), and the
	// free-slot head above is written all the time.
	alignas(64) std::mutex  counterMutex;
	std::unique_ptr<CounterState[]> counters;
	uint32_t                counterCount = 0;
	std::vector<uint32_t>   freeCounters;

	// Limiters, the same way. The table never moves: a task's slot keeps its
	// limiter's index.
	std::mutex              limiterMutex;
	std::unique_ptr<LimiterState[]> limiters;
	uint32_t                limiterCount = 0;
	std::vector<uint32_t>   freeLimiters;

	// false from the first instruction of Shutdown (TryHelp, PumpMainThreadTasks
	// and HasActivePool then report the pool as down while it drains).
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
	std::deque<uint32_t>    injectQueue[3]; // by Priority
	std::atomic<int>        injectSize{ 0 }; // all three

	// High and Low entries queued anywhere (deques and injection): a search
	// looks at those queues only when there is something in them, so Normal
	// work, the default, pays nothing for priorities.
	alignas(64) std::atomic<int> queuedHigh{ 0 };
	alignas(64) std::atomic<int> queuedLow{ 0 };

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
	// Ids are handed out once: a stale id never reaches a newer thread.
	uint32_t                nextDedicatedId = 0;
	std::atomic<int>        lanesHeld{ 0 };

	// Thread stats (the pool visualizer). Rows are only ever added until
	// Shutdown, so t_ThreadStats pointers stay valid while their thread runs.
	std::mutex              statsMutex;
	std::vector<std::unique_ptr<PaddedStats>> threadStats;
	std::vector<std::thread> workers;
	int                     workerCount = 0;
	// The main thread has no stats row; its debug record lives here.
	ThreadDebug             mainDebug;
	int                     mainThreadIndex = 0;

	// Runs first on every pool thread.
	void (*onThreadStart)() = nullptr;
	uint32_t                shutdownTimeoutMs = 5000;

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

	// The next Isolated region's tag (16 bits, 0 skipped: a tag is reused
	// only after 65535 regions, and a clash only widens what a wait helps with).
	std::atomic<uint16_t>   nextIsolation{ 1 };
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

// The indices of threads that have exited, for new threads to take (the
// lowest first), and the next index never handed out. Made once and never
// destroyed: a thread may exit after main has returned.
struct ThreadIndices
{
	std::mutex       mutex;
	std::vector<int> free; // a min-heap
	int              next = 0;
};
static ThreadIndices& TheThreadIndices()
{
	static ThreadIndices* indices = new ThreadIndices();
	return *indices;
}

// A worker's own state. Worker threads die with their pool, so a stale value
// can never point at a later pool.
static thread_local PoolState*      t_WorkerPool = nullptr;
static thread_local WorkerCtx*      t_WorkerCtx = nullptr;
static thread_local DedicatedState* t_CurrentDedicated = nullptr;
// A pool thread's stats row (workers and dedicated threads; null elsewhere).
static thread_local ThreadStats*    t_ThreadStats = nullptr;
// That row's debug record (null elsewhere; the main thread's is in the pool).
static thread_local ThreadDebug*    t_Debug = nullptr;
// The Isolated region the calling thread is in (0: none). A running task's
// thread is in that task's region, so its children inherit it.
static thread_local uint16_t        t_Isolation = 0;
// The running task's limiter tag (a RunSlot local), so a task that completes
// its own counter from inside, as a future's does, can give its unit back
// first. Null outside a task; saved and restored around nested runs.
static thread_local uint16_t*       t_RunningLimiter = nullptr;
// Its unit index once CurrentLimiterUnit asked for one (kNoUnitIndex before), beside it.
static thread_local uint32_t*       t_RunningUnit = nullptr;

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

// The calling thread's debug record: a pool thread's own, the main
// thread's, or none (a thread outside the pool shows up only as a sleeper
// on the counters it waits for).
static ThreadDebug* DebugRow(PoolState& P, WorkerCtx* ctx)
{
	if (t_Debug)
		return t_Debug;
	return ctx && ctx->isMain ? &P.mainDebug : nullptr;
}

// The region a wait on this thread is isolated to. None without workers:
// nobody else could run the work outside it.
static uint16_t ActiveIsolation(PoolState& P)
{
	return P.workerCount > 0 ? t_Isolation : 0;
}

// The live-task count is sharded so that submitting and finishing write only
// the thread's own line. A task submitted on one thread and finished on
// another leaves one shard +1 and the other -1; only the sum means anything.
static std::atomic<int64_t>& LiveShard(PoolState& P, WorkerCtx* ctx)
{
	return ctx ? ctx->live : P.externalLive;
}

// Not a snapshot: the shards are read one after another while tasks move, so
// the sum can be briefly off (never below zero once everything settles).
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

static uint64_t MarkProfilerZone(PoolState& P, int threadIndex, uint32_t nameId, uint8_t profilerTag, const char* name, bool begin, uint64_t handle)
{
	if (ProfilerZoneFn fn = P.profilerZoneHook.load(std::memory_order_acquire))
		return fn(threadIndex, nameId, profilerTag, name, begin, handle);
	return 0;
}

// Counter handle packing: a 20-bit table index (up to ~1M counters) and a
// 12-bit generation above it. Destroying a counter bumps its generation, so
// an old handle to a reused index no longer resolves.
static constexpr uint32_t kCounterGenBits = 20;
static constexpr uint32_t kCounterIndexMask = (1u << kCounterGenBits) - 1;

static CounterHandle PackCounter(uint32_t index, uint32_t generation)
{
	return CounterHandle{ (generation << kCounterGenBits) | index };
}
// Limiter handles pack the same way.
static LimiterHandle PackLimiter(uint32_t index, uint32_t generation)
{
	return LimiterHandle{ (generation << kCounterGenBits) | index };
}
template <typename H>
static uint32_t HandleIndex(H h) { return (uint32_t)h & kCounterIndexMask; }
template <typename H>
static uint32_t HandleGeneration(H h) { return (uint32_t)h >> kCounterGenBits; }

// Handle -> state, or null for a stale or invalid handle. The checks are
// relaxed and lock-free: they catch misuse (asserted by the callers), they do
// not make a racing Destroy safe. Using a counter while another thread
// destroys it is a bug either way.
static CounterState* ResolveCounter(PoolState& P, CounterHandle h)
{
	if (h == kInvalidCounter) return nullptr;
	uint32_t idx = HandleIndex(h);
	if (idx >= P.counterCount) return nullptr;
	CounterState& c = P.counters[idx];
	if (!c.alive.load(std::memory_order_relaxed) || HandleGeneration(h) != c.generation.load(std::memory_order_relaxed)) return nullptr;
	return &c;
}

// Handle -> state, or null, as ResolveCounter (limiter handles pack the same way).
static LimiterState* ResolveLimiter(PoolState& P, LimiterHandle h)
{
	if (h == kInvalidLimiter) return nullptr;
	const uint32_t idx = HandleIndex(h);
	if (idx >= P.limiterCount) return nullptr;
	LimiterState& l = P.limiters[idx];
	if (!l.alive.load(std::memory_order_relaxed) || HandleGeneration(h) != l.generation.load(std::memory_order_relaxed)) return nullptr;
	return &l;
}

// What a slot stores for a task's limiter: its index + 1, or 0 for none.
static uint16_t LimiterTag(PoolState& P, LimiterHandle h)
{
	if (h == kInvalidLimiter)
		return 0;
	LimiterState* l = ResolveLimiter(P, h);
	assert(l && "task uses an invalid/destroyed limiter");
	return l ? uint16_t(l - P.limiters.get() + 1) : 0;
}

// True when the task got a unit; false when it now waits in the limiter.
static bool AcquireUnit(LimiterState& l, uint32_t slot, Priority priority)
{
	if (l.count.fetch_sub(1, std::memory_order_acq_rel) > 0)
		return true;
	std::lock_guard<std::mutex> lock(l.mutex);
	if (l.handoffs > 0)
	{
		l.handoffs--;
		return true;
	}
	if (priority == Priority::High)
		l.waiting.push_front(slot);
	else
		l.waiting.push_back(slot);
	return false;
}

// ---- Test hooks --------------------------------------------------------------------
// With LOOM_TEST_HOOKS (loom's own test build), a test can install a callback
// that runs at the points below, to hold a thread there and widen a race.
// Elsewhere the points compile to nothing.
#if defined(LOOM_TEST_HOOKS)
namespace detail {
using TestHookFn = void (*)(int point);
static std::atomic<TestHookFn> s_TestHook{ nullptr };
void SetTestHook(TestHookFn fn) { s_TestHook.store(fn, std::memory_order_release); }
} // namespace detail
#define LOOM_TEST_POINT(point) \
	do { if (detail::TestHookFn hook = detail::s_TestHook.load(std::memory_order_relaxed)) hook(point); } while (0)
#else
#define LOOM_TEST_POINT(point) do { (void)(point); } while (0)
#endif
// A limiter unit has just become free for anyone to take.
static constexpr int kTestPointUnitFreed = 1;
// A task's function has returned; its runner has given nothing back yet.
static constexpr int kTestPointTaskReturned = 2;

// Gives a unit back. Returns the waiting task it went to (which must now be
// queued), or kNoSlot.
static uint32_t ReleaseUnit(LimiterState& l)
{
	if (l.count.fetch_add(1, std::memory_order_acq_rel) >= 0)
	{
		LOOM_TEST_POINT(kTestPointUnitFreed);
		return kNoSlot;
	}
	std::lock_guard<std::mutex> lock(l.mutex);
	const uint32_t retiring = l.retiring.load(std::memory_order_relaxed);
	if (retiring > 0)
	{
		l.retiring.store(retiring - 1, std::memory_order_relaxed); // the limit was lowered: drop it
		return kNoSlot;
	}
	if (l.waiting.empty())
	{
		l.handoffs++;
		return kNoSlot;
	}
	const uint32_t next = l.waiting.front();
	l.waiting.pop_front();
	return next;
}

static constexpr uint32_t kNoUnitIndex = ~0u;

// A unit index for a task holding a unit: a free one below the limit, or,
// when a lowered limit has more holders than indices, the lowest one unused.
static uint32_t TakeUnitIndex(LimiterState& l)
{
	std::lock_guard<std::mutex> lock(l.mutex);
	uint32_t index;
	if (!l.freeUnits.empty())
	{
		index = l.freeUnits.back();
		l.freeUnits.pop_back();
	}
	else
	{
		index = 0;
		while (index < l.unitInUse.size() && l.unitInUse[index])
			index++;
		if (index == l.unitInUse.size())
			l.unitInUse.push_back(0);
	}
	l.unitInUse[index] = 1;
	return index;
}

static void ReturnUnitIndex(LimiterState& l, uint32_t index)
{
	std::lock_guard<std::mutex> lock(l.mutex);
	l.unitInUse[index] = 0;
	if (index < l.limit.load(std::memory_order_relaxed))
		l.freeUnits.push_back(index);
}

// ---- Forward decls --------------------------------------------------------------

static uint32_t AllocateSlot(PoolState& P, WorkerCtx* ctx);
static bool     EnqueueSlot(PoolState& P, uint32_t slot, WorkerCtx* ctx);
static void     RunSlot(PoolState& P, uint32_t slot, WorkerCtx* ctx);
static void     RunEntry(PoolState& P, uint32_t entry, WorkerCtx* ctx);
#if defined(__cpp_exceptions)
static void     RecordTaskException(PoolState& P, CounterHandle signals, std::exception_ptr error);
#endif
static void     RunTaskFn(PoolState& P, TaskFn fn, void* userData, CounterHandle signals);
static void     ReleaseRunningUnit(PoolState& P, WorkerCtx* ctx);
static void     SignalCounter(PoolState& P, CounterHandle h, int n);
static void     FlushDeferredSignals(PoolState& P, WorkerCtx* ctx);
static void     WakeForWork(PoolState& P, size_t n);
static bool     TryHelpIn(PoolState& P);
static void     ParkHere(PoolState& P, int index, const char* name);
static void     WorkerLoop(PoolState& P, WorkerCtx& ctx);
static void     DedicatedLoop(PoolState& P, DedicatedState& d);
static PaddedStats* RegisterThreadStats(PoolState& P, int index, const char* name);

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
// trades a whole batch per CAS; others trade one slot.
//
// The stack's head carries a 32-bit tag against ABA: without it, a popper
// that read head A and next B could be preempted while A is popped, B is
// popped, and A is pushed back; its CAS would then succeed and install B,
// which is no longer free. Every push and pop bumps the tag, so that CAS fails.

static uint8_t* SlotBytes(TaskSlot& s) { return reinterpret_cast<uint8_t*>(&s); }

// Writes a batch head: its count, and the other members' indices into the
// two free regions of its first line (low first, then from byte 48).
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

// The head of a batch, or kNoSlot when the stack is empty. Acquire on success
// pairs with the pusher's release, so the batch list written before the push
// is visible to TakeBatch.
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

// A batch off the shared stack, growing the pool (with a warning) when it is
// empty. The emptiness check is repeated under growMutex so that several
// threads finding it empty at once grow it once, not once each.
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

// A free slot for a new task. The common case (a worker or the main thread
// with slots cached) is a plain array pop: no atomics at all.
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

// A finished task's slot. A full cache spills one batch, not all of it, so a
// thread that alternates between submitting and finishing keeps a batch on
// hand instead of trading with the shared stack every time.
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

Pool::Pool(const PoolConfig& config)
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
	P.mainThreadIndex = CurrentThreadIndex();

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
	{
		std::lock_guard<std::mutex> lock(P.limiterMutex);
		P.limiterCount = std::min(std::max(1u, config.maxLimiters), 0xFFFFu); // a slot keeps index + 1 in 16 bits
		P.limiters = std::make_unique<LimiterState[]>(P.limiterCount);
		P.freeLimiters.reserve(P.limiterCount);
		for (uint32_t i = P.limiterCount; i-- > 0;)
			P.freeLimiters.push_back(i);
	}

	P.onThreadStart = config.onThreadStart;
	P.shutdownTimeoutMs = config.shutdownTimeoutMs;
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

// Empties the idle list, waking everyone on it to re-check (Shutdown, ParkAll).
static void WakeAllIdle(PoolState& P)
{
	std::lock_guard<std::mutex> lock(P.idleMutex);
	for (Waiter* w : P.idle)
		Wake(*w, kWokeRecheck);
	P.idle.clear();
	P.numIdle.store(0, std::memory_order_seq_cst);
}

// Shutdown's drain: runs until no task is live, the calling thread helping
// (main-thread work first when it is the main thread). Gives up, with a
// report, when no task finishes for shutdownTimeoutMs: what is left can never
// run (tasks waiting on counters nobody will signal), and the drain-or-assert
// checks that follow name it.
static void DrainForShutdown(PoolState& P)
{
	WorkerCtx* ctx = CurrentCtx(P);
	int64_t last = LiveTasksIn(P);
	auto lastProgress = std::chrono::steady_clock::now();
	for (;;)
	{
		const int64_t live = LiveTasksIn(P);
		if (live <= 0)
			return;
		if (live != last)
		{
			last = live;
			lastProgress = std::chrono::steady_clock::now();
		}
		if (TryHelpIn(P))
			continue;
		FlushDeferredSignals(P, ctx);
		if (std::chrono::steady_clock::now() - lastProgress > std::chrono::milliseconds(P.shutdownTimeoutMs))
		{
			std::fprintf(stderr, "loom: Shutdown: %lld task(s) did not finish within %u ms of the last one that did "
				"(waiting on a counter nobody signals?)\n", (long long)live, P.shutdownTimeoutMs);
			return;
		}
		std::this_thread::sleep_for(std::chrono::microseconds(100));
	}
}

void Pool::Shutdown()
{
	PoolState& P = *m_State;
	if (!P.running) return;
	P.running = false; // HasActivePool and PumpMainThreadTasks report the pool as down from here

	// Drain first, with every thread still running: a task may still submit
	// work (main-thread work too) and wait for it, and every wait completes.
	DrainForShutdown(P);

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

	// Lanes join like every dedicated thread; a still-held lane at
	// shutdown is a leak (the drain-or-assert policy).
	assert(P.lanesHeld.load() == 0 && "loom lanes still held at shutdown");

	// Drain-or-assert: the queues must be empty at shutdown.
	{
		std::lock_guard<std::mutex> lock(P.injectMutex);
		for (auto& q : P.injectQueue)
		{
			assert(q.empty() && "AnyWorker tasks abandoned at shutdown");
			(void)q;
		}
	}
	for (WorkerCtx* c : P.deques)
	{
		assert(c->deque.Empty() && c->high.Empty() && c->low.Empty() && "AnyWorker tasks abandoned at shutdown");
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

bool HasActivePool()
{
	PoolState* pool = s_ActivePool.load(std::memory_order_acquire);
	return pool && pool->running;
}

int WorkerCount()
{
	PoolState* pool = s_ActivePool.load(std::memory_order_acquire);
	return pool ? pool->workerCount : 0;
}

// The calling thread's index: -1 before its first call, and kReturnedIndex
// once its exit has given the index back.
static constexpr int kReturnedIndex = -2;
static thread_local int t_ThreadIndex = -1;

// Gives the thread's index back when the thread exits. The thread's writes
// before then (to its PerThread values, its stats row) happen before the next
// thread that takes the index reads them, through the allocator's mutex.
struct ThreadIndexReturn
{
	int index = -1;
	~ThreadIndexReturn()
	{
		ThreadIndices& indices = TheThreadIndices();
		std::lock_guard<std::mutex> lock(indices.mutex);
		indices.free.push_back(index);
		std::push_heap(indices.free.begin(), indices.free.end(), std::greater<int>());
		t_ThreadIndex = kReturnedIndex;
	}
};

// Assigned on a thread's first call and kept for the thread's life; the
// lock is only taken then.
int CurrentThreadIndex()
{
	if (t_ThreadIndex >= 0)
		return t_ThreadIndex;
	const bool exiting = t_ThreadIndex == kReturnedIndex;
	int index;
	{
		ThreadIndices& indices = TheThreadIndices();
		std::lock_guard<std::mutex> lock(indices.mutex);
		if (!indices.free.empty())
		{
			std::pop_heap(indices.free.begin(), indices.free.end(), std::greater<int>());
			index = indices.free.back();
			indices.free.pop_back();
		}
		else
			index = indices.next++;
	}
	t_ThreadIndex = index;
	// A call from a thread-local destructor that runs after the index went
	// back gets one of its own, which is never given back: t_Return is
	// already destroyed and may not be used again.
	if (!exiting)
	{
		static thread_local ThreadIndexReturn t_Return;
		t_Return.index = index;
	}
	return index;
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
		n += c->deque.Size() + c->high.Size() + c->low.Size();
	return n;
}

// Holds at most dedicatedMutex plus one thread's mutex at a time, in the order
// EnqueueSlot takes them.
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

// A sleeper's re-check after it registered as idle. seq_cst so it cannot be
// ordered before that registration (see WakeForWork for the pairing).
static bool HasPoolWork(PoolState& P)
{
	if (P.injectSize.load(std::memory_order_seq_cst) > 0)
		return true;
	for (WorkerCtx* c : P.deques)
		if (!c->deque.Empty() || !c->high.Empty() || !c->low.Empty())
			return true;
	return false;
}

static WorkDeque& DequeFor(WorkerCtx& c, Priority priority)
{
	return priority == Priority::High ? c.high : priority == Priority::Low ? c.low : c.deque;
}

// Called after an entry of this priority left a queue.
static void NoteTaken(PoolState& P, Priority priority)
{
	if (priority == Priority::High)
		P.queuedHigh.fetch_sub(1, std::memory_order_relaxed);
	else if (priority == Priority::Low)
		P.queuedLow.fetch_sub(1, std::memory_order_relaxed);
}

// The oldest injected entry of this priority. The size check keeps an empty
// queue off the lock entirely: every search on every worker passes through here.
static uint32_t PopInjected(PoolState& P, Priority priority)
{
	if (P.injectSize.load(std::memory_order_relaxed) <= 0)
		return kNoSlot;
	std::lock_guard<std::mutex> lock(P.injectMutex);
	auto& q = P.injectQueue[(int)priority];
	if (q.empty())
		return kNoSlot;
	const uint32_t slot = q.front();
	q.pop_front();
	P.injectSize.fetch_sub(1, std::memory_order_relaxed);
	return slot;
}

// The injection queue, then every other thread's deque of this priority,
// from a random victim.
static uint32_t StealAt(PoolState& P, WorkerCtx* self, Priority priority, size_t start)
{
	uint32_t slot = PopInjected(P, priority);
	const size_t n = P.deques.size();
	for (size_t k = 0; k < n && slot == kNoSlot; k++)
	{
		WorkerCtx* victim = P.deques[(start + k) % n];
		if (victim != self)
			slot = DequeFor(*victim, priority).Steal();
	}
	if (slot != kNoSlot)
		NoteTaken(P, priority);
	return slot;
}

// This thread's own work: High first, then Normal. (Its Low work waits for
// StealOnce, which takes Low only when nothing else is queued.)
static uint32_t PopOwn(PoolState& P, WorkerCtx& ctx)
{
	if (P.queuedHigh.load(std::memory_order_relaxed) > 0)
	{
		const uint32_t slot = ctx.high.Pop();
		if (slot != kNoSlot)
		{
			NoteTaken(P, Priority::High);
			return slot;
		}
	}
	return ctx.deque.Pop();
}

// One pass for work queued elsewhere: High anywhere, then Normal anywhere,
// then Low (this thread's own first).
static uint32_t StealOnce(PoolState& P, WorkerCtx* self)
{
	// A random first victim (xorshift32), so thieves spread over the deques
	// instead of all hammering the main thread's first. Threads without a
	// context (dedicated, outside) keep their own state.
	static thread_local uint32_t t_rng = 0x2545F491u;
	uint32_t& rng = self ? self->rng : t_rng;
	rng ^= rng << 13;
	rng ^= rng >> 17;
	rng ^= rng << 5;
	const size_t start = rng % P.deques.size();
	uint32_t slot = kNoSlot;
	if (P.queuedHigh.load(std::memory_order_relaxed) > 0)
		slot = StealAt(P, self, Priority::High, start);
	if (slot == kNoSlot)
		slot = StealAt(P, self, Priority::Normal, start);
	if (slot == kNoSlot && P.queuedLow.load(std::memory_order_relaxed) > 0)
	{
		if (self && (slot = self->low.Pop()) != kNoSlot)
			NoteTaken(P, Priority::Low);
		else
			slot = StealAt(P, self, Priority::Low, start);
	}
	return slot;
}

// The idle list is a plain vector under a mutex: it changes only when a
// thread goes to sleep or is woken, never on the submit/run fast path.
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
	// From the back: the most recent sleeper, whose cache is likeliest still warm.
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

// Queues an entry by its task's priority (a group is always Normal): onto
// this thread's deque, or the injection queue for a thread without one.
static void PushPool(PoolState& P, uint32_t entry, WorkerCtx* ctx)
{
	const Priority priority = (entry & kGroupBit) ? Priority::Normal : Slot(P, entry).priority;
	// Counted before it is queued, so a taker never counts it out first.
	if (priority == Priority::High)
		P.queuedHigh.fetch_add(1, std::memory_order_relaxed);
	else if (priority == Priority::Low)
		P.queuedLow.fetch_add(1, std::memory_order_relaxed);
	if (ctx)
	{
		DequeFor(*ctx, priority).Push(entry);
		return;
	}
	std::lock_guard<std::mutex> lock(P.injectMutex);
	P.injectQueue[(int)priority].push_back(entry);
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
		if (t.priority == Priority::High)
			P.mainQueue.push_front(slot);
		else
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
		// The registry lock stays held through the push: an unregister frees
		// the thread's state as soon as it is out of the registry.
		std::lock_guard<std::mutex> registry(P.dedicatedMutex);
		DedicatedState* d = nullptr;
		for (auto& dd : P.dedicated)
			if (dd->id == t.thread) { d = dd.get(); break; }
		assert(d && "Dedicated task submitted to an unregistered thread");
		if (!d)
			return false;
		std::lock_guard<std::mutex> lock(d->mutex);
		if (t.priority == Priority::High)
			d->queue.push_front(slot);
		else
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

// Makes a slot runnable once it holds a unit of its limiter, if it has one;
// otherwise it waits in the limiter, whose next released unit queues it.
// True when it went into the pool, as EnqueueSlot.
static bool ReadySlot(PoolState& P, uint32_t slot, WorkerCtx* ctx)
{
	const TaskSlot& t = Slot(P, slot);
	if (t.limiter != 0 && !AcquireUnit(P.limiters[t.limiter - 1], slot, t.priority))
		return false;
	return EnqueueSlot(P, slot, ctx);
}

// A waiting task's count of unmet dependencies. It lives in groupNext, which
// only a runnable task uses (a task with dependencies is never grouped).
static std::atomic_ref<uint32_t> PendingDeps(PoolState& P, uint32_t slot)
{
	return std::atomic_ref<uint32_t>(Slot(P, slot).groupNext);
}

static bool HasDependencies(const TaskDesc& desc)
{
	return desc.waitsFor != kInvalidCounter || !desc.waitsForAll.empty();
}

// Parks the slot on each of its counters that is not yet zero, or enqueues it
// when none is. True when it went into the pool (AnyWorker, runnable now).
static bool DispatchSlot(PoolState& P, uint32_t slot, const TaskDesc& desc, WorkerCtx* ctx)
{
	if (!HasDependencies(desc))
		return ReadySlot(P, slot, ctx);

	// One hold of our own while registering, so the task cannot be released
	// (by a counter reaching zero meanwhile) before every counter has it.
	auto deps = PendingDeps(P, slot);
	deps.store(1, std::memory_order_relaxed);
	auto waitOn = [&](CounterHandle h) {
		if (h == kInvalidCounter)
			return;
		CounterState* c = ResolveCounter(P, h);
		assert(c && "task waits on an invalid/destroyed counter");
		if (!c)
			return;
		// Under the lock the count is either settled (0) or the final
		// signaller has yet to take this lock and will release the list.
		std::lock_guard<std::mutex> lock(c->mutex);
		if (c->count.load(std::memory_order_acquire) != 0)
		{
			deps.fetch_add(1, std::memory_order_relaxed);
			c->pending.push_back(slot);
		}
	};
	waitOn(desc.waitsFor);
	for (CounterHandle h : desc.waitsForAll)
		waitOn(h);
	// Drop our hold. Whoever takes the count to zero (us, or the last
	// counter's final signal) enqueues the task, exactly once.
	if (deps.fetch_sub(1, std::memory_order_acq_rel) == 1)
		return ReadySlot(P, slot, ctx); // every counter was already zero
	return false;
}

// Allocates a slot and fills it from the description; does not queue it.
// The task counts as live from here.
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
	t.priority = desc.priority;
	t.owned = false; // byte 29 was part of the free-batch list
	t.isolation = t_Isolation; // a task belongs to the region it was submitted in
	t.nameId = desc.nameId;
	t.profilerTag = desc.profilerTag;
	t.limiter = LimiterTag(P, desc.limiter);
	LiveShard(P, ctx).fetch_add(1, std::memory_order_relaxed);
	return slot;
}

static bool SubmitOne(PoolState& P, const TaskDesc& desc, WorkerCtx* ctx)
{
	assert(desc.fn && "task without a function");
	const uint32_t slot = PrepareSlot(P, desc, ctx);
	return DispatchSlot(P, slot, desc, ctx);
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
		if (groupSize > 1 && d.affinity == Affinity::AnyWorker && d.priority == Priority::Normal && !HasDependencies(d)
			&& d.limiter == kInvalidLimiter)
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

// Submit(F): the callable's bytes are copied into the slot itself, and the
// task's userData points back at them, so the callable lives exactly as long
// as the slot (RunSlot releases the slot only after the call returns).
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
	if (DispatchSlot(P, slot, opts, ctx))
		WakeForWork(P, 1);
}

void SubmitOwnedRaw(void* src, void (*relocate)(void*, void*), void (*run)(void*), void (*destroy)(void*), const TaskDesc& opts)
{
	PoolState& P = ActivePool();
	assert(opts.fn == nullptr && "Submit(F) and TaskDesc::fn are mutually exclusive");
	WorkerCtx* ctx = CurrentCtx(P);
	const uint32_t slot = PrepareSlot(P, opts, ctx);
	TaskSlot& t = Slot(P, slot);
	void* object = src;
	if (relocate)
	{
		relocate(t.inlineData, src);
		object = t.inlineData;
	}
	std::memcpy(t.inlineData + kOwnedDestroyOffset, &destroy, sizeof destroy);
	t.owned = true;
	t.fn = run;
	t.userData = object;
	if (DispatchSlot(P, slot, opts, ctx))
		WakeForWork(P, 1);
}

} // namespace detail

// ---- ParallelFor ---------------------------------------------------------------

namespace {

	// One fixed chunk. The records live in a vector on the caller's stack
	// frame, which outlives the chunks because ParallelFor waits for them.
	struct ParallelRange { size_t begin = 0, end = 0; void* userData = nullptr; ParallelForFn fn = nullptr; };

	void ParallelRangeTrampoline(void* p)
	{
		ParallelRange& range = *static_cast<ParallelRange*>(p);
		range.fn(range.begin, range.end, range.userData);
	}

} // namespace

// ---- ParallelFor with automatic chunking ---------------------------------------
// The range starts as one piece per thread. A piece works through its range a
// grain at a time, and whenever a worker is idle or searching (someone could
// take more), it splits off its upper half as a new task first. A busy pool
// pays one load per grain; an idle worker gets work within a grain.
// Completion counts ITEMS, so a split needs no counter increment.

namespace {

	struct AutoRange
	{
		size_t begin = 0, end = 0, grain = 1;
		ParallelForFn fn = nullptr;
		void* userData = nullptr;
		CounterHandle items = kInvalidCounter;
	};
	static_assert(sizeof(AutoRange) <= 48 && std::is_trivially_copyable_v<AutoRange>, "an AutoRange travels inline in a task slot");

	// Relaxed and racy on purpose: a wrong answer costs one needless split or
	// one grain of delay, while the check runs once per grain on every piece.
	bool SomeoneCouldTakeWork(PoolState& P)
	{
		return P.numIdle.load(std::memory_order_relaxed) > 0 || P.numSearching.load(std::memory_order_relaxed) > 0;
	}

	void AutoRangeRun(AutoRange r);
	void AutoRangeTask(void* p) { AutoRangeRun(*static_cast<AutoRange*>(p)); }

	// `r` is a local copy: the piece shrinks as it runs and splits, and the
	// record it came from is left untouched.
	void AutoRangeRun(AutoRange r)
	{
		PoolState& P = ActivePool();
		while (r.begin < r.end)
		{
			const size_t left = r.end - r.begin;
			if (IsCancelled(r.items))
			{
				Signal(r.items, (int)left); // a chunk threw: the rest is skipped
				return;
			}
			if (left >= 2 * r.grain && SomeoneCouldTakeWork(P))
			{
				AutoRange upper = r;
				upper.begin = r.begin + left / 2;
				r.end = upper.begin;
				detail::SubmitInlineRaw(&upper, sizeof upper, &AutoRangeTask, TaskDesc{});
				continue;
			}
			const size_t step = std::min(r.grain, left);
#if defined(__cpp_exceptions)
			try
			{
				r.fn(r.begin, r.begin + step, r.userData);
			}
			catch (...)
			{
				RecordTaskException(P, r.items, std::current_exception()); // and cancels the rest
				Signal(r.items, (int)left);
				return;
			}
#else
			r.fn(r.begin, r.begin + step, r.userData);
#endif
			Signal(r.items, (int)step);
			r.begin += step;
		}
	}

} // namespace

static void ParallelForFixed(PoolState& P, size_t begin, size_t end, size_t chunkSize, ParallelForFn fn, void* userData);

static void ParallelForAuto(PoolState& P, size_t begin, size_t end, ParallelForFn fn, void* userData)
{
	const size_t n = end - begin;
	const size_t threads = (size_t)P.workerCount + 1;
	// The count is an int: past that, fall back to fixed chunks.
	if (n >= (size_t)INT_MAX / 2)
	{
		ParallelForFixed(P, begin, end, (n - 1) / (threads * 16) + 1, fn, userData);
		return;
	}
	// About 64 grains per thread: small enough that an idle worker waits at
	// most a sixty-fourth of a thread's share for a split, big enough that the
	// per-grain check is noise. A range smaller than the thread count gets one
	// piece per index.
	const size_t grain = std::max<size_t>(1, n / (threads * 64));
	const size_t pieces = std::min(threads, (n - 1) / grain + 1);
	const CounterHandle items = CreateCounter((int)n);
	struct DestroyOnExit
	{
		CounterHandle h;
		~DestroyOnExit() { DestroyCounter(h); }
	} destroy{ items };

	std::vector<AutoRange> roots(pieces);
	std::vector<TaskDesc> descs(pieces);
	size_t cursor = begin;
	for (size_t i = 0; i < pieces; i++)
	{
		const size_t pieceEnd = i + 1 == pieces ? end : cursor + n / pieces;
		roots[i] = AutoRange{ cursor, pieceEnd, grain, fn, userData, items };
		descs[i] = TaskDesc{ .fn = &AutoRangeTask, .userData = &roots[i] };
		cursor = pieceEnd;
	}
	Submit(std::span<const TaskDesc>(descs));
	WaitFor(items);
}

void ParallelFor(size_t begin, size_t end, size_t chunkSize, ParallelForFn fn, void* userData)
{
	PoolState& P = ActivePool(); // the no-pool assert even for an empty range
	if (begin >= end || fn == nullptr)
		return;
	if (chunkSize == 0)
		ParallelForAuto(P, begin, end, fn, userData);
	else
		ParallelForFixed(P, begin, end, chunkSize, fn, userData);
}

static void ParallelForFixed(PoolState& P, size_t begin, size_t end, size_t chunkSize, ParallelForFn fn, void* userData)
{
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

	// Destroyed on the way out even when WaitFor rethrows a chunk's exception
	// (the count is zero by then).
	struct DestroyOnExit
	{
		CounterHandle h;
		~DestroyOnExit() { DestroyCounter(h); }
	} destroy{ counter };
	WaitFor(counter);
}

// ---- Task graphs ---------------------------------------------------------------
// The graph keeps its nodes and edges as built and compiles them once per
// change. Nodes with the same set of predecessors share one *wait set*: one
// count of unfinished predecessors, decremented once by each, whose zero
// releases all its members at once. A dense layer (every node after every
// node of the layer before) then costs one decrement per predecessor, on
// one count, instead of one per edge on as many counts as there are nodes.
// The nodes with no predecessor form the root set.
//
// A run resets the counts, creates a counter of the run's nodes, parks a
// finishing task on it and releases the root set. A node's task runs the
// node, decrements each wait set it feeds and queues the members of those it
// takes to zero (in groups, as Submit(span) does), and returns: the task
// runner signals the run's counter, with each thread's usual batching. The
// finishing task, which runs once every node has signalled, ends the run.

namespace detail {

struct GraphState
{
	struct NodeDesc
	{
		TaskFn      fn = nullptr;
		void*       userData = nullptr;
		void      (*destroy)(void*) = nullptr; // an owned callable (Add(F))
		Affinity    affinity = Affinity::AnyWorker;
		DedicatedId thread = {};
		Priority    priority = Priority::Normal;
		uint8_t     profilerTag = 0;
		uint32_t    nameId = 0;
		LimiterHandle limiter = kInvalidLimiter;
	};
	// A node's task userData: which graph, which node.
	struct NodeRef
	{
		GraphState* graph = nullptr;
		uint32_t    index = 0;
	};
	// Each on its own line: every predecessor of the set writes it.
	struct alignas(64) WaitCount
	{
		std::atomic<uint32_t> remaining{ 0 };
	};

	std::vector<NodeDesc> nodes;
	std::vector<std::pair<uint32_t, uint32_t>> edges; // (before, after)

	// Compiled from the above; valid while `compiled`.
	bool compiled = false;
	bool acyclic = true;
	uint32_t rootSet = 0;
	std::vector<uint32_t> feedsBegin;  // node i decrements sets feeds[feedsBegin[i], feedsBegin[i + 1])
	std::vector<uint32_t> feeds;
	std::vector<uint32_t> setPreds;    // per set: its number of predecessors
	std::vector<uint32_t> memberBegin; // set w releases members[memberBegin[w], memberBegin[w + 1])
	std::vector<uint32_t> members;
	std::unique_ptr<NodeRef[]> refs;
	std::unique_ptr<WaitCount[]> waits; // per run, for the sets with more than one predecessor

	// Per run.
	std::atomic<bool> running{ false };
	CounterHandle     signals = kInvalidCounter; // the caller's
	CounterHandle     nodesLeft = kInvalidCounter; // the run's own: one per node

	~GraphState() { DestroyCallables(); }

	void DestroyCallables()
	{
		for (NodeDesc& n : nodes)
			if (n.destroy)
				n.destroy(n.userData);
	}

	void Compile()
	{
		const uint32_t n = (uint32_t)nodes.size();
		// Each node's predecessors, sorted, without repeats: its set's key.
		std::vector<std::vector<uint32_t>> preds(n);
		for (auto [before, after] : edges)
			preds[after].push_back(before);
		for (auto& p : preds)
		{
			std::sort(p.begin(), p.end());
			p.erase(std::unique(p.begin(), p.end()), p.end());
		}
		std::map<std::vector<uint32_t>, uint32_t> setOf;
		std::vector<uint32_t> nodeSet(n);
		std::vector<const std::vector<uint32_t>*> setKey;
		setOf.emplace(std::vector<uint32_t>{}, 0); // the root set is always set 0, maybe empty
		setKey.push_back(&setOf.begin()->first);
		for (uint32_t i = 0; i < n; i++)
		{
			auto [it, added] = setOf.emplace(preds[i], (uint32_t)setKey.size());
			if (added)
				setKey.push_back(&it->first);
			nodeSet[i] = it->second;
		}
		const uint32_t sets = (uint32_t)setKey.size();
		rootSet = 0;

		setPreds.assign(sets, 0);
		memberBegin.assign(sets + 1, 0);
		feedsBegin.assign(n + 1, 0);
		for (uint32_t w = 0; w < sets; w++)
		{
			setPreds[w] = (uint32_t)setKey[w]->size();
			for (uint32_t p : *setKey[w])
				feedsBegin[p + 1]++;
		}
		for (uint32_t i = 0; i < n; i++)
			memberBegin[nodeSet[i] + 1]++;
		for (uint32_t w = 0; w < sets; w++)
			memberBegin[w + 1] += memberBegin[w];
		for (uint32_t i = 0; i < n; i++)
			feedsBegin[i + 1] += feedsBegin[i];
		members.resize(n);
		feeds.resize(feedsBegin[n]);
		{
			std::vector<uint32_t> fill(memberBegin.begin(), memberBegin.end() - 1);
			for (uint32_t i = 0; i < n; i++)
				members[fill[nodeSet[i]]++] = i;
		}
		{
			std::vector<uint32_t> fill(feedsBegin.begin(), feedsBegin.end() - 1);
			for (uint32_t w = 0; w < sets; w++)
				for (uint32_t p : *setKey[w])
					feeds[fill[p]++] = w;
		}

		// Kahn's algorithm over the sets: every node is reached unless there
		// is a cycle.
		std::vector<uint32_t> left(setPreds);
		std::vector<uint32_t> ready(members.begin() + memberBegin[rootSet], members.begin() + memberBegin[rootSet + 1]);
		uint32_t reached = 0;
		while (!ready.empty())
		{
			const uint32_t i = ready.back();
			ready.pop_back();
			reached++;
			for (uint32_t k = feedsBegin[i]; k < feedsBegin[i + 1]; k++)
				if (--left[feeds[k]] == 0)
					for (uint32_t m = memberBegin[feeds[k]]; m < memberBegin[feeds[k] + 1]; m++)
						ready.push_back(members[m]);
		}
		acyclic = reached == n;

		refs.reset(new NodeRef[n]);
		for (uint32_t i = 0; i < n; i++)
			refs[i] = NodeRef{ this, i };
		waits.reset(new WaitCount[sets]);
		compiled = true;
	}
};

} // namespace detail

using detail::GraphState;

static void GraphNodeTask(void* p);

// Queues the members of a set that has just become ready. Runnable AnyWorker
// Normal nodes go in groups, as Submit(span) chains them; the rest one by one.
// Returns how many entries went into the pool.
static size_t QueueGraphSet(PoolState& P, GraphState& g, uint32_t set, WorkerCtx* ctx)
{
	const uint32_t first = g.memberBegin[set], last = g.memberBegin[set + 1];
	const uint32_t groupSize = GroupSizeFor(last - first);
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
	// Live from here, before any of them is queued (and can finish): one
	// add for the set, not one per node.
	LiveShard(P, ctx).fetch_add(last - first, std::memory_order_relaxed);
	const CounterHandle nodesLeft = g.nodesLeft;
	const uint16_t isolation = t_Isolation; // a node belongs to the region it was queued in
	for (uint32_t m = first; m < last; m++)
	{
		const uint32_t i = g.members[m];
		const GraphState::NodeDesc& n = g.nodes[i];
		// PrepareSlot's fields, written straight from the node.
		const uint32_t slot = AllocateSlot(P, ctx);
		TaskSlot& t = Slot(P, slot);
		t.fn = &GraphNodeTask;
		t.userData = &g.refs[i];
		t.affinity = n.affinity;
		t.thread = n.thread;
		t.signals = nodesLeft;
		t.priority = n.priority;
		t.owned = false;
		t.isolation = isolation;
		t.nameId = n.nameId;
		t.profilerTag = n.profilerTag;
		t.limiter = LimiterTag(P, n.limiter);
		if (groupSize > 1 && n.affinity == Affinity::AnyWorker && n.priority == Priority::Normal && t.limiter == 0)
		{
			if (length == 0)
				head = slot;
			else
				Slot(P, tail).groupNext = slot;
			tail = slot;
			if (++length == groupSize)
				pushGroup();
		}
		else
			entries += ReadySlot(P, slot, ctx) ? 1 : 0;
	}
	pushGroup();
	return entries;
}

// Runs once every node of the run has signalled nodesLeft: ends the run.
static void GraphFinishTask(void* p)
{
	GraphState& g = *static_cast<GraphState*>(p);
	const CounterHandle signals = g.signals, nodesLeft = g.nodesLeft;
	// Released by the final signal, which settled the count first; the
	// destroy waits for that signaller to let go of the counter.
	DestroyCounter(nodesLeft);
	// The graph's last touch: from here it may be run again or destroyed.
	g.running.store(false, std::memory_order_release);
	if (signals != kInvalidCounter)
		Signal(signals);
}

static void StartGraphRun(PoolState& P, GraphState& g)
{
	const uint32_t n = (uint32_t)g.nodes.size();
	if (n == 0)
	{
		const CounterHandle signals = g.signals;
		g.running.store(false, std::memory_order_release);
		if (signals != kInvalidCounter)
			SignalCounter(P, signals, 1);
		return;
	}
	// Plain stores: queueing the roots publishes them to whoever runs a
	// node, and every later node is queued by a thread that ran a predecessor.
	const uint32_t sets = (uint32_t)g.setPreds.size();
	for (uint32_t w = 0; w < sets; w++)
		g.waits[w].remaining.store(g.setPreds[w], std::memory_order_relaxed);
	g.nodesLeft = CreateCounter((int)n);
	// High: it only ends the run, and the caller is waiting for that.
	Submit(TaskDesc{ .fn = &GraphFinishTask, .userData = &g, .waitsFor = g.nodesLeft, .priority = Priority::High });
	WorkerCtx* ctx = CurrentCtx(P);
	WakeForWork(P, QueueGraphSet(P, g, g.rootSet, ctx));
}

static void GraphNodeTask(void* p)
{
	const GraphState::NodeRef ref = *static_cast<GraphState::NodeRef*>(p);
	GraphState& g = *ref.graph;
	PoolState& P = ActivePool();
	const CounterHandle signals = g.signals; // fixed for the run
	const CounterState* counter = signals != kInvalidCounter ? ResolveCounter(P, signals) : nullptr;
	if (!(counter && counter->cancelled.load(std::memory_order_acquire)))
	{
		const GraphState::NodeDesc& n = g.nodes[ref.index];
		// Caught here and sent to the caller's counter: the run's own counter
		// is never cancelled, so this task always runs and releases what
		// follows, whether the node ran, threw or was skipped.
		RunTaskFn(P, n.fn, n.userData, signals);
	}

	WorkerCtx* ctx = CurrentCtx(P);
	size_t entries = 0;
	for (uint32_t k = g.feedsBegin[ref.index]; k < g.feedsBegin[ref.index + 1]; k++)
	{
		const uint32_t w = g.feeds[k];
		// A set with one predecessor is ready now: no count to touch.
		if (g.setPreds[w] == 1 || g.waits[w].remaining.fetch_sub(1, std::memory_order_acq_rel) == 1)
			entries += QueueGraphSet(P, g, w, ctx);
	}
	WakeForWork(P, entries);
	// The runner then signals nodesLeft for this node.
}

// Starts a run whose waitsFor was not yet zero, from a task waiting for it.
static void GraphLaunchTask(void* p)
{
	StartGraphRun(ActivePool(), *static_cast<GraphState*>(p));
}

TaskGraph::TaskGraph() : m_State(std::make_unique<GraphState>()) {}

TaskGraph::~TaskGraph()
{
	assert((!m_State || !m_State->running.load(std::memory_order_acquire)) && "TaskGraph destroyed while it runs");
}

TaskGraph::TaskGraph(TaskGraph&&) noexcept = default;

TaskGraph& TaskGraph::operator=(TaskGraph&& other) noexcept
{
	assert((!m_State || !m_State->running.load(std::memory_order_acquire)) && "TaskGraph replaced while it runs");
	m_State = std::move(other.m_State);
	return *this;
}

TaskGraph::Node TaskGraph::Add(const TaskDesc& desc)
{
	return AddOwned(desc.userData, desc.fn, nullptr, desc);
}

TaskGraph::Node TaskGraph::AddOwned(void* object, void (*run)(void*), void (*destroy)(void*), const TaskDesc& opts)
{
	if (!m_State)
		m_State = std::make_unique<GraphState>(); // a moved-from graph is empty and usable
	GraphState& g = *m_State;
	assert(!g.running.load(std::memory_order_acquire) && "TaskGraph changed while it runs");
	assert(run && "graph node without a function");
	assert(opts.signals == kInvalidCounter && opts.waitsFor == kInvalidCounter && opts.waitsForAll.empty()
		&& "a graph node's counters come from the graph");
	assert((opts.affinity != Affinity::Dedicated) || opts.thread != kInvalidDedicated);
	g.nodes.push_back(GraphState::NodeDesc{ .fn = run, .userData = object, .destroy = destroy, .affinity = opts.affinity,
		.thread = opts.thread, .priority = opts.priority, .profilerTag = opts.profilerTag, .nameId = opts.nameId, .limiter = opts.limiter });
	g.compiled = false;
	return Node(g.nodes.size() - 1);
}

void TaskGraph::Precede(Node before, Node after)
{
	assert(m_State && before < m_State->nodes.size() && after < m_State->nodes.size() && "Precede: no such node");
	GraphState& g = *m_State;
	assert(!g.running.load(std::memory_order_acquire) && "TaskGraph changed while it runs");
	g.edges.emplace_back(before, after);
	g.compiled = false;
}

void TaskGraph::Clear()
{
	if (!m_State)
		return;
	GraphState& g = *m_State;
	assert(!g.running.load(std::memory_order_acquire) && "TaskGraph changed while it runs");
	g.DestroyCallables();
	g.nodes.clear();
	g.edges.clear();
	g.compiled = false;
}

size_t TaskGraph::NodeCount() const
{
	return m_State ? m_State->nodes.size() : 0;
}

bool TaskGraph::IsRunning() const
{
	return m_State && m_State->running.load(std::memory_order_acquire);
}

bool TaskGraph::Validate()
{
	if (!m_State)
		return true;
	GraphState& g = *m_State;
	assert(!g.running.load(std::memory_order_acquire) && "TaskGraph compiled while it runs");
	if (!g.compiled)
		g.Compile();
	return g.acyclic;
}

void Submit(TaskGraph& graph, CounterHandle signals, CounterHandle waitsFor)
{
	PoolState& P = ActivePool();
	if (!graph.m_State)
		graph.m_State = std::make_unique<GraphState>();
	GraphState& g = *graph.m_State;
	assert(!g.running.load(std::memory_order_acquire) && "TaskGraph submitted while a run is in flight");
	if (!g.compiled)
		g.Compile();
	// A cycle's nodes would never run, and the run never finish: fail now,
	// release builds too, rather than hang.
	assert(g.acyclic && "TaskGraph has a cycle");
	if (!g.acyclic)
	{
		std::fprintf(stderr, "loom: Submit of a TaskGraph with a cycle\n");
		std::abort();
	}
	g.running.store(true, std::memory_order_relaxed); // published with the first queued node
	g.signals = signals;
	if (waitsFor == kInvalidCounter)
		StartGraphRun(P, g);
	else
		Submit(TaskDesc{ .fn = &GraphLaunchTask, .userData = &g, .waitsFor = waitsFor });
}

void Run(TaskGraph& graph, Help help)
{
	const CounterHandle done = CreateCounter(1);
	// Destroyed on the way out even when WaitFor rethrows (the count is zero).
	struct DestroyOnExit
	{
		CounterHandle h;
		~DestroyOnExit() { DestroyCounter(h); }
	} destroy{ done };
	Submit(graph, done);
	WaitFor(done, help);
}

// ---- Lanes ---------------------------------------------------------------------
// A lane is a sleeping (never stealing) dedicated thread plus a held count,
// so Shutdown can tell a forgotten lane from an ordinary dedicated thread.

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
	c.failed.store(false, std::memory_order_relaxed); // (error was cleared by the destroy)
	c.cancelled.store(false, std::memory_order_relaxed);
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
		assert(c->pending.empty() && "destroying a counter with waiting tasks");
		assert(c->waiters.empty() && "destroying a counter with waiters");
		c->error = nullptr;
	}
	// Under the lock CreateCounter holds: the handle is invalidated before the
	// index can be handed out again.
	std::lock_guard<std::mutex> lock(P.counterMutex);
	if (!c->alive.load(std::memory_order_relaxed) || c->generation.load(std::memory_order_relaxed) != HandleGeneration(h))
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
	P.freeCounters.push_back(HandleIndex(h));
}

void Signal(CounterHandle h, int n)
{
	SignalCounter(ActivePool(), h, n);
}

// The stored value is the count + 1 while it is nonzero, and the final
// signaller takes it from 2 to 1 before releasing anything. So an add that
// sees more than 1 and wins its exchange comes before that final signal:
// nobody has been released, and the count stays above zero. Deferred task
// signals only hold decrements back, so they never make this unsafe.
bool AddToCounter(CounterHandle h, int n)
{
	CounterState* c = ResolveCounter(ActivePool(), h);
	assert(c && "adding to an invalid/destroyed counter");
	assert(n >= 0 && "AddToCounter adds; Signal takes away");
	if (!c || n < 0)
		return false;
	int v = c->count.load(std::memory_order_acquire);
	for (;;)
	{
		if (v <= 1)
			return false; // reached zero (1: still settling, 0: settled)
		if (n == 0)
			return true;
		assert(v <= INT_MAX - n && "counter overflow");
		if (c->count.compare_exchange_weak(v, v + n, std::memory_order_acq_rel, std::memory_order_relaxed))
			return true;
	}
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

	// The final signal: settle, then release the waiting tasks and the waiters.
	// Settling first means a released task, which may run on another thread
	// before this lock is dropped, already reads the count as zero. Anyone
	// who sees zero and destroys the counter waits for this lock.
	WorkerCtx* ctx = CurrentCtx(P);
	size_t pooled = 0;
	{
		std::lock_guard<std::mutex> lock(c->mutex);
		c->count.store(0, std::memory_order_release);
		// A task waiting for several counters runs when the last one releases it.
		for (uint32_t slot : c->pending)
			if (PendingDeps(P, slot).fetch_sub(1, std::memory_order_acq_rel) == 1)
				pooled += ReadySlot(P, slot, ctx) ? 1 : 0;
		c->pending.clear();
		for (Waiter* w : c->waiters)
			Wake(*w, kWokeRecheck);
		c->waiters.clear();
	}
	WakeForWork(P, pooled);
}

namespace detail {

// A region is just a tag on the thread: PrepareSlot stamps it on every task
// submitted from here, and TryHelpIn reads it to filter what a wait may run.
uint16_t BeginIsolation()
{
	PoolState& P = ActivePool();
	uint16_t region = P.nextIsolation.fetch_add(1, std::memory_order_relaxed);
	if (region == 0)
		region = P.nextIsolation.fetch_add(1, std::memory_order_relaxed);
	const uint16_t outer = t_Isolation;
	t_Isolation = region;
	return outer;
}

void EndIsolation(uint16_t outer)
{
	t_Isolation = outer;
}

} // namespace detail

// Only a flag: RunSlot checks it before calling a task's function, so the
// skipped tasks still flow through the queues and signal as usual.
void Cancel(CounterHandle h)
{
	CounterState* c = ResolveCounter(ActivePool(), h);
	assert(c && "cancelling an invalid/destroyed counter");
	if (c)
		c->cancelled.store(true, std::memory_order_release);
}

bool IsCancelled(CounterHandle h)
{
	const CounterState* c = ResolveCounter(ActivePool(), h);
	assert(c && "querying an invalid/destroyed counter");
	return c && c->cancelled.load(std::memory_order_acquire);
}

// ---- Futures ---------------------------------------------------------------------
// A future's state is shared by reference count: the future, its task or
// promise, and each continuation reading it hold one. A future's own task
// signals the counter itself, at the end of its run, so that it can then
// drop its reference; whoever lets go last destroys the counter and frees
// the state. DestroyCounter takes the counter's lock, which a final
// signaller still releasing waiters holds, so the count is settled by then
// even when the last owner is a continuation released by that signal.

namespace detail {

void CreateFutureCounter(FutureStateBase* state)
{
	state->counter = CreateCounter(1);
}

void StartFuture(FutureStateBase* state, TaskFn run, const TaskDesc& opts, CounterHandle after)
{
	assert(opts.fn == nullptr && opts.userData == nullptr && "Async and Then take the callable; leave fn and userData unset");
	assert(opts.signals == kInvalidCounter && "a future signals its own counter; leave signals unset");
	assert((after == kInvalidCounter || opts.waitsFor == kInvalidCounter) && "Then waits for its future; leave waitsFor unset");
	CreateFutureCounter(state);
	TaskDesc desc = opts;
	desc.fn = run;
	desc.userData = state;
	desc.signals = kInvalidCounter;
	if (after != kInvalidCounter)
		desc.waitsFor = after;
	Submit(desc);
}

void SignalFuture(FutureStateBase* state) noexcept
{
	SignalCounter(ActivePool(), state->counter, 1);
}

void FinishFuture(FutureStateBase* state) noexcept
{
	// As a plain task's runner does, the unit goes back before the signal:
	// once Get returns, the future's task holds none. (A promise set inside a
	// limited task signals through SignalFuture alone: that unit is the
	// task's own.)
	PoolState& P = ActivePool();
	ReleaseRunningUnit(P, CurrentCtx(P));
	SignalCounter(P, state->counter, 1);
	ReleaseFuture(state);
}

void ReleaseFuture(FutureStateBase* state) noexcept
{
	// acq_rel: the last one sees everything the others wrote (the value).
	if (state->refs.fetch_sub(1, std::memory_order_acq_rel) != 1)
		return;
	DestroyCounter(state->counter);
	state->destroy(state);
}

#if defined(LOOM_COROUTINES)
// ---- Coroutines ------------------------------------------------------------------
// A suspended coroutine is a plain task waiting on its counters, whose
// function resumes it. The thread running such a task notes which coroutine
// it resumed, so that the coroutine can give back the task's limiter unit
// where its stretch ends, as it suspends again or before its future is
// ready (as FinishFuture does), while a coroutine that suspends or ends
// inside some other task (one that never suspended, say) leaves that task's
// unit alone.
static thread_local void* t_ResumedCoroutine = nullptr;

static void ResumeCoroutine(void* address)
{
	void* const outer = t_ResumedCoroutine; // a wait inside can resume others
	t_ResumedCoroutine = address;
	// The coroutine may finish, and its frame be freed, inside resume().
	std::coroutine_handle<>::from_address(address).resume();
	t_ResumedCoroutine = outer;
}

void SubmitResume(std::coroutine_handle<> coroutine, const TaskDesc& opts, CounterHandle waitsFor,
	std::span<const CounterHandle> waitsForAll)
{
	assert(opts.fn == nullptr && opts.userData == nullptr && "a coroutine's resume takes no fn or userData");
	assert(opts.signals == kInvalidCounter && opts.waitsFor == kInvalidCounter && opts.waitsForAll.empty()
		&& "a coroutine's resume signals nothing, and waits only for what it awaits");
	// Submit reads the description, waitsForAll included, before the task
	// can run: once it can, it may resume the coroutine and free the frame
	// that holds them.
	TaskDesc desc = opts;
	desc.fn = &ResumeCoroutine;
	desc.userData = coroutine.address();
	desc.waitsFor = waitsFor;
	desc.waitsForAll = waitsForAll;
	// A stretch that a limited task resumed ends here: its unit goes back
	// now, not when the task returns, since the rest may finish on another
	// thread first.
	PoolState& P = ActivePool();
	if (coroutine.address() == t_ResumedCoroutine)
		ReleaseRunningUnit(P, CurrentCtx(P));
	Submit(desc);
}

void FinishCoroutine(FutureStateBase* state, void* coroutine) noexcept
{
	PoolState& P = ActivePool();
	if (coroutine == t_ResumedCoroutine)
		ReleaseRunningUnit(P, CurrentCtx(P));
	SignalCounter(P, state->counter, 1);
	ReleaseFuture(state); // the last reference destroys the frame, state and all
}
#endif

} // namespace detail

// ---- Limiters ------------------------------------------------------------------

LimiterHandle CreateLimiter(uint32_t limit)
{
	PoolState& P = ActivePool();
	assert(limit >= 1 && limit < (uint32_t)INT_MAX && "limiter limit out of range");
	std::lock_guard<std::mutex> lock(P.limiterMutex);
	if (P.freeLimiters.empty())
	{
		assert(false && "limiter pool exhausted (PoolConfig::maxLimiters)");
		return kInvalidLimiter;
	}
	const uint32_t i = P.freeLimiters.back();
	P.freeLimiters.pop_back();
	LimiterState& l = P.limiters[i];
	{
		std::lock_guard<std::mutex> unitLock(l.mutex);
		l.limit.store(limit, std::memory_order_relaxed);
		l.retiring.store(0, std::memory_order_relaxed);
		l.unitInUse.assign(limit, 0);
		l.freeUnits.clear();
		for (uint32_t u = limit; u-- > 0;)
			l.freeUnits.push_back(u); // index 0 on top
	}
	l.count.store((int)limit, std::memory_order_relaxed);
	l.alive.store(true, std::memory_order_release);
	return PackLimiter(i, l.generation.load(std::memory_order_relaxed));
}

void DestroyLimiter(LimiterHandle h)
{
	PoolState& P = ActivePool();
	LimiterState* l = ResolveLimiter(P, h);
	assert(l && "destroying an invalid/destroyed limiter");
	if (!l)
		return;
	{
		std::lock_guard<std::mutex> lock(l->mutex);
		assert(l->count.load(std::memory_order_relaxed) == (int)l->limit.load(std::memory_order_relaxed) && l->waiting.empty()
			&& l->handoffs == 0 && l->retiring.load(std::memory_order_relaxed) == 0 && "destroying a limiter whose tasks are live");
	}
	std::lock_guard<std::mutex> lock(P.limiterMutex);
	if (!l->alive.load(std::memory_order_relaxed) || l->generation.load(std::memory_order_relaxed) != HandleGeneration(h))
	{
		assert(false && "limiter destroyed twice");
		return;
	}
	l->alive.store(false, std::memory_order_relaxed);
	const uint32_t next = (l->generation.load(std::memory_order_relaxed) + 1) & 0xFFF;
	l->generation.store(next == 0 ? 1 : next, std::memory_order_relaxed);
	P.freeLimiters.push_back(HandleIndex(h));
}

// count is free units less waiters less retiring debts. While it is negative
// no unit is free, and the held ones are the limit plus those still to retire.
static uint32_t UnitsHeld(const LimiterState& l)
{
	const int count = l.count.load(std::memory_order_acquire);
	const uint32_t limit = l.limit.load(std::memory_order_relaxed);
	if (count >= 0)
		return (uint32_t)count <= limit ? limit - (uint32_t)count : 0;
	return limit + l.retiring.load(std::memory_order_relaxed);
}

void SetLimit(LimiterHandle h, uint32_t limit)
{
	PoolState& P = ActivePool();
	assert(limit >= 1 && limit < (uint32_t)INT_MAX && "limiter limit out of range");
	LimiterState* l = ResolveLimiter(P, h);
	assert(l && "invalid/destroyed limiter");
	if (!l || limit < 1)
		return;
	uint32_t raise = 0;
	{
		std::lock_guard<std::mutex> lock(l->mutex);
		const uint32_t old = l->limit.load(std::memory_order_relaxed);
		if (limit == old)
			return;
		l->limit.store(limit, std::memory_order_relaxed);
		if (l->unitInUse.size() < limit)
			l->unitInUse.resize(limit, 0);
		if (limit > old)
		{
			raise = limit - old;
			for (uint32_t u = limit; u-- > old;)
				if (!l->unitInUse[u])
					l->freeUnits.push_back(u);
		}
		else
		{
			// A free unit goes at once; a held one when its task gives it back.
			for (uint32_t k = 0; k < old - limit; k++)
				if (l->count.fetch_sub(1, std::memory_order_acq_rel) <= 0)
					l->retiring.store(l->retiring.load(std::memory_order_relaxed) + 1, std::memory_order_relaxed);
			l->freeUnits.erase(std::remove_if(l->freeUnits.begin(), l->freeUnits.end(), [&](uint32_t u) { return u >= limit; }),
				l->freeUnits.end());
		}
	}
	// New units go out as finishing tasks' would: to a retiring debt, to a
	// waiting task, or into the free count.
	WorkerCtx* ctx = CurrentCtx(P);
	size_t woken = 0;
	for (uint32_t k = 0; k < raise; k++)
	{
		const uint32_t next = ReleaseUnit(*l);
		if (next != kNoSlot && EnqueueSlot(P, next, ctx))
			woken++;
	}
	WakeForWork(P, woken);
}

uint32_t CurrentLimiterUnit()
{
	PoolState* pool = s_ActivePool.load(std::memory_order_acquire);
	if (!pool || !t_RunningLimiter || *t_RunningLimiter == 0)
		return kNoUnit;
	if (*t_RunningUnit == kNoUnitIndex)
		*t_RunningUnit = TakeUnitIndex(pool->limiters[*t_RunningLimiter - 1]);
	return *t_RunningUnit;
}

uint32_t Limit(LimiterHandle h)
{
	const LimiterState* l = ResolveLimiter(ActivePool(), h);
	assert(l && "invalid/destroyed limiter");
	return l ? l->limit.load(std::memory_order_relaxed) : 0;
}

uint32_t HeldUnits(LimiterHandle h)
{
	LimiterState* l = ResolveLimiter(ActivePool(), h);
	assert(l && "invalid/destroyed limiter");
	if (!l)
		return 0;
	return UnitsHeld(*l);
}

uint32_t WaitingTasks(LimiterHandle h)
{
	LimiterState* l = ResolveLimiter(ActivePool(), h);
	assert(l && "invalid/destroyed limiter");
	if (!l)
		return 0;
	const int count = l->count.load(std::memory_order_acquire);
	if (count >= 0)
		return 0;
	const uint32_t short_ = (uint32_t)-count, retiring = l->retiring.load(std::memory_order_relaxed);
	return short_ > retiring ? short_ - retiring : 0; // retiring debts are no task's
}

int Count(CounterHandle h)
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
static void SleepOnCounter(PoolState& P, CounterState& c, bool help, Deadline deadline)
{
	Waiter& w = t_Waiter;
	w.isWorker = false;
	// Armed before any registration: a waker may fire the moment we appear in
	// a list, and that wake must not be overwritten.
	w.word.store(kArmed, std::memory_order_relaxed);
	{
		std::lock_guard<std::mutex> lock(c.mutex);
		if (c.count.load(std::memory_order_acquire) == 0)
			return;
		w.threadIndex = CurrentThreadIndex();
		c.waiters.push_back(&w);
	}

	// Register everywhere that could make this thread useful, re-checking each
	// source after registering on it. One Waiter serves every list; a second
	// waker before we unregister only sets the word again, which is harmless.

	bool ready = false;
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
	// An isolated wait takes no pool work, so new pool work need not wake it.
	const bool poolEligible = help && (!d || d->idle == DedicatedConfig::Idle::StealFromPool) && !ActiveIsolation(P);
	if (!ready && poolEligible)
	{
		RegisterIdle(P, w);
		inIdle = true;
		ready = HasPoolWork(P);
	}
	if (!ready)
		ready = c.count.load(std::memory_order_acquire) == 0;
	if (!ready)
		SleepOn(w, deadline); // a timeout is noticed by the caller's loop

	// Unregister from every list under its lock, whether or not that list's
	// waker fired: after this, nobody can still hold a pointer to `w`.
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

// WaitFor and TryWaitFor. True when the count reached zero (and then
// rethrows a task's exception), false when the deadline passed first.
static bool WaitUntil(CounterHandle h, Help help, Deadline deadline)
{
	PoolState& P = ActivePool();
	CounterState* c = ResolveCounter(P, h);
	assert(c && "waiting on an invalid/destroyed counter");
	if (!c)
		return true;

	const int threadIndex = CurrentThreadIndex();
	WorkerCtx* ctx = CurrentCtx(P);
	FlushDeferredSignals(P, ctx); // this thread's own finished tasks may be what we wait for
	MarkProfilerState(P, threadIndex, 0); // waiting
	ThreadDebug* const debug = DebugRow(P, ctx);
	CounterHandle outerWait = kInvalidCounter;
	int64_t outerSince = 0;
	if (debug)
	{
		outerWait = debug->waitingOn.load(std::memory_order_relaxed);
		outerSince = debug->waitingSince.load(std::memory_order_relaxed);
		debug->waitingSince.store(std::chrono::steady_clock::now().time_since_epoch().count(), std::memory_order_relaxed);
		debug->waitingOn.store(h, std::memory_order_relaxed);
	}
	// Untimed, until zero: Shutdown drains before it stops anything, so a
	// wait never needs to give up early.
	bool reached = true;
	while (c->count.load(std::memory_order_acquire) != 0)
	{
		// Checked between tasks, never during one: a helped task can overrun.
		if (deadline != kNoDeadline && std::chrono::steady_clock::now() >= deadline)
		{
			reached = false;
			break;
		}
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
		SleepOnCounter(P, *c, help == Help::WhileWaiting, deadline);
	}
	// Never return to the caller holding signals: it may block on anything.
	FlushDeferredSignals(P, ctx);
	MarkProfilerState(P, threadIndex, 2); // wait over
	if (debug)
	{
		debug->waitingOn.store(outerWait, std::memory_order_relaxed);
		debug->waitingSince.store(outerSince, std::memory_order_relaxed);
	}
#if defined(__cpp_exceptions)
	if (reached && c->failed.load(std::memory_order_acquire))
	{
		std::exception_ptr error;
		{
			std::lock_guard<std::mutex> lock(c->mutex);
			error = c->error;
		}
		if (error)
			std::rethrow_exception(error);
	}
#endif
	return reached;
}

void WaitFor(CounterHandle h, Help help)
{
	WaitUntil(h, help, kNoDeadline);
}

bool TryWaitFor(CounterHandle h, uint32_t timeoutMs, Help help)
{
	return WaitUntil(h, help, std::chrono::steady_clock::now() + std::chrono::milliseconds(timeoutMs));
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

// The oldest entry of a locked queue that belongs to `region`. A linear scan:
// main and dedicated queues are short, and only isolated waits come here.
static uint32_t PopFrontInRegion(PoolState& P, std::mutex& m, std::deque<uint32_t>& q, uint16_t region)
{
	std::lock_guard<std::mutex> lock(m);
	for (auto it = q.begin(); it != q.end(); ++it)
	{
		if (Slot(P, *it).isolation == region)
		{
			const uint32_t slot = *it;
			q.erase(it);
			return slot;
		}
	}
	return kNoSlot;
}

// The newest entry of this thread's own deques that belongs to `region`.
// Entries of other regions it meets on the way go to the injection queue,
// where the other workers take them: an isolated wait never runs them.
static uint32_t PopOwnInRegion(PoolState& P, WorkerCtx& ctx, uint16_t region)
{
	// Bounded, so a wait never spends long moving foreign work about. Past
	// that, it leaves the rest to the other workers (who may steal the
	// region's tasks) and sleeps.
	for (int moved = 0; moved < 64; moved++)
	{
		uint32_t entry = PopOwn(P, ctx);
		if (entry == kNoSlot && (entry = ctx.low.Pop()) != kNoSlot)
			NoteTaken(P, Priority::Low);
		if (entry == kNoSlot)
			return kNoSlot;
		if (Slot(P, entry & ~kGroupBit).isolation == region)
			return entry;
		PushPool(P, entry, nullptr);
		WakeForWork(P, 1);
	}
	return kNoSlot;
}

// One task for a waiting (or explicitly helping) thread. Sources in order:
// the thread's own pinned queue (main or dedicated), its own deques, then a
// steal. An isolated wait narrows all of that to its region, and never steals.
static bool TryHelpIn(PoolState& P)
{
	if (P.stop.load(std::memory_order_relaxed)) return false; // the workers are going

	WorkerCtx* ctx = CurrentCtx(P);
	uint32_t slot = kNoSlot;
	if (const uint16_t region = ActiveIsolation(P))
	{
		// An isolated wait: only this region's tasks, from this thread's own
		// queues; never a steal.
		if (ctx && ctx->isMain)
			slot = PopFrontInRegion(P, P.mainMutex, P.mainQueue, region);
		else if (DedicatedState* d = CurrentDedicated())
			slot = PopFrontInRegion(P, d->mutex, d->queue, region);
		if (slot == kNoSlot && ctx)
			slot = PopOwnInRegion(P, *ctx, region);
	}
	else if (ctx && ctx->isMain)
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
	if (slot == kNoSlot && ctx && !ActiveIsolation(P))
		slot = PopOwn(P, *ctx);
	if (slot == kNoSlot && !ActiveIsolation(P))
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
		stats->tasksHelped.fetch_add(1, std::memory_order_relaxed);
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

#if defined(__cpp_exceptions)
// A task's exception goes to the counter it signals, for WaitFor to rethrow
// (the first one wins). A task that signals no counter has nobody to report
// to: that ends the program, as an uncaught exception would.
static void RecordTaskException(PoolState& P, CounterHandle signals, std::exception_ptr error)
{
	CounterState* c = signals != kInvalidCounter ? ResolveCounter(P, signals) : nullptr;
	if (!c)
	{
		std::fprintf(stderr, "loom: a task that signals no counter threw an exception; nothing can receive it\n");
		std::terminate();
	}
	std::lock_guard<std::mutex> lock(c->mutex);
	if (!c->error)
	{
		c->error = std::move(error);
		c->failed.store(true, std::memory_order_release);
	}
	// The counter's work has failed: what has not started need not run.
	c->cancelled.store(true, std::memory_order_release);
}
#endif

// Calls the task, catching anything it throws so the worker survives and the
// exception reaches the task's counter.
static void RunTaskFn(PoolState& P, TaskFn fn, void* userData, CounterHandle signals)
{
#if defined(__cpp_exceptions)
	try
	{
		fn(userData);
	}
	catch (...)
	{
		RecordTaskException(P, signals, std::current_exception());
	}
#else
	(void)P;
	(void)signals;
	fn(userData);
#endif
}

// Gives back the running task's limiter unit now, or on to the next waiting
// task, instead of when the task ends. At most once per task.
static void ReleaseRunningUnit(PoolState& P, WorkerCtx* ctx)
{
	uint16_t* const limiter = t_RunningLimiter;
	if (!limiter || *limiter == 0)
		return;
	LimiterState& l = P.limiters[*limiter - 1];
	if (*t_RunningUnit != kNoUnitIndex)
	{
		ReturnUnitIndex(l, *t_RunningUnit); // free before the unit, for whoever gets it
		*t_RunningUnit = kNoUnitIndex;
	}
	const uint32_t next = ReleaseUnit(l);
	*limiter = 0;
	if (next != kNoSlot && EnqueueSlot(P, next, ctx))
		WakeForWork(P, 1);
}

static void RunSlot(PoolState& P, uint32_t slot, WorkerCtx* ctx)
{
	// The runner owns the slot until it releases it: no copy, no lock.
	TaskSlot& t = Slot(P, slot);
	const TaskFn fn = t.fn;
	void* const userData = t.userData;
	const CounterHandle signals = t.signals;
	const uint32_t nameId = t.nameId;
	const uint8_t profilerTag = t.profilerTag;
	uint16_t limiter = t.limiter; // cleared if the task gives its unit back early
	TaskFn destroy = nullptr;
	if (t.owned)
		std::memcpy(&destroy, t.inlineData + kOwnedDestroyOffset, sizeof destroy);

	// Deferred signals only ever belong to the counter the RUNNING task
	// signals: that counter cannot reach zero before this task finishes, so
	// holding them back delays nothing. Any other task (another counter, or
	// none, maybe a long one) runs only after they are applied.
	if (ctx && ctx->deferredCount > 0 && ctx->deferredSignal != signals)
		FlushDeferredSignals(P, ctx);

	// A per-task profiler zone (null hook = free no-op). The begin hook
	// returns the handle the end mark pairs with.
	const int threadIndex = CurrentThreadIndex();
	const uint64_t zone = MarkProfilerZone(P, threadIndex, nameId, profilerTag, "Task", true, 0);
	// A task of a cancelled counter is skipped, though it still signals it.
	const CounterState* counter = signals != kInvalidCounter ? ResolveCounter(P, signals) : nullptr;
	const bool cancelled = counter && counter->cancelled.load(std::memory_order_acquire);
	// The task runs in its own region (RunTaskFn catches everything, so the
	// restore below always happens).
	const uint16_t outerRegion = t_Isolation;
	uint16_t* const outerLimiter = t_RunningLimiter;
	uint32_t* const outerUnit = t_RunningUnit;
	uint32_t unitIndex = kNoUnitIndex;
	t_Isolation = t.isolation;
	t_RunningLimiter = &limiter;
	t_RunningUnit = &unitIndex;
	ThreadDebug* const debug = DebugRow(P, ctx);
	uint32_t outerName = 0;
	CounterHandle outerSignals = kInvalidCounter;
	uint16_t outerDebugLimiter = 0;
	if (debug)
	{
		outerName = debug->runningName.load(std::memory_order_relaxed);
		outerSignals = debug->runningSignals.load(std::memory_order_relaxed);
		outerDebugLimiter = debug->runningLimiter.load(std::memory_order_relaxed);
		debug->runningName.store(nameId, std::memory_order_relaxed);
		debug->runningSignals.store(signals, std::memory_order_relaxed);
		debug->runningLimiter.store(limiter, std::memory_order_relaxed);
		debug->running.store(debug->running.load(std::memory_order_relaxed) + 1, std::memory_order_relaxed);
	}
	if (fn && !cancelled)
		RunTaskFn(P, fn, userData, signals); // an inline callable's userData is the slot: release after
	if (debug)
	{
		debug->running.store(debug->running.load(std::memory_order_relaxed) - 1, std::memory_order_relaxed);
		debug->runningName.store(outerName, std::memory_order_relaxed);
		debug->runningSignals.store(outerSignals, std::memory_order_relaxed);
		debug->runningLimiter.store(outerDebugLimiter, std::memory_order_relaxed);
		debug->tasksRun.store(debug->tasksRun.load(std::memory_order_relaxed) + 1, std::memory_order_relaxed);
	}
	t_Isolation = outerRegion;
	t_RunningLimiter = outerLimiter;
	t_RunningUnit = outerUnit;
	// An owned callable dies whether it ran, threw or was skipped, and before
	// the slot it may live in is released.
	if (destroy)
		destroy(userData);
	MarkProfilerZone(P, threadIndex, nameId, profilerTag, "Task", false, zone);
	LOOM_TEST_POINT(kTestPointTaskReturned);
	// The unit goes back (or on to the next waiting task) before the signal:
	// once a WaitFor on the task's counter returns, it holds none.
	if (limiter != 0)
	{
		if (unitIndex != kNoUnitIndex)
			ReturnUnitIndex(P.limiters[limiter - 1], unitIndex); // free before the unit, for whoever gets it
		const uint32_t next = ReleaseUnit(P.limiters[limiter - 1]);
		if (next != kNoSlot && EnqueueSlot(P, next, ctx))
			WakeForWork(P, 1);
	}

	// Live count and slot are given back before the signal: once a WaitFor on
	// this counter returns, the task no longer counts as live.
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
	SignalCounter(P, h, n); // may release waiting tasks onto this deque
}

// ---- Main thread ---------------------------------------------------------------------

void PumpMainThreadTasks()
{
	PoolState* pool = s_ActivePool.load(std::memory_order_acquire);
	if (!pool || !pool->running) return;
	PoolState& P = *pool;
	WorkerCtx* ctx = CurrentCtx(P);
	// Bounded: a main-thread task that resubmits itself must not trap the
	// frame loop here forever.
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

// Rounds of stealing before a worker sleeps: the first kSpinRounds spin, the
// rest yield. A futex sleep and wake costs several microseconds, so a worker
// that finds work again within a few rounds is cheaper than one that slept.
static constexpr int kSearchRounds = 32;
static constexpr int kSpinRounds = 8;

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

// A worker with nothing of its own: steal for a while before giving up.
// kNoSlot means "go to sleep" (or park, when a park is requested).
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
			const uint32_t released = PopOwn(P, ctx);
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

// A pool worker: its own deque, then a search, then sleep. Exits once `stop`
// is set and a full search comes up empty.
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
	PaddedStats* row = RegisterThreadStats(P, index, t_name);
	ThreadStats* stats = &row->s;
	t_ThreadStats = stats;
	t_Debug = &row->debug;
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

		uint32_t slot = PopOwn(P, ctx);
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

// The row for a thread index, created on first use. A thread that takes the
// index of one that has exited takes over its row, under its own name; the
// counts carry on.
static PaddedStats* RegisterThreadStats(PoolState& P, int index, const char* name)
{
	std::lock_guard<std::mutex> lock(P.statsMutex);
	for (auto& t : P.threadStats)
		if (t->s.index == index)
		{
			t->name = name ? name : "";
			t->s.name = t->name.c_str();
			return t.get();
		}
	auto stats = std::make_unique<PaddedStats>();
	stats->s.index = index;
	stats->name = name ? name : "";
	stats->s.name = stats->name.c_str();
	PaddedStats* raw = stats.get();
	P.threadStats.push_back(std::move(stats));
	return raw;
}

PoolSnapshot SnapshotPool()
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
			view.tasksHelped = t.tasksHelped.load();
			view.queueDepth = 0;
			snap.threads.push_back(std::move(view));
		}
	}
	snap.queueDepth = PoolQueueDepth(P);
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
			// Attach the dedicated queue depth to its stats row (none yet if
			// the thread has not started).
			const int index = d->threadIndex.load(std::memory_order_relaxed);
			for (auto& t : snap.threads)
				if (t.index == index)
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

// A dedicated thread: its own queue first, then (if configured) the pool.
// No deque of its own, so anything it submits goes through the injection
// queue. Exits when unregistered, once its own queue is empty.
static void DedicatedLoop(PoolState& P, DedicatedState& d)
{
	if (P.onThreadStart)
		P.onThreadStart();
	t_CurrentDedicated = &d;
	const int index = CurrentThreadIndex(); // reserve this thread's index
	d.threadIndex.store(index, std::memory_order_relaxed);
	const char* name = d.name.empty() ? "loom-dedicated" : d.name.c_str();
	PaddedStats* row = RegisterThreadStats(P, index, name);
	ThreadStats* stats = &row->s;
	t_ThreadStats = stats;
	t_Debug = &row->debug;
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
	const DedicatedId id{ P.nextDedicatedId++ };
	assert(id != kInvalidDedicated && "dedicated thread ids exhausted");
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
	assert((!CurrentDedicated() || CurrentDedicated()->id != id) && "a dedicated thread cannot unregister itself (it would join itself)");
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

// A frame-pointer walk: with frame pointers, each frame begins with
// { caller's frame pointer, return address }. The walk stays inside this
// thread's stack bounds and stops at the first frame that does not point
// further up the stack, so a frame without a frame pointer ends it instead of
// sending it into garbage.
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

// A pool thread's (or participant's) side of a park: record where it is,
// check in, and block until the round it joined is released. The stack is
// captured before taking the lock, so ParkAll's caller can read it at once.
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

// Request, wake everyone who might be asleep, then wait for every thread to
// check in. Running threads see the request at their next loop top.
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

bool IsParkRequested()
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

// ---- Pipelines --------------------------------------------------------------------
// One run's state lives on Run's stack. Every task of the run holds an item
// that has not left the pipeline yet, and touches the run only while it
// does; the task that brings the last item home signals `done` as its very
// last step, so Run can return and free the state. The source keeps filling
// items in one task while free ones are left, and a serial stage keeps
// taking the next item in order when it is already waiting, so a busy
// pipeline does not pay a task per item for them.

namespace detail {

namespace {

struct PipelineSerial
{
	std::mutex mutex;
	uint64_t   next = 0;  // the place of the item it takes next
	bool       busy = false;
	// Items that came early, at their place modulo the item count: the places
	// waiting are always within one item count of `next`.
	std::vector<uint32_t> waiting;
};

struct PipelineRun
{
	const PipelineStage* stages = nullptr;
	size_t               stageCount = 0;
	void* const*         items = nullptr;
	CounterHandle        done = kInvalidCounter;
	std::unique_ptr<PipelineSerial[]> serial; // by stage
	std::vector<uint64_t> order;              // each item's place in the stream (written by the source before it passes the item on)

	std::mutex            mutex;             // the source and the free items
	std::vector<uint32_t> freeItems;
	size_t                inFlight = 0;      // items out of freeItems (the source's included)
	uint64_t              nextOrder = 0;
	bool                  sourceBusy = false;
	bool                  ended = false;     // the source returned false, or a stage threw
	bool                  signalled = false;
	std::atomic<bool>     failed{ false };
#if defined(__cpp_exceptions)
	std::exception_ptr    error;             // under mutex
#endif
};

void PipelineAdvance(PipelineRun& run, uint32_t item, size_t stage);

// Under the lock: the item the source may fill next, or kNoSlot.
uint32_t TakeSourceItem(PipelineRun& run)
{
	if (run.ended || run.freeItems.empty())
		return kNoSlot;
	const uint32_t item = run.freeItems.back();
	run.freeItems.pop_back();
	run.inFlight++;
	return item;
}

// Under the lock: whether this was the run's end (the caller then signals,
// last of all).
bool FinishedLocked(PipelineRun& run)
{
	if (!run.ended || run.inFlight != 0 || run.signalled)
		return false;
	run.signalled = true;
	return true;
}

// Calls a stage's function, catching what it throws into the run. The
// source's false (or a throw) ends the stream.
bool CallStage(PipelineRun& run, size_t stage, uint32_t item)
{
	const PipelineStage& s = run.stages[stage];
#if defined(__cpp_exceptions)
	try
	{
		return s.run(s.stage, run.items[item]);
	}
	catch (...)
	{
		std::lock_guard<std::mutex> lock(run.mutex);
		if (!run.failed.load(std::memory_order_relaxed))
			run.error = std::current_exception();
		run.failed.store(true, std::memory_order_relaxed);
		run.ended = true;
		return false;
	}
#else
	return s.run(s.stage, run.items[item]);
#endif
}

void SubmitStage(PipelineRun& run, uint32_t item, size_t stage);

// A stage task gives its limiter unit back before it lets go of its last
// item, as a future's task does before its signal: once that item moves on,
// the run can end and Run return (and the caller destroy the limiter)
// before the task's runner would give the unit back.
void ReleaseStageUnit()
{
	PoolState& P = ActivePool();
	ReleaseRunningUnit(P, CurrentCtx(P));
}

void RunSource(PipelineRun& run, uint32_t item)
{
	for (;;)
	{
		const bool more = !run.failed.load(std::memory_order_relaxed) && CallStage(run, 0, item);
		uint32_t next = kNoSlot;
		bool finished = false;
		const CounterHandle done = run.done;
		if (!more)
			ReleaseStageUnit();
		{
			std::lock_guard<std::mutex> lock(run.mutex);
			if (more)
			{
				run.order[item] = run.nextOrder++;
				next = TakeSourceItem(run); // keep going while items are free
			}
			else
			{
				run.ended = true;
				run.freeItems.push_back(item);
				run.inFlight--;
				finished = FinishedLocked(run);
			}
			if (next == kNoSlot)
				run.sourceBusy = false;
		}
		if (!more)
		{
			if (finished)
				Signal(done); // the run may be gone after this
			return;
		}
		if (next == kNoSlot)
			ReleaseStageUnit();
		PipelineAdvance(run, item, 1); // with no `next` held, the run may be gone after this
		if (next == kNoSlot)
			return;
		item = next;
	}
}

void RunStage(PipelineRun& run, uint32_t item, size_t stage)
{
	const bool serial = run.stages[stage].serial;
	for (;;)
	{
		if (!run.failed.load(std::memory_order_relaxed))
			(void)CallStage(run, stage, item);
		uint32_t next = kNoSlot;
		if (serial)
		{
			// Take the next item in order straight away if it is waiting.
			PipelineSerial& st = run.serial[stage];
			std::lock_guard<std::mutex> lock(st.mutex);
			st.next++;
			uint32_t& slot = st.waiting[st.next % st.waiting.size()];
			if (slot != kNoSlot)
			{
				next = slot;
				slot = kNoSlot;
			}
			else
				st.busy = false;
		}
		if (next == kNoSlot)
			ReleaseStageUnit();
		PipelineAdvance(run, item, stage + 1); // with no `next` held, the run may be gone after this
		if (next == kNoSlot)
			return;
		item = next;
	}
}

void SubmitStage(PipelineRun& run, uint32_t item, size_t stage)
{
	PipelineRun* r = &run;
	const TaskDesc& desc = run.stages[stage].opts;
	if (stage == 0)
		Submit([r, item] { RunSource(*r, item); }, desc);
	else
		Submit([r, item, stage] { RunStage(*r, item, stage); }, desc);
}

void PipelineAdvance(PipelineRun& run, uint32_t item, size_t stage)
{
	if (stage == run.stageCount)
	{
		// Home: the item goes back to the source, which starts again if idle.
		uint32_t source = kNoSlot;
		bool finished = false;
		const CounterHandle done = run.done;
		{
			std::lock_guard<std::mutex> lock(run.mutex);
			run.freeItems.push_back(item);
			run.inFlight--;
			if (!run.sourceBusy)
			{
				source = TakeSourceItem(run);
				run.sourceBusy = source != kNoSlot;
			}
			finished = FinishedLocked(run);
		}
		if (source != kNoSlot)
			SubmitStage(run, source, 0);
		if (finished)
			Signal(done); // the run may be gone after this
		return;
	}
	if (run.stages[stage].serial)
	{
		PipelineSerial& st = run.serial[stage];
		std::lock_guard<std::mutex> lock(st.mutex);
		if (st.busy || run.order[item] != st.next)
		{
			st.waiting[run.order[item] % st.waiting.size()] = item;
			return;
		}
		st.busy = true;
	}
	SubmitStage(run, item, stage);
}

} // namespace

void RunPipeline(const PipelineStage* stages, size_t stageCount, void* const* items, size_t itemCount)
{
	assert(stageCount >= 1 && itemCount >= 1);
	for (size_t i = 0; i < stageCount; i++)
		assert(stages[i].opts.fn == nullptr && stages[i].opts.signals == kInvalidCounter && stages[i].opts.waitsFor == kInvalidCounter
			&& stages[i].opts.waitsForAll.empty() && "a pipeline stage's TaskDesc gives no fn, signals or waits");
	PipelineRun run;
	run.stages = stages;
	run.stageCount = stageCount;
	run.items = items;
	run.order.assign(itemCount, 0);
	run.serial.reset(new PipelineSerial[stageCount]);
	for (size_t i = 0; i < stageCount; i++)
		run.serial[i].waiting.assign(itemCount, kNoSlot);
	for (uint32_t i = (uint32_t)itemCount; i-- > 0;)
		run.freeItems.push_back(i);
	ScopedCounter done(1);
	run.done = done.Get();
	uint32_t first;
	{
		std::lock_guard<std::mutex> lock(run.mutex);
		first = TakeSourceItem(run);
		run.sourceBusy = true;
	}
	SubmitStage(run, first, 0);
	WaitFor(done);
#if defined(__cpp_exceptions)
	if (run.error)
		std::rethrow_exception(run.error);
#endif
}

} // namespace detail

// ---- Debug dump ---------------------------------------------------------------
// Meant for a pool that may be stuck, so it never waits long for a lock: a
// thread stuck while holding one must not take the dump down with it.

static bool LockBriefly(std::mutex& m)
{
	const auto until = std::chrono::steady_clock::now() + std::chrono::milliseconds(2);
	do
	{
		if (m.try_lock())
			return true;
		std::this_thread::yield();
	} while (std::chrono::steady_clock::now() < until);
	return false;
}

static constexpr size_t kDebugListMax = 64;

static LimiterHandle LimiterOfTag(PoolState& P, uint16_t tag)
{
	if (tag == 0 || tag > P.limiterCount)
		return kInvalidLimiter;
	return PackLimiter(tag - 1u, P.limiters[tag - 1u].generation.load(std::memory_order_relaxed));
}

// A slot that waits for a counter or for a limiter's unit: nobody runs or
// frees it while the lock it is listed under is held.
static DebugTask DebugTaskOf(PoolState& P, uint32_t slot)
{
	const TaskSlot& t = Slot(P, slot);
	DebugTask task;
	task.nameId = t.nameId;
	task.priority = t.priority;
	task.affinity = t.affinity;
	task.limiter = LimiterOfTag(P, t.limiter);
	return task;
}

static void ReadThreadDebug(PoolState& P, const ThreadDebug& d, int64_t now, DebugThread& out)
{
	out.tasksRun = d.tasksRun.load(std::memory_order_relaxed);
	out.runningDepth = d.running.load(std::memory_order_relaxed);
	if (out.runningDepth > 0)
	{
		out.runningNameId = d.runningName.load(std::memory_order_relaxed);
		out.runningSignals = d.runningSignals.load(std::memory_order_relaxed);
		out.runningLimiter = LimiterOfTag(P, d.runningLimiter.load(std::memory_order_relaxed));
	}
	out.waitingOn = d.waitingOn.load(std::memory_order_relaxed);
	if (out.waitingOn != kInvalidCounter)
	{
		const int64_t since = d.waitingSince.load(std::memory_order_relaxed);
		out.waitingSeconds = std::chrono::duration<double>(std::chrono::steady_clock::duration(std::max<int64_t>(0, now - since))).count();
	}
}

DebugState CaptureDebugState()
{
	DebugState s;
	PoolState* pool = s_ActivePool.load(std::memory_order_acquire);
	if (!pool)
		return s;
	PoolState& P = *pool;
	const int64_t now = std::chrono::steady_clock::now().time_since_epoch().count();

	DebugThread mainThread;
	mainThread.index = P.mainThreadIndex;
	mainThread.name = "main";
	mainThread.isMain = true;
	ReadThreadDebug(P, P.mainDebug, now, mainThread);
	s.threads.push_back(std::move(mainThread));
	if (LockBriefly(P.statsMutex))
	{
		for (auto& row : P.threadStats)
		{
			DebugThread t;
			t.index = row->s.index;
			t.name = row->name;
			ReadThreadDebug(P, row->debug, now, t);
			s.threads.push_back(std::move(t));
		}
		P.statsMutex.unlock();
	}
	else
		s.busy = true;
	if (LockBriefly(P.dedicatedMutex))
	{
		for (auto& d : P.dedicated)
		{
			if (!LockBriefly(d->mutex))
				continue;
			const int depth = (int)d->queue.size();
			d->mutex.unlock();
			const int index = d->threadIndex.load(std::memory_order_relaxed);
			for (DebugThread& t : s.threads)
				if (!t.isMain && t.index == index)
					t.queueDepth = depth;
		}
		P.dedicatedMutex.unlock();
	}
	else
		s.busy = true;

	// Counters, under the table lock so none is created or destroyed midway.
	if (LockBriefly(P.counterMutex))
	{
		for (uint32_t i = 0; i < P.counterCount; i++)
		{
			CounterState& c = P.counters[i];
			if (!c.alive.load(std::memory_order_relaxed))
				continue;
			DebugCounter dc;
			dc.handle = PackCounter(i, c.generation.load(std::memory_order_relaxed));
			const int v = c.count.load(std::memory_order_acquire);
			dc.count = v > 1 ? v - 1 : v;
			dc.cancelled = c.cancelled.load(std::memory_order_relaxed);
			dc.failed = c.failed.load(std::memory_order_relaxed);
			if (LockBriefly(c.mutex))
			{
				dc.waitingTotal = (int)c.pending.size();
				for (size_t k = 0; k < c.pending.size() && k < kDebugListMax; k++)
					dc.waiting.push_back(DebugTaskOf(P, c.pending[k]));
				for (const Waiter* w : c.waiters)
					dc.sleepers.push_back(w->threadIndex);
				c.mutex.unlock();
			}
			else
				dc.busy = true;
			s.counters.push_back(std::move(dc));
		}
		P.counterMutex.unlock();
	}
	else
		s.busy = true;

	if (LockBriefly(P.limiterMutex))
	{
		for (uint32_t i = 0; i < P.limiterCount; i++)
		{
			LimiterState& l = P.limiters[i];
			if (!l.alive.load(std::memory_order_relaxed))
				continue;
			DebugLimiter dl;
			dl.handle = PackLimiter(i, l.generation.load(std::memory_order_relaxed));
			dl.limit = l.limit.load(std::memory_order_relaxed);
			dl.held = UnitsHeld(l);
			if (LockBriefly(l.mutex))
			{
				dl.waitingTotal = (int)l.waiting.size();
				for (size_t k = 0; k < l.waiting.size() && k < kDebugListMax; k++)
					dl.waiting.push_back(DebugTaskOf(P, l.waiting[k]));
				l.mutex.unlock();
			}
			else
				dl.busy = true;
			s.limiters.push_back(std::move(dl));
		}
		P.limiterMutex.unlock();
	}
	else
		s.busy = true;

	s.liveTasks = (int)LiveTasksIn(P);
	s.queueDepth = PoolQueueDepth(P);
	s.highQueueDepth = P.queuedHigh.load(std::memory_order_relaxed);
	s.lowQueueDepth = P.queuedLow.load(std::memory_order_relaxed);
	if (LockBriefly(P.mainMutex))
	{
		s.mainQueueDepth = (int)P.mainQueue.size();
		P.mainMutex.unlock();
	}
	else
		s.busy = true;
	return s;
}

namespace {

struct TextOut
{
	std::string text;
	void Line(const char* fmt, ...)
	{
		char buf[512];
		va_list args;
		va_start(args, fmt);
		std::vsnprintf(buf, sizeof buf, fmt, args);
		va_end(args);
		text += buf;
	}
};

std::string Plural(long long n, const char* word)
{
	return std::to_string(n) + " " + word + (n == 1 ? "" : "s");
}

template <typename H>
std::string HandleText(char kind, H h)
{
	char buf[32];
	std::snprintf(buf, sizeof buf, "%c%u.%u", kind, HandleIndex(h), HandleGeneration(h));
	return buf;
}

std::string TaskNameText(uint32_t nameId, TaskNameFn nameOf)
{
	if (nameId == 0)
		return "unnamed";
	if (nameOf)
		if (const char* name = nameOf(nameId))
			return std::string("'") + name + "'";
	return "task#" + std::to_string(nameId);
}

// "'draw' x3, task#17 x2": tasks grouped by name, in order of first appearance.
std::string TaskListText(const std::vector<DebugTask>& tasks, int total, TaskNameFn nameOf)
{
	std::vector<std::pair<uint32_t, int>> groups;
	for (const DebugTask& t : tasks)
	{
		auto it = std::find_if(groups.begin(), groups.end(), [&](const auto& g) { return g.first == t.nameId; });
		if (it == groups.end())
			groups.push_back({ t.nameId, 1 });
		else
			it->second++;
	}
	std::string text;
	for (size_t i = 0; i < groups.size() && i < 8; i++)
	{
		if (!text.empty())
			text += ", ";
		text += TaskNameText(groups[i].first, nameOf);
		if (groups[i].second > 1)
			text += " x" + std::to_string(groups[i].second);
	}
	if (groups.size() > 8 || total > (int)tasks.size())
		text += ", ...";
	return text;
}

} // namespace

std::string FormatDebugState(const DebugState& s, TaskNameFn nameOf)
{
	TextOut out;
	out.Line("loom: %s live; queued %d (high %d, low %d), for the main thread %d%s\n", Plural(s.liveTasks, "task").c_str(),
		s.queueDepth, s.highQueueDepth, s.lowQueueDepth, s.mainQueueDepth, s.busy ? "; some tables were busy and are missing" : "");

	out.Line("threads:\n");
	bool anyRunning = false, anyWaiting = false, dedicatedQueued = false;
	const DebugThread* mainThread = nullptr;
	for (const DebugThread& t : s.threads)
	{
		if (t.isMain)
			mainThread = &t;
		anyRunning = anyRunning || t.runningDepth > 0;
		anyWaiting = anyWaiting || t.waitingOn != kInvalidCounter;
		dedicatedQueued = dedicatedQueued || t.queueDepth > 0;
		std::string what;
		if (t.runningDepth > 0)
		{
			what = "running " + TaskNameText(t.runningNameId, nameOf);
			if (t.runningSignals != kInvalidCounter)
				what += " (signals " + HandleText('c', t.runningSignals) + ")";
			if (t.runningLimiter != kInvalidLimiter)
				what += " (holds a unit of " + HandleText('l', t.runningLimiter) + ")";
			if (t.runningDepth > 1)
				what += ", inside " + std::to_string(t.runningDepth - 1) + " more";
		}
		if (t.waitingOn != kInvalidCounter)
		{
			char buf[96];
			std::snprintf(buf, sizeof buf, "waiting %.3f s on %s", t.waitingSeconds, HandleText('c', t.waitingOn).c_str());
			if (!what.empty())
				what = std::string(buf) + ", meanwhile " + what;
			else
				what = buf;
		}
		if (what.empty())
			what = "idle";
		out.Line("  [%d] %s: %s; ran %s", t.index, t.name.c_str(), what.c_str(), Plural((long long)t.tasksRun, "task").c_str());
		if (t.queueDepth > 0)
			out.Line("; %d queued for it", t.queueDepth);
		out.Line("\n");
	}

	out.Line("counters (%zu live):\n", s.counters.size());
	for (const DebugCounter& c : s.counters)
	{
		out.Line("  %s: count %d%s%s", HandleText('c', c.handle).c_str(), c.count, c.cancelled ? ", cancelled" : "",
			c.failed ? ", a task threw" : "");
		if (c.busy)
			out.Line("; busy");
		if (c.waitingTotal > 0)
			out.Line("; %d waiting: %s", c.waitingTotal, TaskListText(c.waiting, c.waitingTotal, nameOf).c_str());
		if (!c.sleepers.empty())
		{
			std::string who;
			for (int w : c.sleepers)
				who += (who.empty() ? "[" : ", [") + std::to_string(w) + "]";
			out.Line("; asleep on it: %s", who.c_str());
		}
		out.Line("\n");
	}

	if (!s.limiters.empty())
	{
		out.Line("limiters (%zu live):\n", s.limiters.size());
		for (const DebugLimiter& l : s.limiters)
		{
			out.Line("  %s: limit %u, held %u", HandleText('l', l.handle).c_str(), l.limit, l.held);
			if (l.busy)
				out.Line("; busy");
			if (l.waitingTotal > 0)
				out.Line("; %d waiting: %s", l.waitingTotal, TaskListText(l.waiting, l.waitingTotal, nameOf).c_str());
			out.Line("\n");
		}
	}

	// The common ways a wait never returns that the capture can see. A
	// thread outside the pool shows up only asleep on a counter.
	for (const DebugCounter& c : s.counters)
		anyWaiting = anyWaiting || !c.sleepers.empty();
	if (anyWaiting && !anyRunning && s.queueDepth == 0 && s.mainQueueDepth == 0 && !dedicatedQueued)
		out.Line("hint: nothing is running or queued, so the counters waited on are waiting for a Signal or a Submit that has not happened\n");
	if (s.mainQueueDepth > 0 && mainThread && mainThread->waitingOn == kInvalidCounter && mainThread->runningDepth == 0)
		out.Line("hint: main-thread tasks are queued; they run only while the main thread waits (helping) or calls PumpMainThreadTasks\n");
	for (const DebugLimiter& l : s.limiters)
	{
		if (l.held < l.limit || l.waitingTotal == 0)
			continue;
		for (const DebugThread& t : s.threads)
			if (t.runningDepth > 0 && t.runningLimiter == l.handle && t.waitingOn != kInvalidCounter)
				out.Line("hint: [%d] holds a unit of %s and waits while %s for that limiter; a limited task must not wait for its own limiter's work\n",
					t.index, HandleText('l', l.handle).c_str(), (Plural(l.waitingTotal, "task") + (l.waitingTotal == 1 ? " waits" : " wait")).c_str());
	}
	return out.text;
}

std::string DebugDump(TaskNameFn nameOf)
{
	return FormatDebugState(CaptureDebugState(), nameOf);
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
