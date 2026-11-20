#include "loom/loom.h"
#include "loom/loom_profiler_shim.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <deque>
#include <mutex>
#include <thread>
#include <vector>

#if defined(__linux__)
#include <pthread.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace loom {

// ---- Task slots (fixed pool + freelist; no per-task allocation) -------------

struct TaskSlot
{
	TaskFn        fn       = nullptr;
	void*         userData = nullptr;
	Affinity      affinity = Affinity::AnyWorker;
	DedicatedId   thread   = 0;
	CounterHandle signals  = kInvalidCounter;
	CounterHandle waitsFor = kInvalidCounter;
	uint32_t      nameId   = 0;
	uint8_t       window   = 0;
	alignas(16) uint8_t inlineData[48]; // the Submit(F) convenience
	uint32_t nextFree = ~0u;            // freelist link (unused, kept for M2)
};

// ---- Counters ---------------------------------------------------------------

struct CounterState
{
	std::atomic<int>      count{ 0 };
	std::mutex            mutex;
	std::condition_variable cv;
	std::vector<uint32_t> pending; // slot indices parked on this counter
	// alive/generation are read lock-free by
	// ResolveCounter from the worker threads (Signal/Load/WaitFor) while
	// CreateCounter/DestroyCounter write them on the submitting thread -
	// the old plain fields were the data race the dev-tsan suite caught.
	std::atomic<bool>     alive{ false };
	std::atomic<uint32_t> generation{ 0 };
};

// ---- Dedicated threads --------------------------------------------------------

struct DedicatedState
{
	std::thread            thread;
	std::string            name;
	DedicatedConfig::Idle  idle = DedicatedConfig::Idle::Sleep;
	std::deque<uint32_t>   queue; // MPSC: anyone submits, only it pops
	std::mutex             mutex;
	std::condition_variable cv;
	std::atomic<bool>      stop{ false };
	DedicatedId            id = 0;
};

// ---- The pool state --------------------------------------------------------------
// Every former loom.cpp static, owned by one Pool (the engine's LoomPool
// subsystem owns the app's; the standalone suites own theirs). Internal
// functions take the pool explicitly - the workers and the dedicated threads
// receive their own at spawn; the public free functions resolve the active
// pool.

struct PoolState
{
	std::mutex              poolMutex;
	std::deque<TaskSlot>    slots; // a deque: its elements keep stable addresses across the growth (the vector's resize moved the slots under the submit/run threads' live references)
	std::vector<uint32_t>   freeSlots;
	std::deque<CounterState> counters;
	// The old s_Init: false from the first instruction of Shutdown (TryHelp/
	// PumpMainThreadTasks/IsInitialised then report the pool as down while it
	// drains). Atomic: the workers read it.
	std::atomic<bool>       running{ false };

	std::deque<uint32_t>    anyQueue;   // MPMC run queue (M1: mutex)
	std::mutex              anyMutex;
	std::condition_variable anyCV;
	std::deque<uint32_t>    mainQueue;  // MPSC
	std::mutex              mainMutex;

	std::mutex              dedicatedMutex;
	std::vector<std::unique_ptr<DedicatedState>> dedicated;

	std::atomic<bool>       stop{ false };
	std::thread::id         mainThreadId;
	std::atomic<int>        lanesHeld{ 0 };

	// Thread stats (the pool visualizer).
	std::mutex              statsMutex;
	std::vector<std::unique_ptr<ThreadStats>> threadStats;
	std::vector<std::thread> workers;
	int                     workerCount = 0;
	std::atomic<uint64_t>   liveTasks{ 0 };

	// Runs first on every pool thread.
	void (*onThreadStart)() = nullptr;

	// The park. parkRequested is written under
	// parkMutex (a thread checks it again there before it counts itself).
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

	// The profiler shim's hooks (loom_profiler_shim.h; the old function-local
	// statics). Atomic: the owner unhooks while the workers are still alive.
	std::atomic<ProfilerStateFn> profilerStateHook{ nullptr };
	std::atomic<ProfilerZoneFn>  profilerZoneHook{ nullptr };
};

// The two process-scope values:
// - the active pool, a NON-owning access root for the free API (the role
//   Application::s_Instance plays for the engine): installed by the Pool
//   constructor, cleared at the end of its Shutdown. No lazy auto-init - a
//   call with no active pool is a bug (asserted).
// - the thread-identity allocator behind CurrentThreadIndex(): the index is a
//   thread_local that outlives any one pool, so its allocator must too (a
//   per-pool counter restarting at 0 would hand a fresh
//   worker the main thread's 0 after a re-init).
static std::atomic<PoolState*> s_ActivePool{ nullptr };
static std::atomic<int>         s_NextThreadIndex{ 0 };

static PoolState& ActivePool()
{
	PoolState* pool = s_ActivePool.load(std::memory_order_acquire);
	assert(pool && "loom: no active pool - the engine's LoomPool (or an explicit loom::Pool) is the only init");
	return *pool;
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
	if (idx >= P.counters.size()) return nullptr;
	CounterState& c = P.counters[idx];
	if (!c.alive.load(std::memory_order_relaxed) || CounterGeneration(h) != c.generation.load(std::memory_order_relaxed)) return nullptr;
	return &c;
}

// ---- Forward decls --------------------------------------------------------------

static uint32_t AllocateSlot(PoolState& P);
static void     ReleaseSlot(PoolState& P, uint32_t slot);
static void     EnqueueSlot(PoolState& P, uint32_t slot);
static void     RunSlot(PoolState& P, uint32_t slot);
static void     SignalCounter(PoolState& P, CounterHandle h, int n);
static bool     TryHelpIn(PoolState& P);
static void     ParkHere(PoolState& P, int index, const char* name);
static void     WorkerLoop(PoolState& P);
static void     DedicatedLoop(PoolState& P, DedicatedState& d);
static DedicatedState* CurrentDedicated();
static ThreadStats* RegisterThreadStats(PoolState& P, int index, const char* name);

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
	// The old atexit(Shutdown) is gone: the workers are joined by the owner's
	// Shutdown/destructor (the engine's LoomPool, a suite's local Pool), and
	// the owned std::thread objects only destruct there.
	P.running = true;
	P.stop = false;
	P.mainThreadId = std::this_thread::get_id(); // the initializing thread is "main"
	// The initializing thread claims index 0 (the
	// per-thread records' owner - the command pools, the debug_draw
	// chunks) BEFORE the workers spawn; the counter is monotonic across
	// re-inits (the dead workers' indices stay claimed, never reused -
	// a re-init resetting it to 0 would hand a fresh worker the main
	// thread's 0).
	(void)CurrentThreadIndex();

	{
		std::lock_guard<std::mutex> lock(P.poolMutex);
		uint32_t n = std::max(1u, config.maxTasks);
		P.freeSlots.reserve(n);
		for (uint32_t i = 0; i < n; i++)
		{
			P.slots.emplace_back();
			P.slots.back().fn = nullptr;
			P.freeSlots.push_back(n - 1 - i);
		}
		for (uint32_t i = 0; i < std::max(1u, config.maxCounters); i++)
			P.counters.emplace_back();
	}

	P.onThreadStart = config.onThreadStart;
	P.workerCount = config.workerCount >= 0
		? config.workerCount
		: std::max(1, (int)std::thread::hardware_concurrency() - 1);
	for (int i = 0; i < P.workerCount; i++)
		P.workers.emplace_back(WorkerLoop, std::ref(P));
}

Pool::~Pool()
{
	Shutdown();
}

void Pool::Shutdown()
{
	PoolState& P = *m_State;
	if (!P.running) return;
	P.running = false;

	// The stop flag + the notify
	// must happen under the SAME mutex the waiters pair with their
	// predicate check (the old flag-set-then-notify outside the lock
	// lost the wakeup when the notify landed between the check and the
	// wait - the worker slept forever and join() blocked).
	{
		std::lock_guard<std::mutex> lock(P.anyMutex);
		P.stop = true;
		P.anyCV.notify_all();
	}

	for (auto& w : P.workers)
		if (w.joinable()) w.join();
	P.workers.clear();

	// Dedicated threads last.
	{
		std::lock_guard<std::mutex> lock(P.dedicatedMutex);
		for (auto& d : P.dedicated)
		{
			// The dedicated waiter pairs d->cv with d->mutex - the stop
			// flag and the notify go under THAT mutex (the registry lock
			// alone left the same lost-wakeup window).
			{
				std::lock_guard<std::mutex> dlock(d->mutex);
				d->stop = true;
				d->cv.notify_all();
			}
		}
		for (auto& d : P.dedicated)
			if (d->thread.joinable()) d->thread.join();
		P.dedicated.clear();
	}

	// The counter waiters' predicates also test P.stop - wake every
	// counter's cv under its own mutex so a worker parked in
	// WaitFor(BlockOnly) can't strand the join (the same window).
	for (auto& c : P.counters)
	{
		std::lock_guard<std::mutex> lock(c.mutex);
		c.cv.notify_all();
	}

	// Lanes join like every dedicated thread; a still-held lane at
	// shutdown is a leak (the drain-or-assert policy).
	assert(P.lanesHeld.load() == 0 && "loom lanes still held at shutdown");

	// Drain-or-assert: the queues must be empty at shutdown.
	{
		std::lock_guard<std::mutex> lock(P.anyMutex);
		assert(P.anyQueue.empty() && "AnyWorker tasks abandoned at shutdown");
	}
	{
		std::lock_guard<std::mutex> lock(P.mainMutex);
		assert(P.mainQueue.empty() && "MainThread tasks abandoned at shutdown");
	}
	assert(P.liveTasks == 0 && "tasks still live at shutdown");

	{
		std::lock_guard<std::mutex> lock(P.poolMutex);
		P.slots.clear();
		P.freeSlots.clear();
		P.counters.clear();
		P.stop = false;
		{
			std::lock_guard<std::mutex> slock(P.statsMutex);
			P.threadStats.clear();
		}
	}

	// Uninstalled only now: a task still running on a draining worker
	// resolves this pool, not none.
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
	return pool ? (int)pool->liveTasks.load() : 0;
}

static int PendingTaskCountIn(PoolState& P)
{
	int n = 0;
	{
		std::lock_guard<std::mutex> lock(P.anyMutex);
		n += (int)P.anyQueue.size();
	}
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

// ---- Submission -------------------------------------------------------------------

static uint32_t AllocateSlot(PoolState& P)
{
	std::lock_guard<std::mutex> lock(P.poolMutex);
	if (!P.freeSlots.empty())
	{
		uint32_t s = P.freeSlots.back();
		P.freeSlots.pop_back();
		return s;
	}
	// Grow with a warning. The deque's push_backs never move the live
	// slots.
	uint32_t s = (uint32_t)P.slots.size();
	for (int i = 0; i < 1024; i++)
		P.slots.emplace_back();
	std::fprintf(stderr, "loom: task pool grown to %zu slots\n", P.slots.size());
	P.slots[s].fn = nullptr;
	return s;
}

static void ReleaseSlot(PoolState& P, uint32_t slot)
{
	std::lock_guard<std::mutex> lock(P.poolMutex);
	P.slots[slot].fn = nullptr;
	P.slots[slot].userData = nullptr;
	P.slots[slot].signals = kInvalidCounter;
	P.slots[slot].waitsFor = kInvalidCounter;
	P.freeSlots.push_back(slot);
}

static void EnqueueSlot(PoolState& P, uint32_t slot)
{
	TaskSlot& t = P.slots[slot];
	switch (t.affinity)
	{
	case Affinity::AnyWorker:
	{
		std::lock_guard<std::mutex> lock(P.anyMutex);
		P.anyQueue.push_back(slot);
	}
	P.anyCV.notify_one();
	break;
	case Affinity::MainThread:
	{
		std::lock_guard<std::mutex> lock(P.mainMutex);
		P.mainQueue.push_back(slot);
	}
	break;
	case Affinity::Dedicated:
	{
		DedicatedState* d = nullptr;
		{
			std::lock_guard<std::mutex> lock(P.dedicatedMutex);
			for (auto& dd : P.dedicated)
				if (dd->id == t.thread) { d = dd.get(); break; }
		}
		assert(d && "Dedicated task submitted to an unregistered thread");
		{
			std::lock_guard<std::mutex> lock(d->mutex);
			d->queue.push_back(slot);
		}
		d->cv.notify_one();
		break;
	}
	}
}

static void SubmitOne(PoolState& P, const TaskDesc& desc)
{
	assert(desc.fn && "task without a function");
	assert((desc.affinity != Affinity::Dedicated) || desc.thread != kInvalidDedicated);

	uint32_t slot = AllocateSlot(P);
	TaskSlot& t = P.slots[slot];
	t.fn = desc.fn;
	t.userData = desc.userData;
	t.affinity = desc.affinity;
	t.thread = desc.thread;
	t.signals = desc.signals;
	t.waitsFor = desc.waitsFor;
	t.nameId = desc.nameId;
	t.window = desc.window;
	P.liveTasks++;

	// waitsFor: park on the counter while nonzero, else runnable now.
	if (desc.waitsFor != kInvalidCounter)
	{
		CounterState* c = ResolveCounter(P, desc.waitsFor);
		assert(c && "task waits on an invalid/destroyed counter");
		bool runnable = false;
		{
			std::lock_guard<std::mutex> lock(c->mutex);
			if (c->count.load() > 0)
				c->pending.push_back(slot);
			else
				runnable = true;
		}
		if (runnable)
			EnqueueSlot(P, slot);
		return;
	}
	EnqueueSlot(P, slot);
}

void Submit(const TaskDesc& desc) { SubmitOne(ActivePool(), desc); }

void Submit(std::span<const TaskDesc> descs)
{
	PoolState& P = ActivePool();
	for (const TaskDesc& d : descs)
		SubmitOne(P, d);
}

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
	(void)ActivePool(); // the no-pool assert (the old lazy auto-init) even for an empty range
	if (begin >= end || chunkSize == 0 || fn == nullptr)
		return;

	size_t chunkCount = (end - begin - 1) / chunkSize + 1;
	std::vector<ParallelRange> ranges(chunkCount);
	CounterHandle counter = CreateCounter((int)chunkCount);

	size_t cursor = begin;
	for (size_t i = 0; i < chunkCount; i++)
	{
		size_t chunkEnd = cursor + chunkSize < end ? cursor + chunkSize : end;
		ranges[i] = { cursor, chunkEnd, userData, fn };
		Submit(TaskDesc{ .fn = &ParallelRangeTrampoline, .userData = &ranges[i],
			.affinity = Affinity::AnyWorker, .signals = counter });
		cursor = chunkEnd;
	}

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

namespace detail {

void SubmitInlineRaw(const void* data, size_t size, void (*trampoline)(void*), const TaskDesc& opts)
{
	PoolState& P = ActivePool();
	assert(opts.fn == nullptr && "Submit(F) and TaskDesc::fn are mutually exclusive");
	assert(size <= 48);

	uint32_t slot = AllocateSlot(P);
	TaskSlot& t = P.slots[slot];
	std::memcpy(t.inlineData, data, size);
	t.fn = trampoline;
	t.userData = t.inlineData;
	t.affinity = opts.affinity;
	t.thread = opts.thread;
	t.signals = opts.signals;
	t.waitsFor = opts.waitsFor;
	P.liveTasks++;

	if (opts.waitsFor != kInvalidCounter)
	{
		CounterState* c = ResolveCounter(P, opts.waitsFor);
		assert(c && "task waits on an invalid/destroyed counter");
		bool runnable = false;
		{
			std::lock_guard<std::mutex> lock(c->mutex);
			if (c->count.load() > 0)
				c->pending.push_back(slot);
			else
				runnable = true;
		}
		if (runnable)
			EnqueueSlot(P, slot);
		return;
	}
	EnqueueSlot(P, slot);
}

} // namespace detail

// ---- Counters -----------------------------------------------------------------------

CounterHandle CreateCounter(int initial)
{
	PoolState& P = ActivePool();
	std::lock_guard<std::mutex> lock(P.poolMutex);
	for (uint32_t i = 0; i < P.counters.size(); i++)
	{
		CounterState& c = P.counters[i];
		if (!c.alive.load(std::memory_order_relaxed))
		{
			c.alive.store(true, std::memory_order_relaxed);
			c.count.store(initial, std::memory_order_relaxed);
			// The reused counter has no waiters by contract (Destroy
			// asserts it), but a stale in-flight Signal could still touch
			// the vector - clear under the counter's own mutex (the same
			// lock Signal holds).
			{
				std::lock_guard<std::mutex> lock(c.mutex);
				c.pending.clear();
			}
			// No generation bump here: the DESTROY path invalidates stale
			// handles. Bumping on create too doubled the churn and
			// overflowed the packed generation field under per-frame
			// create/destroy cycles.
			return PackCounter(i, c.generation.load(std::memory_order_relaxed));
		}
	}
	assert(false && "counter pool exhausted");
	return kInvalidCounter;
}

void DestroyCounter(CounterHandle h)
{
	PoolState& P = ActivePool();
	CounterState* c = ResolveCounter(P, h);
	assert(c && "destroying an invalid/destroyed counter");
	assert(c->count.load() == 0 && "destroying a counter with nonzero count");
	assert(c->pending.empty() && "destroying a counter with parked waiters");
	// The pool lock CreateCounter holds: a create seeing alive == false before
	// the bump below would claim the slot with the old generation, and the bump
	// would then invalidate that live handle (the first Signal/WaitFor asserts).
	std::lock_guard<std::mutex> lock(P.poolMutex);
	c->alive.store(false, std::memory_order_relaxed);
	// Wrap the packed generation explicitly: the 12-bit field must never
	// exceed 0xFFF or the (gen << 20) pack overflows into the index bits
	// (the sandbox's per-frame counter churn hit this at ~4096 lifecycles).
	// A wrapped generation is safe: no stale handle survives 4096
	// lifecycles of its slot.
	uint32_t next = (c->generation.load(std::memory_order_relaxed) + 1) & 0xFFF;
	c->generation.store(next == 0 ? 1 : next, std::memory_order_relaxed);
}

void Signal(CounterHandle h, int n)
{
	SignalCounter(ActivePool(), h, n);
}

static void SignalCounter(PoolState& P, CounterHandle h, int n)
{
	CounterState* c = ResolveCounter(P, h);
	assert(c && "signalling an invalid/destroyed counter");
	bool hitZero = false;
	{
		std::lock_guard<std::mutex> lock(c->mutex);
		int before = c->count.fetch_sub(n);
		assert(before >= n && "counter underflow");
		if (before == n)
		{
			hitZero = true;
			for (uint32_t slot : c->pending)
				EnqueueSlot(P, slot);
			c->pending.clear();
		}
	}
	if (hitZero)
		c->cv.notify_all();
}

int Load(CounterHandle h)
{
	CounterState* c = ResolveCounter(ActivePool(), h);
	assert(c && "loading an invalid/destroyed counter");
	return c->count.load();
}

void WaitFor(CounterHandle h, Help help)
{
	PoolState& P = ActivePool();
	CounterState* c = ResolveCounter(P, h);
	assert(c && "waiting on an invalid/destroyed counter");

	if (help == Help::BlockOnly)
	{
		MarkProfilerState(P, CurrentThreadIndex(), 0); // waiting
		std::unique_lock<std::mutex> lock(c->mutex);
		c->cv.wait(lock, [&] { return c->count.load() == 0 || P.stop; });
		MarkProfilerState(P, CurrentThreadIndex(), 2);
		return;
	}

	// Help-while-waiting: run eligible tasks while the counter is nonzero.
	MarkProfilerState(P, CurrentThreadIndex(), 0); // waiting
	while (c->count.load() > 0 && !P.stop)
	{
		if (!TryHelpIn(P))
		{
			std::unique_lock<std::mutex> lock(c->mutex);
			c->cv.wait_for(lock, std::chrono::milliseconds(1),
				[&] { return c->count.load() == 0 || P.stop; });
		}
		else
			MarkProfilerState(P, CurrentThreadIndex(), 1); // helping
	}
	MarkProfilerState(P, CurrentThreadIndex(), 2); // wait over
}

// Run at most one eligible task on the CALLING thread.
bool TryHelp()
{
	PoolState* pool = s_ActivePool.load(std::memory_order_acquire);
	return pool && TryHelpIn(*pool);
}

static bool TryHelpIn(PoolState& P)
{
	if (!P.running) return false;

	if (std::this_thread::get_id() == P.mainThreadId)
	{
		// Main-affinity first (keeps main-only work deadlock-free while
		// the main thread waits), then AnyWorker. Pop under the lock,
		// run OUTSIDE it - a task may submit more work (nested
		// ParallelFor), which re-locks these queues.
		{
			uint32_t slot = ~0u;
			{
				std::lock_guard<std::mutex> lock(P.mainMutex);
				if (!P.mainQueue.empty())
				{
					slot = P.mainQueue.front();
					P.mainQueue.pop_front();
				}
			}
			if (slot != ~0u)
			{
				RunSlot(P, slot);
				return true;
			}
		}
		{
			uint32_t slot = ~0u;
			{
				std::lock_guard<std::mutex> lock(P.anyMutex);
				if (!P.anyQueue.empty())
				{
					slot = P.anyQueue.front();
					P.anyQueue.pop_front();
				}
			}
			if (slot != ~0u)
			{
				RunSlot(P, slot);
				return true;
			}
		}
		return false;
	}

	if (DedicatedState* d = CurrentDedicated())
	{
		{
			uint32_t slot = ~0u;
			{
				std::lock_guard<std::mutex> lock(d->mutex);
				if (!d->queue.empty())
				{
					slot = d->queue.front();
					d->queue.pop_front();
				}
			}
			if (slot != ~0u)
			{
				RunSlot(P, slot);
				return true;
			}
		}
		if (d->idle == DedicatedConfig::Idle::StealFromPool)
		{
			uint32_t slot = ~0u;
			{
				std::lock_guard<std::mutex> lock(P.anyMutex);
				if (!P.anyQueue.empty())
				{
					slot = P.anyQueue.front();
					P.anyQueue.pop_front();
				}
			}
			if (slot != ~0u)
			{
				RunSlot(P, slot);
				return true;
			}
		}
		return false;
	}

	{
		uint32_t slot = ~0u;
		{
			std::lock_guard<std::mutex> lock(P.anyMutex);
			if (!P.anyQueue.empty())
			{
				slot = P.anyQueue.front();
				P.anyQueue.pop_front();
			}
		}
		if (slot != ~0u)
		{
			RunSlot(P, slot);
			return true;
		}
	}
	return false;
}

static void RunSlot(PoolState& P, uint32_t slot)
{
	TaskSlot t;
	{
		std::lock_guard<std::mutex> lock(P.poolMutex);
		t = P.slots[slot];
	}
	// A per-task profiler zone (null hook = free no-op). The
	// begin hook returns the handle the end mark pairs with; the task's
	// interned name id + the window tag ride alongside (0 = unnamed).
	int threadIndex = CurrentThreadIndex();
	uint64_t zone = MarkProfilerZone(P, threadIndex, t.nameId, t.window, "Task", true, 0);
	if (t.fn)
		t.fn(t.userData);
	MarkProfilerZone(P, threadIndex, t.nameId, t.window, "Task", false, zone);
	P.liveTasks--;
	ReleaseSlot(P, slot);

	if (t.signals != kInvalidCounter)
		SignalCounter(P, t.signals, 1);
}

// ---- Main thread ---------------------------------------------------------------------

void PumpMainThreadTasks()
{
	PoolState* pool = s_ActivePool.load(std::memory_order_acquire);
	if (!pool || !pool->running) return;
	PoolState& P = *pool;
	for (int guard = 0; guard < 1000000; guard++)
	{
		uint32_t slot = ~0u;
		{
			std::lock_guard<std::mutex> lock(P.mainMutex);
			if (P.mainQueue.empty()) break;
			slot = P.mainQueue.front();
			P.mainQueue.pop_front();
		}
		RunSlot(P, slot);
	}
}

// ---- Workers --------------------------------------------------------------------------

static void WorkerLoop(PoolState& P)
{
	if (P.onThreadStart)
		P.onThreadStart();
	int index = CurrentThreadIndex(); // reserve this worker's index
	// A unique per-worker name (the pool visualizer keys its per-thread
	// history on it, and the profiler's lane names benefit too).
	static thread_local char t_name[32];
	std::snprintf(t_name, sizeof(t_name), "loom-worker-%d", index);
	ThreadStats* stats = RegisterThreadStats(P, index, t_name);
#if defined(__linux__)
	if (stats) pthread_setname_np(pthread_self(), stats->name);
#endif
	for (;;)
	{
		uint32_t slot = ~0u;
		{
			if (stats) stats->state = ThreadState::Idle;
			MarkProfilerState(P, index, 0); // waiting (the idle sleep)
			std::unique_lock<std::mutex> lock(P.anyMutex);
			P.anyCV.wait(lock, [&] { return !P.anyQueue.empty() || P.stop || P.parkRequested.load(); });
			if (P.parkRequested.load() && !P.stop)
			{
				lock.unlock();
				MarkProfilerState(P, index, 2);
				ParkHere(P, index, stats ? stats->name : "loom-worker");
				continue;
			}
			if (!P.anyQueue.empty())
			{
				slot = P.anyQueue.front();
				P.anyQueue.pop_front();
			}
			lock.unlock();
			// The app hook (the profiler's thread registration) locks the
			// pool snapshot mutexes - it must never run while this thread
			// still holds P.anyMutex (the self-deadlock).
			MarkProfilerState(P, index, 2); // wait over
			if (slot == ~0u)
			{
				if (P.stop) break;
				continue;
			}
		}
		if (stats) { stats->state = ThreadState::Running; }
		RunSlot(P, slot);
		if (stats) { stats->tasksRun++; }
	}
}

// ---- Thread stats (the pool visualizer) ------------------------------------------------

static ThreadStats* RegisterThreadStats(PoolState& P, int index, const char* name)
{
	std::lock_guard<std::mutex> lock(P.statsMutex);
	for (auto& t : P.threadStats)
		if (t->index == index)
			return t.get();
	auto stats = std::make_unique<ThreadStats>();
	stats->index = index;
	stats->name = name;
	ThreadStats* raw = stats.get();
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
		for (auto& t : P.threadStats)
		{
			ThreadStatsView view;
			view.index = t->index;
			view.name = t->name;
			view.state = t->state.load();
			view.tasksRun = t->tasksRun.load();
			view.helpedRun = t->helpedRun.load();
			view.queueDepth = 0;
			snap.threads.push_back(std::move(view));
		}
	}
	{
		std::lock_guard<std::mutex> lock(P.anyMutex);
		snap.anyQueueDepth = (int)P.anyQueue.size();
	}
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
	snap.liveTasks = (int)P.liveTasks.load();
	snap.pendingTasks = PendingTaskCountIn(P);
	return snap;
}

// ---- Dedicated threads --------------------------------------------------------------------

static thread_local DedicatedState* t_CurrentDedicated = nullptr;

static DedicatedState* CurrentDedicated() { return t_CurrentDedicated; }

static void DedicatedLoop(PoolState& P, DedicatedState& d)
{
	if (P.onThreadStart)
		P.onThreadStart();
	t_CurrentDedicated = &d;
	int index = CurrentThreadIndex(); // reserve this thread's index
	ThreadStats* stats = RegisterThreadStats(P, index, d.name.empty() ? "loom-dedicated" : d.name.c_str());
	if (!d.name.empty())
	{
#if defined(__linux__)
		pthread_setname_np(pthread_self(), d.name.c_str());
#endif
	}
	for (;;)
	{
		uint32_t slot = ~0u;
		{
			MarkProfilerState(P, index, 0); // waiting (the idle sleep)
			std::unique_lock<std::mutex> lock(d.mutex);
			d.cv.wait(lock, [&] { return !d.queue.empty() || d.stop || P.parkRequested.load(); });
			if (P.parkRequested.load() && !d.stop)
			{
				lock.unlock();
				MarkProfilerState(P, index, 2);
				ParkHere(P, index, d.name.empty() ? "loom-dedicated" : d.name.c_str());
				continue;
			}
			if (!d.queue.empty())
			{
				slot = d.queue.front();
				d.queue.pop_front();
			}
			lock.unlock();
			// The app hook must not run while this thread holds d.mutex.
			MarkProfilerState(P, index, 2); // wait over
			if (slot == ~0u)
			{
				if (d.stop) break;
				if (d.idle == DedicatedConfig::Idle::StealFromPool)
					TryHelpIn(P);
				continue;
			}
		}
		if (stats) { stats->state = ThreadState::Running; }
		RunSlot(P, slot);
		if (stats) { stats->tasksRun++; }
		if (stats) { stats->state = ThreadState::Idle; }
	}
	t_CurrentDedicated = nullptr;
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
	// The stop flag + the notify
	// under the waiter's own mutex (the old bare flag-set-then-notify
	// lost the wakeup - the same window Shutdown's worker loop had).
	{
		std::lock_guard<std::mutex> lock(d->mutex);
		d->stop = true;
		d->cv.notify_all();
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
	// Wake the idle ones under their waiters' own mutexes (the lost-wakeup
	// rule Shutdown follows).
	{
		std::lock_guard<std::mutex> lock(P.anyMutex);
		P.anyCV.notify_all();
	}
	{
		std::lock_guard<std::mutex> lock(P.dedicatedMutex);
		for (auto& d : P.dedicated)
		{
			std::lock_guard<std::mutex> dlock(d->mutex);
			d->cv.notify_all();
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
