#include "loom.h"
#include "loom_profiler_shim.h"

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

// ---- Globals ------------------------------------------------------------------

static std::mutex              s_PoolMutex;
static std::deque<TaskSlot>    s_Slots; // a deque: its elements keep stable addresses across the growth (the vector's resize moved the slots under the submit/run threads' live references)
static std::vector<uint32_t>   s_FreeSlots;
static std::deque<CounterState> s_Counters;
static bool                    s_Init = false;

static std::deque<uint32_t>    s_AnyQueue;   // MPMC run queue (M1: mutex)
static std::mutex              s_AnyMutex;
static std::condition_variable s_AnyCV;
static std::deque<uint32_t>    s_MainQueue;  // MPSC
static std::mutex              s_MainMutex;

static std::mutex              s_DedicatedMutex;
static std::vector<std::unique_ptr<DedicatedState>> s_Dedicated;

static std::atomic<bool>       s_Stop{ false };
static std::thread::id          s_MainThreadId;
static std::atomic<int>         s_NextThreadIndex{ 0 };
static std::atomic<int>         s_LanesHeld{ 0 };

// Thread stats (the pool visualizer).
static std::mutex s_StatsMutex;
static std::vector<std::unique_ptr<ThreadStats>> s_ThreadStats;
static std::vector<std::thread> s_Workers;
static int                     s_WorkerCount = 0;
static std::atomic<uint64_t>   s_LiveTasks{ 0 };

// Counter handle packing: generation in the high bits, index low.
static constexpr uint32_t kCounterGenBits = 20;
static constexpr uint32_t kCounterIndexMask = (1u << kCounterGenBits) - 1;

static CounterHandle PackCounter(uint32_t index, uint32_t generation)
{
	return (generation << kCounterGenBits) | index;
}
static uint32_t CounterIndex(CounterHandle h) { return h & kCounterIndexMask; }
static uint32_t CounterGeneration(CounterHandle h) { return h >> kCounterGenBits; }

static CounterState* ResolveCounter(CounterHandle h)
{
	if (h == kInvalidCounter) return nullptr;
	uint32_t idx = CounterIndex(h);
	if (idx >= s_Counters.size()) return nullptr;
	CounterState& c = s_Counters[idx];
	if (!c.alive.load(std::memory_order_relaxed) || CounterGeneration(h) != c.generation.load(std::memory_order_relaxed)) return nullptr;
	return &c;
}

// ---- Forward decls --------------------------------------------------------------

static uint32_t AllocateSlot();
static void     ReleaseSlot(uint32_t slot);
static void     EnqueueSlot(uint32_t slot);
static void     RunSlot(uint32_t slot);
static void     WorkerLoop();
static void     DedicatedLoop(DedicatedState& d);
static DedicatedState* CurrentDedicated();
static ThreadStats* RegisterThreadStats(int index, const char* name);

// ---- Lifecycle -------------------------------------------------------------------

void Init(const LoomConfig& config)
{
	if (s_Init) return;
	// The workers MUST be joined before main returns or the std::thread
	// destructors std::terminate (the editor's exit hang). Registering the
	// idempotent shutdown with atexit covers every teardown path, including
	// ones that never reach an app layer's OnDetach.
	{
		static bool registered = false;
		if (!registered)
		{
			registered = true;
			std::atexit([]() { Shutdown(); });
		}
	}
	s_Init = true;
	s_Stop = false;
	s_MainThreadId = std::this_thread::get_id(); // the initializing thread is "main"
	// The initializing thread claims index 0 (the
	// per-thread records' owner - the command pools, the debug_draw
	// chunks) BEFORE the workers spawn; the counter is monotonic across
	// re-inits (the dead workers' indices stay claimed, never reused -
	// a re-init resetting it to 0 would hand a fresh worker the main
	// thread's 0).
	(void)CurrentThreadIndex();

	{
		std::lock_guard<std::mutex> lock(s_PoolMutex);
		uint32_t n = std::max(1u, config.maxTasks);
		s_Slots.clear();
		s_FreeSlots.clear();
		s_FreeSlots.reserve(n);
		for (uint32_t i = 0; i < n; i++)
		{
			s_Slots.emplace_back();
			s_Slots.back().fn = nullptr;
			s_FreeSlots.push_back(n - 1 - i);
		}
		s_Counters.clear();
		for (uint32_t i = 0; i < std::max(1u, config.maxCounters); i++)
			s_Counters.emplace_back();
	}

	s_WorkerCount = config.workerCount >= 0
		? config.workerCount
		: std::max(1, (int)std::thread::hardware_concurrency() - 1);
	for (int i = 0; i < s_WorkerCount; i++)
		s_Workers.emplace_back(WorkerLoop);
}

void Shutdown()
{
	if (!s_Init) return;
	s_Init = false;

	// The stop flag + the notify
	// must happen under the SAME mutex the waiters pair with their
	// predicate check (the old flag-set-then-notify outside the lock
	// lost the wakeup when the notify landed between the check and the
	// wait - the worker slept forever and join() blocked).
	{
		std::lock_guard<std::mutex> lock(s_AnyMutex);
		s_Stop = true;
		s_AnyCV.notify_all();
	}

	for (auto& w : s_Workers)
		if (w.joinable()) w.join();
	s_Workers.clear();

	// Dedicated threads last.
	{
		std::lock_guard<std::mutex> lock(s_DedicatedMutex);
		for (auto& d : s_Dedicated)
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
		for (auto& d : s_Dedicated)
			if (d->thread.joinable()) d->thread.join();
		s_Dedicated.clear();
	}

	// The counter waiters' predicates also test s_Stop - wake every
	// counter's cv under its own mutex so a worker parked in
	// WaitFor(BlockOnly) can't strand the join (the same window).
	for (auto& c : s_Counters)
	{
		std::lock_guard<std::mutex> lock(c.mutex);
		c.cv.notify_all();
	}

	// Lanes join like every dedicated thread; a still-held lane at
	// shutdown is a leak (the drain-or-assert policy).
	assert(s_LanesHeld.load() == 0 && "loom lanes still held at shutdown");

	// Drain-or-assert: the queues must be empty at shutdown.
	{
		std::lock_guard<std::mutex> lock(s_AnyMutex);
		assert(s_AnyQueue.empty() && "AnyWorker tasks abandoned at shutdown");
	}
	{
		std::lock_guard<std::mutex> lock(s_MainMutex);
		assert(s_MainQueue.empty() && "MainThread tasks abandoned at shutdown");
	}
	assert(s_LiveTasks == 0 && "tasks still live at shutdown");

	std::lock_guard<std::mutex> lock(s_PoolMutex);
	s_Slots.clear();
	s_FreeSlots.clear();
	s_Counters.clear();
	s_Stop = false;
	{
		std::lock_guard<std::mutex> slock(s_StatsMutex);
		s_ThreadStats.clear();
	}
}

bool IsInitialised() { return s_Init; }
int  WorkerCount()   { return s_WorkerCount; }

int CurrentThreadIndex()
{
	static thread_local int t_index = -1;
	if (t_index >= 0)
		return t_index;
	t_index = s_NextThreadIndex.fetch_add(1);
	return t_index;
}
int  LiveTaskCount() { return (int)s_LiveTasks.load(); }

int PendingTaskCount()
{
	int n = 0;
	{
		std::lock_guard<std::mutex> lock(s_AnyMutex);
		n += (int)s_AnyQueue.size();
	}
	{
		std::lock_guard<std::mutex> lock(s_MainMutex);
		n += (int)s_MainQueue.size();
	}
	std::lock_guard<std::mutex> lock(s_DedicatedMutex);
	for (auto& d : s_Dedicated)
	{
		std::lock_guard<std::mutex> dl(d->mutex);
		n += (int)d->queue.size();
	}
	return n;
}

// ---- Submission -------------------------------------------------------------------

static uint32_t AllocateSlot()
{
	std::lock_guard<std::mutex> lock(s_PoolMutex);
	if (!s_FreeSlots.empty())
	{
		uint32_t s = s_FreeSlots.back();
		s_FreeSlots.pop_back();
		return s;
	}
	// Grow with a warning. The deque's push_backs never move the live
	// slots.
	uint32_t s = (uint32_t)s_Slots.size();
	for (int i = 0; i < 1024; i++)
		s_Slots.emplace_back();
	std::fprintf(stderr, "loom: task pool grown to %zu slots\n", s_Slots.size());
	s_Slots[s].fn = nullptr;
	return s;
}

static void ReleaseSlot(uint32_t slot)
{
	std::lock_guard<std::mutex> lock(s_PoolMutex);
	s_Slots[slot].fn = nullptr;
	s_Slots[slot].userData = nullptr;
	s_Slots[slot].signals = kInvalidCounter;
	s_Slots[slot].waitsFor = kInvalidCounter;
	s_FreeSlots.push_back(slot);
}

static void EnqueueSlot(uint32_t slot)
{
	TaskSlot& t = s_Slots[slot];
	switch (t.affinity)
	{
	case Affinity::AnyWorker:
	{
		std::lock_guard<std::mutex> lock(s_AnyMutex);
		s_AnyQueue.push_back(slot);
	}
	s_AnyCV.notify_one();
	break;
	case Affinity::MainThread:
	{
		std::lock_guard<std::mutex> lock(s_MainMutex);
		s_MainQueue.push_back(slot);
	}
	break;
	case Affinity::Dedicated:
	{
		DedicatedState* d = nullptr;
		{
			std::lock_guard<std::mutex> lock(s_DedicatedMutex);
			for (auto& dd : s_Dedicated)
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

static void SubmitOne(const TaskDesc& desc)
{
	if (!s_Init) Init();
	assert(desc.fn && "task without a function");
	assert((desc.affinity != Affinity::Dedicated) || desc.thread != kInvalidDedicated);

	uint32_t slot = AllocateSlot();
	TaskSlot& t = s_Slots[slot];
	t.fn = desc.fn;
	t.userData = desc.userData;
	t.affinity = desc.affinity;
	t.thread = desc.thread;
	t.signals = desc.signals;
	t.waitsFor = desc.waitsFor;
	t.nameId = desc.nameId;
	t.window = desc.window;
	s_LiveTasks++;

	// waitsFor: park on the counter while nonzero, else runnable now.
	if (desc.waitsFor != kInvalidCounter)
	{
		CounterState* c = ResolveCounter(desc.waitsFor);
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
			EnqueueSlot(slot);
		return;
	}
	EnqueueSlot(slot);
}

void Submit(const TaskDesc& desc) { SubmitOne(desc); }

void Submit(std::span<const TaskDesc> descs)
{
	if (!s_Init) Init();
	for (const TaskDesc& d : descs)
		SubmitOne(d);
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
	if (!s_Init) Init();
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
	if (!s_Init) Init();
	s_LanesHeld.fetch_add(1, std::memory_order_relaxed);
	return RegisterDedicatedThread(DedicatedConfig{ .name = name, .idle = DedicatedConfig::Idle::Sleep });
}

void ReleaseLane(LaneId lane)
{
	// Only decrement when the lane actually existed - a double release or
	// an invalid id must not corrupt the shutdown-held-lanes counter.
	{
		std::lock_guard<std::mutex> lock(s_DedicatedMutex);
		bool found = false;
		for (const auto& d : s_Dedicated)
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
	s_LanesHeld.fetch_sub(1, std::memory_order_relaxed);
}

namespace detail {

void SubmitInlineRaw(const void* data, size_t size, void (*trampoline)(void*), const TaskDesc& opts)
{
	if (!s_Init) Init();
	assert(opts.fn == nullptr && "Submit(F) and TaskDesc::fn are mutually exclusive");
	assert(size <= 48);

	uint32_t slot = AllocateSlot();
	TaskSlot& t = s_Slots[slot];
	std::memcpy(t.inlineData, data, size);
	t.fn = trampoline;
	t.userData = t.inlineData;
	t.affinity = opts.affinity;
	t.thread = opts.thread;
	t.signals = opts.signals;
	t.waitsFor = opts.waitsFor;
	s_LiveTasks++;

	if (opts.waitsFor != kInvalidCounter)
	{
		CounterState* c = ResolveCounter(opts.waitsFor);
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
			EnqueueSlot(slot);
		return;
	}
	EnqueueSlot(slot);
}

} // namespace detail

// ---- Counters -----------------------------------------------------------------------

CounterHandle CreateCounter(int initial)
{
	if (!s_Init) Init();
	std::lock_guard<std::mutex> lock(s_PoolMutex);
	for (uint32_t i = 0; i < s_Counters.size(); i++)
	{
		CounterState& c = s_Counters[i];
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
	CounterState* c = ResolveCounter(h);
	assert(c && "destroying an invalid/destroyed counter");
	assert(c->count.load() == 0 && "destroying a counter with nonzero count");
	assert(c->pending.empty() && "destroying a counter with parked waiters");
	// The pool lock CreateCounter holds: a create seeing alive == false before
	// the bump below would claim the slot with the old generation, and the bump
	// would then invalidate that live handle (the first Signal/WaitFor asserts).
	std::lock_guard<std::mutex> lock(s_PoolMutex);
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
	CounterState* c = ResolveCounter(h);
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
				EnqueueSlot(slot);
			c->pending.clear();
		}
	}
	if (hitZero)
		c->cv.notify_all();
}

int Load(CounterHandle h)
{
	CounterState* c = ResolveCounter(h);
	assert(c && "loading an invalid/destroyed counter");
	return c->count.load();
}

void WaitFor(CounterHandle h, Help help)
{
	CounterState* c = ResolveCounter(h);
	assert(c && "waiting on an invalid/destroyed counter");

	if (help == Help::BlockOnly)
	{
		MarkProfilerState(CurrentThreadIndex(), 0); // waiting
		std::unique_lock<std::mutex> lock(c->mutex);
		c->cv.wait(lock, [&] { return c->count.load() == 0 || s_Stop; });
		MarkProfilerState(CurrentThreadIndex(), 2);
		return;
	}

	// Help-while-waiting: run eligible tasks while the counter is nonzero.
	MarkProfilerState(CurrentThreadIndex(), 0); // waiting
	while (c->count.load() > 0 && !s_Stop)
	{
		if (!TryHelp())
		{
			std::unique_lock<std::mutex> lock(c->mutex);
			c->cv.wait_for(lock, std::chrono::milliseconds(1),
				[&] { return c->count.load() == 0 || s_Stop; });
		}
		else
			MarkProfilerState(CurrentThreadIndex(), 1); // helping
	}
	MarkProfilerState(CurrentThreadIndex(), 2); // wait over
}

// Run at most one eligible task on the CALLING thread.
bool TryHelp()
{
	if (!s_Init) return false;

	if (std::this_thread::get_id() == s_MainThreadId)
	{
		// Main-affinity first (keeps main-only work deadlock-free while
		// the main thread waits), then AnyWorker. Pop under the lock,
		// run OUTSIDE it - a task may submit more work (nested
		// ParallelFor), which re-locks these queues.
		{
			uint32_t slot = ~0u;
			{
				std::lock_guard<std::mutex> lock(s_MainMutex);
				if (!s_MainQueue.empty())
				{
					slot = s_MainQueue.front();
					s_MainQueue.pop_front();
				}
			}
			if (slot != ~0u)
			{
				RunSlot(slot);
				return true;
			}
		}
		{
			uint32_t slot = ~0u;
			{
				std::lock_guard<std::mutex> lock(s_AnyMutex);
				if (!s_AnyQueue.empty())
				{
					slot = s_AnyQueue.front();
					s_AnyQueue.pop_front();
				}
			}
			if (slot != ~0u)
			{
				RunSlot(slot);
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
				RunSlot(slot);
				return true;
			}
		}
		if (d->idle == DedicatedConfig::Idle::StealFromPool)
		{
			uint32_t slot = ~0u;
			{
				std::lock_guard<std::mutex> lock(s_AnyMutex);
				if (!s_AnyQueue.empty())
				{
					slot = s_AnyQueue.front();
					s_AnyQueue.pop_front();
				}
			}
			if (slot != ~0u)
			{
				RunSlot(slot);
				return true;
			}
		}
		return false;
	}

	{
		uint32_t slot = ~0u;
		{
			std::lock_guard<std::mutex> lock(s_AnyMutex);
			if (!s_AnyQueue.empty())
			{
				slot = s_AnyQueue.front();
				s_AnyQueue.pop_front();
			}
		}
		if (slot != ~0u)
		{
			RunSlot(slot);
			return true;
		}
	}
	return false;
}

static void RunSlot(uint32_t slot)
{
	TaskSlot t;
	{
		std::lock_guard<std::mutex> lock(s_PoolMutex);
		t = s_Slots[slot];
	}
	// A per-task profiler zone (null hook = free no-op). The
	// begin hook returns the handle the end mark pairs with; the task's
	// interned name id + the window tag ride alongside (0 = unnamed).
	int threadIndex = CurrentThreadIndex();
	uint64_t zone = MarkProfilerZone(threadIndex, t.nameId, t.window, "Task", true, 0);
	if (t.fn)
		t.fn(t.userData);
	MarkProfilerZone(threadIndex, t.nameId, t.window, "Task", false, zone);
	s_LiveTasks--;
	ReleaseSlot(slot);

	if (t.signals != kInvalidCounter)
		Signal(t.signals, 1);
}

// ---- Main thread ---------------------------------------------------------------------

void PumpMainThreadTasks()
{
	if (!s_Init) return;
	for (int guard = 0; guard < 1000000; guard++)
	{
		uint32_t slot = ~0u;
		{
			std::lock_guard<std::mutex> lock(s_MainMutex);
			if (s_MainQueue.empty()) break;
			slot = s_MainQueue.front();
			s_MainQueue.pop_front();
		}
		RunSlot(slot);
	}
}

// ---- Workers --------------------------------------------------------------------------

static void WorkerLoop()
{
	int index = CurrentThreadIndex(); // reserve this worker's index
	// A unique per-worker name (the pool visualizer keys its per-thread
	// history on it, and the profiler's lane names benefit too).
	static thread_local char t_name[32];
	std::snprintf(t_name, sizeof(t_name), "loom-worker-%d", index);
	ThreadStats* stats = RegisterThreadStats(index, t_name);
#if defined(__linux__)
	if (stats) pthread_setname_np(pthread_self(), stats->name);
#endif
	for (;;)
	{
		uint32_t slot = ~0u;
		{
			if (stats) stats->state = ThreadState::Idle;
			MarkProfilerState(index, 0); // waiting (the idle sleep)
			std::unique_lock<std::mutex> lock(s_AnyMutex);
			s_AnyCV.wait(lock, [&] { return !s_AnyQueue.empty() || s_Stop; });
			if (!s_AnyQueue.empty())
			{
				slot = s_AnyQueue.front();
				s_AnyQueue.pop_front();
			}
			lock.unlock();
			// The app hook (the profiler's thread registration) locks the
			// pool snapshot mutexes - it must never run while this thread
			// still holds s_AnyMutex (the self-deadlock).
			MarkProfilerState(index, 2); // wait over
			if (slot == ~0u)
			{
				if (s_Stop) break;
				continue;
			}
		}
		if (stats) { stats->state = ThreadState::Running; }
		RunSlot(slot);
		if (stats) { stats->tasksRun++; }
	}
}

// ---- Thread stats (the pool visualizer) ------------------------------------------------

static ThreadStats* RegisterThreadStats(int index, const char* name)
{
	std::lock_guard<std::mutex> lock(s_StatsMutex);
	for (auto& t : s_ThreadStats)
		if (t->index == index)
			return t.get();
	auto stats = std::make_unique<ThreadStats>();
	stats->index = index;
	stats->name = name;
	ThreadStats* raw = stats.get();
	s_ThreadStats.push_back(std::move(stats));
	return raw;
}

PoolSnapshot SnapshotThreadStats()
{
	PoolSnapshot snap;
	{
		std::lock_guard<std::mutex> lock(s_StatsMutex);
		for (auto& t : s_ThreadStats)
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
		std::lock_guard<std::mutex> lock(s_AnyMutex);
		snap.anyQueueDepth = (int)s_AnyQueue.size();
	}
	{
		std::lock_guard<std::mutex> lock(s_MainMutex);
		snap.mainQueueDepth = (int)s_MainQueue.size();
	}
	{
		std::lock_guard<std::mutex> lock(s_DedicatedMutex);
		for (auto& d : s_Dedicated)
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
	snap.liveTasks = (int)s_LiveTasks.load();
	snap.pendingTasks = PendingTaskCount();
	return snap;
}

// ---- Dedicated threads --------------------------------------------------------------------

static thread_local DedicatedState* t_CurrentDedicated = nullptr;

static DedicatedState* CurrentDedicated() { return t_CurrentDedicated; }

static void DedicatedLoop(DedicatedState& d)
{
	t_CurrentDedicated = &d;
	int index = CurrentThreadIndex(); // reserve this thread's index
	ThreadStats* stats = RegisterThreadStats(index, d.name.empty() ? "loom-dedicated" : d.name.c_str());
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
			MarkProfilerState(index, 0); // waiting (the idle sleep)
			std::unique_lock<std::mutex> lock(d.mutex);
			d.cv.wait(lock, [&] { return !d.queue.empty() || d.stop; });
			if (!d.queue.empty())
			{
				slot = d.queue.front();
				d.queue.pop_front();
			}
			lock.unlock();
			// The app hook must not run while this thread holds d.mutex.
			MarkProfilerState(index, 2); // wait over
			if (slot == ~0u)
			{
				if (d.stop) break;
				if (d.idle == DedicatedConfig::Idle::StealFromPool)
					TryHelp();
				continue;
			}
		}
		if (stats) { stats->state = ThreadState::Running; }
		RunSlot(slot);
		if (stats) { stats->tasksRun++; }
		if (stats) { stats->state = ThreadState::Idle; }
	}
	t_CurrentDedicated = nullptr;
}

DedicatedId RegisterDedicatedThread(const DedicatedConfig& config)
{
	if (!s_Init) Init();
	std::lock_guard<std::mutex> lock(s_DedicatedMutex);
	uint32_t id = 0;
	for (;;)
	{
		bool taken = false;
		for (auto& d : s_Dedicated)
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
	d->thread = std::thread(DedicatedLoop, std::ref(*d));
	s_Dedicated.push_back(std::move(d));
	return handle;
}

void UnregisterDedicatedThread(DedicatedId id)
{
	std::unique_ptr<DedicatedState> d;
	{
		std::lock_guard<std::mutex> lock(s_DedicatedMutex);
		for (auto it = s_Dedicated.begin(); it != s_Dedicated.end(); ++it)
		{
			if ((*it)->id == id)
			{
				d = std::move(*it);
				s_Dedicated.erase(it);
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

} // namespace loom
