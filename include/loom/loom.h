#pragma once

// Loom — a task & threading framework.
//
// Three execution models under one API:
//   1. Dedicated threads — long-lived, named, owned threads with their own
//      queue. Tasks can be pinned to one (Affinity::Dedicated).
//   2. Pool tasks — bursty per-frame work executed by a worker pool
//      (Affinity::AnyWorker).
//   3. Any-thread tasks — run by whoever has cycles: pool workers,
//      stealing dedicated threads, and the MAIN thread itself while it
//      waits on a dependency (help-while-waiting).
//
// Rules & hazards:
//   - Vulkan: no task may touch the allocator/device unless that path is
//     proven thread-safe; GPU-facing glue is Affinity::MainThread.
//   - Blocking IO never runs on pool workers — a dedicated thread or
//     nothing.
//   - Do NOT wait with Help::WhileWaiting while holding a lock (a wait can
//     execute arbitrary tasks on your stack); use BlockOnly or restructure.
//   - Helping means a wait can run unrelated tasks on your thread — tasks
//     must not assume ambient thread-local context beyond loom's own.
//   - Determinism: loom guarantees WHAT completes (counter semantics),
//     never scheduling ORDER. Order-independent outputs only.
//
// Internals: per-worker work-stealing deques, lock-free task slots,
// counters that lock only on their final signal; docs/design.md has the
// details. Shutdown is drain-or-assert: abandoned queued tasks are a bug.

#include <atomic>
#include <cstdint>
#include <memory>
#include <new>
#include <span>
#include <string>
#include <vector>
#include <type_traits>
#include <utility>

namespace loom {

using TaskFn = void (*)(void* userData);
using DedicatedId = uint32_t;
using CounterHandle = uint32_t;

inline constexpr DedicatedId kInvalidDedicated = ~0u;
inline constexpr CounterHandle kInvalidCounter = ~0u;

enum class Affinity : uint8_t
{
	AnyWorker,   // pool workers, helping waiters, stealing dedicated threads
	MainThread,  // ONLY the main thread (GPU submit glue, OS/Wayland calls)
	Dedicated,   // ONLY the named dedicated thread in TaskDesc::thread
};

// Which queued work runs first. High runs before Normal; Low runs only when
// no other pool work is queued anywhere (background work). A running task is
// never interrupted. For MainThread and Dedicated tasks, High goes to the
// front of that thread's queue.
enum class Priority : uint8_t
{
	High,
	Normal,
	Low,
};

enum class Help : uint8_t
{
	WhileWaiting, // execute eligible tasks while the counter is nonzero
	BlockOnly,    // sleep until the counter reaches zero; never reenter
};

struct TaskDesc
{
	TaskFn        fn       = nullptr;
	void*         userData = nullptr;   // caller-owned until completion
	Affinity      affinity = Affinity::AnyWorker;
	DedicatedId   thread   = {};        // required iff Dedicated
	CounterHandle signals  = kInvalidCounter; // decremented by 1 on completion; receives the task's exception
	CounterHandle waitsFor = kInvalidCounter; // runnable when it hits 0
	// Profiler metadata, passed through to the zone hook untouched: an
	// interned name id (0 = unnamed) and a window-slot tag.
	uint32_t      nameId   = 0;
	uint8_t       window   = 0;
	// More counters to wait for: the task becomes runnable once waitsFor and
	// every counter here have reached zero. Read only during Submit (the
	// array need not outlive it); a counter may be listed more than once.
	std::span<const CounterHandle> waitsForAll = {};
	Priority      priority = Priority::Normal;
};

struct LoomConfig
{
	int      workerCount = -1;   // default max(1, hw_concurrency - 1)
	uint32_t maxTasks    = 4096; // task-slot pool size, rounded up to 4096s (grows with a warning)
	uint32_t maxCounters = 2048;
	// Runs first on every worker and dedicated thread (the engine's crash
	// handler gives each its alternate signal stack here). Optional.
	void (*onThreadStart)() = nullptr;
	// Shutdown first drains: it runs every task (helping on the calling
	// thread) until none is live. When no task finishes for this long, what
	// is left can never run; Shutdown reports it and its asserts fire.
	uint32_t shutdownTimeoutMs = 5000;
};

// ---- Lifecycle -------------------------------------------------------------
// A Pool owns all of loom's state: task slots, queues, counters, workers,
// dedicated threads, stats and the profiler hooks. In the engine, the LoomPool
// subsystem owns the app's pool; standalone users (the loom and debug_draw
// test suites) construct one themselves. Constructing a Pool spawns its
// threads and installs it as the ACTIVE pool that every free function below
// uses. There is no lazy init: calling them with no active pool asserts. One
// pool at a time.
struct PoolState;

class Pool
{
public:
	explicit Pool(const LoomConfig& config = {});
	~Pool(); // Shutdown(), then frees the state
	Pool(const Pool&) = delete;
	Pool& operator=(const Pool&) = delete;

	// Idempotent. Drains first: every submitted task runs, including work
	// submitted while draining (main-thread work runs on the calling thread,
	// which should be the main thread), and every WaitFor completes. Then
	// joins the workers and dedicated threads and uninstalls the pool.
	// Drain-or-assert: tasks that could never run, held lanes and live tasks
	// assert (see LoomConfig::shutdownTimeoutMs).
	void Shutdown();

private:
	std::unique_ptr<PoolState> m_State;
};

// ---- Submission ------------------------------------------------------------
void Submit(const TaskDesc&);
// A batch: much cheaper per task than one Submit per task when there are many
// (runnable AnyWorker tasks are handed to the workers in groups; see
// docs/design.md). Prefer it whenever the tasks are known together.
void Submit(std::span<const TaskDesc>);

// Convenience: submits any callable. Where it is stored:
//   - trivially copyable, up to 48 bytes: its bytes, in the task slot;
//   - otherwise up to 40 bytes and nothrow-movable (a lambda capturing a
//     std::string or a shared_ptr, say): moved into the task slot;
//   - anything bigger, or aligned past 16 bytes: one heap allocation.
// The first two never allocate. A stored callable is destroyed once the task
// has run, or when it is skipped because its counter was cancelled.
namespace detail {
// Copies `size` bytes from `data` into the task slot's inline storage and
// enqueues a task that calls `trampoline` with the slot's storage pointer.
void SubmitInlineRaw(const void* data, size_t size, void (*trampoline)(void*), const TaskDesc& opts);
// Enqueues a task that owns a callable. With `relocate`, the callable is
// move-constructed from `src` into the slot (it must fit kOwnedInlineSize);
// without, `src` is a heap object the task takes over. `run` calls it and
// `destroy` ends it (freeing a heap one), after the run or the skip.
void SubmitOwnedRaw(void* src, void (*relocate)(void* dst, void* src), void (*run)(void*), void (*destroy)(void*), const TaskDesc& opts);
inline constexpr size_t kOwnedInlineSize = 40; // the slot's 48, less the destroy pointer
template <typename G>
void InlineTrampoline(void* p) { (*static_cast<G*>(p))(); }
template <typename G>
void OwnedRelocate(void* dst, void* src) noexcept { ::new (dst) G(std::move(*static_cast<G*>(src))); }
template <typename G>
void OwnedDestroyInPlace(void* p) noexcept { static_cast<G*>(p)->~G(); }
template <typename G>
void OwnedDestroyHeap(void* p) noexcept { delete static_cast<G*>(p); }
} // namespace detail

template <typename F> requires std::is_invocable_v<std::decay_t<F>&>
void Submit(F&& fn, const TaskDesc& opts = {})
{
	using G = std::decay_t<F>;
	if constexpr (std::is_trivially_copyable_v<G> && sizeof(G) <= 48 && alignof(G) <= 16)
	{
		// Materialise the decayed callable, then copy its bytes into the slot
		// (no constructor or destructor ever runs there).
		G copy(std::forward<F>(fn));
		detail::SubmitInlineRaw(&copy, sizeof(G), &detail::InlineTrampoline<G>, opts);
	}
	else if constexpr (sizeof(G) <= detail::kOwnedInlineSize && alignof(G) <= 16 && std::is_nothrow_move_constructible_v<G>)
	{
		// Built here, where a throwing copy cannot leave a half-made task, then
		// moved into the slot (nothrow) and destroyed with it.
		G copy(std::forward<F>(fn));
		detail::SubmitOwnedRaw(&copy, &detail::OwnedRelocate<G>, &detail::InlineTrampoline<G>, &detail::OwnedDestroyInPlace<G>, opts);
	}
	else
	{
		G* heap = new G(std::forward<F>(fn)); // over-aligned types get aligned new
		detail::SubmitOwnedRaw(heap, nullptr, &detail::InlineTrampoline<G>, &detail::OwnedDestroyHeap<G>, opts);
	}
}

// ---- Counters (the only synchronization primitive) -------------------------
CounterHandle CreateCounter(int initial);
void           DestroyCounter(CounterHandle);  // asserts count == 0, no waiters
void           Signal(CounterHandle, int n = 1);
// The current count. It can lag briefly behind tasks that have finished on
// other threads (a worker applies its finished tasks' signals to a counter in
// one go, at the latest when it runs out of work at hand or runs a task of
// another counter); it never runs ahead, and WaitFor is never delayed by it.
int            Load(CounterHandle);
// Returns when the count reaches zero. If a task that signals this counter
// threw, WaitFor then rethrows the first such exception (every WaitFor on the
// counter does, until it is destroyed); the throw cancelled the counter, so
// its tasks that had not started were skipped (see Cancel). Tasks a
// thread runs while helping report to their own counters, never to its wait.
// (A task that throws and signals no counter ends the program.)
void           WaitFor(CounterHandle, Help help = Help::WhileWaiting);
bool           TryHelp(); // run at most one eligible task; true if one ran
// Cancels the work behind a counter: tasks that signal it and have not
// started are skipped (they still signal it, so WaitFor returns as usual).
// Running tasks finish; a long one can poll IsCancelled to stop early. A
// task's exception cancels its counter too. A new counter is never cancelled.
void           Cancel(CounterHandle);
bool           IsCancelled(CounterHandle);

// RAII wrapper over a counter handle: destroys on scope exit; non-copyable.
class ScopedCounter
{
public:
	ScopedCounter() = default;
	explicit ScopedCounter(int initial) : m_Handle(CreateCounter(initial)) {}
	~ScopedCounter() { if (m_Handle != kInvalidCounter) DestroyCounter(m_Handle); }
	ScopedCounter(const ScopedCounter&) = delete;
	ScopedCounter& operator=(const ScopedCounter&) = delete;

	CounterHandle Get() const { return m_Handle; }
	operator CounterHandle() const { return m_Handle; }

private:
	CounterHandle m_Handle = kInvalidCounter;
};

// ---- Isolation ---------------------------------------------------------------
// Runs fn() as an isolated region. The tasks submitted inside it, and the
// tasks those submit, belong to the region; a WaitFor inside it helps only
// with the region's tasks, so a frame-critical wait never picks up an
// unrelated long task. Other work it meets in its own queues is handed to the
// other workers, and it never steals. So the region's work must not depend on
// work outside it that only this thread could run. With no workers
// (workerCount 0) a region does not isolate: nobody else could run the rest.
// Regions nest; the innermost one counts.
namespace detail {
	uint16_t BeginIsolation();   // returns the region the thread was in
	void     EndIsolation(uint16_t outer);
}

template <typename F> requires std::is_invocable_v<F&&>
void Isolated(F&& fn)
{
	// Restores the outer region even if fn throws.
	struct Scope
	{
		uint16_t outer = detail::BeginIsolation();
		~Scope() { detail::EndIsolation(outer); }
	} scope;
	std::forward<F>(fn)();
}

// ---- ParallelFor ------------------------------------------------------------
// Runs fn over [begin, end) on the pool and waits. `fn` is invoked as
// fn(subBegin, subEnd, userData) on disjoint pieces covering the range, each
// index exactly once, in no particular order.
//
// chunkSize 0 (or the overloads without one) chunks automatically: the range
// starts as one piece per thread, and a piece splits off half its remaining
// work whenever a worker is idle, so an uneven workload still spreads and a
// busy pool pays almost nothing. Otherwise the range is cut into
// ceil((end-begin)/chunkSize) fixed chunks of at most chunkSize indices
// (the last takes the remainder), one task each.
//
// If fn throws, the pieces not yet started are skipped and ParallelFor
// rethrows the first exception. Each call allocates one small array of
// piece records; the tasks themselves never allocate. The lambda forms call
// the callable in place (it lives on the caller's stack), never a heap copy.
using ParallelForFn = void (*)(size_t begin, size_t end, void* userData);

void ParallelFor(size_t begin, size_t end, size_t chunkSize, ParallelForFn fn, void* userData);
inline void ParallelFor(size_t begin, size_t end, ParallelForFn fn, void* userData) { ParallelFor(begin, end, 0, fn, userData); }

template <typename F> requires std::is_invocable_v<std::decay_t<F>&, size_t, size_t>
void ParallelFor(size_t begin, size_t end, size_t chunkSize, F&& fn)
{
	using R = std::remove_reference_t<F>;
	if constexpr (std::is_invocable_v<R&, size_t, size_t>)
	{
		// Every chunk calls the caller's own callable (alive until the wait returns).
		ParallelFor(begin, end, chunkSize,
			[](size_t b, size_t e, void* user) { (*static_cast<R*>(user))(b, e); },
			const_cast<void*>(static_cast<const void*>(std::addressof(fn))));
	}
	else
	{
		// A const callable with a non-const call operator: call a stack copy.
		using G = std::decay_t<F>;
		G copy(std::forward<F>(fn));
		ParallelFor(begin, end, chunkSize,
			[](size_t b, size_t e, void* user) { (*static_cast<G*>(user))(b, e); }, &copy);
	}
}

template <typename F> requires std::is_invocable_v<std::decay_t<F>&, size_t, size_t>
void ParallelFor(size_t begin, size_t end, F&& fn)
{
	ParallelFor(begin, end, size_t(0), std::forward<F>(fn));
}

// ---- ParallelReduce ---------------------------------------------------------
// Reduces [begin, end) to one value, reproducibly: map(chunkBegin, chunkEnd)
// gives each chunk's partial result, and the partials are combined in index
// order, result = combine(combine(identity, p0), p1)..., on the calling
// thread. The chunks depend only on the range and chunkSize, never on the
// worker count or on timing, so the same inputs give the same bits (a
// floating-point sum included). chunkSize 0 picks a size from the range length
// alone (about 256 chunks). Exceptions propagate as in ParallelFor.
template <typename T, typename Map, typename Combine>
	requires std::is_invocable_r_v<T, Map&, size_t, size_t> && std::is_invocable_r_v<T, Combine&, T, T>
T ParallelReduce(size_t begin, size_t end, size_t chunkSize, T identity, Map&& map, Combine&& combine)
{
	if (begin >= end)
		return identity;
	const size_t n = end - begin;
	if (chunkSize == 0)
		chunkSize = (n - 1) / 256 + 1;
	const size_t chunks = (n - 1) / chunkSize + 1;
	// One slot per fixed chunk, indexed by the chunk's position: each chunk
	// writes only its own slot, so no locking, and the combine order below is
	// the index order whichever thread ran what.
	std::vector<T> partials(chunks, identity);
	ParallelFor(begin, end, chunkSize, [&](size_t b, size_t e) {
		partials[(b - begin) / chunkSize] = map(b, e);
	});
	T result = std::move(identity);
	for (T& partial : partials)
		result = combine(std::move(result), std::move(partial));
	return result;
}

// ---- Lanes ------------------------------------------------------------------
// A lane is a long-lived worker reserved for ONE owner (e.g. a simulation
// tick) — implemented as a dedicated thread that never steals. Shutdown
// joins lanes like every dedicated thread and asserts none are still held.
using LaneId = uint32_t;
inline constexpr LaneId kInvalidLane = ~0u;

LaneId AcquireLane(const char* name);
void   ReleaseLane(LaneId lane);

// ---- Main thread -----------------------------------------------------------
// Call once per frame from the engine loop: a bounded drain of the
// main-affinity queue (WaitFor on the main thread pumps it too).
void PumpMainThreadTasks();

// ---- Dedicated threads -----------------------------------------------------
struct DedicatedConfig
{
	const char* name = nullptr; // thread + profiler name
	enum class Idle : uint8_t { Sleep, StealFromPool };
	Idle idle = Idle::Sleep;
};
// Spawns the thread. Each id is handed out once per pool, so a stale id
// (of an unregistered thread) never reaches a newer thread; ids are not small
// indices. Submitting to an unregistered id is a bug (asserted).
DedicatedId RegisterDedicatedThread(const DedicatedConfig&);
void         UnregisterDedicatedThread(DedicatedId);         // drains + joins

// ---- Thread identity -------------------------------------------------------
// A stable small index for the CALLING thread: 0 = the initialising
// (main) thread, 1..N = the pool workers and dedicated threads, and any
// other thread gets one on first use. Consumers key per-thread scratch
// (debug_draw's command chunks) on this index.
int CurrentThreadIndex();

// ---- Thread-pool introspection (the editor's thread-pool visualizer) --------
// Per-frame snapshots drive the CTOP-style panel: each pool thread
// reports its state (Idle/Running/Helping), the tasks it has run, and its
// queue depth; the global totals complete the header row.
enum class ThreadState : uint8_t { Idle, Running, Helping };

struct ThreadStats
{
	int         index = 0;
	const char* name = "";     // owned by the pool, valid until Shutdown
	std::atomic<ThreadState> state{ ThreadState::Idle };
	std::atomic<uint64_t> tasksRun{ 0 };  // every task this thread ran
	std::atomic<uint64_t> helpedRun{ 0 }; // the ones it ran while waiting (state Helping)
	int         queueDepth = 0; // snapshot of the owned queue (0 = shared pool)
};

// Plain-value copies (ThreadStats itself holds atomics).
struct ThreadStatsView
{
	int         index = 0;
	std::string name;
	ThreadState state = ThreadState::Idle;
	uint64_t    tasksRun = 0;
	uint64_t    helpedRun = 0;
	int         queueDepth = 0;
};

struct PoolSnapshot
{
	std::vector<ThreadStatsView> threads; // pool workers + dedicated threads
	int         anyQueueDepth = 0;     // the shared AnyWorker run queue
	int         mainQueueDepth = 0;    // the main-affinity queue
	int         liveTasks = 0;
	int         pendingTasks = 0;
	int         counterCount = 0;
};

PoolSnapshot SnapshotThreadStats();

// ---- Parking ----------------------------------------------------------------
// Stops every pool thread (the workers, the dedicated threads, the lanes) at
// a known point: each finishes its current task, records its own stack and
// blocks at its loop top until UnparkAll. Queued tasks stay queued; the
// caller keeps running and must not be a pool thread. ParkAll returns false,
// with every thread released again, when one did not park within
// `timeoutMs` (a long task).
struct ParkedThread
{
	int index = 0;
	std::string name;
	long osThreadId = 0;                    // the kernel tid (/proc/self/task/<tid>)
	std::vector<uintptr_t> returnAddresses; // innermost first
};
bool ParkAll(uint32_t timeoutMs);
void UnparkAll();
// A thread outside the pool that runs engine code (the hot-reload listener,
// the profiler writer) joins the park as a participant: ParkAll calls its
// `wake` (from the requesting thread) so it reaches its known wait point,
// where it calls ParkPoint. No-ops with no active pool.
using ParkWakeFn = void (*)(void* user);
int  RegisterParkParticipant(ParkWakeFn wake, void* user);
void UnregisterParkParticipant(int id);
bool ParkRequested();
void ParkPoint(const char* name);
// Valid between a successful ParkAll and UnparkAll.
std::vector<ParkedThread> ParkedThreads();
// The calling thread's return addresses, innermost first: a frame-pointer
// walk bounded by the thread's stack (every dev TU keeps frame pointers).
void CaptureReturnAddresses(std::vector<uintptr_t>& out, size_t max = 256);
long CurrentOsThreadId();

// ---- Introspection (tests / debug) -----------------------------------------
// All four answer for the active pool (0/false with none).
int  WorkerCount();
bool IsInitialised();
// Tasks submitted and not yet finished. A task counts until its bookkeeping
// is done, which can be a moment after the work it was waited for (an
// automatically chunked ParallelFor counts items, not tasks).
int  LiveTaskCount();
int  PendingTaskCount(); // tasks queued but not started

} // namespace loom
