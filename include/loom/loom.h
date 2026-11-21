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
	// The app-supplied task metadata,
	// forwarded verbatim to the profiler zone hook (loom never interprets
	// it) - the interned name id (0 = unnamed) + the window-slot tag.
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
// The pool's whole state (task slots,
// queues, counters, workers, dedicated threads, stats, the profiler hooks)
// is one PoolState owned by a Pool. The engine's LoomPool subsystem owns the
// app's (its Init is the only init in an app); a standalone user - the
// loom/debug_draw suites - owns one explicitly. Constructing a Pool installs
// it as the ACTIVE pool every free function below resolves and spawns its
// threads; there is no lazy auto-init (a call with no active pool asserts).
// One pool at a time.
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

// Convenience: inline-stores a trivially-copyable callable of <= 48 bytes
// into the task slot — zero allocation.
namespace detail {
// Copies `size` bytes from `data` into the task slot's inline storage and
// enqueues a task that calls `trampoline` with the slot's storage pointer.
void SubmitInlineRaw(const void* data, size_t size, void (*trampoline)(void*), const TaskDesc& opts);
template <typename G>
void InlineTrampoline(void* p) { (*static_cast<G*>(p))(); }
} // namespace detail

template <typename F> requires std::is_invocable_v<std::decay_t<F>&>
void Submit(F&& fn, const TaskDesc& opts = {})
{
	using G = std::decay_t<F>;
	static_assert(std::is_trivially_copyable_v<G>,
		"loom::Submit(F) requires a trivially-copyable callable");
	static_assert(sizeof(G) <= 48,
		"loom::Submit(F) callables must fit in 48 bytes");
	G copy(std::forward<F>(fn));
	detail::SubmitInlineRaw(&copy, sizeof(G), &detail::InlineTrampoline<G>, opts);
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
// counter does, until it is destroyed). The other tasks still ran. Tasks a
// thread runs while helping report to their own counters, never to its wait.
// (A task that throws and signals no counter ends the program.)
void           WaitFor(CounterHandle, Help help = Help::WhileWaiting);
bool           TryHelp(); // run at most one eligible task; true if one ran

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

// ---- ParallelFor ------------------------------------------------------------
// Splits [begin, end) into ceil((end-begin)/chunkSize) contiguous chunks -
// each task processes AT MOST chunkSize indices (the last chunk takes the
// remainder) - fans each out as an AnyWorker task signalling one counter,
// and waits for all of them. `fn` is invoked as fn(chunkBegin, chunkEnd,
// userData). Each call allocates one array of chunk records (one entry per
// chunk, freed when the wait returns); the chunks themselves take task
// slots, which never allocate. The lambda convenience calls the callable in
// place (it lives on the caller's stack until the wait returns), never a
// heap copy. Pick chunkSize so there are a small multiple of WorkerCount()
// chunks, each with enough work to dwarf the per-task tax (see
// bench/loom_bench.cpp).
using ParallelForFn = void (*)(size_t begin, size_t end, void* userData);

void ParallelFor(size_t begin, size_t end, size_t chunkSize, ParallelForFn fn, void* userData);

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
int  LiveTaskCount();   // allocated slots not yet completed
int  PendingTaskCount(); // tasks queued but not started

} // namespace loom
