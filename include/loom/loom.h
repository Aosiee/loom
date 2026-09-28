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
// M1 internals are mutex+condvar and correct (M2 swaps work-stealing
// deques behind the same API). Shutdown is drain-or-assert: abandoned
// queued tasks are a bug.

#include <atomic>
#include <cstdint>
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

enum class Help : uint8_t
{
	WhileWaiting, // execute eligible tasks while the counter is nonzero
	BlockOnly,    // sleep on the condvar; never reenter
};

struct TaskDesc
{
	TaskFn        fn       = nullptr;
	void*         userData = nullptr;   // caller-owned until completion
	Affinity      affinity = Affinity::AnyWorker;
	DedicatedId   thread   = {};        // required iff Dedicated
	CounterHandle signals  = kInvalidCounter; // decremented by 1 on completion
	CounterHandle waitsFor = kInvalidCounter; // runnable when it hits 0
};

struct LoomConfig
{
	int      workerCount = -1;   // default max(1, hw_concurrency - 1)
	uint32_t maxTasks    = 4096; // task-slot pool size (grows with a warning)
	uint32_t maxCounters = 2048;
};

// ---- Lifecycle -------------------------------------------------------------
void Init(const LoomConfig& config = {});
void Shutdown(); // idempotent; drains workers + dedicated threads, joins all

// ---- Submission ------------------------------------------------------------
void Submit(const TaskDesc&);
void Submit(std::span<const TaskDesc>); // batch

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
int            Load(CounterHandle);
void           WaitFor(CounterHandle, Help help = Help::WhileWaiting);
bool           TryHelp(); // run at most one eligible task; true if one ran

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
DedicatedId RegisterDedicatedThread(const DedicatedConfig&); // spawns
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
	const char* name = "";     // static storage, valid until Shutdown
	std::atomic<ThreadState> state{ ThreadState::Idle };
	std::atomic<uint64_t> tasksRun{ 0 };
	std::atomic<uint64_t> helpedRun{ 0 };
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

// ---- Introspection (tests / debug) -----------------------------------------
int  WorkerCount();
bool IsInitialised();
int  LiveTaskCount();   // allocated slots not yet completed
int  PendingTaskCount(); // tasks queued but not started

} // namespace loom
