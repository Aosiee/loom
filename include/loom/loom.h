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
#include <exception>
#include <memory>
#include <new>
#include <optional>
#include <span>
#include <string>
#include <vector>
#include <type_traits>
#include <utility>

namespace loom {

using TaskFn = void (*)(void* userData);
using DedicatedId = uint32_t;
using CounterHandle = uint32_t;
using LimiterHandle = uint32_t;

inline constexpr DedicatedId kInvalidDedicated = ~0u;
inline constexpr CounterHandle kInvalidCounter = ~0u;
inline constexpr LimiterHandle kInvalidLimiter = ~0u;

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
	// Caps how many tasks of this limiter run at once (see CreateLimiter).
	LimiterHandle limiter  = kInvalidLimiter;
};

struct LoomConfig
{
	int      workerCount = -1;   // default max(1, hw_concurrency - 1)
	uint32_t maxTasks    = 4096; // task-slot pool size, rounded up to 4096s (grows with a warning)
	uint32_t maxCounters = 2048;
	uint32_t maxLimiters = 64;
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
// WaitFor with a time limit: true once the count reached zero (rethrowing as
// WaitFor does), false if `timeoutMs` passed first. 0 only checks, without
// helping. While helping, the limit is checked between tasks, so a long task
// run on the way can overrun it.
[[nodiscard]] bool TryWaitFor(CounterHandle, uint32_t timeoutMs, Help help = Help::WhileWaiting);
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

// ---- Concurrency limits -------------------------------------------------------
// A limiter caps how many of its tasks are in flight at once, so a kind of
// work cannot take every worker: asset decompression on at most 4 threads
// while the frame keeps the rest, or uploads that each need one of N staging
// buffers. A task with TaskDesc::limiter takes one of the limiter's units when
// it becomes runnable (at Submit, or once its counters reach zero) and gives
// it back when it finishes. While none is free it waits in the limiter, not
// in a queue, and the task that finishes hands its unit to the next one:
// High tasks first, otherwise in the order they became runnable. A limit of 1
// makes a serial queue.
//
// A unit is held from queuing to the end of the run, so at most `limit` of
// the tasks are queued or running, never more running. Low priority decides
// when background work may start; a limiter decides how many workers it may
// hold once it has. Tasks without a limiter pay one untaken branch.
//
// A limited task must not wait for another task of its own limiter: with
// every unit held, that task never becomes runnable. Limited tasks may be of
// any affinity, may be TaskGraph nodes, and are skipped by Cancel as usual
// (a skipped or throwing task gives its unit back too).
LimiterHandle CreateLimiter(uint32_t limit); // limit >= 1; LoomConfig::maxLimiters at once
void          DestroyLimiter(LimiterHandle); // asserts none of its tasks is live
// The units held right now (tasks queued or running), and the tasks waiting
// for one. Snapshots for tests and debug views.
uint32_t      LimiterHeld(LimiterHandle);
uint32_t      LimiterWaiting(LimiterHandle);

// Changes the limit while tasks use the limiter, from any thread. Raising it
// starts waiting tasks at once. Lowering it never stops a task: free units go
// at once, held ones as their tasks finish, and no task takes a unit until
// fewer than the new limit are held. LimiterHeld can exceed the limit
// meanwhile.
void          SetLimiterLimit(LimiterHandle, uint32_t limit); // limit >= 1
uint32_t      LimiterLimit(LimiterHandle);

// Which of its limiter's units the running task holds: an index in
// [0, limit), the same for the whole run and different from every other task
// of that limiter running at the same time, so it can pick one of `limit`
// scratch buffers or staging slots without a lock. Indices are reused, the
// most recently freed first (0 first while all are free), so the buffers in
// use stay few and warm. kNoUnit outside a limited task.
// After the limit is lowered, a task over the new limit can get an index at
// or above it (until enough tasks have finished); size buffers for the
// highest limit used. A future's task has none once its value is ready.
inline constexpr uint32_t kNoUnit = ~0u;
uint32_t      LimiterUnit();

// RAII wrapper over a limiter handle: destroys on scope exit; non-copyable.
class ScopedLimiter
{
public:
	ScopedLimiter() = default;
	explicit ScopedLimiter(uint32_t limit) : m_Handle(CreateLimiter(limit)) {}
	~ScopedLimiter() { if (m_Handle != kInvalidLimiter) DestroyLimiter(m_Handle); }
	ScopedLimiter(const ScopedLimiter&) = delete;
	ScopedLimiter& operator=(const ScopedLimiter&) = delete;

	LimiterHandle Get() const { return m_Handle; }
	operator LimiterHandle() const { return m_Handle; }

private:
	LimiterHandle m_Handle = kInvalidLimiter;
};

// ---- Futures ------------------------------------------------------------------
// Async runs a callable as a task and returns a Future for its result: the
// asset that a load parses, the path a search finds. Get() waits (helping,
// by default) and returns the value, or rethrows what the callable threw.
//
//   loom::Future<Mesh> mesh = loom::Async([path] { return ParseMesh(path); });
//   ...
//   Draw(mesh.Get());
//
// Then(fn) runs fn(value) as a task once the value is ready and returns a
// future for fn's result, so steps chain without writing the waits:
//
//   loom::Future<GpuMesh> gpu = loom::Async([path] { return ReadFile(path); })
//       .Then([](Bytes& bytes) { return ParseMesh(bytes); })
//       .Then([](Mesh& mesh) { return Upload(mesh); }, { .affinity = loom::Affinity::MainThread });
//
// A step whose input failed is skipped and its future fails the same way, so
// one Get at the end of a chain sees the first error. A continuation keeps
// what it reads alive: the futures before it may be dropped.
//
// A Promise is a future whose value comes from outside loom: an IO
// callback, a GPU fence, another library's thread. SetValue (or
// SetException) completes the future, from any thread. A promise destroyed
// without either completes it with BrokenPromise, so a Get never hangs.
//
// A future is also a counter: Counter() is zero once the value is ready, so
// a plain task can wait for it (TaskDesc::waitsFor, or waitsForAll for
// several) and read Get() without blocking. Keep the future alive until that
// task has read it.
//
// A future's state (the callable, the result, a counter) is one heap
// allocation, freed once the future, its task or promise, and any
// continuation reading it have all let go. Dropping a future before its
// value is there is fine: the task still runs. A future's task always runs:
// it ignores Cancel. Each pending future holds one counter
// (LoomConfig::maxCounters), and like a counter it must not outlive the pool.
//
// The TaskDesc of Async and Then gives the task's affinity, thread,
// priority, limiter, waitsForAll and profiler fields; fn, userData and
// signals must be unset (the future signals its own counter), and so must
// waitsFor for Then (it waits for the future).

// What Get throws when the promise behind a future was destroyed unset.
struct BrokenPromise : std::exception
{
	const char* what() const noexcept override { return "loom: a promise was destroyed without a value"; }
};

template <typename T> class Future;
template <typename T> class Promise;

namespace detail {
	struct FutureStateBase
	{
		CounterHandle    counter = kInvalidCounter;
		std::atomic<int> refs{ 2 }; // one per owner: the future, its task or promise, continuations
		void (*destroy)(FutureStateBase*) noexcept = nullptr;
		bool             broken = false; // completed without a value (also `error`, with exceptions)
#if defined(__cpp_exceptions)
		std::exception_ptr error;
#endif
		bool Failed() const
		{
#if defined(__cpp_exceptions)
			return broken || error != nullptr;
#else
			return broken;
#endif
		}
	};
	// Creates the state's counter (count 1: its value).
	void CreateFutureCounter(FutureStateBase* state);
	// CreateFutureCounter, then submits `run` with the state as its userData,
	// waiting also for `after` when given.
	void StartFuture(FutureStateBase* state, TaskFn run, const TaskDesc& opts, CounterHandle after = kInvalidCounter);
	// The end of a future's own task: gives back its limiter unit, signals
	// the counter, then drops the task's reference.
	void FinishFuture(FutureStateBase* state) noexcept;
	// Marks the value there (a promise): signals the counter. Drops nothing.
	void SignalFuture(FutureStateBase* state) noexcept;
	// Drops one reference; the last frees the counter and the state.
	void ReleaseFuture(FutureStateBase* state) noexcept;
	inline void RetainFuture(FutureStateBase* state) noexcept { state->refs.fetch_add(1, std::memory_order_relaxed); }

	// A continuation's input failed: its own future fails the same way.
	inline void CopyFailure(FutureStateBase& to, const FutureStateBase& from)
	{
		to.broken = from.broken;
#if defined(__cpp_exceptions)
		to.error = from.error;
#endif
	}

	struct NoValue {};
	template <typename T>
	struct FutureState : FutureStateBase
	{
		std::conditional_t<std::is_void_v<T>, NoValue, std::optional<T>> value;
	};

	// Calls fn, with the input's value unless it is void, into `out`.
	template <typename U, typename G, typename... In>
	void RunInto(FutureState<U>& out, G& fn, In&... in)
	{
#if defined(__cpp_exceptions)
		try
		{
#endif
			if constexpr (std::is_void_v<U>)
				fn(in...);
			else
				out.value.emplace(fn(in...));
#if defined(__cpp_exceptions)
		}
		catch (...)
		{
			out.error = std::current_exception();
		}
#endif
	}

	template <typename T, typename F>
	struct FutureTask final : FutureState<T>
	{
		union { F fn; }; // ended by Run, as soon as it has run
		template <typename A>
		explicit FutureTask(A&& a) : fn(std::forward<A>(a)) {}
		~FutureTask() {}

		static void Run(void* p)
		{
			FutureTask& t = *static_cast<FutureTask*>(p);
			RunInto(t, t.fn);
			t.fn.~F();
			FinishFuture(&t);
		}
		static void Destroy(FutureStateBase* b) noexcept { delete static_cast<FutureTask*>(b); }
	};

	// Then: fn(input value) once the input is ready.
	template <typename T, typename G> struct ThenInvocable : std::is_invocable<G&, T&> {};
	template <typename G> struct ThenInvocable<void, G> : std::is_invocable<G&> {};
	template <typename T, typename G> struct ThenResult { using type = std::invoke_result_t<G&, T&>; };
	template <typename G> struct ThenResult<void, G> { using type = std::invoke_result_t<G&>; };

	template <typename T, typename U, typename F>
	struct ContinuationTask final : FutureState<U>
	{
		union { F fn; };
		FutureState<T>* input = nullptr; // one reference, dropped once read
		template <typename A>
		explicit ContinuationTask(A&& a) : fn(std::forward<A>(a)) {}
		~ContinuationTask() {}

		static void Run(void* p)
		{
			ContinuationTask& t = *static_cast<ContinuationTask*>(p);
			FutureState<T>& in = *t.input;
			if (in.Failed())
				CopyFailure(t, in);
			else if constexpr (std::is_void_v<T>)
				RunInto(t, t.fn);
			else
				RunInto(t, t.fn, *in.value);
			t.fn.~F();
			ReleaseFuture(&in);
			FinishFuture(&t);
		}
		static void Destroy(FutureStateBase* b) noexcept { delete static_cast<ContinuationTask*>(b); }
	};

	template <typename T>
	struct PromiseState final : FutureState<T>
	{
		static void Destroy(FutureStateBase* b) noexcept { delete static_cast<PromiseState*>(b); }
	};
} // namespace detail

template <typename T>
class Future
{
	static_assert(std::is_void_v<T> || (std::is_object_v<T> && !std::is_array_v<T>),
		"a Future holds a value or nothing: return by value, not by reference");

public:
	Future() = default;
	~Future() { Reset(); }
	Future(Future&& other) noexcept : m_State(std::exchange(other.m_State, nullptr)) {}
	Future& operator=(Future&& other) noexcept
	{
		if (this != &other)
		{
			Reset();
			m_State = std::exchange(other.m_State, nullptr);
		}
		return *this;
	}
	Future(const Future&) = delete;
	Future& operator=(const Future&) = delete;

	bool Valid() const { return m_State != nullptr; }
	// True once the value (or the error) is there; never waits.
	bool Ready() const { return Load(m_State->counter) == 0; }
	// Waits for the value; Get calls it. Rethrows nothing: Get does.
	void Wait(Help help = Help::WhileWaiting) const { WaitFor(m_State->counter, help); }
	// Wait with a time limit: true once ready, false if `timeoutMs` passed first.
	[[nodiscard]] bool TryWait(uint32_t timeoutMs, Help help = Help::WhileWaiting) const
	{
		return TryWaitFor(m_State->counter, timeoutMs, help);
	}
	// Waits, then returns the value (a reference into the future, valid while
	// it lives; move from it to take it), or rethrows the error. Can be called
	// again: it returns the same value, or throws the same exception. (Built
	// without exceptions, a failed future's Get ends the program.)
	std::add_lvalue_reference_t<T> Get(Help help = Help::WhileWaiting)
	{
		Wait(help);
#if defined(__cpp_exceptions)
		if (m_State->error)
			std::rethrow_exception(m_State->error);
#else
		if (m_State->broken)
			std::terminate();
#endif
		if constexpr (!std::is_void_v<T>)
			return *m_State->value;
	}
	// fn(value) (fn() for a void future) as a task once the value is ready;
	// returns the future of its result. Skipped, with the same error, if this
	// future fails. This future may be dropped afterwards.
	template <typename F> requires detail::ThenInvocable<T, std::decay_t<F>>::value
	[[nodiscard]] Future<typename detail::ThenResult<T, std::decay_t<F>>::type> Then(F&& fn, const TaskDesc& opts = {}) const
	{
		using G = std::decay_t<F>;
		using U = typename detail::ThenResult<T, G>::type;
		using Task = detail::ContinuationTask<T, U, G>;
		Task* task = new Task(std::forward<F>(fn)); // nothing started yet if this throws
		task->destroy = &Task::Destroy;
		task->input = m_State;
		detail::RetainFuture(m_State);
		detail::StartFuture(task, &Task::Run, opts, m_State->counter);
		return Future<U>(task);
	}
	// Zero once ready: for a task's waitsFor. Valid while the future lives.
	CounterHandle Counter() const { return m_State ? m_State->counter : kInvalidCounter; }
	// Drops the future (Valid() is then false); its task, if still to come,
	// runs anyway.
	void Reset() noexcept
	{
		if (m_State)
			detail::ReleaseFuture(std::exchange(m_State, nullptr));
	}

private:
	explicit Future(detail::FutureState<T>* state) : m_State(state) {}
	template <typename> friend class Future;
	template <typename> friend class Promise;
	template <typename F> requires std::is_invocable_v<std::decay_t<F>&>
	friend Future<std::invoke_result_t<std::decay_t<F>&>> Async(F&& fn, const TaskDesc& opts);

	detail::FutureState<T>* m_State = nullptr;
};

template <typename F> requires std::is_invocable_v<std::decay_t<F>&>
[[nodiscard]] Future<std::invoke_result_t<std::decay_t<F>&>> Async(F&& fn, const TaskDesc& opts = {})
{
	using G = std::decay_t<F>;
	using T = std::invoke_result_t<G&>;
	using Task = detail::FutureTask<T, G>;
	Task* task = new Task(std::forward<F>(fn)); // nothing started yet if this throws
	task->destroy = &Task::Destroy;
	detail::StartFuture(task, &Task::Run, opts);
	return Future<T>(task);
}

// The producing side of a future, for a value that comes from outside loom.
// Set it once, from any thread, while the pool lives; the Promise object
// itself is not shared between threads.
template <typename T>
class Promise
{
	static_assert(std::is_void_v<T> || (std::is_object_v<T> && !std::is_array_v<T>),
		"a Promise holds a value or nothing");

public:
	Promise() : m_State(new detail::PromiseState<T>())
	{
		m_State->destroy = &detail::PromiseState<T>::Destroy;
		m_State->refs.store(1, std::memory_order_relaxed); // the promise; GetFuture adds the future's
		detail::CreateFutureCounter(m_State);
	}
	// Completes the future with BrokenPromise if it was never set.
	~Promise() { Abandon(); }
	Promise(Promise&& other) noexcept : m_State(std::exchange(other.m_State, nullptr)), m_Set(other.m_Set), m_Taken(other.m_Taken) {}
	Promise& operator=(Promise&& other) noexcept
	{
		if (this != &other)
		{
			Abandon();
			m_State = std::exchange(other.m_State, nullptr);
			m_Set = other.m_Set;
			m_Taken = other.m_Taken;
		}
		return *this;
	}
	Promise(const Promise&) = delete;
	Promise& operator=(const Promise&) = delete;

	// The future it completes. Once per promise, before or after it is set.
	[[nodiscard]] Future<T> GetFuture()
	{
		CheckUsable(!m_Taken);
		m_Taken = true;
		detail::RetainFuture(m_State);
		return Future<T>(m_State);
	}
	// Stores the value (constructed from `args`) and completes the future.
	template <typename... A>
		requires (std::is_void_v<T> ? sizeof...(A) == 0 : std::is_constructible_v<T, A&&...>)
	void SetValue(A&&... args)
	{
		CheckUsable(!m_Set);
		if constexpr (!std::is_void_v<T>)
			m_State->value.emplace(std::forward<A>(args)...);
		Complete();
	}
#if defined(__cpp_exceptions)
	// Completes the future with an error for Get to rethrow.
	void SetException(std::exception_ptr error)
	{
		CheckUsable(!m_Set && error != nullptr);
		m_State->error = std::move(error);
		Complete();
	}
#endif
	bool IsSet() const { return m_Set; }

private:
	static void CheckUsable(bool ok)
	{
		if (!ok)
			std::terminate(); // a moved-from promise, a second set, or a second GetFuture
	}
	void Complete()
	{
		CheckUsable(m_State != nullptr);
		m_Set = true;
		detail::SignalFuture(m_State);
	}
	void Abandon() noexcept
	{
		if (!m_State)
			return;
		if (!m_Set)
		{
			m_State->broken = true;
#if defined(__cpp_exceptions)
			m_State->error = std::make_exception_ptr(BrokenPromise{});
#endif
			m_Set = true;
			detail::SignalFuture(m_State);
		}
		detail::ReleaseFuture(std::exchange(m_State, nullptr));
	}

	detail::PromiseState<T>* m_State = nullptr;
	bool m_Set = false;
	bool m_Taken = false;
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

// ---- Task graphs ------------------------------------------------------------
// A DAG of tasks built once and run many times: the frame's systems (input,
// then physics and animation, then culling, then the render submit on the
// main thread), declared once with their order. Nodes with the same
// predecessors share one wait, so a run costs one task per node and one
// atomic decrement per predecessor of each such wait (a layer after a layer:
// one per node of the layer before), plus one counter and one small task to
// end the run. Ready nodes are queued in groups, as Submit(span) does. The
// order is checked for cycles once, when the graph changes.
//
// A run behaves as one task. It starts once `waitsFor` is zero and signals
// `signals` once, when its last node has finished. A node's exception goes to
// `signals` (WaitFor rethrows it) and cancels it, as Cancel(signals) does:
// the nodes not yet started are skipped and the run still completes. (With
// no `signals`, a node that throws ends the program, as a task does.)
//
// A node takes the fn, userData, affinity, thread, priority, limiter, nameId
// and window of its TaskDesc; its counters come from the graph, so `signals` and
// the waits must be unset. One run at a time per graph (a frame that overlaps
// the next one uses two graphs), and the graph must not change or be
// destroyed while it runs.
namespace detail { struct GraphState; }

class TaskGraph
{
public:
	using Node = uint32_t;

	TaskGraph();
	~TaskGraph(); // asserts it is not running; destroys its callables
	TaskGraph(TaskGraph&&) noexcept;
	TaskGraph& operator=(TaskGraph&&) noexcept;
	TaskGraph(const TaskGraph&) = delete;
	TaskGraph& operator=(const TaskGraph&) = delete;

	Node Add(const TaskDesc& desc);
	// A node that calls a callable on every run. The graph keeps it (one heap
	// allocation, at Add) and destroys it with the graph or on Clear.
	template <typename F> requires std::is_invocable_v<std::decay_t<F>&>
	Node Add(F&& fn, const TaskDesc& opts = {});
	// `after` starts only once `before` has finished. Listing an edge twice
	// is harmless.
	void Precede(Node before, Node after);
	void Clear();

	size_t NodeCount() const;
	// True from Submit until the last node has finished (just before the
	// run's counter is signalled).
	bool Running() const;
	// False when the order has a cycle; Submit asserts there is none. A graph
	// is checked once per change, here or on its next Submit.
	bool Validate();

private:
	friend void Submit(TaskGraph&, CounterHandle, CounterHandle);
	Node AddOwned(void* object, void (*run)(void*), void (*destroy)(void*), const TaskDesc& opts);
	std::unique_ptr<detail::GraphState> m_State;
};

template <typename F> requires std::is_invocable_v<std::decay_t<F>&>
TaskGraph::Node TaskGraph::Add(F&& fn, const TaskDesc& opts)
{
	using G = std::decay_t<F>;
	// Freed here if AddOwned throws; owned by the graph once it returns.
	std::unique_ptr<G> object(new G(std::forward<F>(fn)));
	const Node node = AddOwned(object.get(), &detail::InlineTrampoline<G>, &detail::OwnedDestroyHeap<G>, opts);
	object.release();
	return node;
}

// Starts a run and returns at once (see above).
void Submit(TaskGraph& graph, CounterHandle signals = kInvalidCounter, CounterHandle waitsFor = kInvalidCounter);
// Runs the graph and waits for it, rethrowing a node's exception.
void Run(TaskGraph& graph, Help help = Help::WhileWaiting);

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

// ---- Per-thread storage --------------------------------------------------------
// One T per thread, made on that thread's first Local() call: a scratch
// buffer per worker, or a per-thread partial result (a histogram, a list of
// hits) merged once the work is done. Each T sits on its own cache lines, so
// threads never share one, and Local() after the first call is an index
// lookup with no lock.
//
// Local() may be called from any thread at once. ForEach, Size and Clear may
// not run while any thread can still call Local(): call them after the wait
// for that work has returned (the wait orders every thread's writes before
// them). A T outlives its thread (a finished worker's partial stays until
// Clear), and every one is destroyed with the PerThread. Keyed by
// CurrentThreadIndex, so a process that creates more than 65536 threads in
// all cannot use it.
template <typename T>
class PerThread
{
public:
	PerThread() = default;
	// Every thread's T starts as a copy of `exemplar` (a sized buffer, a zeroed tally).
	explicit PerThread(const T& exemplar) requires std::is_copy_constructible_v<T> : m_Exemplar(exemplar) {}
	~PerThread() { Destroy(); }
	PerThread(const PerThread&) = delete;
	PerThread& operator=(const PerThread&) = delete;

	// The calling thread's T, made on first use.
	T& Local()
	{
		const int index = CurrentThreadIndex();
		if (index < 0 || (size_t)index >= kChunks * kChunkSize)
			std::terminate(); // more threads than PerThread can key
		Chunk* chunk = m_Chunks[(size_t)index / kChunkSize].load(std::memory_order_acquire);
		if (!chunk)
			chunk = AddChunk((size_t)index / kChunkSize);
		Cell& cell = chunk->cells[(size_t)index % kChunkSize];
		if (!cell.constructed) // only this thread touches its cell
		{
			if constexpr (std::is_copy_constructible_v<T>)
			{
				if (m_Exemplar)
					::new (static_cast<void*>(cell.storage)) T(*m_Exemplar);
				else
					::new (static_cast<void*>(cell.storage)) T();
			}
			else
				::new (static_cast<void*>(cell.storage)) T();
			cell.constructed = true;
		}
		return *std::launder(reinterpret_cast<T*>(cell.storage));
	}

	// fn(T&) for every T made so far, in thread-index order.
	template <typename F>
	void ForEach(F&& fn)
	{
		for (auto& slot : m_Chunks)
			if (Chunk* chunk = slot.load(std::memory_order_acquire))
				for (Cell& cell : chunk->cells)
					if (cell.constructed)
						fn(*std::launder(reinterpret_cast<T*>(cell.storage)));
	}

	size_t Size()
	{
		size_t n = 0;
		ForEach([&](T&) { n++; });
		return n;
	}

	// Destroys every T; the next Local() on a thread makes a new one.
	void Clear()
	{
		for (auto& slot : m_Chunks)
			if (Chunk* chunk = slot.load(std::memory_order_acquire))
				for (Cell& cell : chunk->cells)
					if (cell.constructed)
					{
						std::launder(reinterpret_cast<T*>(cell.storage))->~T();
						cell.constructed = false;
					}
	}

private:
	static constexpr size_t kChunkSize = 64;
	static constexpr size_t kChunks = 1024;
	static constexpr size_t kCellAlign = alignof(T) > 64 ? alignof(T) : 64;
	struct alignas(kCellAlign) Cell
	{
		alignas(T) unsigned char storage[sizeof(T)];
		bool constructed = false;
	};
	struct Chunk
	{
		Cell cells[kChunkSize];
	};

	Chunk* AddChunk(size_t i)
	{
		Chunk* fresh = new Chunk();
		Chunk* expected = nullptr;
		if (m_Chunks[i].compare_exchange_strong(expected, fresh, std::memory_order_acq_rel, std::memory_order_acquire))
			return fresh;
		delete fresh; // another thread of the same chunk got there first
		return expected;
	}

	void Destroy()
	{
		Clear();
		for (auto& slot : m_Chunks)
			delete slot.load(std::memory_order_relaxed);
	}

	std::atomic<Chunk*>  m_Chunks[kChunks] = {};
	std::optional<T>     m_Exemplar;
};

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

// ---- Debug dump ---------------------------------------------------------------
// Why does a wait not return? CaptureDebugState records what every pool
// thread (and the main thread) is running or waiting on, every live counter
// with the tasks parked on it and the threads asleep on it, and every
// limiter with the tasks waiting for a unit. DebugDump formats that as text.
//
// Call it from any thread while the pool lives: a watchdog thread, after a
// TryWaitFor timed out, or from a debugger. It takes each lock briefly and,
// rather than hang on a lock held by a stuck thread, marks what it could not
// read as busy. Things keep moving while it reads, so it is a best-effort
// picture, not one instant.
//
// Tasks are known by their TaskDesc::nameId; pass a TaskNameFn (the
// profiler's name table) to print names instead of ids. The text shows a
// counter as c<index>.<generation> and a limiter as l<index>.<generation>,
// and threads by their CurrentThreadIndex in brackets.
struct DebugTask
{
	uint32_t      nameId   = 0;
	Priority      priority = Priority::Normal;
	Affinity      affinity = Affinity::AnyWorker;
	LimiterHandle limiter  = kInvalidLimiter;
};

struct DebugThread
{
	int         index = 0;
	std::string name;
	bool        isMain = false;
	uint64_t    tasksRun = 0;   // compare two captures to see whether it moves
	int         queueDepth = 0; // a dedicated thread's own queue
	// The tasks this thread is inside: more than one when a wait inside a
	// task ran another while helping. The innermost one's name and counter.
	int           running = 0;
	uint32_t      runningName = 0;
	CounterHandle runningSignals = kInvalidCounter;
	LimiterHandle runningLimiter = kInvalidLimiter; // the unit the innermost task holds
	// The innermost WaitFor or TryWaitFor this thread is in, and for how long.
	CounterHandle waitingOn = kInvalidCounter;
	double        waitingSeconds = 0.0;
};

struct DebugCounter
{
	CounterHandle handle = kInvalidCounter;
	int  count = 0;
	bool cancelled = false;
	bool failed = false;  // a task signalling it threw
	bool busy = false;    // its lock was held throughout: parked and waiters unread
	std::vector<DebugTask> parked; // tasks waiting for it to reach zero (the first 64)
	int  parkedTotal = 0;
	std::vector<int> waiters;      // thread indices asleep in a wait on it
};

struct DebugLimiter
{
	LimiterHandle handle = kInvalidLimiter;
	uint32_t limit = 0;
	uint32_t held = 0;
	bool     busy = false;
	std::vector<DebugTask> waiting; // tasks waiting for a unit (the first 64)
	int      waitingTotal = 0;
};

struct DebugState
{
	std::vector<DebugThread>  threads; // the main thread first
	std::vector<DebugCounter> counters;
	std::vector<DebugLimiter> limiters;
	int liveTasks = 0;
	int queued = 0;       // in the AnyWorker queues (every priority)
	int queuedHigh = 0;
	int queuedLow = 0;
	int mainQueued = 0;   // waiting for the main thread to pump or wait
	bool busy = false;    // some table could not be read at all
};

using TaskNameFn = const char* (*)(uint32_t nameId);

DebugState  CaptureDebugState();
std::string FormatDebugState(const DebugState&, TaskNameFn nameOf = nullptr);
std::string DebugDump(TaskNameFn nameOf = nullptr); // both of the above

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
