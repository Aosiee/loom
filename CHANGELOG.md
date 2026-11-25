# Changelog

Each release is an annotated tag (`v0.x.0`, or `v0.x.y` for a release of
fixes only) whose message is the entry below.
Versions before 1.0 may change the API; every entry says whether existing
callers need to change anything.

## Updating from an older version

Three releases changed things for existing callers. Updating from 0.9.0 to
any later version, check these:

- 0.10.0: `Shutdown` drains every live task first, and may block for up to
  `LoomConfig::shutdownTimeoutMs`. `DedicatedId`s are handed out once per pool
  and are no longer small, dense indices. `ThreadStats::tasksRun` includes
  tasks run while helping. `LiveTaskCount` and `Load` can lag briefly behind
  finished tasks.
- 0.11.0: `ParallelFor` with `chunkSize` 0 now chunks automatically (it did
  nothing). A task's exception now cancels its counter: that counter's tasks
  that have not started are skipped.
- 0.19.0: names changed across the API, all at once (the full list is in
  its entry below; the compiler finds every use). Handles are their own
  types: code that stores one as an integer needs a cast
  (`static_cast<uint32_t>(h)`, `CounterHandle{ v }`), and a positional
  `TaskDesc{ ... }` that gives 0 for `thread` gives `{}`. A Dedicated task
  with no `thread` now asserts instead of going to the first dedicated
  thread.

## 0.20.1 (2026-11-25): thread indices are given back, and GitHub releases

- A thread gives its index back when it exits, and a new thread takes
  the lowest index free, so indices stay below the most threads alive at
  once. A process that kept creating and destroying pools used to run
  through them: `PerThread` keys at most 65536, and past that `Local()`
  terminated. What was keyed on an index passes to the next thread that
  takes it, whole: a `PerThread` value carries on, and a dedicated thread
  takes over the stats row of the thread whose index it took, under its
  own name.
- `hamr.cluau` runs, locally, the configurations the Linux workflow checks
  (`hamr run`; `--tier nightly` adds the portable sleep, the sanitizers,
  the no-exceptions embedding and the installed package).
- Releases: `main` holds tagged releases only, work gathers on `dev` (where
  CI runs) and is built on streams (`CONTRIBUTING.md`). A version tag
  publishes a GitHub release.

Nothing changes for existing callers.

## 0.20.0 (2026-11-23): coroutines, an installable package, and a stress test

- Coroutines: a coroutine can return a `Future<T>`, and `co_await` a
  future (`co_await f`, or `co_await After(f, opts)`), a counter
  (`After(counter)`) or several (`After({ a, b })`) without holding a
  thread while it waits: a task resumes it once they are ready.
  `co_await ResumeOn(opts)` moves the rest of a coroutine onto a task with
  those options, to the main thread, a dedicated thread, a priority or a
  limiter. Errors reach the coroutine at its `co_await`, and its future if
  it does not catch them. `LOOM_COROUTINES` says whether the compiler
  supports them. docs/rules.md has what not to hold across a `co_await`.
- `loom/loom_version.h` (included by `loom/loom.h`) defines
  `LOOM_VERSION_MAJOR`, `_MINOR`, `_PATCH`, `LOOM_VERSION_STRING`, and
  `LOOM_VERSION` with `LOOM_MAKE_VERSION` to compare it against.
- `cmake --install` installs the library, headers and a CMake package:
  `find_package(loom 0.20)` and the `loom::loom` target, which
  `add_subdirectory` users get too. The package accepts only its own minor
  version, since a minor release before 1.0 may change the API.
- The library a host links, installs or benchmarks never has the test
  hooks: the suite now links its own copy that does.
- `loom_stress` runs random mixes of every feature on random pools for as
  long as asked, checking what each promises, and names the seed of a
  failing round so it can be run again. CI runs it for a minute in every
  configuration.

No changes for existing callers.

## 0.19.0 (2026-11-23): consistent names, and handles as types

Every rename is in this one release, with no old names kept alongside, so
a caller updates once.

- `CounterHandle`, `LimiterHandle` and `DedicatedId` are each an enum class
  over 32 bits, so passing one where another belongs, or a plain integer,
  no longer compiles. `LaneId` is a `DedicatedId`. `TaskDesc::thread`
  defaults to `kInvalidDedicated`, so a Dedicated task with no thread
  asserts instead of going to the first dedicated thread.
- "Park" now means only what `ParkAll` does to threads: the debug dump
  calls the tasks held back by a counter "waiting", as it already did for a
  limiter's, and the threads asleep on it "sleepers".
- Yes-or-no queries all start with `Is` or `Has`, and the limiter queries,
  which no longer need the type in their names, drop it.

Changes for existing callers (old name, then new):

- Pool: `LoomConfig` is `PoolConfig`; `IsInitialised()` is
  `HasActivePool()`; `SnapshotThreadStats()` is `SnapshotPool()`;
  `PoolSnapshot::anyQueueDepth` is `queueDepth`; `ThreadStats::helpedRun`
  and `ThreadStatsView::helpedRun` are `tasksHelped`.
- Counters: `Load(counter)` is `Count(counter)`.
- Limiters: `LimiterHeld` is `HeldUnits`, `LimiterWaiting` is
  `WaitingTasks`, `SetLimiterLimit` is `SetLimit`, `LimiterLimit` is
  `Limit`, and `LimiterUnit()` is `CurrentLimiterUnit()`.
- Futures and graphs: `Future::Valid()` and `Ready()` are `IsValid()` and
  `IsReady()`; `TaskGraph::Running()` is `IsRunning()`.
- Parking: `ParkRequested()` is `IsParkRequested()`.
- Tasks: `TaskDesc::window` is `profilerTag`, as is the matching
  `ProfilerZoneFn` parameter.
- Debug dump: `DebugCounter::parked`, `parkedTotal` and `waiters` are
  `waiting`, `waitingTotal` and `sleepers` (and the text says "N waiting");
  `DebugThread::running` and `runningName` are `runningDepth` and
  `runningNameId`; `DebugState::queued`, `queuedHigh`, `queuedLow` and
  `mainQueued` are `queueDepth`, `highQueueDepth`, `lowQueueDepth` and
  `mainQueueDepth`.
- Handles: see the first point above.

## 0.18.1 (2026-11-23): a pipeline fix, and CI

- A pipeline stage with a limiter could still hold its unit for a moment
  after `Run` returned, so a caller that destroyed the limiter straight away
  could have a task give a unit back to a limiter that was gone (a Debug
  build asserted in `DestroyLimiter`). A stage task now gives its unit back
  before it hands on its last item, as a future's task does before its
  signal.
- GitHub Actions builds and tests every push on Linux with GCC and Clang,
  in Debug and Release, with the portable sleep instead of the futex, and
  under the address, undefined-behaviour and thread sanitizers, and builds
  loom as an engine embeds it, without exceptions. The README shows a badge
  per platform: Linux tested, Windows and macOS not supported yet.
- A debug-dump test no longer fails when it captures a wait between naming
  its counter and going to sleep on it.

No changes for existing callers.

## 0.18.0 (2026-11-23): counter increments

- `AddToCounter(counter, n)` raises a count that has not reached zero yet, so
  work that finds more work as it runs (a tree walk, culling a
  bounding-volume hierarchy) adds 1 for each task it submits and the whole
  tree runs on one counter and one wait, with no task waiting for its
  children. It returns false, changing nothing, once the count has reached
  zero; a running task of that counter always succeeds.
- The README has examples of everything added since 0.9.0 (each checked to
  compile and run against this release), and this changelog collects the
  release notes.
- loom's own test build has hooks to hold a thread at a point inside loom;
  a test uses one to check that a limiter unit's index is free before the
  unit is. Builds that embed loom do not have them.

No changes for existing callers.

## 0.17.0 (2026-11-23): parallel algorithms, pipelines, per-thread storage, limiter units

- LimiterUnit() gives a running limited task a stable index in [0, limit),
  unique among that limiter's running tasks, to pick a scratch or staging
  buffer without a lock. SetLimiterLimit changes a limit while tasks run;
  lowering it never stops a task. LimiterLimit reads it back.
- PerThread<T>: one T per thread, made on first use on its own cache lines,
  merged with ForEach after the work's wait.
- ParallelInclusiveScan / ParallelExclusiveScan: the arguments of the
  std scans with an initial value, in place allowed, the same bits whatever
  the pool. Faster than std on cache-sized arrays; bandwidth bound beyond.
- ParallelSort: a drop-in for std::sort, sample sort into a three-way
  parallel quicksort, the same order whatever the pool; ~12.6x std::sort
  on 4M keys with 31 workers. A throwing comparator is rethrown, leaving
  the range valid in no particular order, as std::sort does.
- Pipeline<T>(inFlight): a source, serial stages in source order and
  parallel stages, each with its own TaskDesc (a main-thread upload stage,
  a limited decode stage); a few hundred ns per item.

No changes for existing callers.

## 0.16.0 (2026-11-22): debug dump

- DebugDump(nameOf) prints what every pool thread and the main thread is
  running and waiting on (and for how long), every live counter with its
  count, the tasks parked on it and the threads asleep on it, and every
  limiter with its held units and waiting tasks. nameOf maps a task's
  nameId to a name, such as the profiler's name table.
- CaptureDebugState returns the same as plain structs (for an editor
  view); FormatDebugState prints one.
- Hints for the common reasons a wait never returns: nothing running or
  queued, main-thread work nobody pumps, and a limited task waiting on
  work of its own limiter.
- Safe to call from any thread on a stuck pool: each lock is tried for
  about 2 ms and marked busy rather than waited on.
- A few relaxed stores per task and one clock read per wait; no change in
  loom_bench's submit cost.

No changes for existing callers.

## 0.15.0 (2026-11-22): futures, promises and continuations

- Async(fn, opts) runs fn as a task and returns a Future<T>. Get() waits
  (helping, by default) and returns the value, or rethrows what fn threw,
  every time it is called. Ready, Wait and TryWait check or wait without
  reading. Works for void and move-only results.
- A future is also a counter: a task can wait for Counter() through
  waitsFor or waitsForAll and then read Get() without blocking. A future
  can be dropped before its task runs; the task still runs and frees it.
- Promise<T> makes a future for a value produced outside loom, such as an
  IO callback. SetValue or SetException completes it from any thread; a
  promise destroyed unset completes it with BrokenPromise.
- future.Then(fn) runs fn on the value as a task once it is ready and
  returns the future of its result. A chain may be built from
  temporaries; a failure skips the rest of the chain and reaches its end.
- The TaskDesc passes affinity, priority, limiter and waits on to a
  future's task. Futures ignore Cancel. A limited future gives its unit
  back before its value is ready.
- About 2x a plain task per result (~130-155 ns), and ~1.7-1.95x a
  hand-written counter per step for Then chains (~190-260 ns per step).
- Fix: a task released by a counter now always reads that counter as
  zero; before, it could briefly read 1 while the final signal was still
  releasing the counter's other tasks.

No changes for existing callers.

## 0.14.0 (2026-11-22): limiters

- CreateLimiter(n), or ScopedLimiter: at most n tasks with
  TaskDesc::limiter run at once, so background work (asset loading,
  decompression, uploads) cannot take every worker. A limit of 1 is a
  serial queue.
- A task takes a unit when it becomes runnable and gives it back when it
  finishes; while none is free it waits in the limiter, not in a run
  queue. High tasks go first, otherwise in the order they became runnable.
- Works with every affinity, dependencies, Cancel, exceptions and
  TaskGraph nodes. LimiterHeld and LimiterWaiting report its state;
  LoomConfig::maxLimiters sizes the table (64 by default).
- About 4-15 ns per task while units are free. In loom_bench, a frame
  beside 1 ms background tasks on every worker went from a ~986 us median
  to ~124 us (the same as with no background) with the background limited
  to 4 workers.

No changes for existing callers.

## 0.13.0 (2026-11-22): task graphs

- TaskGraph: a dependency graph declared once and run every frame, with
  no counters to rebuild. Nodes take a TaskDesc or a callable; Precede
  orders them; cycles are found once per change.
- A run acts as one task: Submit(graph, signals, waitsFor) waits on a
  counter and signals one when done, and a node's exception or a Cancel
  skips the nodes not yet started. Run(graph) submits and waits.
- Nodes with the same predecessors share one wait, ready nodes are queued
  in groups and a run's completion is batched, so a run matches or beats
  the cheapest hand-built counter form on every shape in loom_bench.
- loom_bench times a frame-shaped DAG and four graph shapes against their
  counter equivalents.

No changes for existing callers.

## 0.12.0 (2026-11-21): any callable, timed waits, a portable waiter

- Submit(F) takes any callable. A nothrow-movable one of up to 40 bytes is
  moved into the task slot; anything larger or over-aligned goes to the
  heap. It is destroyed after the task runs, throws or is skipped.
  Trivially copyable callables of up to 48 bytes are stored as before.
- TryWaitFor(counter, timeoutMs, help): a wait with a time limit, returning
  false if the time runs out first; 0 only checks.
- Off Linux, threads sleep on a mutex and condition variable (it supports
  a timeout, unlike std::atomic::wait); LOOM_NO_FUTEX tests that path on
  Linux.
- Fix: two dedicated threads with the same name no longer share a stats
  row's queue depth.

No changes for existing callers.

## 0.11.0 (2026-11-21): dependencies, priorities, cancellation, isolation, auto-chunking

- TaskDesc::waitsForAll: a task can wait for several counters.
- TaskDesc::priority: High runs before Normal; Low only when no other pool
  work is queued. High goes to the front of main and dedicated queues.
- Cancel and IsCancelled: skip a counter's tasks that have not started.
- Isolated(fn): a wait inside the region helps only with the region's own
  tasks, so a frame-critical wait never picks up an unrelated long task.
- ParallelFor chunks automatically with chunkSize 0 (or no chunk size),
  splitting work whenever a worker is idle.
- ParallelReduce: a reduction that gives the same bits on every run and
  with any number of workers.
- The counter table moved off the free-slot stack's cache line (about
  40 ns per task back to 20-28 ns at 16 workers).
- docs/design.md and docs/rules.md cover all of the above, and the
  scheduler internals are commented throughout.

**Changes for existing callers:**
- ParallelFor with chunkSize 0 now chunks automatically (it did nothing).
- A task's exception now cancels its counter: that counter's tasks that
  have not started are skipped.

## 0.10.0 (2026-11-21): a work-stealing scheduler

The same API on a lock-free core, and the correctness fixes found while
building it.

- Task slots in chunks that never move, with per-thread free-slot caches:
  submitting and finishing a task takes no lock.
- One Chase-Lev deque per worker and one for the main thread; idle threads
  steal. Submit(span) hands tasks over in groups of up to 32.
- A counter is one atomic; only the final signal locks, and a worker
  applies its finished tasks' signals to a counter in one go.
- A task's exception is kept on its counter and rethrown by WaitFor
  (before, it terminated the program). Builds without exceptions still work.
- Fixes: long thread names are cut to the OS limit; thread stats count
  helped tasks and own their names (a use-after-free); a submit to a
  dedicated thread can no longer race its unregister.

**Changes for existing callers:**
- Shutdown drains every live task first, and may block for up to
  LoomConfig::shutdownTimeoutMs.
- DedicatedIds are handed out once per pool and are no longer small,
  dense indices.
- ThreadStats::tasksRun includes the tasks a thread ran while helping.
- LiveTaskCount and Load can lag briefly behind finished tasks.

## 0.9.0 (2026-11-20): the library as extracted from TinyEngine, before generalisation
