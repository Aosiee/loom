# loom

A task and threading library for C++20: one pool of worker threads,
dedicated threads and lanes, atomic counters as the only synchronisation
primitive, and **help-while-waiting**, so a thread that waits on work runs
eligible tasks instead of sleeping. It has no dependencies beyond the
standard library and POSIX threads.

loom was written for [TinyEngine](https://github.com/Aosiee/TinyEngine), whose
renderer, physics, editor and server have run on it since 2026-09. It knows
nothing about the engine.

## A taste

```cpp
#include "loom/loom.h"

loom::Pool pool;                          // workers = hardware threads - 1

loom::ScopedCounter done(2);
loom::Submit([&]() { BuildShadows(); },   { .signals = done });
loom::Submit([&]() { BuildParticles(); }, { .signals = done });
loom::WaitFor(done);                      // runs eligible tasks while it waits

loom::ParallelFor(0, meshes.size(), 64, [&](size_t begin, size_t end) {
    for (size_t i = begin; i < end; i++)
        Cull(meshes[i]);
});
```

## The model

- **Pool tasks** (`Affinity::AnyWorker`) run on any worker, or on a waiting
  thread that helps.
- **Main-thread tasks** (`Affinity::MainThread`) run only on the thread that
  created the pool, when it waits or calls `PumpMainThreadTasks()`: the place
  for APIs that are tied to one thread (a GPU device, a windowing system).
- **Dedicated threads** (`RegisterDedicatedThread`) are long-lived, named
  threads with their own queue; they can sleep when idle or help the pool.
  **Lanes** (`AcquireLane`) are dedicated threads reserved for one owner.
- **Counters** are the only synchronisation: a task can signal one when it
  finishes and wait for one or several before it may start, so a dependency
  graph is a set of counters. `WaitFor` either helps (`Help::WhileWaiting`)
  or blocks (`Help::BlockOnly`), and rethrows an exception a task of that
  counter threw; `TryWaitFor` does the same with a time limit. `Cancel`
  skips a counter's tasks that have not started.
- **Task graphs** (`TaskGraph`): a dependency graph declared once and run
  every frame, without rebuilding counters. A run acts as one task: it can
  wait on a counter, signals one when done, and reports exceptions and
  cancellation through it.
- **Priorities**: `Priority::High` work runs before Normal, and `Low` only
  when nothing else is queued.
- **Futures** (`Async`): a task that returns a value; `Get()` waits for it
  or rethrows, and `Counter()` lets another task wait for it first.
  `Then` runs a task on the value once it is ready, and a `Promise` makes a
  future for a value that comes from outside loom, such as an IO callback.
- **Limiters** (`CreateLimiter`): at most N tasks of a kind run at once, so
  background work (asset decompression, uploads) cannot take every worker;
  a limit of 1 is a serial queue. `LimiterUnit()` tells a task which of the
  N units it holds (one staging buffer each), and `SetLimiterLimit` changes
  N while the work runs.
- **Isolation** (`Isolated`): a wait inside an isolated region helps only
  with that region's tasks, so a frame-critical wait never picks up an
  unrelated long task.
- **Parallel loops**: `ParallelFor` chunks automatically (or by a chunk size
  you give), and `ParallelReduce` combines per-chunk results in a fixed
  order, so the result is the same on every run.
- **Parking** (`ParkAll` / `UnparkAll`) stops every pool thread at a known
  point with its stack recorded: the hook a live code reloader needs.
- **Introspection**: `SnapshotThreadStats()` for a thread-pool view, and
  profiler hooks (`loom_profiler_shim.h`) for wait/help states and per-task
  zones.
- **Debug dump** (`DebugDump`, `CaptureDebugState`): when a wait does not
  return, what each thread is running and waiting on, every live counter
  with the tasks parked on it and who sleeps on it, and every limiter's
  waiting tasks, with hints for the common mistakes.

The rules that keep this safe (no helping while holding a lock, no blocking
IO on workers, no ambient thread-local assumptions) are in
[`docs/rules.md`](docs/rules.md); the internals in
[`docs/design.md`](docs/design.md).

## Process state

The free-function API resolves one **active pool** per process: constructing a
`Pool` installs it, and `Shutdown()` (or the destructor) uninstalls it. loom's
only process-wide state is that pointer and the counter that hands out thread
indices (so a thread keeps its index across pools). A host that audits its
globals lists those two.

## Building

```sh
cmake -S . -B build && cmake --build build
./build/loom_tests                 # the suite (doctest, vendored)
./build/loom_bench [tasks] [runs]  # per-task cost and contention by worker count
```

`-DLOOM_SANITIZE=thread|address|undefined` builds everything with a
sanitizer; under TSAN run the suite with
`TSAN_OPTIONS=suppressions=tests/tsan.supp`. A host build adds the library
with `add_subdirectory` (tests off by default there) or compiles
`src/loom.cpp` with `include/` on its include path.

Linux is supported and tested. Other platforms are not yet: the thread
naming, the OS thread id and the stack walk used by parking are Linux and
POSIX calls today.

## Licence

MIT with one condition: a product that includes loom credits it ("Uses loom -
Copyright (c) 2026 Felipe Lara") in its credits, about box or documentation. No
logo or splash screen. See [`LICENSE`](LICENSE).
