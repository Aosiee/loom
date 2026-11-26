# loom

[![Linux](https://github.com/Aosiee/loom/actions/workflows/linux.yml/badge.svg?branch=dev)](https://github.com/Aosiee/loom/actions/workflows/linux.yml)
[![Windows: not supported](https://img.shields.io/badge/Windows-not%20supported-lightgrey)](#building)
[![macOS: not supported](https://img.shields.io/badge/macOS-not%20supported-lightgrey)](#building)

[![Release](https://img.shields.io/github/v/release/Aosiee/loom)](https://github.com/Aosiee/loom/releases/latest)

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

## More examples

A frame's systems declared once as a graph, then run every frame:

```cpp
loom::TaskGraph frame;
auto input   = frame.Add([&] { PollInput(); });
auto physics = frame.Add([&] { StepPhysics(); });
auto anim    = frame.Add([&] { Animate(); });
auto submit  = frame.Add([&] { SubmitDraws(); }, { .affinity = loom::Affinity::MainThread });
frame.Precede(input, physics);
frame.Precede(input, anim);
frame.Precede(physics, submit);
frame.Precede(anim, submit);
loom::Run(frame);                         // each frame
```

A value from a task, a continuation on it, and a value from outside loom:

```cpp
loom::Future<Mesh> mesh = loom::Async([&] { return LoadMesh(path); });
loom::Future<size_t> bytes = mesh.Then([](Mesh& m) { return Upload(m); });

loom::Promise<Texture> texture;
loom::Future<Texture> ready = texture.GetFuture();
io.ReadAsync(file, [p = std::move(texture)](Texture t) mutable { p.SetValue(std::move(t)); });
```

A coroutine that waits without holding a thread, and moves to the main
thread for its last step:

```cpp
loom::Future<GpuMesh> LoadMesh(std::string path)
{
    Bytes bytes = co_await loom::Async([&] { return ReadFile(path); });
    Mesh mesh = co_await loom::Async([&] { return ParseMesh(bytes); });
    co_await loom::ResumeOn({ .affinity = loom::Affinity::MainThread });
    co_return Upload(mesh);
}
```

Background work on at most four workers, each with a staging buffer of its own:

```cpp
loom::ScopedLimiter decoders(4);
std::vector<StagingBuffer> staging(4);
loom::ScopedCounter decoded((int)assets.size());
for (Asset& a : assets)
    loom::Submit([&] { Decode(a, staging[loom::CurrentLimiterUnit()]); },
                 { .signals = decoded, .priority = loom::Priority::Low, .limiter = decoders });
loom::WaitFor(decoded);
```

Work that finds more work as it goes, on one counter and one wait:

```cpp
void Visit(Node* node, loom::CounterHandle done)
{
    if (!Visible(node))
        return;
    for (Node* child : node->children)
    {
        loom::AddToCounter(done);         // before the child's task can signal
        loom::Submit([=] { Visit(child, done); }, { .signals = done });
    }
}
loom::ScopedCounter done(1);
loom::Submit([&] { Visit(root, done); }, { .signals = done });
loom::WaitFor(done);
```

Per-thread results, sorting and prefix sums:

```cpp
loom::PerThread<std::vector<Hit>> hits;
loom::ParallelFor(0, rays.size(), [&](size_t b, size_t e) {
    for (size_t i = b; i < e; i++)
        Trace(rays[i], hits.Local());
});
hits.ForEach([&](std::vector<Hit>& h) { all.insert(all.end(), h.begin(), h.end()); });

loom::ParallelSort(keys.begin(), keys.end());
loom::ParallelExclusiveScan(counts.begin(), counts.end(), offsets.begin(), size_t(0));
```

A stream of items through stages, uploads in order on the main thread:

```cpp
struct Chunk { std::vector<char> raw; Mesh mesh; };
loom::Pipeline<Chunk> stream(8);          // at most 8 chunks in flight
stream.Source([&](Chunk& c) { return file.ReadNext(c.raw); })
      .Parallel([](Chunk& c) { c.mesh = Decode(c.raw); })
      .Serial([&](Chunk& c) { Upload(c.mesh); }, { .affinity = loom::Affinity::MainThread });
stream.Run();
```

When a wait does not return, `loom::DebugDump()` says what every thread,
counter and limiter is doing.

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
  skips a counter's tasks that have not started. `AddToCounter` raises a
  count that has not reached zero, so work that finds more work as it runs
  (a tree walk) needs one counter and one wait.
- **Handles** (`CounterHandle`, `LimiterHandle`, `DedicatedId`) are each
  their own type, so passing a limiter where a counter belongs does not
  compile.
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
- **Coroutines**: a coroutine can return a `Future` and `co_await` futures
  and counters (`After`) without holding a thread while it waits, and
  `ResumeOn` moves its rest to the main thread, a dedicated thread or a
  limiter.
- **Limiters** (`CreateLimiter`): at most N tasks of a kind run at once, so
  background work (asset decompression, uploads) cannot take every worker;
  a limit of 1 is a serial queue. `CurrentLimiterUnit()` tells a task which
  of the N units it holds (one staging buffer each), and `SetLimit` changes
  N while the work runs.
- **Isolation** (`Isolated`): a wait inside an isolated region helps only
  with that region's tasks, so a frame-critical wait never picks up an
  unrelated long task.
- **Parallel loops**: `ParallelFor` chunks automatically (or by a chunk size
  you give), and `ParallelReduce` combines per-chunk results in a fixed
  order, so the result is the same on every run.
- **Parallel algorithms**: `ParallelInclusiveScan` / `ParallelExclusiveScan`
  (prefix sums, reproducible bit for bit) and `ParallelSort` (a sample sort
  into a parallel quicksort; the same order whatever the pool).
- **Pipelines** (`Pipeline<T>`): a stream of items through serial and
  parallel stages, at most N items in flight: read, decode and upload
  overlapping, uploads in order and on the main thread if need be.
- **Per-thread storage** (`PerThread<T>`): one scratch buffer or partial
  result per thread, made on first use, merged once the work is done.
- **Parking** (`ParkAll` / `UnparkAll`) stops every pool thread at a known
  point with its stack recorded: the hook a live code reloader needs.
- **Introspection**: `SnapshotPool()` for a thread-pool view, and
  profiler hooks (`loom_profiler_shim.h`) for wait/help states and per-task
  zones.
- **Debug dump** (`DebugDump`, `CaptureDebugState`): when a wait does not
  return, what each thread is running and waiting on, every live counter
  with the tasks waiting for it and who sleeps on it, and every limiter's
  waiting tasks, with hints for the common mistakes.

The rules that keep this safe (no helping while holding a lock, no blocking
IO on workers, no ambient thread-local assumptions) are in
[`docs/rules.md`](docs/rules.md); the internals in
[`docs/design.md`](docs/design.md). [`CHANGELOG.md`](CHANGELOG.md) lists
what each release added and what, if anything, an existing caller must
change.

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
./build/loom_faults                # every allocation of loom's calls failed in turn
./build/loom_stress --seconds 60   # random mixes of every feature on random pools
./build/loom_bench [tasks] [runs]  # per-task cost and contention by worker count
```

`-DLOOM_SANITIZE=thread|address|undefined` builds everything with a
sanitizer; under TSAN run the suite with
`TSAN_OPTIONS=suppressions=tests/tsan.supp`.

A host project uses loom in one of three ways, and links `loom::loom` in
the first two:

- installed: `cmake --install build --prefix <dir>`, then
  `find_package(loom 0.20 REQUIRED)` with `<dir>` on `CMAKE_PREFIX_PATH`.
  Before 1.0 the package accepts only its own minor version, since a minor
  release may change the API.
- `add_subdirectory(loom)`, which builds neither the tests nor the install
  rules;
- or compiling `src/loom.cpp` with `include/` on the include path.

`LOOM_VERSION` (from `loom/loom_version.h`, which `loom/loom.h` includes)
lets code build against more than one release:
`#if LOOM_VERSION >= LOOM_MAKE_VERSION(0, 20, 0)`.

The configurations the CI checks also run locally as a
[hamr](https://github.com/Aosiee/hamr) suite, `hamr.cluau`: `hamr run` builds
and tests the GCC release and debug builds and a Clang release, and
`hamr run --tier nightly` adds the portable sleep, the three sanitizers,
the no-exceptions embedding and the installed package. Each build lives in
its own `build-hamr-<name>` directory, and the report times every test
case. `hamr list` shows each step's budget and recent times.

Linux is supported, and every push to `dev` is built and tested there with
GCC and Clang, in Debug and Release and under each sanitizer (the Linux
badge above). A release on `main` is a commit that passed there first. Windows and macOS are not supported yet: the thread naming, the OS
thread id and the stack walk used by parking are Linux and POSIX calls
today.

## Branches and releases

`main` holds releases only, each one tagged. New work is built on a stream
branched from `dev`, lands on `dev` once it passes locally, and is checked
there by CI; when enough has gathered, `dev` becomes the next release. See
[`CONTRIBUTING.md`](CONTRIBUTING.md).

## Licence

MIT with one condition: a product that includes loom credits it ("Uses loom -
Copyright (c) 2026 Felipe Lara") in its credits, about box or documentation. No
logo or splash screen. See [`LICENSE`](LICENSE).
