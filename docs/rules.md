# The rules

loom keeps its own state consistent under any interleaving. These rules keep
*your* code correct on top of it. Each comes from a bug class that loom makes
easy to write.

## Thread-affine APIs belong on the main thread

A GPU device, a windowing system or any other API that must be called from one
thread is used only from `Affinity::MainThread` tasks (or from the main thread
directly), unless that exact path is documented as thread-safe.

## No blocking IO on pool workers

A worker blocked on a socket, a file or a process stops every task behind it.
Blocking work runs on a dedicated thread, which owns its own queue, or off loom
entirely.

## Never help while holding a lock

`WaitFor(counter, Help::WhileWaiting)` may run any eligible task on the waiting
thread's stack, including one that takes the lock you hold. Wait with
`Help::BlockOnly`, or release the lock before waiting.

## No ambient thread-local context

Because a wait can run unrelated tasks, a task must not assume thread-local
state set by whoever ran before it. Pass context through the task's
`userData` (or the captured state of an inline callable).

## Determinism is about what, not when

loom guarantees *what* completes (a counter reaches zero when its signallers
have run). It never guarantees the order tasks run in. Results must not depend
on scheduling order: give each task its own output slot and combine in a fixed
order afterwards.

## Shutdown is drain-or-assert

Every submitted task must be able to run before the pool shuts down. Tasks
still queued, lanes still held, or counters still waited on at `Shutdown()`
are bugs, and assert.

## Inline callables are small and trivially copyable

`Submit(F)` stores the callable in the task slot (48 bytes, no allocation),
so it must be trivially copyable. Capture pointers, not owning objects.
