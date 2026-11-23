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
have run). It never guarantees the order tasks run in, priorities included
(they decide what is taken next, not when it finishes). Results must not
depend on scheduling order: give each task its own output slot and combine in
a fixed order afterwards, which is what `ParallelReduce` does.

## Shutdown is drain-or-assert

`Shutdown()` first runs everything that was submitted, including work those
tasks submit while it drains, and every wait completes. So every submitted
task must be able to run: a task waiting on a counter nobody will signal
stops the drain (after `PoolConfig::shutdownTimeoutMs`), and what is left,
along with lanes still held, asserts.

## An exception needs a counter to go to

A task's exception is kept on the counter it signals and rethrown by
`WaitFor` on that counter; it also cancels the counter's tasks that have not
started. A task that signals no counter has nowhere to send an exception: if
it throws, the program ends. Give a task that may throw a counter, and wait
on it.

## An isolated region must not need outside work only it can run

Inside `Isolated(...)`, a wait helps only with the region's own tasks and
hands anything else to the other workers. If the region's work waits on
work outside the region, the other threads must be free to run it: an
isolated wait never will. (With no workers there is no isolation.)

## Keep hot-path callables small

`Submit(F)` takes any callable, but only small ones avoid an allocation: a
trivially copyable callable of up to 48 bytes, or a nothrow-movable one of up
to 40 bytes (capturing a `std::string` or a `shared_ptr`, say), lives in the
task slot. Anything bigger, or aligned past 16 bytes, costs one heap
allocation per task. For per-frame work, capture pointers and indices. An
owned callable is destroyed after its task runs, or when the task is skipped
by cancellation, before the task signals its counter.

## A graph runs once at a time and does not change while it runs

A `TaskGraph` keeps one set of per-run counts, so a second `Submit` before
the first run has finished asserts, as does adding a node or an edge, or
destroying the graph, while it runs. Wait for the run's counter first. A
frame that overlaps the next one (simulation of frame N+1 while frame N
renders) uses two graphs.

## A limited task must not wait for its own limiter's work

A task holds its limiter's unit while it runs, so a task that waits for
another task of the same limiter can wait forever: with every unit held,
that other task never becomes runnable, and helping cannot run it (it waits
in the limiter, not in a queue). Split the work into tasks that hand on
through counters (`waitsFor`) instead of one that waits in the middle. A
limited task may wait for anything else.

## Keep a future alive while anything reads it

A future's value lives in the future, and its counter is destroyed once the
future is dropped and its task has finished. A task that waits on
`future.Counter()` and reads `future.Get()` needs the future to outlive it:
keep the future where both can reach it (a member, or the frame's state),
and drop it only after that task's own counter has reached zero. Each
pending future holds one counter, so thousands in flight at once need a
larger `PoolConfig::maxCounters`.

A continuation is the exception: `Then` holds its input itself, so a chain
may be built from temporaries and only its last future kept.

## Set every promise once, while the pool lives

A promise's future is completed only by `SetValue`, `SetException` or the
promise's destructor (which completes it with `BrokenPromise`). Setting it
twice terminates; setting or destroying it after `Shutdown` touches a
counter of a pool that is gone, so finish or drop it first. An IO
callback that may never fire should still destroy its promise, so whoever
waits on the future sees the broken promise instead of waiting forever.

## Hold nothing thread-bound across a co_await

The part of a coroutine after a `co_await` that suspends may run on
another thread. A lock held across it is unlocked on a thread that does
not own it (undefined for `std::mutex`), and a thread-local read after it
is another thread's. Release locks before the `co_await`, and read
`CurrentThreadIndex`, `PerThread::Local` and `CurrentLimiterUnit` again
after it.

A coroutine's parameters are copied into its frame, but a reference
parameter (a `const std::string&`, a lambda's captures by reference) stays
a reference: what it names must outlive the coroutine, not just the call
that started it. Pass what the coroutine reads after a suspension by
value. A counter awaited with `After` must live until the coroutine has
resumed.

## Merge per-thread storage only after the work is done

`PerThread<T>::Local()` may run on many threads at once, but `ForEach`,
`Size` and `Clear` walk every thread's cell without a lock. Call them after
the wait for the work that writes the cells, never while a task may still
call `Local()`. That wait is also what makes each thread's writes visible
to the merge.

## Add to a counter before submitting what will signal it

`AddToCounter` succeeds only while the count is above zero, and only a
signal still to come can keep it there. Add for a child before submitting
the child: once the child is submitted it may finish, and if it was the
last thing holding the count up the counter reaches zero and the next add
fails. A running task that signals the counter holds it open for its own
children, so the tree-walk pattern (add, then submit, from inside a task of
that counter) always works. Check the result where nothing else holds the
count up.
