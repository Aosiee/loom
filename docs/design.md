# Design

This is a draft of how loom works inside; the header (`include/loom/loom.h`)
is the contract.

## Task slots

Tasks live in a fixed pool of slots (`LoomConfig::maxTasks`, grown with a
warning when exhausted). A slot holds the function, its user data or up to 48
bytes of an inline callable, the affinity, and the counters it signals and
waits on. Submitting a task never allocates once the pool has reached its
working size.

## Counters

A counter is an index into a table plus a generation. `CreateCounter` hands
out a free entry with a fresh generation; `DestroyCounter` asserts the count
is zero and nobody waits on it. A stale handle (an old generation) can never
reach a re-used entry. A task with `waitsFor` sits in its counter's waiting
list until the count reaches zero, then becomes runnable; `signals` is
decremented when the task finishes.

## Queues

There are three kinds of run queue: the shared AnyWorker queue, the
main-thread queue, and one queue per dedicated thread. In this version they
are mutex and condition-variable queues: simple and correct, and fast enough
while the per-task work dwarfs the queue cost (`loom_bench` measures where
that stops being true). Work-stealing deques would replace them behind the
same API.

## Help-while-waiting

`WaitFor` with `Help::WhileWaiting` loops: if the counter is zero it returns;
otherwise it takes one eligible task (AnyWorker work, plus main-thread work
when the waiter is the main thread) and runs it on its own stack. Only when
nothing is eligible does it sleep on the counter. This is what makes
fork-and-join from inside a task safe: the waiter never idles a thread while
the work it waits for is queued.

## Thread identity

Every thread loom runs gets a small index (`CurrentThreadIndex`): 0 for the
main thread, then the workers and dedicated threads; any other thread gets
one on first use. The index allocator outlives pools, so a thread keeps its
index, and per-thread scratch keyed on it stays valid.

## Parking

`ParkAll` raises a park request: every pool thread finishes its current task,
records its return addresses (a frame-pointer walk bounded by its stack) and
blocks at its loop top; threads outside the pool that registered as park
participants are woken to reach their own park points. Queued work stays
queued. `UnparkAll` releases them all. A host uses this to change code or
data no thread may be running, such as a live code patch.

## Shutdown

`Shutdown` drains every queue, joins the workers and dedicated threads, and
asserts nothing was abandoned. It is idempotent, and the `Pool` destructor
calls it.
