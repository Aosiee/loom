# Design

This is a draft of how loom works inside; the header (`include/loom/loom.h`)
is the contract.

## Task slots

Tasks live in slots (`LoomConfig::maxTasks`, rounded up to chunks of 4096
and grown a chunk at a time, with a warning, when exhausted). A slot holds the
function, its user data or up to 48 bytes of an inline callable (an owned,
non-trivial callable keeps its destroy function in the last 8), the
affinity, and the counters it signals and waits on; each is two cache lines
so adjacent slots run on different threads never share one. Every field a
submit writes and a run reads sits in the first line (the second holds only
inline-callable storage), so a plain task moves one line between cores. Chunks never
move, so a slot index is valid on any thread without a lock.

Free slots live in two places. Each thread with a deque (the main thread and
the workers) keeps a small cache of up to 30 slots: it allocates from the
cache and finished tasks land in it. The rest sit on one shared lock-free
stack of *batches* of up to 15 slots (a Treiber stack whose head carries a
32-bit tag against ABA). A cache trades with the stack a whole batch per
compare-and-swap: an empty cache pops a batch, a full one pushes its newest
15. A worker going to sleep flushes its whole cache, so idle threads never
sit on slots another thread needs. Threads without a cache take one slot from
a batch (pushing back the rest) and return slots one at a time.

A batch is a head slot that *lists* the other 14 in its first cache line, in
bytes a free slot does not use (the task fields and the start of the inline
storage); the head's count and its link to the next batch sit beside them.
Taking a batch therefore costs one cache miss, on a line the submit is about
to write anyway. (A chain through the slots cost one *dependent* miss per
slot, each on a line another core, often on the other die, had written last;
it was the main thread's largest cost with every worker running.)

So a submit touches the shared stack once per 15 tasks, not once per task:
the stack's head is the line every worker writes when it finishes a task,
and contending on it per task made creation about 20 times slower with every
worker running. No lock and no allocation once the pool is at its working
size; the runner reads the slot in place and owns it until it releases it.

## Counters

A counter is an index into a table plus a generation. `CreateCounter` takes
an entry off a free list; `DestroyCounter` asserts the count is zero and
nobody waits on it, then bumps the generation, so a stale handle never reaches
a re-used entry.

`Signal` is one atomic decrement. Only the decrement that reaches zero takes
the counter's lock: it publishes zero, then releases the tasks parked on the
counter (`waitsFor`) and wakes its waiters. Zero comes first so that a
released task, which may start on another thread at once, sees the counter
it waited for as zero (a future it waited for is `Ready`, and `Get` returns
without waiting). The stored value is the count plus one while it is
nonzero, so between the decrement and the lock the final signaller leaves 1
behind (`Load` reports it as 1). Someone who sees zero can destroy the
counter while that signaller still holds the lock: `DestroyCounter` takes
the lock first, so it waits for the signaller to finish.

### Deferred task signals

When tasks signal one counter, that counter's line is the hottest in the
pool: every task on every worker writes it. So a thread with a deque (the
main thread, a worker) does not signal per task. It counts its finished
tasks' signals while consecutive tasks signal the same counter, and applies
them as one `Signal(counter, n)`:

- before running a task that signals a different counter, or none;
- when it runs out of work at hand (after its first failed steal pass),
  before it parks, sleeps or exits;
- before `WaitFor` waits and before `WaitFor`, `TryHelp` and
  `PumpMainThreadTasks` return; `Load` applies the caller's own first;
- after 64 at most.

The first rule is what makes this invisible. While a task runs, the only
signals its thread holds are for the counter that task itself signals, and
that counter cannot reach zero before the task finishes. So holding them back
never delays a zero, and a task that blocks (on anything) never blocks a
counter it does not itself count toward. Dedicated threads and outside
threads signal per task. `Signal` called by user code is always immediate.

### Exceptions and cancellation

A task's function runs inside a catch-all (when exceptions are on). The
first exception from a task that signals a counter is kept on that counter,
and the counter is marked cancelled; `WaitFor` rethrows it once the count is
zero. Before running a task, its thread reads the cancelled flag of the
counter it signals and skips the function if it is set (the task still
signals). `Cancel` sets that flag. The flag lives on its own cache line with
the counter's alive flag and generation, away from the count that every
signal writes, so that check is a read of a line that is rarely written.

### Several dependencies

A task may wait for `waitsFor` and every counter in `waitsForAll`. Its slot
then holds a count of unmet counters (in the group link, which only a
runnable task uses). The submitter takes one count of its own, parks the
slot on each counter that is not yet zero (adding one per counter), then
drops its own count: the task becomes runnable at once if all were zero.
Each counter that reaches zero decrements the count for the tasks parked on
it, and the decrement that takes it to zero queues the task.

### Task graphs

A `TaskGraph` keeps its nodes and edges as built and compiles them once per
change (Kahn's algorithm checks for a cycle on the way). Nodes with the same
set of predecessors share one *wait set*: a count of unfinished
predecessors, decremented once by each of them, whose zero releases all its
members at once. A dense layer (every node after every node of the layer
before) then costs one decrement per predecessor on one count, where one
count per node would take a decrement per edge on as many lines. A set with
one predecessor has no count: that predecessor releases it directly. A node
holds no task slot until its set is ready.

A run resets the counts, creates a counter holding the number of nodes,
parks a small finishing task on it and queues the root set. A node's task
runs the node's function (unless the caller's counter is cancelled),
decrements the sets it feeds and queues the members of those it takes to
zero, then returns; the task runner signals the run's counter, batched per
thread like any task's signal. The finishing task runs once every node has
signalled: it destroys the run's counter, clears the running flag and
signals the caller's counter, after which the graph may run again or be
destroyed. A node catches its own exception (for the caller's counter)
rather than leaving it to the runner, and the run's counter is never
cancelled, so every node's task runs and releases its successors whether the
node ran, threw or was skipped: a failed run still completes.

Queuing a set is the hot path when a graph is wide: it writes each slot
straight from the node, adds the set to the live count once, and chains
runnable AnyWorker Normal members into groups as `Submit(span)` does, so it
costs less than a `Submit(span)` of the same tasks. A node's task reads its
node through a pointer into the graph; copying what it needs into the slot
instead made the submitter write the slot's second cache line, which slowed
a wide graph by about a third.

### Futures

`Async` allocates one block holding the callable, room for the result, a
counter handle and a reference count of two (the future and its task), then
submits a task with the block as its userData. The task runs the callable,
stores its result or its exception in the block, destroys the callable and
signals the counter itself rather than through the task runner, then drops
its reference. Signalling from inside the task is what lets it drop the
reference afterwards: whichever of the two lets go last destroys the counter
and frees the block, so a future can be dropped before its task has run. At
that point the counter is always settled: either the task's own final signal
has returned, or the future's wait has. A limited future's task gives its
unit back just before that signal (the runner would do it only after the
function returns), so once `Get` returns the limiter is free, as it is for
a plain task once its counter's wait returns.

`Get` waits on the counter (helping, by default) and rethrows the exception
from the block, so a future's exception never goes through the counter's
own error and cancellation. Because the task signals no counter through
the runner, nothing can skip it: a future's task ignores `Cancel`, and
always completes its future.

A `Promise` uses the same block with no task behind it: the promise holds
the reference the task would, `SetValue` or `SetException` stores into the
block and signals the counter, and the promise's destructor drops its
reference. A promise destroyed unset first stores `BrokenPromise` and
signals, so nothing waiting on it hangs. Setting a promise never touches a
limiter: a task that sets one keeps its own unit until it returns, unlike
a future's own task, which gives its unit back with its signal.

`Then` is a future whose task holds a reference to its input and waits on
the input's counter through `waitsFor`, so it is queued only once the
input is ready and never blocks a worker. When it runs it either copies the
input's failure (error or broken promise) into its own block, so a failure
skips the rest of a chain, or calls the callable with the input's value;
then it drops the input and completes as `Async` does. Because the
continuation holds its input, a chain's earlier futures may be dropped as
soon as `Then` returns.

## Queues

AnyWorker work lives in work-stealing deques, one per worker and one for the
main thread: Chase-Lev deques after Le et al. (PPoPP 2013), with seq_cst
operations in place of the fences so ThreadSanitizer can check them. The owner
pushes and pops the bottom (newest first, good for nested fork-join), and
every other thread steals the oldest from the top. A deque's buffer doubles
when full; old buffers stay until the pool dies, so a thief holding one still
reads valid entries. The owner keeps its last view of the top and re-reads
the real one (the line every thief writes) only when the buffer looks full:
the top only grows, so the stale view can only overstate how full it is.

A thread with no deque (a dedicated thread, or any thread loom did not
start) submits AnyWorker work into one shared injection queue, a mutex queue
every searcher checks first. The main-thread queue and each dedicated
thread's queue are mutex queues too: they are per-thread and not on the hot
path.

### Groups

A queue entry is a task slot, or (with its top bit set) the first slot of a
*group*: tasks chained through the slots and handed over as one entry.
`Submit(span)` chains its runnable AnyWorker tasks (no `waitsFor`) into
groups of `n / 8`, at most 32, so a batch always leaves at least 8 entries
for thieves to take at once. Whoever takes a group pushes the rest onto its
own deque, one entry each, and runs the first; idle workers then steal them
from there. (A thread without a deque pushes the rest back to the injection
queue as one group.)

This is what keeps one submitting thread from being the bottleneck. Handing
tasks over one at a time, every task cost a steal on the submitter's deque,
and its top and bottom bounced between up to 31 cores: with 23 ns tasks the
workers spent about 92% of their time searching. Groups make that one steal
per 32 tasks and spread the rest across the workers' own deques.
`ParallelFor` does not group: its chunks are already coarse. Queue depths
(`PendingTaskCount`, `PoolSnapshot::anyQueueDepth`) count entries, so a
queued group counts once.

### Priorities

Each deque owner has three deques (High, Normal, Low) and the injection
queue has one per priority. A thread takes its own High work, then its own
Normal work; a search takes High work anywhere, then Normal anywhere, then
Low (its own first). Two pool-wide counts of queued High and Low entries let
a search skip those queues while they are empty, so Normal work, the
default, pays nothing for priorities. The main-thread and dedicated queues
put a High task at the front. Groups are always Normal.

### Limiters

A limiter is a count of free units, a lock and a queue of waiting tasks. A
task with a limiter takes a unit at the point it becomes runnable: in
`Submit` when it has no counters to wait for, or when the last of them
reaches zero; a task parked on a counter holds none. Taking is one
decrement of the count while it is positive. At zero or below the task waits:
under the lock it puts itself in the queue (a High task at the front), and
the slot stays out of every run queue, so no worker pops it only to put it
back. A finishing task gives its unit back with one increment, before it
signals its counter, so a `WaitFor` that returns has every unit back. Only
when the count was negative (someone waits) does it take the lock and hand
the unit over: it queues the first waiting task itself, onto its own deque.
A waiter that has decremented but not yet queued itself is covered by a
*handoff*: the finisher, finding the queue empty, leaves the unit on the
limiter, and the waiter takes it under the lock instead of queueing.

The unit is held from queuing to the end of the run, so the limit counts
queued tasks as well as running ones. That is what keeps the waiting tasks
out of the deques, and it bounds what the work can occupy just as well: a
held unit whose task sits in a queue is a worker the work is not using.
Limited tasks are never grouped (each takes its own unit); a task's limiter
is a 16-bit index in a free byte pair of the slot's first line, so a task
without one pays a load it already makes and an untaken branch.

## Sleeping and waking

Every sleep in loom goes through one `Waiter` per thread, a 32-bit word the
thread sleeps on: a raw futex on Linux, a mutex and condition variable
elsewhere (the `LOOM_NO_FUTEX` CMake option selects the portable one on
Linux, for testing). A sleep can carry a deadline, which is how
`TryWaitFor` gives up. A sleeper sets the word to armed,
registers wherever a waker will look for it, re-checks for work, and only
then sleeps; a waker takes it off the list, sets the word and wakes it while
holding that list's lock. The sleeper takes the same lock to unregister, so
it is never woken after it has moved on.

- Idle workers register in the pool's idle list. A worker that runs dry
  searches first (spinning, then yielding, across the injection queue and
  every deque) and counts itself as *searching*; a submit wakes an idle
  worker only when no worker is searching, and the last searcher to find work
  wakes the next one. A batch (`Submit(span)`, `ParallelFor`) wakes as many
  workers as it pushed queue entries, up to the worker count; a worker that
  unpacks a group wakes more for the tasks it spread.
- A helper asleep in `WaitFor` registers on the counter, in the idle list
  (when it may run pool work), and on the main-thread queue (the main
  thread) or its dedicated queue (a dedicated thread). Whichever comes first,
  zero or eligible work, wakes it; there is no polling.

The lost-wakeup pair: a sleeper bumps the idle count (a seq_cst
read-modify-write) and then re-checks the queues; a submitter pushes (a
plain release store) and then reads the idle count with a seq_cst
read-modify-write, not a load. Both RMWs sit in the idle count's modification
order: if the submitter's comes first, its push happens-before the sleeper's
re-check, which sees the work; if the sleeper's comes first, the submitter
reads a nonzero count and wakes it. So a push needs no fence of its own: the
ordering costs one RMW per batch instead of a full fence per task.

## Help-while-waiting

`WaitFor` with `Help::WhileWaiting` loops: if the counter is zero it returns;
otherwise it takes one eligible task (main-thread work first on the main
thread, then its own deque, then the injection queue and the other deques) and
runs it on its own stack. Only when nothing is eligible does it sleep, as
above. This is what makes fork-and-join from inside a task safe: the waiter
never idles a thread while the work it waits for is queued. Outside an
isolated region a helper takes any eligible task, not only its own subtree,
so a wait can run unrelated work; inside one it runs only the region's (see
Isolation).

### Isolation

`Isolated(fn)` gives the calling thread a fresh 16-bit region tag for the
duration of `fn`. Every task records the tag of the thread that submitted
it, and a thread running a task takes that task's tag, so a region's tasks
and their descendants share it. A wait whose thread has a tag helps only
with tasks of that tag, and only from its own queues: entries of other
regions it pops are pushed to the injection queue for the other workers, and
it never steals. It sleeps registered on its counter (and its main-thread or
dedicated queue) but not as a helper for pool work, since none would be its
own. Without workers a region does not isolate: no other thread could run
the work it would set aside.

## Parallel loops

`ParallelFor` with a chunk size cuts the range into fixed chunks, one task
each, pushed as one batch. With chunk size 0 it chunks automatically: one
piece per thread to start with, each working through its range a grain
(about 1/64 of its share) at a time and splitting off half of what is left
whenever a worker is idle or searching. Completion is counted in items, by
an explicit signal per grain, so a split needs no counter increment, and a
cancelled count (a piece threw) makes the other pieces signal the rest and
stop.

`ParallelReduce` uses fixed chunks whose size depends only on the range (and
the chunk size, if given), keeps one partial per chunk, and folds the
partials in index order on the calling thread. Nothing about the result
depends on timing or the number of workers.

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

## Debug dump

Each pool thread, and the main thread, keeps a small record of what it is
doing on a cache line of its own: the innermost task it runs (name id,
counter, limiter unit), how many tasks it is inside, and the innermost wait
with the time it began. Only that thread writes it, with relaxed stores; a
task run while helping saves the outer values and puts them back. That is a
few stores per task and one clock read per wait. A thread outside the pool
has no record, but each sleeper notes its thread index in its `Waiter`, so
it still shows up asleep on the counter it waits for.

`CaptureDebugState` reads those records, then walks the counter and limiter
tables under their table locks, and each live entry under its own lock: the
tasks parked on a counter or waiting for a limiter are listed in a lock
they cannot leave without, so their slots are stable while it reads them.
Every lock is taken with a try-lock that gives up after about two
milliseconds and marks that part busy, so a capture never hangs on a thread
stuck holding a lock. Nothing is frozen while it reads; it is a picture
taken piece by piece.

`FormatDebugState` prints the capture and adds hints for what it can
recognise: something waits but nothing runs or is queued (a missing
`Signal` or `Submit`), main-thread tasks queued while the main thread
neither waits nor pumps, and a task holding a limiter's unit that waits
while tasks of that limiter wait for a unit.

## Shutdown

`Shutdown` drains first: with every thread still running, the calling
thread helps (main-thread work first) until no task is live, so work
submitted while draining runs and every wait completes. If no task finishes
for `shutdownTimeoutMs`, it reports how many are left and stops draining.
Then it stops and joins the workers (each exits after a search finds
nothing) and the dedicated threads, and asserts nothing was abandoned. It is idempotent, and the `Pool` destructor
calls it.
