# Design

This is a draft of how loom works inside; the header (`include/loom/loom.h`)
is the contract.

## Task slots

Tasks live in slots (`PoolConfig::maxTasks`, rounded up to chunks of 4096
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
nobody waits on it, then bumps the generation, so a stale handle does not
reach a re-used entry. The generation is 12 bits, so after 4095 reuses of
one entry an old handle can match again; the free list hands out the entry
freed longest ago, which spreads the reuses over the whole table (with 2048
counters, about eight million create-and-destroy cycles). Limiters pack their handles the same way. Each kind of
handle is an enum class over its 32 bits, so the compiler keeps them apart;
inside loom they are unpacked with a cast.

`Signal` is one atomic decrement. Only the decrement that reaches zero takes
the counter's lock: it publishes zero, then releases the tasks waiting for the
counter (`waitsFor`) and wakes its waiters. Zero comes first so that a
released task, which may start on another thread at once, sees the counter
it waited for as zero (a future it waited for reports `IsReady`, and `Get`
returns without waiting). The stored value is the count plus one while it is
nonzero, so between the decrement and the lock the final signaller leaves 1
behind (`Count` reports it as 1). Someone who sees zero can destroy the
counter while that signaller still holds the lock: `DestroyCounter` takes
the lock first, so it waits for the signaller to finish.

`AddToCounter` is a compare-and-swap that adds only while the stored value
is above 1. The final decrement takes it from 2 to 1 before releasing
anyone, so an add that wins comes before that signal: no task or waiter has
been released, and the count stays above zero. Once the value is 1 or 0 the
add fails and changes nothing. Deferred task signals only hold decrements
back, so they can keep a count up a little longer but never let an add
reopen one that has been released.

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
  `PumpMainThreadTasks` return; `Count` applies the caller's own first;
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
Each counter that reaches zero decrements the count for the tasks waiting on
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

### Coroutines

A coroutine's wait is a plain task. Where it awaits a future or counters
that are not yet zero, `await_suspend` submits a task whose function
resumes the coroutine, with the counters as its `waitsFor` and
`waitsForAll` and the awaited `TaskDesc` for the rest, and returns. So a
suspended coroutine costs one task slot and its frame, holds no thread,
and is released by the counters' final signal like any waiting task; the
drain at `Shutdown` and the debug dump see it as one. The resume task may
start on another thread before `await_suspend` has returned: `Submit` reads
the description, the counter list included, before the task can run, and
nothing in the awaiter is touched after it.

A coroutine that returns a `Future` keeps its future's state in its
frame: the promise type holds the block a promise would allocate, with a
reference for the future and one for the coroutine, and creates its
counter before the body starts, so the frame is the coroutine's only
allocation. `co_return` stores the value and an uncaught exception stores
the error. At the final suspension, its locals already gone, the coroutine
signals the counter and drops its reference, and stays suspended there;
whichever reference goes last destroys the frame, its parameters and the
state with it. The frame is not destroyed as the coroutine ends because
the code that starts a coroutine may still touch the frame after its first
suspension (GCC keeps a count in the frame that it updates there), and a
coroutine resumed on another thread can reach its end before that code is
done; the future, which that code has just returned to its caller, keeps
the frame alive past it.

The thread that runs a resume task notes which coroutine it resumed; a
coroutine that suspends again there gives back that task's limiter unit
before it submits its next resume, and one that ends there gives it back
before the signal, as a future's task does, so the rest of the coroutine,
which may run on another thread at once, never finds the unit still taken.
A coroutine that suspends or ends inside some other task (one that never
suspended) leaves that task's unit alone.

An awaited future or counter that is already zero does not suspend, unless
the await gives a `TaskDesc`: then it always suspends, so the rest of the
coroutine always runs where that description says.

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
`ParallelFor` with a chunk size does not group: its chunks are already
coarse. Automatic chunking queues its first pieces as one batch, so they
group like any other (each then splits as workers ask). Queue depths
(`PendingTaskCount`, `PoolSnapshot::queueDepth`) count entries, so a
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
reaches zero; a task waiting for a counter holds none. Taking is one
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

`SetLimit` raising the limit gives out the new units the way
finishing tasks give theirs back, so waiting tasks start at once: one add
to the count for the whole raise, and as many units as the count was below
zero go to the debts and waiting tasks. Lowering
it takes free units out of the count at once. For each held unit it would
need, it leaves a *retiring* debt instead: the count goes down as if a task
waited, and the next finisher that takes the lock to hand its unit on finds
the debt and drops the unit. So no task waits behind a debt, and the count
stays the one number the fast paths read. `HeldUnits` is the limit plus
the debts still open while the count is negative.

A unit's index (`CurrentLimiterUnit`) is handed out only when a task asks for one,
under the limiter's lock: from a stack of indices below the limit that were
used and given back, or else the next one never used. So the index tables
grow with the most tasks that have held units at once, not with the limit,
and a large limit costs nothing until that many run.
The index goes back before the unit does, so whoever takes the unit next
finds one free. After the limit is lowered, more tasks can hold units than
there are indices below it; such a task gets the lowest unused index above
the limit, and that index is dropped again when its task finishes.

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
  zero or eligible work, wakes it; there is no polling. A submit wakes idle
  workers first and helpers for the rest. A helper woken for pool work may
  leave its wait without using the wake (its counter reached zero first, or
  it ran one task of several); it then wakes the next sleeper if pool work
  is still queued, as the last searcher does, so the work never waits for a
  wake that went to a thread that is gone. A helper woken for pool work that
  takes main-thread or dedicated work first passes the wake on before it
  runs that, since it may be long or block. An isolated wait counts only its
  own region's entries in its main-thread or dedicated queue as work: an
  entry of another region there is no reason to stay awake, or to wake it.

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
with tasks of that tag, from its own queues and the injection queue: entries
of other regions it pops from its own deques are pushed to the injection
queue for the other workers (all of them, so none of its own is left below),
and it never steals. A region's task made runnable by a thread outside the
region (the last signal of its counter, a limiter's unit handed on) goes to
the injection queue rather than that thread's deque, where the region's wait
could not take it. The wait sleeps registered on its counter, on its
main-thread or dedicated queue (woken only for its region's entries there),
and on a list of isolated sleepers that an injected entry of its region
wakes; not as a helper for pool work, since most of it would not be its own.
Without workers a region does not isolate: no other thread could run the
work it would set aside.

### Bounded nesting

A task run while waiting may itself wait, and run another task in that
wait, so waits pile up on one thread's stack. For a ParallelFor inside the
chunks of another one that pile has no natural end: each inner wait picks
up another outer chunk, which starts another inner ParallelFor. One thread
could reach a thousand levels, each holding a counter and a stack frame.
Each thread counts the waits it is inside, and from 16 on a ParallelFor
(and what is built on it) runs its pieces on the calling thread, in order,
with no wait: nothing is added to the pile, and the thread's waits further
up still help as usual. The threads further down have plenty to do. The
sort does the same, splitting its pieces on the calling thread exactly as
it would have split them into tasks, so equal elements come out in the
same order however deep it was called. Pipelines and graph runs cannot run
that way (their steps may be pinned to other threads); from 64 waits on
they open an isolated region instead, whose wait runs only their own tasks,
unless the thread is in a region already.

Isolating sooner, or the data-parallel fork-joins at all, had a cost. An
isolated wait helps nobody else: a main thread waiting BlockOnly for work
that only the helping of a deep worker would have run waited for good, and
a fresh region for every deep fork-join made a main-thread step that a
deeper region's work needed not the outer region's, so the main thread,
waiting isolated in it, never ran it.

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

### Scans

A scan reduces each chunk, combines the chunk totals into offsets on the
calling thread, and scans each chunk from its offset. Chunks are about
256 KB of input, fixed by the length, so the grouping, and with it every
bit of the result, never depends on the pool. They go in rounds of up to 32
chunks, reduced and then scanned while the round is still in the shared
cache, so the input comes from memory about once. A scan of an array that
does not fit in the cache is bound by memory bandwidth, and in loom_bench
it is no faster than `std::inclusive_scan`; one that fits gains with the
workers.

### Sorting

`ParallelSort` on 128K elements or more (whose moves cannot throw) starts as
a sample sort. Splitters are elements at fixed positions, sorted by index
so nothing is copied. A parallel pass counts each chunk's elements per
bucket, keeping each element's bucket in a 16-bit id. The counts give each
chunk its place in each bucket, and a second pass moves the elements into
a buffer of the same size. The comparator runs only while counting and
sorting, so if it throws, every element is in the range or in the buffer,
and the buffer is moved back. As after `std::sort`, which sorts the small
pieces, the range is then valid but in no particular order.

Each bucket, or a smaller range as a whole, is then sorted as quicksort
pieces. A piece is split three ways around a median-of-three pivot; the
larger outer part becomes a new task and the smaller is split on in the same
loop. The pieces never wait for each other. As with `ParallelFor`,
completion is counted in elements: one counter starts at the length, each
split signals its run of pivot-equal elements, and each small piece signals
its length once `std::sort` has placed it. Only the caller waits, so a sort
holds one counter and no nested waits however large it is. A piece whose
comparator throws records the error and still signals its elements, so the
count reaches zero and the sort rethrows. Leaf size, buckets and splits
depend on the data and the length only, so the order of equal keys is the
same whatever the pool.

### Pipelines

A pipeline run keeps its state on `Run`'s stack: the free items, each
item's place in the stream, and for each serial stage the place it takes
next and the items that came early, in a ring indexed by place (at most
the item count of places can be waiting, so it never collides). Every task
of a run holds an item that is still in the pipeline and touches the run
only while it does. The task that brings the last item home after the
source has ended signals the run's counter as its last step. A stage task
with a limiter gives its unit back before it lets go of its last item, as
a future's task does before its signal: once that item moves on, the run
can end and `Run` return before the task's runner would otherwise give the
unit back, and the caller may destroy the limiter straight away.

The source fills items in one task for as long as free ones are left; an
item coming home restarts it if it stopped. A serial stage likewise takes
the next item in place order straight away when it is already waiting, and
otherwise lets the item that arrives in its turn start it as a new task.
So a busy pipeline pays a task per parallel step, and the serial steps
mostly run in a row on one thread. If a stage throws, the source stops and
every item in flight passes the remaining stages without running them, so
all of them come home and the counter still reaches zero.

### Per-thread storage

`PerThread<T>` is two levels of cells indexed by `CurrentThreadIndex`, a
fixed array of chunk pointers and chunks of 64 cells allocated on first
use. Each cell is on cache lines of its own, and only its thread constructs
or touches it, so `Local()` takes no lock once its chunk exists. The rest
of the API walks the cells and so must not race with `Local()`.

## Thread identity

Every thread loom runs gets a small index (`CurrentThreadIndex`) on its
first call, usually 0 for the main thread, then the workers and dedicated
threads, though nothing fixes which thread has which. The index allocator outlives pools, so a thread keeps its
index, and per-thread scratch keyed on it stays valid.

A thread gives its index back when it exits, from the destructor of a
pthread key set on its first call, and a new thread takes the lowest index
free. glibc runs key destructors after every C++ thread-local destructor,
so one of those that calls loom still finds the thread's own index. The allocator's mutex is taken only then, and it orders the
exited thread's writes before the next owner's reads, so a `PerThread`
value or a stats row keyed on the index passes from one to the next
whole. The indices therefore stay below the most threads alive at once
rather than growing with every pool a process creates. A key destructor
that runs after ours and asks for an index again sets the key again, and
glibc runs ours once more.

## Parking

`ParkAll` raises a park request: every pool thread finishes its current task,
records its return addresses (a frame-pointer walk bounded by its stack) and
blocks at its loop top; threads outside the pool that registered as park
participants are woken to reach their own park points. Queued work stays
queued. `UnparkAll` releases them all. A host uses this to change code or
data no thread may be running, such as a live code patch.

`ParkAll` waits for threads, not for a number of arrivals. Each worker and
each dedicated thread notes, under the park lock, the round it parked in
(the count of releases so far); the request is complete once every worker
and every dedicated thread registered at that moment shows the current
round. A dedicated thread registered mid-park joins the threads waited for;
one being unregistered leaves them, so neither can complete the park in
place of a worker still in a long task. Participants are only counted.
A parked thread also leaves the park when it is told to stop, so a
dedicated thread can be unregistered while parked, and Shutdown releases a
park still in effect before it drains.

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
tasks waiting for a counter or for a limiter are listed in a lock
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
