# Design

This is a draft of how loom works inside; the header (`include/loom/loom.h`)
is the contract.

## Task slots

Tasks live in slots (`LoomConfig::maxTasks`, rounded up to chunks of 4096
and grown a chunk at a time, with a warning, when exhausted). A slot holds the
function, its user data or up to 48 bytes of an inline callable, the
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
the counter's lock: it releases the tasks parked on the counter (`waitsFor`),
wakes its waiters, and only then publishes zero. The stored value is the
count plus one while it is nonzero, so the final signaller leaves 1 behind
until it is done; nobody can observe zero, and destroy or re-use the counter,
while that signaller still touches it (`Load` reports the settling 1 as 1).

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

## Sleeping and waking

Every sleep in loom goes through one `Waiter` per thread, a 32-bit word the
thread sleeps on (a raw futex on Linux). A sleeper sets the word to armed,
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
never idles a thread while the work it waits for is queued. A helper takes
any eligible task, not only its own subtree, so a wait can run unrelated
work.

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

`Shutdown` drains every queue (a worker exits only after a full search finds
nothing), joins the workers and dedicated threads, and asserts nothing was
abandoned. It is idempotent, and the `Pool` destructor
calls it.
