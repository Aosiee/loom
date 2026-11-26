// loom_fuzz: model-based differential fuzzing of loom's public API.
//
// A seed and a size make a *program*: a list of pool lifetimes (epochs), each
// with a random PoolConfig, dedicated threads, lanes and limiters, and a list
// of operations run by the main thread. Some operations finish before the
// next starts (algorithms, graphs, pipelines, parking, registering threads);
// others start work and are joined a few operations later (dependency graphs
// of tasks, future chains, coroutines, outside threads), so their tasks run
// while the main thread does everything else. Every operation knows from its
// own seed what loom's contract says must happen, and checks it:
//
//   - every task runs once, or is skipped only when its counter was
//     cancelled (and must be skipped when the cancel happened before it could
//     start); a skipped owned callable is still destroyed, exactly once;
//   - a task starts only once every counter it waits for is zero, and after
//     every task (and manual signal) behind those counters has finished;
//   - a limiter never has more tasks running than the highest limit in
//     effect since the newest of them was submitted, and their unit indices
//     are distinct and below the highest limit used;
//   - a main-thread task runs on the main thread, a dedicated one on its
//     thread, and a Sleep dedicated thread or a lane never runs pool work
//     while there are workers;
//   - WaitFor rethrows exactly when a task of the counter threw, an error
//     from one of those tasks, the same one every time; IsCancelled is
//     exactly "cancelled or a task threw";
//   - algorithm results equal sequential std:: equivalents (with
//     non-commutative operators, so the combine order is checked too);
//   - futures deliver the right value or the right error, continuations of
//     a failed step never run, promises set or broken from other threads;
//   - coroutines resume where they asked to and return the right value;
//   - parked pools run nothing; Shutdown drains what was left; at the end no
//     task is live and every limiter is free.
//
// Usage:
//   loom_fuzz --seed N [--size N]     generate and run one program
//   loom_fuzz --program TEXT          run a program printed by --print (the
//                                     shrinker edits these)
//   loom_fuzz ... --print             print the program and exit
//   loom_fuzz ... --timeout SECONDS   abort with a debug dump after this long
//   loom_fuzz ... --verbose           log each operation as it starts
//   loom_fuzz ... --block-only        also wait with BlockOnly (see g_BlockOnly)
//
// Exit status 0 when every check passed, 1 when one failed; a hang aborts.
// fuzz.py runs many seeds in parallel and shrinks failures.
#include "loom/loom.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <set>
#include <string>
#include <thread>
#include <vector>

#if defined(__linux__)
#include <sys/prctl.h>
#include <unistd.h>
#endif

namespace fz {

using Clock = std::chrono::steady_clock;
using loom::Affinity;
using loom::CounterHandle;
using loom::Priority;

// ---- Randomness ----------------------------------------------------------------

struct Rng
{
	uint64_t s;
	// Scrambled, so that nearby seeds give unrelated streams (a plain
	// seed * golden start would make seed s + 1's stream seed s's, shifted).
	explicit Rng(uint64_t seed) : s(Scramble(seed)) {}
	static uint64_t Scramble(uint64_t z)
	{
		z ^= 0x632BE59BD9B4E019ull;
		z = (z ^ (z >> 33)) * 0xFF51AFD7ED558CCDull;
		z = (z ^ (z >> 33)) * 0xC4CEB9FE1A85EC53ull;
		return z ^ (z >> 33);
	}
	uint64_t Next()
	{
		uint64_t z = (s += 0x9E3779B97F4A7C15ull);
		z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
		z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
		return z ^ (z >> 31);
	}
	uint32_t Below(uint32_t n) { return n ? (uint32_t)(Next() % n) : 0; }
	int Range(int lo, int hi) { return hi <= lo ? lo : lo + (int)Below((uint32_t)(hi - lo + 1)); }
	bool Chance(double p) { return (double)(Next() >> 11) * (1.0 / 9007199254740992.0) < p; }
	uint64_t Fork() { return Next(); }
};

uint64_t Hash(uint64_t a, uint64_t b = 0)
{
	Rng r(a ^ (b * 0xD6E8FEB86659FD93ull));
	return r.Next();
}

extern bool g_BlockOnly; // see its definition

// ---- Failures ------------------------------------------------------------------

std::atomic<int> g_Failures{ 0 };
std::string g_ProgramText;

[[gnu::format(printf, 3, 4)]] void Fail(int line, const char* cond, const char* fmt, ...)
{
	char buf[1024];
	va_list ap;
	va_start(ap, fmt);
	std::vsnprintf(buf, sizeof buf, fmt, ap);
	va_end(ap);
	std::fprintf(stderr, "FUZZ CHECK FAILED (fuzz.cpp:%d): %s: %s\n", line, cond, buf);
	std::fflush(stderr);
	if (g_Failures.fetch_add(1) + 1 >= 20)
	{
		std::fprintf(stderr, "too many failures; program: %s\n", g_ProgramText.c_str());
		std::_Exit(1);
	}
}

#define CHECK(cond, ...) \
	do { if (!(cond)) ::fz::Fail(__LINE__, #cond, __VA_ARGS__); } while (0)

// A task's exception: which operation and which task threw it.
struct FuzzError
{
	int op;
	int id;
};

// ---- Threads ---------------------------------------------------------------------

enum class Role : uint8_t { Pool, Main, Outside };
thread_local Role t_Role = Role::Pool;
// The dedicated slot (Env::ded index) this thread is, once its probe ran.
thread_local int t_DedSlot = -1;
std::thread::id g_MainTid;
// Instrumented task bodies running on pool threads right now (a park checks it).
std::atomic<int> g_PoolRunning{ 0 };
// A global clock of submissions and limit changes, for the limiter check.
std::atomic<uint64_t> g_Seq{ 1 };

uint64_t Tick() { return g_Seq.fetch_add(1, std::memory_order_seq_cst); }

void SpinFor(int iterations)
{
	volatile uint64_t x = 0;
	for (int i = 0; i < iterations; i++)
		x = x + (uint64_t)i;
}

// Help until `done` holds: the main thread pumps its queue too. For waiting
// on an outside thread or a task with no counter without blocking the main
// thread's own work.
template <typename Pred>
void HelpUntil(Pred done)
{
	while (!done())
	{
		bool ran = loom::TryHelp();
		if (std::this_thread::get_id() == g_MainTid)
		{
			loom::PumpMainThreadTasks();
		}
		if (!ran)
			std::this_thread::sleep_for(std::chrono::microseconds(50));
	}
}

// An outside thread: role Outside, joined while helping.
class Outside
{
public:
	template <typename F>
	explicit Outside(F&& fn)
	{
		m_Thread = std::thread([this, f = std::forward<F>(fn)]() mutable {
			t_Role = Role::Outside;
			f();
			m_Done.store(true, std::memory_order_release);
		});
	}
	~Outside() { Join(); }
	void Join()
	{
		if (!m_Thread.joinable())
			return;
		HelpUntil([&] { return m_Done.load(std::memory_order_acquire); });
		m_Thread.join();
	}

private:
	std::atomic<bool> m_Done{ false };
	std::thread m_Thread;
};

// ---- The epoch's environment ---------------------------------------------------------

struct Ded
{
	loom::DedicatedId id{};
	bool lane = false;
	bool steal = false;
	std::thread::id tid;
	int users = 0; // operations still running that may submit to it (main thread only)
	bool alive = true;
};

constexpr int kMaxUnits = 256;

struct Lim
{
	loom::LimiterHandle h = loom::kInvalidLimiter;
	bool leaf = false;     // only for tasks that never wait inside
	bool blockers = false; // the BlockOnly waiters' own limiter, never changed
	std::mutex mu;
	std::vector<std::pair<uint64_t, uint32_t>> log; // (tick, limit) in tick order
	uint32_t maxEver = 0;
	uint32_t current = 0;
	std::atomic<int> running{ 0 };
	std::atomic<uint8_t> busy[kMaxUnits] = {};

	// The highest limit in effect at any point in [since, now], and the highest ever.
	std::pair<uint32_t, uint32_t> Cap(uint64_t since)
	{
		std::lock_guard<std::mutex> lock(mu);
		uint32_t cap = 0;
		for (size_t i = 0; i < log.size(); i++)
		{
			const bool inEffectAtSince = log[i].first <= since && (i + 1 == log.size() || log[i + 1].first > since);
			if (log[i].first > since || inEffectAtSince)
				cap = std::max(cap, log[i].second);
		}
		return { cap, maxEver };
	}
	// A raise counts from before the call (tasks may start under it at
	// once), a lowering from after it (until it returns, tasks may still take
	// units under the old limit).
	void Set(uint32_t limit)
	{
		std::lock_guard<std::mutex> lock(mu);
		const bool raise = limit > current;
		maxEver = std::max(maxEver, limit);
		current = limit;
		if (raise)
			log.push_back({ Tick(), limit });
		loom::SetLimit(h, limit);
		if (!raise)
			log.push_back({ Tick(), limit });
	}
	std::string History()
	{
		std::lock_guard<std::mutex> lock(mu);
		std::string s;
		for (auto& [t, l] : log)
			s += std::to_string(l) + "@" + std::to_string(t) + " ";
		return s;
	}
};

struct Env
{
	int workers = 0;
	std::vector<std::unique_ptr<Ded>> ded;
	std::vector<std::unique_ptr<Lim>> lims; // [0], [1]: leaf; then outer; maybe a blockers one last
	Lim* blockers = nullptr;
	std::set<uint32_t> dedIdsSeen;
	int opIndex = 0;
};
Env* g_Env = nullptr;

// Where a task runs.
struct Place
{
	Affinity aff = Affinity::AnyWorker;
	int ded = -1;
	Priority pri = Priority::Normal;
	int lim = -1;
};

enum : unsigned
{
	kMain = 1,      // may be MainThread
	kDedAny = 2,    // may be any alive dedicated thread (generation on the main thread only)
	kLeafLim = 4,   // may take a leaf limiter
	kOuterLim = 8,  // may take an outer limiter (tasks that wait inside)
};

// `ownDed`: a dedicated slot known alive for as long as the work runs.
Place RandomPlace(Rng& r, unsigned flags, int ownDed = -1, std::vector<int>* users = nullptr)
{
	Place p;
	const int a = (int)r.Below(10);
	if (a < 2 && (flags & kMain))
		p.aff = Affinity::MainThread;
	else if (a < 4)
	{
		int slot = -1;
		if (flags & kDedAny)
		{
			std::vector<int> alive;
			for (size_t i = 0; i < g_Env->ded.size(); i++)
				if (g_Env->ded[i]->alive)
					alive.push_back((int)i);
			if (!alive.empty())
				slot = alive[r.Below((uint32_t)alive.size())];
		}
		else
			slot = ownDed;
		if (slot >= 0)
		{
			p.aff = Affinity::Dedicated;
			p.ded = slot;
			if (users)
				users->push_back(slot);
		}
	}
	const int pr = (int)r.Below(6);
	p.pri = pr == 0 ? Priority::High : pr == 1 ? Priority::Low : Priority::Normal;
	if (r.Chance(0.3))
	{
		std::vector<int> ok;
		for (size_t i = 0; i < g_Env->lims.size(); i++)
		{
			const Lim& l = *g_Env->lims[i];
			if (l.blockers)
				continue;
			if ((l.leaf && (flags & kLeafLim)) || (!l.leaf && (flags & kOuterLim)))
				ok.push_back((int)i);
		}
		if (!ok.empty())
			p.lim = ok[r.Below((uint32_t)ok.size())];
	}
	return p;
}

loom::TaskDesc Desc(const Place& p)
{
	loom::TaskDesc d;
	d.affinity = p.aff;
	if (p.aff == Affinity::Dedicated)
		d.thread = g_Env->ded[(size_t)p.ded]->id;
	d.priority = p.pri;
	if (p.lim >= 0)
		d.limiter = g_Env->lims[(size_t)p.lim]->h;
	return d;
}

const char* AffName(Affinity a)
{
	return a == Affinity::MainThread ? "main" : a == Affinity::Dedicated ? "dedicated" : "any";
}

void CheckPlace(const Place& p, const char* what)
{
	const auto self = std::this_thread::get_id();
	if (p.aff == Affinity::MainThread)
		CHECK(self == g_MainTid, "%s: a MainThread task ran off the main thread", what);
	else if (p.aff == Affinity::Dedicated)
		CHECK(self == g_Env->ded[(size_t)p.ded]->tid, "%s: a task for dedicated slot %d ran on another thread (slot %d)", what,
			p.ded, t_DedSlot);
	else if (t_DedSlot >= 0 && g_Env->workers > 0)
	{
		const Ded& d = *g_Env->ded[(size_t)t_DedSlot];
		CHECK(d.steal, "%s: an AnyWorker task ran on a %s that does not take pool work", what, d.lane ? "lane" : "Sleep dedicated thread");
	}
}

// Brackets an instrumented task body: checks where it runs and, for a
// limited one, the limit and its unit index.
class TaskScope
{
public:
	TaskScope(const Place& p, uint64_t submitTick, const char* what, bool checkNoUnit = true)
		: m_What(what)
	{
		CheckPlace(p, what);
		m_Pool = t_Role == Role::Pool;
		if (m_Pool)
			g_PoolRunning.fetch_add(1, std::memory_order_seq_cst);
		if (p.lim >= 0)
		{
			m_Lim = g_Env->lims[(size_t)p.lim].get();
			const int running = m_Lim->running.fetch_add(1, std::memory_order_seq_cst) + 1;
			// The index is handed out lazily, here: read the limits after it,
			// so a raise in between (from another thread) is counted.
			m_Unit = loom::CurrentLimiterUnit();
			const auto [cap, maxEver] = m_Lim->Cap(submitTick);
			CHECK((uint32_t)running <= cap, "%s: %d tasks of limiter %d running, limit at most %u since submit", what, running, p.lim, cap);
			CHECK(m_Unit != loom::kNoUnit && m_Unit < maxEver && m_Unit < kMaxUnits, "%s: unit index %u, highest limit %u", what,
				m_Unit, maxEver);
			if (m_Unit < kMaxUnits)
				CHECK(m_Lim->busy[m_Unit].exchange(1) == 0, "%s: unit %u of limiter %d held by two running tasks", what, m_Unit, p.lim);
		}
		else if (checkNoUnit)
			CHECK(loom::CurrentLimiterUnit() == loom::kNoUnit, "%s: an unlimited task has a unit", what);
	}
	~TaskScope()
	{
		if (m_Lim)
		{
			const uint32_t now = loom::CurrentLimiterUnit();
			CHECK(now == m_Unit, "%s: unit index changed during the run (%u, then %u)", m_What, m_Unit, now);
			if (m_Unit < kMaxUnits)
				m_Lim->busy[m_Unit].store(0);
			m_Lim->running.fetch_sub(1, std::memory_order_seq_cst);
		}
		if (m_Pool)
			g_PoolRunning.fetch_sub(1, std::memory_order_seq_cst);
	}
	TaskScope(const TaskScope&) = delete;
	TaskScope& operator=(const TaskScope&) = delete;

private:
	const char* m_What;
	Lim* m_Lim = nullptr;
	uint32_t m_Unit = loom::kNoUnit;
	bool m_Pool = false;
};

// ---- Waiting -------------------------------------------------------------------------

enum class WaitKind : uint8_t { Help, Block, Timed, Poll, Isolated, Zero };

// Waits for `c` in one of several ways. Returns the error WaitFor rethrew, if
// any. `pure`: nothing behind c needs this thread (BlockOnly and an isolated
// wait are safe).
std::optional<FuzzError> WaitCounter(CounterHandle c, WaitKind kind, Rng& r, const char* what)
{
	std::optional<FuzzError> got;
	try
	{
		switch (kind)
		{
		case WaitKind::Help: loom::WaitFor(c, loom::Help::WhileWaiting); break;
		case WaitKind::Block: loom::WaitFor(c, loom::Help::BlockOnly); break;
		case WaitKind::Isolated: loom::Isolated([&] { loom::WaitFor(c); }); break;
		case WaitKind::Timed:
			for (;;)
			{
				const uint32_t ms = r.Below(4) == 0 ? 0 : r.Range(1, 15);
				const loom::Help help = r.Chance(0.3) ? loom::Help::BlockOnly : loom::Help::WhileWaiting;
				const auto t0 = Clock::now();
				// BlockOnly only with a limit: nothing behind c may need this thread otherwise.
				const bool done = loom::TryWaitFor(c, ms, ms == 0 ? loom::Help::WhileWaiting : help);
				const double took = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
				if (done)
					break;
				CHECK(took + 0.05 >= ms, "%s: TryWaitFor(%u ms) gave up after %.3f ms", what, ms, took);
				if (ms == 0 || help == loom::Help::BlockOnly)
					HelpUntil([&, n = 0]() mutable { return ++n > 3; });
			}
			break;
		case WaitKind::Poll:
		case WaitKind::Zero:
			HelpUntil([&] { return loom::Count(c) == 0; });
			CHECK(loom::TryWaitFor(c, 0), "%s: Count is 0 but TryWaitFor(0) is false", what);
			break;
		}
	}
	catch (const FuzzError& e)
	{
		got = e;
	}
	return got;
}

WaitKind RandomWait(Rng& r, bool pure)
{
	for (;;)
	{
		const auto k = (WaitKind)r.Below(6);
		if ((k == WaitKind::Block || k == WaitKind::Isolated) && !pure)
			continue;
		return k;
	}
}

// ---- Operations ------------------------------------------------------------------------

enum OpKind : int
{
	kOpDag,
	kOpWork,
	kOpFutures,
	kOpCoro,
	kOpGraphAsync,
	kOpRegister,
	kOpUnregister,
	kOpPark,
	kOpOutside,
	kOpLimits,
	kOpHelp,
	kOpDebug,
	kOpBigFan,
	kOpOnDedicated,
	kOpCount
};
const char* kOpNames[kOpCount] = { "dag", "work", "futures", "coro", "graph-async", "register", "unregister", "park",
	"outside", "limits", "help", "debug", "bigfan", "on-dedicated" };

struct OpSpec
{
	int kind = 0;
	uint64_t seed = 0;
	int size = 1;
	int join = -1; // async ones: joined after this many more ops; -1 at the epoch's end; -2 left to Shutdown's drain
};

// An operation still running.
struct Pending
{
	virtual ~Pending() = default;
	virtual void Join() = 0;
	// After Shutdown, for one left to the drain.
	virtual void AfterDrain() {}
	virtual bool CanDrain() const { return false; }
	int joinAt = -1;
	int op = 0;
	std::vector<int> users; // dedicated slots it may submit to
};

// What a piece of work may use. The restrictions keep the generated program
// within the rules (docs/rules.md), so that a hang is loom's:
//   - a BlockOnly wait (canBlock) only waits for AnyWorker work with no
//     limiter: a unit can be held by a task queued for the main thread or a
//     dedicated thread, which would then be needed;
//   - an isolated region's work needs nothing outside it (no main thread
//     unless the region is on it, no dedicated thread, no limiter);
//   - a task that waits inside holds an outer limiter's unit at most, and
//     what it waits for takes leaf limiters only.
struct WorkCtx
{
	int depth = 0;
	int ded = -1;        // a dedicated slot alive while this work runs
	bool nested = false; // inside a task: leaf limiters only
	bool canBlock = false;
	bool noMain = false; // nothing below may need the main thread
	bool noLim = false;  // nothing below may take a limiter
	bool noDed = false;  // nothing below may need a dedicated thread
	int op = 0;
};

unsigned PlaceFlags(const WorkCtx& c, bool leafTask)
{
	unsigned f = 0;
	if (!c.noMain)
		f |= kMain;
	if (!c.noLim && leafTask)
		f |= kLeafLim | (c.nested ? 0u : (unsigned)kOuterLim);
	return f;
}

int OwnDed(const WorkCtx& c) { return c.noDed ? -1 : c.ded; }

bool BlockSafe(const Place& p) { return p.aff == Affinity::AnyWorker && p.lim < 0; }

void RunWork(uint64_t seed, int size, const WorkCtx& ctx);

// ======================================================================================
// Work: synchronous, self-checking pieces that run on whatever thread calls them
// (the main thread, an outside thread, or a task's body).
// ======================================================================================

// ---- Fork-join with Submit and a counter ----

struct Leaf
{
	Place place;
	bool throws = false;
	int sub = 0; // > 0: runs nested work of this size
	uint64_t subSeed = 0;
	WorkCtx subCtx;
	uint64_t tick = 0;
	std::atomic<int> runs{ 0 };
	int id = 0;
	int op = 0;
	void Run()
	{
		CHECK(runs.fetch_add(1) == 0, "fork-join leaf %d ran twice", id);
		TaskScope scope(place, tick, "fork-join leaf");
		if (sub > 0)
			RunWork(subSeed, sub, subCtx);
		else
			SpinFor((int)(Hash(subSeed) % 2000));
		if (throws)
			throw FuzzError{ op, id };
	}
};

void WorkForkJoin(Rng& r, int size, const WorkCtx& ctx)
{
	const int k = r.Range(1, 2 + size);
	// 0 help, 1 timed, 2 isolated, 3 block, 4 poll
	int mode = (int)r.Below(5);
	if (mode == 3 && !ctx.canBlock)
		mode = 0;
	const bool onMain = std::this_thread::get_id() == g_MainTid;
	const bool leavesOnly = mode == 3 || ctx.depth >= 3;
	WorkCtx childCtx = ctx;
	childCtx.depth++;
	childCtx.nested = true;
	childCtx.canBlock = false;
	if (mode == 2 || mode == 3)
	{
		childCtx.noMain = childCtx.noMain || !onMain || mode == 3;
		childCtx.noLim = true;
		childCtx.noDed = true;
	}
	const bool cancel = r.Chance(0.1);
	const int extra = r.Chance(0.2) ? r.Range(1, 3) : 0;
	std::vector<std::unique_ptr<Leaf>> leaves;
	for (int i = 0; i < k; i++)
	{
		auto leaf = std::make_unique<Leaf>();
		leaf->id = i;
		leaf->op = ctx.op;
		leaf->throws = r.Chance(0.07);
		leaf->subSeed = r.Fork();
		const bool sub = !leavesOnly && r.Chance(0.25);
		leaf->place = RandomPlace(r, PlaceFlags(childCtx, !sub), OwnDed(childCtx));
		if (sub)
		{
			leaf->sub = r.Range(1, std::max(1, size / 2));
			leaf->subCtx = childCtx;
			leaf->subCtx.ded = leaf->place.aff == Affinity::Dedicated ? leaf->place.ded : childCtx.ded;
		}
		leaves.push_back(std::move(leaf));
	}
	loom::ScopedCounter c(k + extra);
	auto submitAll = [&] {
		for (auto& leaf : leaves)
		{
			loom::TaskDesc d = Desc(leaf->place);
			d.signals = c;
			leaf->tick = Tick();
			Leaf* p = leaf.get();
			loom::Submit([p] { p->Run(); }, d);
		}
		if (cancel)
			loom::Cancel(c);
		for (int i = 0; i < extra; i++)
			loom::Signal(c);
	};
	std::optional<FuzzError> got;
	if (mode == 2)
	{
		// The region's own tasks: submitted inside it.
		try
		{
			loom::Isolated([&] {
				submitAll();
				loom::WaitFor(c);
			});
		}
		catch (const FuzzError& e)
		{
			got = e;
		}
	}
	else
	{
		submitAll();
		const WaitKind wk = mode == 0 ? WaitKind::Help : mode == 1 ? WaitKind::Timed : mode == 3 ? WaitKind::Block : WaitKind::Poll;
		got = WaitCounter(c, wk, r, "fork-join");
	}
	std::optional<FuzzError> again = WaitCounter(c, WaitKind::Help, r, "fork-join again");
	bool anyThrew = false, anySkipped = false;
	for (auto& leaf : leaves)
	{
		const int runs = leaf->runs.load();
		CHECK(runs <= 1, "fork-join leaf ran %d times", runs);
		if (runs == 1 && leaf->throws)
			anyThrew = true;
		if (runs == 0)
			anySkipped = true;
	}
	CHECK(!anySkipped || cancel || anyThrew, "fork-join: a leaf was skipped with no cancel and no throw");
	CHECK(got.has_value() == anyThrew, "fork-join: WaitFor %s but %s leaf threw", got ? "threw" : "did not throw", anyThrew ? "a" : "no");
	if (got)
	{
		CHECK(got->op == ctx.op && got->id >= 0 && got->id < k && leaves[(size_t)got->id]->throws && leaves[(size_t)got->id]->runs == 1,
			"fork-join: rethrew an error (%d,%d) that is not one of its leaves'", got->op, got->id);
		CHECK(again && again->id == got->id, "fork-join: a second WaitFor rethrew another error");
	}
	CHECK(loom::IsCancelled(c) == (cancel || anyThrew), "fork-join: IsCancelled is %d", (int)loom::IsCancelled(c));
	CHECK(loom::Count(c) == 0, "fork-join: count %d after the wait", loom::Count(c));
	CHECK(!loom::AddToCounter(c, 1), "fork-join: AddToCounter succeeded on a zero counter");
}

// ---- ParallelFor ----

void WorkParallelFor(Rng& r, int size, const WorkCtx& ctx)
{
	size_t n = r.Below(4) == 0 ? r.Below(8) : r.Below((uint32_t)(64 * size + 1));
	if (!ctx.nested && r.Chance(0.05))
		n = 50000 + r.Below(200000);
	const size_t begin = r.Chance(0.5) ? 0 : r.Below(1000);
	const size_t chunk = r.Chance(0.4) ? 0 : r.Chance(0.2) ? n + r.Below(10) : 1 + r.Below((uint32_t)std::max<size_t>(1, n / 3 + 1));
	const bool throws = n > 0 && r.Chance(0.1);
	const size_t throwAt = throws ? begin + r.Below((uint32_t)n) : ~size_t(0);
	const bool inner = ctx.depth < 3 && r.Chance(0.25);
	const uint64_t innerSeed = r.Fork();
	WorkCtx innerCtx = ctx;
	innerCtx.depth++;
	innerCtx.nested = true;
	innerCtx.canBlock = false;
	std::unique_ptr<std::atomic<uint8_t>[]> hits(new std::atomic<uint8_t>[n + 1]);
	for (size_t i = 0; i < n; i++)
		hits[i].store(0, std::memory_order_relaxed);
	std::atomic<int> badRange{ 0 };
	std::optional<FuzzError> got;
	auto body = [&](size_t b, size_t e) {
		if (b < begin || e > begin + n || b >= e)
			badRange++;
		if (chunk != 0 && (e - b > chunk || (b - begin) % chunk != 0))
			badRange++;
		for (size_t i = b; i < e && i < begin + n; i++)
		{
			hits[i - begin].fetch_add(1, std::memory_order_relaxed);
			if (i == throwAt)
				throw FuzzError{ ctx.op, (int)(i & 0x7fffffff) };
		}
		if (inner && (Hash(innerSeed, b) % 4) == 0)
			RunWork(Hash(innerSeed, b), std::max(1, size / 3), innerCtx);
	};
	try
	{
		if (r.Chance(0.3))
		{
			struct U { decltype(body)* f; };
			U u{ &body };
			loom::ParallelFor(begin, begin + n, chunk, [](size_t b, size_t e, void* p) { (*static_cast<U*>(p)->f)(b, e); }, &u);
		}
		else
			loom::ParallelFor(begin, begin + n, chunk, body);
	}
	catch (const FuzzError& e)
	{
		got = e;
	}
	CHECK(badRange.load() == 0, "ParallelFor: %d pieces outside the range or the chunking (n %zu, chunk %zu)", badRange.load(), n, chunk);
	size_t missing = 0, twice = 0;
	for (size_t i = 0; i < n; i++)
	{
		const int h = hits[i].load();
		if (h == 0)
			missing++;
		if (h > 1)
			twice++;
	}
	CHECK(twice == 0, "ParallelFor: %zu indices ran twice (n %zu, chunk %zu)", twice, n, chunk);
	if (!throws)
	{
		CHECK(!got, "ParallelFor: threw with no throwing index");
		CHECK(missing == 0, "ParallelFor: %zu of %zu indices never ran (chunk %zu)", missing, n, chunk);
	}
	else
	{
		CHECK(got && got->id == (int)(throwAt & 0x7fffffff), "ParallelFor: the throw at %zu was not rethrown", throwAt);
	}
}

// ---- ParallelReduce ----

void WorkReduce(Rng& r, int size, const WorkCtx& ctx)
{
	const size_t n = r.Below(4) == 0 ? r.Below(5) : r.Below((uint32_t)(200 * size + 1));
	const size_t begin = r.Below(100);
	size_t chunk = r.Chance(0.4) ? 0 : 1 + r.Below((uint32_t)(n + 3));
	const bool throws = n > 0 && r.Chance(0.08);
	const size_t throwChunkAt = throws ? begin + r.Below((uint32_t)n) : ~size_t(0);
	auto map = [&](size_t b, size_t e) -> uint64_t {
		if (throwChunkAt >= b && throwChunkAt < e)
			throw FuzzError{ ctx.op, 77 };
		uint64_t h = 0;
		for (size_t i = b; i < e; i++)
			h = h * 31 + Hash(i);
		return h ^ (e - b);
	};
	auto combine = [](uint64_t a, uint64_t b) { return a * 1000003u + (b ^ (a >> 7)); }; // neither commutative nor associative
	uint64_t want = 0x1234;
	if (n > 0)
	{
		const size_t c = chunk == 0 ? (n - 1) / 256 + 1 : chunk;
		for (size_t b = begin; b < begin + n; b += c)
			want = combine(want, [&] {
				uint64_t h = 0;
				const size_t e = std::min(begin + n, b + c);
				for (size_t i = b; i < e; i++)
					h = h * 31 + Hash(i);
				return h ^ (e - b);
			}());
	}
	try
	{
		const uint64_t got = loom::ParallelReduce<uint64_t>(begin, begin + n, chunk, uint64_t(0x1234), map, combine);
		CHECK(!throws, "ParallelReduce: no exception from a throwing map");
		CHECK(got == want, "ParallelReduce: %llx, want %llx (n %zu, chunk %zu)", (unsigned long long)got, (unsigned long long)want, n, chunk);
	}
	catch (const FuzzError& e)
	{
		CHECK(throws && e.id == 77, "ParallelReduce: an unexpected exception");
	}
}

// ---- Scans ----

struct Mat2
{
	uint64_t a = 1, b = 0, c = 0, d = 1;
	bool operator==(const Mat2&) const = default;
};
Mat2 Mul(const Mat2& x, const Mat2& y) // associative, not commutative (mod 2^64)
{
	return { x.a * y.a + x.b * y.c, x.a * y.b + x.b * y.d, x.c * y.a + x.d * y.c, x.c * y.b + x.d * y.d };
}

void WorkScan(Rng& r, int size, const WorkCtx& ctx)
{
	const int type = (int)r.Below(3);
	const bool inclusive = r.Chance(0.5);
	const bool inPlace = r.Chance(0.3);
	const bool big = !ctx.nested ? r.Chance(0.3) : r.Chance(0.05);
	(void)size;
	if (type == 0) // Mat2: chunks of 8192, parallel from 16384
	{
		const size_t n = big ? 8192 * (size_t)r.Range(2, 6) + r.Below(9000) : r.Below(3000);
		std::vector<Mat2> in(n);
		for (size_t i = 0; i < n; i++)
		{
			const uint64_t h = Hash(i, 5);
			in[i] = { h | 1, h >> 3, h >> 7, (h >> 11) | 1 };
		}
		Mat2 init{ 3, 1, 4, 1 };
		std::vector<Mat2> want(n), got(n);
		if (inclusive)
			std::inclusive_scan(in.begin(), in.end(), want.begin(), Mul, init);
		else
			std::exclusive_scan(in.begin(), in.end(), want.begin(), init, Mul);
		std::vector<Mat2>& out = inPlace ? in : got;
		if (inclusive)
			loom::ParallelInclusiveScan(in.begin(), in.end(), out.begin(), init, Mul);
		else
			loom::ParallelExclusiveScan(in.begin(), in.end(), out.begin(), init, Mul);
		CHECK(out == want, "%s scan of %zu Mat2%s differs from std::", inclusive ? "inclusive" : "exclusive", n, inPlace ? " in place" : "");
	}
	else if (type == 1) // uint32 in, uint64 out: chunks of 65536
	{
		const size_t n = big ? 65536 * (size_t)r.Range(2, 4) + r.Below(70000) : r.Below(5000);
		std::vector<uint32_t> in(n);
		for (size_t i = 0; i < n; i++)
			in[i] = (uint32_t)Hash(i, 9);
		std::vector<uint64_t> want(n), got(n);
		if (inclusive)
			std::inclusive_scan(in.begin(), in.end(), want.begin(), std::plus<>{}, uint64_t(7));
		else
			std::exclusive_scan(in.begin(), in.end(), want.begin(), uint64_t(7), std::plus<>{});
		if (inclusive)
			loom::ParallelInclusiveScan(in.begin(), in.end(), got.begin(), uint64_t(7));
		else
			loom::ParallelExclusiveScan(in.begin(), in.end(), got.begin(), uint64_t(7));
		CHECK(got == want, "%s scan of %zu uint32 into uint64 differs from std::", inclusive ? "inclusive" : "exclusive", n);
	}
	else // a throwing op
	{
		const size_t n = big ? 8192 * (size_t)r.Range(2, 5) : r.Below(3000) + 1;
		std::vector<Mat2> in(n, Mat2{ 1, 1, 0, 1 }), out(n);
		const size_t at = r.Below((uint32_t)n);
		std::atomic<size_t> calls{ 0 };
		auto op = [&](const Mat2& x, const Mat2& y) {
			if (calls.fetch_add(1) == at)
				throw FuzzError{ ctx.op, 88 };
			return Mul(x, y);
		};
		bool threw = false;
		try
		{
			if (inclusive)
				loom::ParallelInclusiveScan(in.begin(), in.end(), out.begin(), Mat2{}, op);
			else
				loom::ParallelExclusiveScan(in.begin(), in.end(), out.begin(), Mat2{}, op);
		}
		catch (const FuzzError& e)
		{
			threw = e.id == 88;
		}
		const bool shouldThrow = calls.load() > at;
		CHECK(threw == shouldThrow, "scan with a throwing op: threw %d, op called %zu times, throw at call %zu", (int)threw,
			calls.load(), at);
	}
}

// ---- ParallelSort ----

struct KeyVal
{
	int64_t key;
	uint32_t val;
};

struct NoThrowMove // moves that may throw: the sort never sample-sorts these
{
	int64_t key = 0;
	NoThrowMove() = default;
	explicit NoThrowMove(int64_t k) : key(k) {}
	NoThrowMove(const NoThrowMove&) = default;
	NoThrowMove(NoThrowMove&& o) noexcept(false) : key(o.key) {}
	NoThrowMove& operator=(const NoThrowMove&) = default;
	NoThrowMove& operator=(NoThrowMove&& o) noexcept(false) { key = o.key; return *this; }
};

void WorkSort(Rng& r, int size, const WorkCtx& ctx)
{
	size_t n;
	const int band = (int)r.Below(10);
	if (band < 4)
		n = r.Below(3000);
	else if (band < 9 || ctx.nested)
		n = 2049 + r.Below(ctx.nested ? 20000 : 60000);
	else
		n = 131072 + r.Below(150000);
	(void)size;
	const int dist = (int)r.Below(5); // random, few keys, sorted, reversed, organ pipe
	auto key = [&](size_t i) -> int64_t {
		switch (dist)
		{
		case 0: return (int64_t)Hash(i, 1);
		case 1: return (int64_t)(Hash(i, 2) % 7);
		case 2: return (int64_t)i;
		case 3: return -(int64_t)i;
		default: return (int64_t)std::min(i, n - i);
		}
	};
	const int type = (int)r.Below(4);
	if (type == 0)
	{
		std::vector<int64_t> v(n);
		for (size_t i = 0; i < n; i++)
			v[i] = key(i);
		std::vector<int64_t> want = v;
		std::sort(want.begin(), want.end());
		loom::ParallelSort(v.begin(), v.end());
		CHECK(v == want, "ParallelSort of %zu int64 (dist %d) differs from std::sort", n, dist);
	}
	else if (type == 1)
	{
		std::vector<KeyVal> v(n);
		for (size_t i = 0; i < n; i++)
			v[i] = { key(i), (uint32_t)i };
		auto greater = [](const KeyVal& a, const KeyVal& b) { return a.key > b.key; };
		loom::ParallelSort(v.begin(), v.end(), greater);
		bool sorted = std::is_sorted(v.begin(), v.end(), greater);
		std::vector<uint8_t> seen(n, 0);
		bool perm = true;
		for (const KeyVal& kv : v)
		{
			if (kv.val >= n || seen[kv.val]++ || kv.key != key(kv.val))
				perm = false;
		}
		CHECK(sorted && perm, "ParallelSort of %zu KeyVal descending: sorted %d, permutation %d", n, (int)sorted, (int)perm);
	}
	else if (type == 2)
	{
		std::vector<NoThrowMove> v;
		v.reserve(n);
		for (size_t i = 0; i < n; i++)
			v.emplace_back(key(i));
		loom::ParallelSort(v.begin(), v.end(), [](const NoThrowMove& a, const NoThrowMove& b) { return a.key < b.key; });
		bool sorted = std::is_sorted(v.begin(), v.end(), [](const NoThrowMove& a, const NoThrowMove& b) { return a.key < b.key; });
		CHECK(sorted, "ParallelSort of %zu throwing-move elements is not sorted", n);
	}
	else // a throwing comparator: the range keeps its elements
	{
		std::vector<int64_t> v(n);
		for (size_t i = 0; i < n; i++)
			v[i] = key(i);
		std::vector<int64_t> before = v;
		std::sort(before.begin(), before.end());
		std::atomic<uint64_t> calls{ 0 };
		const uint64_t at = r.Below((uint32_t)(n * 4 + 1));
		bool threw = false;
		try
		{
			loom::ParallelSort(v.begin(), v.end(), [&](int64_t a, int64_t b) {
				if (calls.fetch_add(1, std::memory_order_relaxed) == at)
					throw FuzzError{ ctx.op, 99 };
				return a < b;
			});
		}
		catch (const FuzzError& e)
		{
			threw = e.id == 99;
		}
		CHECK(threw == (calls.load() > at), "ParallelSort with a throwing comparator: threw %d after %llu calls (throw at %llu)",
			(int)threw, (unsigned long long)calls.load(), (unsigned long long)at);
		std::vector<int64_t> after = v;
		std::sort(after.begin(), after.end());
		// "a throw inside std::sort can leave one value copied over another": only the size is promised then.
		if (!threw)
			CHECK(std::is_sorted(v.begin(), v.end()) && after == before, "ParallelSort (no throw in the end) lost elements or is unsorted");
		CHECK(v.size() == n, "ParallelSort changed the size");
	}
}

// ---- Futures in one piece (Async, Then, Get) ----

void WorkAsync(Rng& r, int size, const WorkCtx& ctx)
{
	const int steps = r.Range(0, 3 + size / 4);
	const bool block = ctx.canBlock && r.Chance(0.3);
	const unsigned flags = block ? 0u : PlaceFlags(ctx, true);
	struct Step
	{
		Place place;
		bool throws;
		uint64_t tick;
		std::atomic<int> runs{ 0 };
	};
	std::vector<std::unique_ptr<Step>> s;
	for (int i = 0; i <= steps; i++)
	{
		auto st = std::make_unique<Step>();
		st->place = RandomPlace(r, flags, block ? -1 : OwnDed(ctx));
		st->throws = r.Chance(0.06);
		s.push_back(std::move(st));
	}
	int64_t want = 0;
	int firstThrow = -1;
	for (int i = 0; i <= steps; i++)
	{
		if (firstThrow < 0 && s[(size_t)i]->throws)
			firstThrow = i;
		want = (int64_t)((uint64_t)want * 3 + (uint64_t)i);
	}
	s[0]->tick = Tick();
	Step* s0 = s[0].get();
	const int op = ctx.op;
	loom::Future<int64_t> f = loom::Async([s0, op]() -> int64_t {
		s0->runs++;
		TaskScope scope(s0->place, s0->tick, "Async");
		if (s0->throws)
			throw FuzzError{ op, 0 };
		return 0;
	}, Desc(s0->place));
	for (int i = 1; i <= steps; i++)
	{
		Step* si = s[(size_t)i].get();
		si->tick = Tick();
		f = f.Then([si, i, op](int64_t& v) -> int64_t {
			si->runs++;
			TaskScope scope(si->place, si->tick, "Then");
			if (si->throws)
				throw FuzzError{ op, i };
			return (int64_t)((uint64_t)v * 3 + (uint64_t)i);
		}, Desc(si->place));
	}
	try
	{
		const loom::Help help = block ? loom::Help::BlockOnly : loom::Help::WhileWaiting;
		int64_t got = f.Get(help);
		CHECK(firstThrow < 0 && got == want, "Async chain: %lld, want %lld (first throw %d)", (long long)got, (long long)want, firstThrow);
	}
	catch (const FuzzError& e)
	{
		CHECK(firstThrow >= 0 && e.id == firstThrow, "Async chain: error from step %d, want %d", e.id, firstThrow);
	}
	for (int i = 0; i <= steps; i++)
	{
		const int runs = s[(size_t)i]->runs.load();
		const int expect = firstThrow < 0 || i <= firstThrow ? 1 : 0;
		CHECK(runs == expect, "Async chain step %d ran %d times, want %d (first throw %d)", i, runs, expect, firstThrow);
	}
}

// ---- TaskGraph ----

struct GraphModel
{
	struct NodeRec
	{
		Place place;
		std::vector<int> preds;
		std::atomic<int> lastRun{ -1 };
		std::atomic<int> runsThisRun{ 0 };
		uint64_t seed = 0;
		bool raw = false;
	};
	std::vector<std::unique_ptr<NodeRec>> nodes;
	std::atomic<int> run{ -1 };
	double throwP = 0;
	uint64_t tick = 0;
	int op = 0;
	std::atomic<int> badOrder{ 0 };

	bool Throws(int node, int runIndex) const
	{
		return throwP > 0 && (double)(Hash(nodes[(size_t)node]->seed, (uint64_t)runIndex) % 1000) < throwP * 1000;
	}
	void RunNode(int i)
	{
		NodeRec& n = *nodes[(size_t)i];
		const int cur = run.load();
		TaskScope scope(n.place, tick, "graph node");
		for (int p : n.preds)
			if (nodes[(size_t)p]->lastRun.load() != cur)
				badOrder++;
		n.runsThisRun++;
		n.lastRun.store(cur);
		if (Throws(i, cur))
			throw FuzzError{ op, i };
	}
};

struct GraphNodeArg
{
	GraphModel* m;
	int i;
};

void BuildGraph(loom::TaskGraph& g, GraphModel& m, Rng& r, int n, unsigned flags, int ded, std::vector<std::unique_ptr<GraphNodeArg>>& args)
{
	m.nodes.clear();
	g.Clear();
	CHECK(g.NodeCount() == 0, "TaskGraph::Clear left %zu nodes", g.NodeCount());
	for (int i = 0; i < n; i++)
	{
		auto rec = std::make_unique<GraphModel::NodeRec>();
		rec->place = RandomPlace(r, flags, ded);
		rec->seed = r.Fork();
		rec->raw = r.Chance(0.3);
		m.nodes.push_back(std::move(rec));
	}
	for (int i = 0; i < n; i++)
	{
		loom::TaskDesc d = Desc(m.nodes[(size_t)i]->place);
		loom::TaskGraph::Node node;
		if (m.nodes[(size_t)i]->raw)
		{
			args.push_back(std::make_unique<GraphNodeArg>(GraphNodeArg{ &m, i }));
			d.fn = [](void* p) { auto* a = static_cast<GraphNodeArg*>(p); a->m->RunNode(a->i); };
			d.userData = args.back().get();
			node = g.Add(d);
		}
		else
			node = g.Add([&m, i] { m.RunNode(i); }, d);
		CHECK(node == (loom::TaskGraph::Node)i, "TaskGraph::Add returned node %u for the %d-th", node, i);
	}
	const double density = r.Chance(0.3) ? 0.5 : 2.0 / std::max(1, n);
	for (int j = 1; j < n; j++)
		for (int i = 0; i < j; i++)
			if (r.Chance(density))
			{
				g.Precede((loom::TaskGraph::Node)i, (loom::TaskGraph::Node)j);
				m.nodes[(size_t)j]->preds.push_back(i);
				if (r.Chance(0.1))
					g.Precede((loom::TaskGraph::Node)i, (loom::TaskGraph::Node)j); // twice is harmless
			}
	CHECK(g.NodeCount() == (size_t)n, "TaskGraph::NodeCount %zu, want %d", g.NodeCount(), n);
}

// After a run that ended: every node once unless skipped by a throw; a node after a thrower is skipped.
void CheckGraphRun(GraphModel& m, int runIndex, std::optional<FuzzError> got, const char* what)
{
	const int n = (int)m.nodes.size();
	std::vector<char> ran(n), after(n, 0);
	bool anyThrew = false;
	for (int i = 0; i < n; i++)
	{
		const int runs = m.nodes[(size_t)i]->runsThisRun.exchange(0);
		CHECK(runs <= 1, "%s: node %d ran %d times in run %d", what, i, runs, runIndex);
		ran[(size_t)i] = runs > 0;
		if (runs > 0 && m.Throws(i, runIndex))
			anyThrew = true;
	}
	for (int i = 0; i < n; i++)
		for (int p : m.nodes[(size_t)i]->preds)
			if ((ran[(size_t)p] && m.Throws(p, runIndex)) || after[(size_t)p])
				after[(size_t)i] = 1;
	for (int i = 0; i < n; i++)
	{
		if (after[(size_t)i])
			CHECK(!ran[(size_t)i], "%s: node %d ran after its predecessor threw (run %d)", what, i, runIndex);
		else if (!ran[(size_t)i])
			CHECK(anyThrew, "%s: node %d was skipped with no throw (run %d)", what, i, runIndex);
	}
	CHECK(got.has_value() == anyThrew, "%s: run %d %s but %s node threw", what, runIndex, got ? "threw" : "did not throw", anyThrew ? "a" : "no");
	if (got)
		CHECK(got->op == m.op && got->id >= 0 && got->id < n && ran[(size_t)got->id] && m.Throws(got->id, runIndex),
			"%s: rethrew (%d,%d), not a node that threw", what, got->op, got->id);
	CHECK(m.badOrder.exchange(0) == 0, "%s: a node started before its predecessors finished (run %d)", what, runIndex);
}

void WorkGraph(Rng& r, int size, const WorkCtx& ctx)
{
	loom::TaskGraph g;
	GraphModel m;
	m.op = ctx.op;
	m.throwP = r.Chance(0.3) ? 0.05 : 0;
	std::vector<std::unique_ptr<GraphNodeArg>> args;
	const bool block = ctx.canBlock && r.Chance(0.3);
	const unsigned flags = block ? 0u : PlaceFlags(ctx, true);
	const int ded = block ? -1 : OwnDed(ctx);
	int n = r.Range(1, 3 + size);
	BuildGraph(g, m, r, n, flags, ded, args);
	if (r.Chance(0.1))
	{
		// A cycle: Validate says so (nothing is submitted).
		loom::TaskGraph cyc;
		const auto a = cyc.Add([] {}), b = cyc.Add([] {});
		cyc.Precede(a, b);
		cyc.Precede(b, a);
		CHECK(!cyc.Validate(), "TaskGraph::Validate accepted a cycle");
	}
	CHECK(g.Validate(), "TaskGraph::Validate rejected a DAG");
	const int runs = r.Range(1, 4);
	for (int k = 0; k < runs; k++)
	{
		if (k > 0 && r.Chance(0.2))
		{
			n = r.Range(1, 3 + size);
			BuildGraph(g, m, r, n, flags, ded, args);
		}
		if (r.Chance(0.15))
		{
			loom::TaskGraph moved(std::move(g)); // a move keeps the nodes
			g = std::move(moved);
		}
		const int runIndex = m.run.load() + 1;
		m.run.store(runIndex);
		m.tick = Tick();
		std::optional<FuzzError> got;
		const int how = (int)r.Below(3);
		if (how == 0)
		{
			try
			{
				loom::Run(g, block ? loom::Help::BlockOnly : loom::Help::WhileWaiting);
			}
			catch (const FuzzError& e)
			{
				got = e;
			}
		}
		else
		{
			// Submit with a gate: nothing may start before it opens.
			loom::ScopedCounter done(1);
			loom::ScopedCounter gate(how == 2 ? 1 : 0);
			loom::Submit(g, done, gate);
			if (how == 2)
			{
				CHECK(g.IsRunning(), "TaskGraph::IsRunning false after Submit, with its waitsFor not zero");
				SpinFor(2000);
				for (auto& node : m.nodes)
					CHECK(node->lastRun.load() != runIndex, "a graph node ran before the run's waitsFor was zero");
				loom::Signal(gate);
			}
			got = WaitCounter(done, r.Chance(0.5) ? WaitKind::Help : WaitKind::Timed, r, "graph run");
		}
		CHECK(!g.IsRunning(), "TaskGraph::IsRunning true after the run's wait returned");
		CheckGraphRun(m, runIndex, got, "graph");
	}
}

// ---- Pipeline ----

struct PipeItem
{
	int64_t seq = -1;
	uint32_t trail = 0;
	int run = -1;
};

void WorkPipeline(Rng& r, int size, const WorkCtx& ctx)
{
	const uint32_t inFlight = (uint32_t)r.Range(0, 8);
	const uint32_t effective = std::max<uint32_t>(1, inFlight);
	const int stages = r.Range(0, 4);
	const unsigned flags = PlaceFlags(ctx, true);
	struct StageRec
	{
		bool serial;
		Place place;
		std::vector<std::atomic<uint8_t>>* seen = nullptr;
		std::atomic<int64_t> next{ 0 };
		std::atomic<int> outOfOrder{ 0 };
		int throwAt = -1;
	};
	std::vector<std::unique_ptr<StageRec>> st;
	for (int s = 0; s < stages; s++)
	{
		auto rec = std::make_unique<StageRec>();
		rec->serial = r.Chance(0.5);
		rec->place = RandomPlace(r, flags, OwnDed(ctx));
		st.push_back(std::move(rec));
	}
	Place srcPlace = RandomPlace(r, flags, OwnDed(ctx));
	loom::Pipeline<PipeItem> p(inFlight);
	std::atomic<int64_t> produced{ 0 };
	int64_t items = 0;
	int srcThrowAt = -1;
	std::atomic<int> inside{ 0 }, maxInside{ 0 }, badItem{ 0 };
	int runIndex = 0;
	bool runThrows = false;
	bool lastRunClean = true;
	uint64_t tick = 0;
	const uint32_t full = stages == 0 ? 0 : (1u << stages) - 1;
	std::vector<std::unique_ptr<std::vector<std::atomic<uint8_t>>>> seen;
	p.Source([&](PipeItem& item) {
		TaskScope scope(srcPlace, tick, "pipeline source");
		if (!runThrows && item.run >= 0 && item.run == runIndex && item.trail != full)
			badItem++;
		const int64_t s = produced.load();
		if (s >= items)
			return false;
		if (s == srcThrowAt)
			throw FuzzError{ ctx.op, 1000 };
		produced.store(s + 1);
		item.seq = s;
		item.trail = 0;
		item.run = runIndex;
		const int now = inside.fetch_add(1) + 1;
		int m = maxInside.load();
		while (now > m && !maxInside.compare_exchange_weak(m, now)) {}
		if (stages == 0)
			inside.fetch_sub(1);
		return true;
	}, Desc(srcPlace));
	for (int s = 0; s < stages; s++)
	{
		StageRec* rec = st[(size_t)s].get();
		auto fn = [&, rec, s](PipeItem& item) {
			TaskScope scope(rec->place, tick, "pipeline stage");
			if (item.trail != (1u << s) - 1 || item.seq < 0 || item.seq >= items)
				badItem++;
			item.trail |= 1u << s;
			if (item.seq >= 0 && item.seq < items)
				(*rec->seen)[(size_t)item.seq]++;
			if (rec->serial)
			{
				const int64_t want = rec->next.load();
				if (item.seq < want)
					rec->outOfOrder++;
				rec->next.store(item.seq + 1);
			}
			if (item.seq == rec->throwAt)
				throw FuzzError{ ctx.op, 2000 + s };
			if (s == stages - 1)
				inside.fetch_sub(1);
		};
		if (rec->serial)
			p.Serial(fn, Desc(rec->place));
		else
			p.Parallel(fn, Desc(rec->place));
	}
	const int runs = r.Range(1, 3);
	for (int k = 0; k < runs; k++)
	{
		runIndex = k;
		items = r.Below((uint32_t)(8 * size + 2));
		produced.store(0);
		srcThrowAt = items > 0 && r.Chance(0.05) ? (int)r.Below((uint32_t)items) : -1;
		int stageThrow = -1;
		for (int s = 0; s < stages; s++)
		{
			StageRec* rec = st[(size_t)s].get();
			rec->next.store(0);
			rec->throwAt = items > 0 && r.Chance(0.05) ? (int)r.Below((uint32_t)items) : -1;
			if (rec->throwAt >= 0)
				stageThrow = s;
			seen.push_back(std::make_unique<std::vector<std::atomic<uint8_t>>>((size_t)items));
			rec->seen = seen.back().get();
		}
		inside.store(0);
		maxInside.store(0);
		runThrows = srcThrowAt >= 0 || stageThrow >= 0;
		tick = Tick();
		std::optional<FuzzError> got;
		try
		{
			p.Run();
		}
		catch (const FuzzError& e)
		{
			got = e;
		}
		const bool throwing = srcThrowAt >= 0 || stageThrow >= 0;
		CHECK(maxInside.load() <= (int)effective, "pipeline: %d items inside at once, inFlight %u", maxInside.load(), effective);
		CHECK(badItem.exchange(0) == 0, "pipeline: an item reached a stage (or came home) in the wrong state");
		if (!throwing)
		{
			CHECK(!got, "pipeline: Run threw with no throwing stage");
			CHECK(produced.load() == items, "pipeline: the source filled %lld of %lld items", (long long)produced.load(), (long long)items);
			for (int s = 0; s < stages; s++)
			{
				StageRec* rec = st[(size_t)s].get();
				size_t bad = 0;
				for (auto& x : *rec->seen)
					bad += x.load() != 1;
				CHECK(bad == 0, "pipeline: stage %d saw %zu items other than once", s, bad);
				CHECK(rec->outOfOrder.exchange(0) == 0, "pipeline: serial stage %d took items out of order", s);
			}
		}
		else
		{
			if (got)
				CHECK(got->op == ctx.op && (got->id == 1000 || (got->id >= 2000 && got->id < 2000 + stages)), "pipeline: an unexpected error");
			// A thrower may not get to run (an earlier one stopped the stream), but a throw always reaches Run.
			for (int s = 0; s < stages; s++)
			{
				StageRec* rec = st[(size_t)s].get();
				bool dup = false;
				for (auto& x : *rec->seen)
					dup |= x.load() > 1;
				CHECK(!dup, "pipeline: stage %d saw an item twice in a failed run", s);
				if (rec->throwAt >= 0 && (*rec->seen)[(size_t)rec->throwAt].load() == 1)
					CHECK(got.has_value(), "pipeline: stage %d threw but Run did not rethrow", s);
				rec->outOfOrder.store(0);
			}
			if (srcThrowAt >= 0 && produced.load() == srcThrowAt)
				CHECK(got.has_value(), "pipeline: the source threw but Run did not rethrow");
		}
		lastRunClean = !got;
	}
	(void)lastRunClean;
}

// ---- Coroutines ----

#if defined(LOOM_COROUTINES)
struct CoroStep
{
	int kind = 0;
	Place place;
	bool throws = false;
	int k = 0;          // tasks behind a counter
	int child = -1;     // a nested coroutine plan
	uint64_t seed = 0;
	std::atomic<int> runs{ 0 };
};

struct CoroPlan
{
	std::vector<std::vector<std::unique_ptr<CoroStep>>> bodies; // body 0 is the top
	int op = 0;
	int ded = -1;
	unsigned leafFlags = kMain | kLeafLim;
	std::vector<std::unique_ptr<Leaf>> leaves;
	std::mutex mu;
};

enum CoroKind
{
	kAwaitAsync,      // co_await Async(...) (a temporary: moved out)
	kAwaitLvalue,     // auto f = Async; co_await f, and a Then of it
	kAwaitMoveReaders,// co_await std::move(f) while a Then reads it (a move-only value)
	kAfterCounter,    // co_await After(counter[, opts]) with tasks behind it
	kAfterList,       // co_await After({a, b})
	kAfterFuture,     // co_await After(Async, opts)
	kResumeOn,        // co_await ResumeOn(opts)
	kAwaitCoro,       // co_await a nested coroutine
	kThrow,           // throws out of the coroutine
	kPromise,         // co_await a promise's future set from another thread
	kCoroKinds
};

// The value a body returns if no step throws, and whether one does (which, first).
int64_t CoroExpect(CoroPlan& plan, int body, int* firstThrow)
{
	int64_t v = body;
	*firstThrow = -1;
	auto& steps = plan.bodies[(size_t)body];
	for (size_t i = 0; i < steps.size(); i++)
	{
		CoroStep& s = *steps[i];
		int64_t add = 0;
		bool fails = false;
		switch (s.kind)
		{
		case kAwaitAsync: case kAfterFuture: add = (int64_t)(s.seed % 1000); fails = s.throws; break;
		case kAwaitLvalue: add = 2 * (int64_t)(s.seed % 1000) + 1; fails = s.throws; break;
		case kAwaitMoveReaders: add = 2 * (int64_t)(s.seed % 1000); break;
		case kAfterCounter: case kAfterList: add = s.k; break;
		case kResumeOn: add = 5; break;
		case kAwaitCoro:
		{
			int ft;
			add = CoroExpect(plan, s.child, &ft);
			fails = ft >= 0;
			break;
		}
		case kThrow: fails = true; break;
		case kPromise: add = (int64_t)(s.seed % 1000); fails = s.throws; break;
		}
		if (fails)
		{
			int ft = (int)(body * 100 + (int)i);
			if (s.kind == kAwaitCoro)
				CoroExpect(plan, s.child, &ft); // the nested coroutine's own error comes through
			*firstThrow = ft;
			return 0;
		}
		v = (int64_t)((uint64_t)v * 7 + (uint64_t)add); // wraps
	}
	return v;
}

void CheckResumedOn(const Place& p, const char* what)
{
	CheckPlace(p, what);
	if (p.lim >= 0)
	{
		// The rest holds the unit until its next suspension.
		const uint32_t u = loom::CurrentLimiterUnit();
		CHECK(u != loom::kNoUnit, "%s: a limited resume holds no unit", what);
	}
}

loom::Future<int64_t> CoroBody(CoroPlan* plan, int body)
{
	int64_t v = body;
	auto& steps = plan->bodies[(size_t)body];
	for (size_t i = 0; i < steps.size(); i++)
	{
		CoroStep& s = *steps[i];
		s.runs++;
		const int errId = body * 100 + (int)i;
		const int op = plan->op;
		int64_t add = 0;
		switch (s.kind)
		{
		case kAwaitAsync:
		{
			CoroStep* sp = &s;
			const uint64_t tick = Tick();
			try
			{
				add = co_await loom::Async([sp, tick, op, errId]() -> int64_t {
					TaskScope scope(sp->place, tick, "coroutine's Async");
					if (sp->throws)
						throw FuzzError{ op, errId };
					return (int64_t)(sp->seed % 1000);
				}, Desc(s.place));
			}
			catch (const FuzzError& e)
			{
				CHECK(s.throws && e.id == errId, "coroutine: co_await Async rethrew (%d,%d)", e.op, e.id);
				throw;
			}
			break;
		}
		case kAwaitLvalue:
		{
			CoroStep* sp = &s;
			const uint64_t tick = Tick();
			loom::Future<int64_t> f = loom::Async([sp, tick, op, errId]() -> int64_t {
				TaskScope scope(sp->place, tick, "coroutine's Async (lvalue)");
				if (sp->throws)
					throw FuzzError{ op, errId };
				return (int64_t)(sp->seed % 1000);
			}, Desc(s.place));
			loom::Future<int64_t> g = f.Then([](int64_t& x) { return x + 1; });
			add = co_await f;
			add += co_await g;
			break;
		}
		case kAwaitMoveReaders:
		{
			const int64_t x = (int64_t)(s.seed % 1000);
			loom::Future<std::unique_ptr<int64_t>> f = loom::Async([x] { return std::make_unique<int64_t>(x); });
			loom::Future<int64_t> g = f.Then([](std::unique_ptr<int64_t>& p) { SpinFor(500); return *p; });
			std::unique_ptr<int64_t> mine = co_await std::move(f);
			CHECK(mine && *mine == x, "coroutine: a moved-out value is wrong");
			add = (mine ? *mine : 0) + co_await g;
			break;
		}
		case kAfterCounter:
		case kAfterList:
		{
			loom::ScopedCounter a(s.k), b(s.kind == kAfterList ? 1 : 0);
			bool anyThrows = false;
			std::vector<Leaf*> mine;
			{
				std::lock_guard<std::mutex> lock(plan->mu);
				for (int t = 0; t < s.k; t++)
				{
					auto leaf = std::make_unique<Leaf>();
					leaf->op = op;
					leaf->id = errId;
					leaf->throws = s.throws && t == 0;
					anyThrows |= leaf->throws;
					Rng lr(Hash(s.seed, (uint64_t)t));
					leaf->place = RandomPlace(lr, plan->leafFlags, plan->ded);
					mine.push_back(leaf.get());
					plan->leaves.push_back(std::move(leaf));
				}
			}
			for (Leaf* leaf : mine)
			{
				loom::TaskDesc d = Desc(leaf->place);
				d.signals = a;
				leaf->tick = Tick();
				loom::Submit([leaf] { leaf->Run(); }, d);
			}
			if (s.kind == kAfterList)
				loom::Signal(b);
			bool threw = false;
			try
			{
				if (s.kind == kAfterList)
					co_await loom::After({ a.Get(), b.Get() });
				else if (s.place.aff != Affinity::AnyWorker || s.place.lim >= 0)
				{
					co_await loom::After(a.Get(), Desc(s.place));
					CheckResumedOn(s.place, "coroutine after After(counter, opts)");
				}
				else
					co_await loom::After(a.Get());
			}
			catch (const FuzzError& e)
			{
				threw = true;
				CHECK(e.id == errId, "coroutine: After(counter) rethrew another error");
			}
			CHECK(threw == anyThrows, "coroutine: After(counter) threw %d, a task threw %d", (int)threw, (int)anyThrows);
			CHECK(loom::Count(a) == 0, "coroutine resumed before its counter reached zero (count %d)", loom::Count(a));
			for (Leaf* leaf : mine)
				CHECK(leaf->runs.load() == 1 || anyThrows, "coroutine: a task behind After(counter) did not run");
			add = s.k;
			break;
		}
		case kAfterFuture:
		{
			CoroStep* sp = &s;
			const uint64_t tick = Tick();
			add = co_await loom::After(loom::Async([sp, tick, op, errId]() -> int64_t {
				(void)tick;
				if (sp->throws)
					throw FuzzError{ op, errId };
				return (int64_t)(sp->seed % 1000);
			}), Desc(s.place));
			CheckResumedOn(s.place, "coroutine after After(future, opts)");
			break;
		}
		case kResumeOn:
		{
			const uint64_t tick = Tick();
			co_await loom::ResumeOn(Desc(s.place));
			CheckResumedOn(s.place, "coroutine after ResumeOn");
			if (s.place.lim >= 0)
			{
				TaskScope scope(s.place, tick, "coroutine stretch after ResumeOn");
				SpinFor(1000);
			}
			add = 5;
			break;
		}
		case kAwaitCoro:
			add = co_await CoroBody(plan, s.child);
			break;
		case kThrow:
			throw FuzzError{ op, errId };
		case kPromise:
		{
			loom::Promise<int64_t> pr;
			loom::Future<int64_t> f = pr.GetFuture();
			const int64_t x = (int64_t)(s.seed % 1000);
			const bool throws = s.throws;
			// Set by a task (any thread) a little later.
			loom::Submit([pr = std::move(pr), x, throws, op, errId]() mutable {
				SpinFor(300);
				if (throws)
					pr.SetException(std::make_exception_ptr(FuzzError{ op, errId }));
				else
					pr.SetValue(x);
			});
			add = co_await f;
			break;
		}
		}
		v = (int64_t)((uint64_t)v * 7 + (uint64_t)add);
	}
	co_return v;
}

void MakeCoroPlan(CoroPlan& plan, Rng& r, int size, unsigned flags, int ded, int depth, std::vector<int>* users)
{
	const int body = (int)plan.bodies.size();
	plan.bodies.emplace_back();
	const int n = r.Range(0, 2 + size / 3);
	for (int i = 0; i < n; i++)
	{
		auto s = std::make_unique<CoroStep>();
		s->kind = (int)r.Below(kCoroKinds);
		if (s->kind == kAwaitCoro && depth >= 3)
			s->kind = kResumeOn;
		if (s->kind == kThrow && !r.Chance(0.3))
			s->kind = kAwaitAsync;
		s->place = RandomPlace(r, flags, ded, users);
		s->throws = r.Chance(0.06);
		s->k = r.Range(1, 4);
		s->seed = r.Fork();
		if (s->kind == kAwaitCoro)
		{
			plan.bodies[(size_t)body].push_back(std::move(s));
			const int child = (int)plan.bodies.size();
			plan.bodies[(size_t)body].back()->child = child;
			MakeCoroPlan(plan, r, size / 2, flags, ded, depth + 1, users);
			continue;
		}
		plan.bodies[(size_t)body].push_back(std::move(s));
	}
}

void CheckCoroResult(CoroPlan& plan, loom::Future<int64_t>& f, const char* what)
{
	int firstThrow = -1;
	const int64_t want = CoroExpect(plan, 0, &firstThrow);
	try
	{
		const int64_t got = f.Get();
		CHECK(firstThrow < 0 && got == want, "%s: returned %lld, want %lld (first throw %d)", what, (long long)got, (long long)want, firstThrow);
	}
	catch (const FuzzError& e)
	{
		CHECK(firstThrow >= 0 && e.id == firstThrow, "%s: error %d, want %d", what, e.id, firstThrow);
	}
}

void WorkCoro(Rng& r, int size, const WorkCtx& ctx)
{
	CoroPlan plan;
	plan.op = ctx.op;
	plan.ded = OwnDed(ctx);
	plan.leafFlags = PlaceFlags(ctx, true) & ~(unsigned)kOuterLim;
	MakeCoroPlan(plan, r, size, PlaceFlags(ctx, true), OwnDed(ctx), ctx.depth, nullptr);
	loom::Future<int64_t> f = CoroBody(&plan, 0);
	CheckCoroResult(plan, f, "coroutine");
	f.Reset();
}
#endif

// ---- PerThread ----

void WorkPerThread(Rng& r, int size, const WorkCtx& ctx)
{
	(void)ctx;
	const int64_t exemplar = r.Below(5);
	loom::PerThread<int64_t> pt(exemplar);
	const size_t n = r.Below((uint32_t)(500 * size + 1));
	std::mutex mu;
	std::map<int, std::thread::id> owners; // index -> thread, while alive in this work
	std::atomic<int> clash{ 0 };
	const bool onMain = std::this_thread::get_id() == g_MainTid;
	loom::ParallelFor(0, n, r.Chance(0.5) ? 0 : 1 + r.Below(64), [&](size_t b, size_t e) {
		int64_t& mine = pt.Local();
		for (size_t i = b; i < e; i++)
			mine += (int64_t)i;
		CHECK(&pt.Local() == &mine, "PerThread::Local moved within a task");
		if (onMain && t_Role != Role::Outside)
		{
			std::lock_guard<std::mutex> lock(mu);
			auto [it, fresh] = owners.emplace(loom::CurrentThreadIndex(), std::this_thread::get_id());
			if (!fresh && it->second != std::this_thread::get_id())
				clash++;
		}
	});
	int64_t sum = 0;
	pt.ForEach([&](int64_t& x) { sum += x; });
	const int64_t want = (int64_t)(n * (n ? n - 1 : 0) / 2) + exemplar * (int64_t)pt.Size();
	CHECK(sum == want, "PerThread: sum %lld, want %lld over %zu cells", (long long)sum, (long long)want, pt.Size());
	CHECK(clash.load() == 0, "two live threads shared a CurrentThreadIndex");
	pt.Clear();
	CHECK(pt.Size() == 0, "PerThread::Clear left %zu cells", pt.Size());
}

// ---- Dispatch ----

// Work nests through helping: a wait runs another piece of work, which
// waits in turn. The rules leave a hand-written fork-join unbounded, each
// level holding counters, so past this many levels on one thread a piece
// only spins, or a small pool's counters would run out. Well past loom's
// own bound (16 waits), so its inline paths still run.
constexpr int kMaxNesting = 24;
thread_local int t_Nesting = 0;

void RunWork(uint64_t seed, int size, const WorkCtx& given)
{
	if (t_Nesting >= kMaxNesting)
	{
		SpinFor((int)(seed % 2000));
		return;
	}
	struct Nest
	{
		Nest() { t_Nesting++; }
		~Nest() { t_Nesting--; }
	} nest;
	Rng r(seed);
	size = std::max(1, size);
	WorkCtx ctx = given;
	const int kind = (int)r.Below(10);
	switch (kind)
	{
	case 0: case 1: WorkForkJoin(r, size, ctx); break;
	case 2: WorkParallelFor(r, size, ctx); break;
	case 3: WorkReduce(r, size, ctx); break;
	case 4: WorkScan(r, size, ctx); break;
	case 5: WorkSort(r, size, ctx); break;
	case 6: WorkAsync(r, size, ctx); break;
	case 7: WorkGraph(r, size, ctx); break;
	case 8: WorkPipeline(r, size, ctx); break;
	case 9:
#if defined(LOOM_COROUTINES)
		if (r.Chance(0.7)) { WorkCoro(r, size, ctx); break; }
#endif
		WorkPerThread(r, size, ctx);
		break;
	}
}

// ======================================================================================
// Dag: a random DAG of tasks over counters, joined later.
// ======================================================================================

struct DagOp;

enum CallKind : uint8_t { kRaw, kRawBatch, kInline, kOwned, kHeap };

struct DagTask
{
	DagOp* dag = nullptr;
	int id = 0;
	int sig = -1;
	std::vector<int> waits;
	std::vector<CounterHandle> waitHandles;
	bool noWaitsFor = false; // every wait in waitsForAll
	Place place;
	CallKind call = kOwned;
	bool throws = false;
	int cancel = -1;
	int sigTarget = -1, sigAmount = 0;
	std::vector<int> children;
	int parent = -1;
	int nested = 0;
	uint64_t nestedSeed = 0;
	bool canBlock = false;
	uint64_t tick = 0;
	std::atomic<int> runs{ 0 }, destroys{ 0 };
	std::atomic<bool> submitted{ false }, threw{ false }, ended{ false }, destroyDone{ false };

	void Run();
	void Destroyed();
	void CheckDeps(const char* what);
	void SubmitSelf(std::vector<loom::TaskDesc>* batch);
};

// Destroyed with the owned callable: tells a skip from a run.
struct Probe
{
	DagTask* t;
	explicit Probe(DagTask* task) : t(task) {}
	Probe(Probe&& o) noexcept : t(std::exchange(o.t, nullptr)) {}
	Probe(const Probe&) = delete;
	~Probe() { if (t) t->Destroyed(); }
};

struct DagCounter
{
	CounterHandle h = loom::kInvalidCounter;
	int initial = 0;
	int manual = 0;
	int manualWhen = 0; // 0 at start, 1 at join, 2 an outside thread, 3 a task (sigTarget)
	bool precancel = false;
	bool observable = true;
	std::atomic<int> finished{ 0 };
	std::atomic<int> added{ 0 };
	std::vector<int> producers; // static ones and children
	std::vector<int> cancellers;
};

struct DagOp : Pending
{
	std::vector<std::unique_ptr<DagTask>> tasks;
	std::vector<std::unique_ptr<DagCounter>> ctrs;
	bool pure = false;
	bool joined = false;
	std::unique_ptr<Outside> signaller;
	Rng rng{ 0 };

	void Generate(Rng& r, int size);
	void Start();
	void Join() override;
	bool CanDrain() const override { return true; }
	void AfterDrain() override { CheckTasks(false); }
	void CheckTasks(bool afterJoin);
	bool CancelledModel(int c) const;
	bool After(int s, int t, std::vector<int8_t>& memo) const;
};

void DagTask::CheckDeps(const char* what)
{
	for (int w : waits)
	{
		DagCounter& c = *dag->ctrs[(size_t)w];
		const int count = loom::Count(c.h);
		CHECK(count == 0, "dag op %d task %d %s while counter %d is %d", dag->op, id, what, w, count);
		if (c.observable)
		{
			const int fin = c.finished.load(std::memory_order_seq_cst);
			const int want = c.initial + c.added.load(std::memory_order_seq_cst);
			CHECK(fin == want, "dag op %d task %d %s before counter %d's work finished (%d of %d)", dag->op, id, what, w, fin, want);
		}
	}
}

void DagTask::Run()
{
	CHECK(runs.fetch_add(1) == 0, "dag op %d task %d ran twice", dag->op, id);
	TaskScope scope(place, tick, "dag task");
	CheckDeps("started");
	if (!children.empty())
	{
		DagCounter& c = *dag->ctrs[(size_t)sig];
		const bool ok = loom::AddToCounter(c.h, (int)children.size());
		CHECK(ok, "dag op %d task %d: AddToCounter failed while it ran", dag->op, id);
		if (ok)
		{
			c.added.fetch_add((int)children.size(), std::memory_order_seq_cst);
			for (int ch : children)
			{
				DagTask& child = *dag->tasks[(size_t)ch];
				child.submitted.store(true);
				child.SubmitSelf(nullptr);
			}
		}
	}
	if (cancel >= 0)
		loom::Cancel(dag->ctrs[(size_t)cancel]->h);
	if (nested > 0)
	{
		WorkCtx ctx;
		ctx.depth = 1;
		ctx.nested = true;
		ctx.ded = place.aff == Affinity::Dedicated ? place.ded : -1;
		ctx.canBlock = canBlock;
		ctx.noDed = canBlock;
		ctx.op = dag->op;
		RunWork(nestedSeed, nested, ctx);
	}
	else
		SpinFor((int)(Hash(nestedSeed) % 3000));
	if (call < kOwned)
	{
		if (sigTarget >= 0)
		{
			dag->ctrs[(size_t)sigTarget]->finished.fetch_add(sigAmount, std::memory_order_seq_cst);
			loom::Signal(dag->ctrs[(size_t)sigTarget]->h, sigAmount);
		}
		if (sig >= 0)
			dag->ctrs[(size_t)sig]->finished.fetch_add(1, std::memory_order_seq_cst);
	}
	const bool willThrow = throws;
	const FuzzError error{ dag->op, id };
	if (willThrow)
		threw.store(true);
	ended.store(true); // a task with no counter may be freed from here on
	if (willThrow)
		throw error;
}

void DagTask::Destroyed()
{
	CHECK(destroys.fetch_add(1) == 0, "dag op %d task %d: callable destroyed twice", dag->op, id);
	if (runs.load() == 0)
		CheckDeps("skipped");
	if (sigTarget >= 0)
	{
		dag->ctrs[(size_t)sigTarget]->finished.fetch_add(sigAmount, std::memory_order_seq_cst);
		loom::Signal(dag->ctrs[(size_t)sigTarget]->h, sigAmount);
	}
	if (sig >= 0)
		dag->ctrs[(size_t)sig]->finished.fetch_add(1, std::memory_order_seq_cst);
	destroyDone.store(true); // the last touch: a task with no counter is waited for by this
}

void DagTask::SubmitSelf(std::vector<loom::TaskDesc>* batch)
{
	loom::TaskDesc d = Desc(place);
	if (sig >= 0)
		d.signals = dag->ctrs[(size_t)sig]->h;
	if (!waitHandles.empty())
	{
		if (noWaitsFor)
			d.waitsForAll = waitHandles;
		else
		{
			d.waitsFor = waitHandles[0];
			d.waitsForAll = std::span<const CounterHandle>(waitHandles).subspan(1);
		}
	}
	tick = Tick();
	DagTask* self = this;
	switch (call)
	{
	case kRaw:
	case kRawBatch:
		d.fn = [](void* p) { static_cast<DagTask*>(p)->Run(); };
		d.userData = self;
		if (batch)
			batch->push_back(d);
		else
			loom::Submit(d);
		break;
	case kInline:
		loom::Submit([self] { self->Run(); }, d);
		break;
	case kOwned:
		loom::Submit([p = Probe(self)] { p.t->Run(); }, d);
		break;
	case kHeap:
		loom::Submit([p = Probe(self), pad = std::array<uint64_t, 8>{}] { (void)pad; p.t->Run(); }, d);
		break;
	}
}

void DagOp::Generate(Rng& r, int size)
{
	const int K = r.Range(1, std::min(40, 2 + size));
	const int N = r.Range(1, 3 + size * 3);
	pure = g_Env->workers >= 1 && r.Chance(0.25) && g_BlockOnly;
	for (int c = 0; c < K; c++)
	{
		auto ctr = std::make_unique<DagCounter>();
		if (r.Chance(0.2))
		{
			ctr->manual = r.Range(1, 3);
			ctr->manualWhen = (int)r.Below(4);
		}
		ctr->precancel = r.Chance(0.03);
		ctrs.push_back(std::move(ctr));
	}
	auto makeTask = [&](int sig, int level, int parent, int depth) -> int {
		auto t = std::make_unique<DagTask>();
		t->dag = this;
		t->id = (int)tasks.size();
		t->sig = sig;
		t->parent = parent;
		t->nestedSeed = r.Fork();
		if (parent < 0 && level > 0)
		{
			const int nw = r.Range(0, 3);
			for (int i = 0; i < nw; i++)
				t->waits.push_back((int)r.Below((uint32_t)level));
			t->noWaitsFor = r.Chance(0.2);
		}
		if (!pure && g_Env->blockers && g_BlockOnly && sig >= 0 && r.Chance(0.05))
		{
			// A BlockOnly waiter: its own limiter, its own AnyWorker leaves.
			t->canBlock = true;
			t->nested = r.Range(1, 6);
			t->place.lim = (int)(std::find_if(g_Env->lims.begin(), g_Env->lims.end(), [](auto& l) { return l->blockers; }) - g_Env->lims.begin());
			t->place.pri = r.Chance(0.3) ? Priority::High : Priority::Normal;
		}
		else
		{
			const bool nests = !pure && r.Chance(0.1) && depth == 0;
			// A pure DAG may be waited for with BlockOnly: AnyWorker, no limiter.
			unsigned flags = pure ? 0u : kMain | (nests ? (unsigned)kOuterLim : kLeafLim | kOuterLim);
			if (parent < 0)
				flags |= pure ? 0u : (unsigned)kDedAny;
			const int parentDed = parent >= 0 && tasks[(size_t)parent]->place.aff == Affinity::Dedicated ? tasks[(size_t)parent]->place.ded : -1;
			t->place = RandomPlace(r, flags, parentDed, &users);
			if (sig < 0)
				t->place.lim = -1; // nothing to wait on before a DestroyLimiter
			if (nests)
				t->nested = r.Range(1, std::max(1, size / 3));
		}
		if (sig >= 0)
		{
			t->throws = r.Chance(0.06);
			if (level + 1 < K && r.Chance(0.05))
				t->cancel = r.Range(level + 1, K - 1);
		}
		const int kind = (int)r.Below(5);
		t->call = (CallKind)kind;
		const int id = t->id;
		tasks.push_back(std::move(t));
		return id;
	};
	for (int i = 0; i < N; i++)
	{
		const int sig = r.Chance(0.05) ? -1 : (int)r.Below((uint32_t)K);
		const int level = sig < 0 ? K : sig;
		makeTask(sig, level, -1, 0);
	}
	// Children (AddToCounter in a running task) and manual signallers.
	const size_t statics = tasks.size();
	std::vector<std::pair<int, int>> todo; // (task, depth)
	for (size_t i = 0; i < statics; i++)
		todo.push_back({ (int)i, 0 });
	for (size_t q = 0; q < todo.size(); q++)
	{
		const auto [id, depth] = todo[q];
		DagTask* t = tasks[(size_t)id].get();
		if (t->sig < 0 || depth >= 3 || !r.Chance(depth == 0 ? 0.08 : 0.3))
			continue;
		if (t->call < kOwned)
			t->call = kOwned;
		const int k = r.Range(1, 4);
		for (int c = 0; c < k; c++)
		{
			const int child = makeTask(t->sig, t->sig, id, depth + 1);
			tasks[(size_t)id]->children.push_back(child);
			tasks[(size_t)child]->call = r.Chance(0.5) ? kOwned : kHeap;
			todo.push_back({ child, depth + 1 });
		}
	}
	for (int c = 0; c < K; c++)
	{
		DagCounter& ctr = *ctrs[(size_t)c];
		if (ctr.manual > 0 && ctr.manualWhen == 3)
		{
			// A static task of a lower level signals it (from its callable's destructor, so a skip signals too).
			std::vector<int> lower;
			for (size_t i = 0; i < statics; i++)
				if (tasks[i]->sig >= 0 && tasks[i]->sig < c && tasks[i]->sigTarget < 0)
					lower.push_back((int)i);
			if (lower.empty())
				ctr.manualWhen = 0;
			else
			{
				DagTask& s = *tasks[(size_t)lower[r.Below((uint32_t)lower.size())]];
				s.sigTarget = c;
				s.sigAmount = ctr.manual;
				if (s.call < kOwned)
					s.call = kOwned;
			}
		}
		if (ctr.manualWhen == 2 && pure)
			ctr.manualWhen = 0;
	}
	for (size_t i = 0; i < tasks.size(); i++)
	{
		DagTask& t = *tasks[i];
		if (t.sig >= 0)
		{
			ctrs[(size_t)t.sig]->producers.push_back((int)i);
			if (t.call < kOwned)
				ctrs[(size_t)t.sig]->observable = false;
		}
		if (t.cancel >= 0)
			ctrs[(size_t)t.cancel]->cancellers.push_back((int)i);
		if (t.parent < 0 && t.sig >= 0)
			ctrs[(size_t)t.sig]->initial++;
	}
	for (auto& c : ctrs)
		c->initial += c->manual;
	rng = Rng(r.Fork());
}

void DagOp::Start()
{
	for (auto& c : ctrs)
	{
		c->h = loom::CreateCounter(c->initial);
		CHECK(!loom::IsCancelled(c->h), "a new counter is cancelled");
		CHECK(loom::Count(c->h) == c->initial, "a new counter counts %d, want %d", loom::Count(c->h), c->initial);
		if (c->precancel)
			loom::Cancel(c->h);
	}
	for (auto& t : tasks)
		for (int w : t->waits)
			t->waitHandles.push_back(ctrs[(size_t)w]->h);
	std::vector<int> order;
	for (auto& t : tasks)
		if (t->parent < 0)
			order.push_back(t->id);
	for (size_t i = order.size(); i > 1; i--)
		std::swap(order[i - 1], order[rng.Below((uint32_t)i)]);
	std::vector<loom::TaskDesc> batch;
	for (int id : order)
	{
		DagTask& t = *tasks[(size_t)id];
		t.submitted.store(true);
		t.SubmitSelf(t.call == kRawBatch ? &batch : nullptr);
		if (batch.size() > 8 || (batch.size() > 0 && rng.Chance(0.2)))
		{
			loom::Submit(std::span<const loom::TaskDesc>(batch));
			batch.clear();
		}
	}
	if (!batch.empty())
		loom::Submit(std::span<const loom::TaskDesc>(batch));
	std::vector<DagCounter*> later;
	for (auto& c : ctrs)
	{
		if (c->manual > 0 && c->manualWhen == 0)
		{
			c->finished.fetch_add(c->manual);
			loom::Signal(c->h, c->manual);
		}
		if (c->manual > 0 && c->manualWhen == 2)
			later.push_back(c.get());
	}
	if (!later.empty())
	{
		const uint64_t seed = rng.Fork();
		signaller = std::make_unique<Outside>([later, seed] {
			Rng r(seed);
			for (DagCounter* c : later)
			{
				std::this_thread::sleep_for(std::chrono::microseconds(r.Below(2000)));
				for (int i = 0; i < c->manual; i++)
				{
					c->finished.fetch_add(1);
					loom::Signal(c->h);
				}
			}
		});
	}
}

bool DagOp::CancelledModel(int c) const
{
	const DagCounter& ctr = *ctrs[(size_t)c];
	if (ctr.precancel)
		return true;
	for (int t : ctr.cancellers)
		if (tasks[(size_t)t]->runs.load() > 0)
			return true;
	for (int p : ctr.producers)
		if (tasks[(size_t)p]->threw.load())
			return true;
	return false;
}

// Whether task s could only start after task t had finished.
bool DagOp::After(int s, int t, std::vector<int8_t>& memo) const
{
	if (memo[(size_t)s] >= 0)
		return memo[(size_t)s] != 0;
	memo[(size_t)s] = 0;
	const DagTask& S = *tasks[(size_t)s];
	bool after = false;
	if (S.parent >= 0)
		after = S.parent != t && After(S.parent, t, memo);
	const int tsig = tasks[(size_t)t]->sig;
	for (int w : S.waits)
	{
		if (after)
			break;
		if (w == tsig)
		{
			after = true;
			break;
		}
		for (int p : ctrs[(size_t)w]->producers)
			if (tasks[(size_t)p]->submitted.load() && After(p, t, memo))
			{
				after = true;
				break;
			}
	}
	memo[(size_t)s] = after ? 1 : 0;
	return after;
}

void DagOp::Join()
{
	joined = true;
	for (auto& c : ctrs)
		if (c->manual > 0 && c->manualWhen == 1)
		{
			c->finished.fetch_add(c->manual);
			loom::Signal(c->h, c->manual);
		}
	if (signaller)
		signaller->Join();
	std::vector<std::optional<FuzzError>> errors(ctrs.size());
	for (size_t c = 0; c < ctrs.size(); c++)
	{
		const WaitKind wk = RandomWait(rng, pure);
		errors[c] = WaitCounter(ctrs[c]->h, wk, rng, "dag join");
		std::optional<FuzzError> again = WaitCounter(ctrs[c]->h, WaitKind::Zero, rng, "dag join again");
		CHECK(again.has_value() == errors[c].has_value() && (!again || again->id == errors[c]->id),
			"dag op %d counter %zu: a second wait rethrew differently", op, c);
	}
	// Tasks that signal nothing: help until they have run.
	const auto deadline = Clock::now() + std::chrono::seconds(60);
	HelpUntil([&] {
		for (auto& t : tasks)
			if (t->sig < 0 && (!t->ended.load() || (t->call >= kOwned && !t->destroyDone.load())))
				return Clock::now() > deadline;
		return true;
	});
	for (size_t c = 0; c < ctrs.size(); c++)
	{
		DagCounter& ctr = *ctrs[c];
		bool anyThrew = false;
		for (int p : ctr.producers)
			anyThrew |= tasks[(size_t)p]->threw.load();
		CHECK(errors[c].has_value() == anyThrew, "dag op %d counter %zu: WaitFor %s, but %s task of it threw", op, c,
			errors[c] ? "threw" : "did not throw", anyThrew ? "a" : "no");
		if (errors[c])
		{
			const FuzzError e = *errors[c];
			const bool mine = e.op == op && e.id >= 0 && e.id < (int)tasks.size() && tasks[(size_t)e.id]->sig == (int)c
				&& tasks[(size_t)e.id]->threw.load();
			CHECK(mine, "dag op %d counter %zu: rethrew (%d,%d), not one of its tasks' errors", op, c, e.op, e.id);
		}
		CHECK(loom::IsCancelled(ctr.h) == CancelledModel((int)c), "dag op %d counter %zu: IsCancelled %d, model %d", op, c,
			(int)loom::IsCancelled(ctr.h), (int)CancelledModel((int)c));
		CHECK(loom::Count(ctr.h) == 0, "dag op %d counter %zu: count %d after its wait", op, c, loom::Count(ctr.h));
		CHECK(!loom::AddToCounter(ctr.h, 1), "dag op %d counter %zu: AddToCounter succeeded at zero", op, c);
	}
	CheckTasks(true);
	for (auto& c : ctrs)
		loom::DestroyCounter(c->h);
}

void DagOp::CheckTasks(bool afterJoin)
{
	for (auto& tp : tasks)
	{
		DagTask& t = *tp;
		const bool sub = t.submitted.load();
		const int runs = t.runs.load(), destroys = t.destroys.load();
		CHECK(runs <= 1, "dag op %d task %d ran %d times", op, t.id, runs);
		if (t.call >= kOwned)
			CHECK(destroys == (sub ? 1 : 0), "dag op %d task %d: callable destroyed %d times (submitted %d)", op, t.id, destroys, (int)sub);
		if (!sub)
		{
			CHECK(runs == 0, "dag op %d task %d ran but was never submitted", op, t.id);
			continue;
		}
		if (runs == 0)
		{
			CHECK(t.sig >= 0, "dag op %d task %d (no counter) never ran", op, t.id);
			if (t.sig >= 0 && afterJoin)
				CHECK(CancelledModel(t.sig), "dag op %d task %d was skipped but counter %d was never cancelled", op, t.id, t.sig);
		}
		if (t.sig >= 0)
		{
			for (int cn : ctrs[(size_t)t.sig]->cancellers)
			{
				if (tasks[(size_t)cn]->runs.load() == 0)
					continue;
				std::vector<int8_t> memo(tasks.size(), -1);
				if (After(t.id, cn, memo))
					CHECK(runs == 0, "dag op %d task %d ran after task %d cancelled its counter %d", op, t.id, cn, t.sig);
			}
			if (ctrs[(size_t)t.sig]->precancel)
				CHECK(runs == 0, "dag op %d task %d ran although its counter was cancelled before it was submitted", op, t.id);
		}
	}
}

// ======================================================================================
// Futures: chains, fan-outs and promises, joined later.
// ======================================================================================

struct FutStep
{
	Place place;
	bool throws = false;
	int parent = -1;     // -1: a source
	int source = 0;      // 0 Async, 1 promise set by main at join, 2 by an outside thread, 3 by a task, 4 SetException, 5 broken
	uint64_t tick = 0;
	std::atomic<int> runs{ 0 };
	std::atomic<bool> done{ false }; // its callable's last touch of this step
	int64_t value = 0;   // the model's
	int error = -2;      // the model's: -2 none, -1 broken promise, else the step that threw
	bool keep = false;   // the op keeps its future (else it is dropped, the chain still runs)
	bool plainReader = false; // a plain task waits on its Counter() and reads Get()
};

struct FuturesOp : Pending
{
	std::vector<std::unique_ptr<FutStep>> steps;
	std::vector<loom::Future<int64_t>> futs;
	std::vector<std::optional<loom::Promise<int64_t>>> promises;
	std::unique_ptr<Outside> setter;
	struct Reader
	{
		std::atomic<int> runs{ 0 };
		std::atomic<int> bad{ 0 };
		int step = 0;
	};
	std::vector<std::unique_ptr<Reader>> readers;
	std::unique_ptr<loom::ScopedCounter> readersDone;
	Rng rng{ 0 };

	void Generate(Rng& r, int size)
	{
		const int n = r.Range(1, 2 + std::min(size, 60)); // a counter each, while kept
		for (int i = 0; i < n; i++)
		{
			auto s = std::make_unique<FutStep>();
			s->parent = i == 0 || r.Chance(0.25) ? -1 : (int)r.Below((uint32_t)i);
			s->source = s->parent < 0 ? (r.Chance(0.5) ? 0 : r.Range(1, 5)) : 0;
			s->place = RandomPlace(r, kMain | kDedAny | kLeafLim | kOuterLim, -1, &users);
			s->throws = r.Chance(0.06);
			s->keep = r.Chance(0.6) || i == n - 1;
			s->plainReader = s->keep && r.Chance(0.15);
			// The model.
			if (s->parent < 0)
			{
				s->value = (int64_t)i;
				if (s->source == 0 && s->throws)
					s->error = i;
				else if (s->source == 4)
					s->error = i;
				else if (s->source == 5)
					s->error = -1;
			}
			else
			{
				const FutStep& p = *steps[(size_t)s->parent];
				if (p.error != -2)
					s->error = p.error;
				else if (s->throws)
					s->error = i;
				else
					s->value = (int64_t)((uint64_t)p.value * 5 + (uint64_t)i);
			}
			steps.push_back(std::move(s));
		}
		rng = Rng(r.Fork());
	}

	void Start()
	{
		futs.resize(steps.size());
		promises.resize(steps.size());
		std::vector<std::pair<int, loom::Promise<int64_t>>> forOutside;
		int nReaders = 0;
		for (auto& s : steps)
			nReaders += s->plainReader;
		readersDone = std::make_unique<loom::ScopedCounter>(nReaders);
		const int opIndex = op;
		for (size_t i = 0; i < steps.size(); i++)
		{
			FutStep* s = steps[i].get();
			s->tick = Tick();
			if (s->parent < 0 && s->source == 0)
			{
				futs[i] = loom::Async([s, opIndex, i]() -> int64_t {
					s->runs++;
					int64_t v;
					bool throws;
					{
						TaskScope scope(s->place, s->tick, "future source");
						throws = s->throws;
						v = s->value;
					}
					s->done.store(true); // the op may be freed from here on
					if (throws)
						throw FuzzError{ opIndex, (int)i };
					return v;
				}, Desc(s->place));
			}
			else if (s->parent < 0)
			{
				loom::Promise<int64_t> p;
				const bool setFirst = s->source == 1 && rng.Chance(0.5);
				if (setFirst)
					p.SetValue(s->value); // GetFuture after the set
				futs[i] = p.GetFuture();
				switch (s->source)
				{
				case 1:
					if (!setFirst)
						promises[i].emplace(std::move(p)); // set at the join
					break;
				case 2: case 4: forOutside.push_back({ (int)i, std::move(p) }); break;
				case 3:
					loom::Submit([p = std::move(p), v = s->value]() mutable { SpinFor(200); p.SetValue(v); });
					break;
				case 5:
					if (rng.Chance(0.5))
						promises[i].emplace(std::move(p)); // broken at the join
					break;                                 // or right here
				}
			}
			else
			{
				FutStep* parent = steps[(size_t)s->parent].get();
				(void)parent;
				futs[i] = futs[(size_t)s->parent].Then([s, opIndex, i](int64_t& v) -> int64_t {
					s->runs++;
					bool throws;
					{
						TaskScope scope(s->place, s->tick, "future continuation");
						throws = s->throws;
					}
					s->done.store(true); // the op may be freed from here on
					if (throws)
						throw FuzzError{ opIndex, (int)i };
					return (int64_t)((uint64_t)v * 5 + (uint64_t)i);
				}, Desc(s->place));
			}
		}
		for (size_t i = 0; i < steps.size(); i++)
		{
			FutStep* s = steps[i].get();
			if (s->plainReader)
			{
				auto rd = std::make_unique<Reader>();
				rd->step = (int)i;
				Reader* rp = rd.get();
				loom::Future<int64_t>* f = &futs[i];
				loom::TaskDesc d;
				d.waitsFor = f->Counter();
				d.signals = *readersDone;
				loom::Submit([rp, f, s] {
					rp->runs++;
					if (!f->IsReady())
						rp->bad++;
					try
					{
						if (f->Get() != s->value || s->error != -2)
							rp->bad++;
					}
					catch (const FuzzError& e)
					{
						if (e.id != s->error)
							rp->bad++;
					}
					catch (const loom::BrokenPromise&)
					{
						if (s->error != -1)
							rp->bad++;
					}
				}, d);
				readers.push_back(std::move(rd));
			}
		}
		for (size_t i = 0; i < steps.size(); i++)
			if (!steps[i]->keep && !steps[i]->plainReader)
				futs[i].Reset(); // dropped: what depends on it still runs
		if (!forOutside.empty())
		{
			const uint64_t seed = rng.Fork();
			std::vector<FutStep*> stepPtrs;
			for (auto& s : steps)
				stepPtrs.push_back(s.get());
			const int opIndex2 = op;
			setter = std::make_unique<Outside>([ps = std::move(forOutside), seed, stepPtrs, opIndex2]() mutable {
				Rng r(seed);
				for (auto& [i, p] : ps)
				{
					std::this_thread::sleep_for(std::chrono::microseconds(r.Below(1500)));
					if (stepPtrs[(size_t)i]->source == 4)
						p.SetException(std::make_exception_ptr(FuzzError{ opIndex2, i }));
					else
						p.SetValue(stepPtrs[(size_t)i]->value);
					CHECK(p.IsSet(), "Promise::IsSet false after a set");
				}
				ps.clear();
			});
		}
	}

	void Join() override
	{
		for (size_t i = 0; i < steps.size(); i++)
			if (promises[i])
			{
				if (steps[i]->source == 1)
					promises[i]->SetValue(steps[i]->value);
				promises[i].reset(); // source 5: broken here
			}
		if (setter)
			setter->Join();
		for (size_t i = 0; i < steps.size(); i++)
		{
			FutStep& s = *steps[i];
			if (!futs[i].IsValid())
				continue;
			try
			{
				int64_t got;
				if (rng.Chance(0.3))
				{
					while (!futs[i].TryWait(rng.Below(3)))
						loom::TryHelp();
					got = futs[i].Get();
				}
				else
					got = futs[i].Get();
				CHECK(s.error == -2 && got == s.value, "futures op %d step %zu: %lld, want %lld (error %d)", op, i, (long long)got,
					(long long)s.value, s.error);
				CHECK(futs[i].Get() == got, "futures op %d step %zu: a second Get differs", op, i);
			}
			catch (const FuzzError& e)
			{
				CHECK(e.id == s.error && e.op == op, "futures op %d step %zu: error (%d,%d), want %d", op, i, e.op, e.id, s.error);
			}
			catch (const loom::BrokenPromise&)
			{
				CHECK(s.error == -1, "futures op %d step %zu: BrokenPromise, want %d", op, i, s.error);
			}
			CHECK(futs[i].IsReady(), "futures op %d step %zu: not ready after Get", op, i);
		}
		WaitCounter(*readersDone, WaitKind::Help, rng, "future readers");
		for (auto& rd : readers)
			CHECK(rd->runs.load() == 1 && rd->bad.load() == 0, "futures op %d: a plain task reading step %d's future saw it wrong", op, rd->step);
		// Every step whose input succeeded ran exactly once (dropped futures included): wait for those not kept.
		const auto deadline = Clock::now() + std::chrono::seconds(60);
		HelpUntil([&] {
			for (size_t i = 0; i < steps.size(); i++)
			{
				const FutStep& s = *steps[i];
				const bool shouldRun = (s.parent < 0 && s.source == 0) || (s.parent >= 0 && steps[(size_t)s.parent]->error == -2);
				if (shouldRun && !s.done.load())
					return Clock::now() > deadline;
			}
			return true;
		});
		for (size_t i = 0; i < steps.size(); i++)
		{
			const FutStep& s = *steps[i];
			const bool shouldRun = (s.parent < 0 && s.source == 0) || (s.parent >= 0 && steps[(size_t)s.parent]->error == -2);
			const bool isTask = s.parent >= 0 || s.source == 0;
			if (isTask)
				CHECK(s.runs.load() == (shouldRun ? 1 : 0), "futures op %d step %zu ran %d times, want %d", op, i, s.runs.load(), shouldRun ? 1 : 0);
		}
		futs.clear();
		readersDone.reset();
	}
};

// ======================================================================================
// Coroutines started now, joined later.
// ======================================================================================

#if defined(LOOM_COROUTINES)
struct CoroOp : Pending
{
	std::vector<std::unique_ptr<CoroPlan>> plans;
	std::vector<loom::Future<int64_t>> futs;
	void Start(Rng& r, int size)
	{
		const int n = r.Range(1, 4);
		for (int i = 0; i < n; i++)
		{
			auto plan = std::make_unique<CoroPlan>();
			plan->op = op;
			MakeCoroPlan(*plan, r, size, kMain | kDedAny | kLeafLim | kOuterLim, -1, 0, &users);
			plans.push_back(std::move(plan));
		}
		for (auto& p : plans)
			futs.push_back(CoroBody(p.get(), 0));
	}
	void Join() override
	{
		for (size_t i = 0; i < futs.size(); i++)
			CheckCoroResult(*plans[i], futs[i], "async coroutine");
		futs.clear();
	}
};
#endif

// A graph submitted now and waited for later.
struct GraphAsyncOp : Pending
{
	loom::TaskGraph g;
	GraphModel m;
	std::vector<std::unique_ptr<GraphNodeArg>> args;
	std::unique_ptr<loom::ScopedCounter> done, gate;
	Rng rng{ 0 };
	void Start(Rng& r, int size)
	{
		rng = Rng(r.Fork());
		m.op = op;
		m.throwP = r.Chance(0.3) ? 0.08 : 0;
		BuildGraph(g, m, r, r.Range(1, 3 + 2 * size), kMain | kDedAny | kLeafLim | kOuterLim, -1, args);
		for (auto& n : m.nodes)
			if (n->place.ded >= 0)
				users.push_back(n->place.ded);
		m.run.store(0);
		m.tick = Tick();
		done = std::make_unique<loom::ScopedCounter>(1);
		gate = std::make_unique<loom::ScopedCounter>(r.Chance(0.5) ? 1 : 0);
		loom::Submit(g, *done, *gate);
	}
	void Join() override
	{
		if (loom::Count(*gate) != 0)
		{
			for (auto& node : m.nodes)
				CHECK(node->lastRun.load() != 0, "graph-async: a node ran before the gate opened");
			loom::Signal(*gate);
		}
		std::optional<FuzzError> got = WaitCounter(*done, WaitKind::Help, rng, "graph-async");
		CHECK(!g.IsRunning(), "graph-async: IsRunning after its counter's wait");
		CheckGraphRun(m, 0, got, "graph-async");
	}
};

// An outside thread doing work of its own, changing limits, reading the debug state.
struct OutsideOp : Pending
{
	std::unique_ptr<Outside> thread;
	void Start(Rng& r, int size)
	{
		const uint64_t seed = r.Fork();
		const int kind = (int)r.Below(4);
		std::vector<Lim*> lims;
		for (auto& l : g_Env->lims)
			if (!l->blockers)
				lims.push_back(l.get());
		const int opIndex = op;
		int ded = -1;
		if (r.Chance(0.3))
		{
			std::vector<int> alive;
			for (size_t i = 0; i < g_Env->ded.size(); i++)
				if (g_Env->ded[i]->alive)
					alive.push_back((int)i);
			if (!alive.empty())
			{
				ded = alive[r.Below((uint32_t)alive.size())];
				users.push_back(ded);
			}
		}
		thread = std::make_unique<Outside>([seed, size, kind, lims, opIndex, ded] {
			Rng r(seed);
			if (kind == 0 || kind == 3)
			{
				const int n = r.Range(1, 3);
				for (int i = 0; i < n; i++)
				{
					WorkCtx ctx;
					ctx.op = opIndex;
					ctx.nested = true; // leaf limiters only
					ctx.ded = ded;
					RunWork(r.Fork(), size, ctx);
				}
			}
			if (kind == 1 || kind == 3)
			{
				const int n = r.Range(1, 2 + size / 4);
				for (int i = 0; i < n && !lims.empty(); i++)
				{
					Lim* l = lims[r.Below((uint32_t)lims.size())];
					l->Set((uint32_t)r.Range(1, 12));
					std::this_thread::sleep_for(std::chrono::microseconds(r.Below(3000)));
				}
			}
			if (kind == 2)
			{
				for (int i = 0; i < 3; i++)
				{
					loom::DebugState s = loom::CaptureDebugState();
					std::string text = loom::FormatDebugState(s);
					CHECK(!text.empty(), "an empty debug dump");
					loom::PoolSnapshot snap = loom::SnapshotPool();
					CHECK(snap.liveTasks >= 0 && snap.counterCount >= 0, "a snapshot with negative counts");
					std::this_thread::sleep_for(std::chrono::microseconds(r.Below(2000)));
				}
			}
		});
	}
	void Join() override { thread->Join(); }
};

// ======================================================================================
// Epochs
// ======================================================================================

struct EpochSpec
{
	int workers = 2;
	uint32_t maxTasks = 4096;
	uint32_t maxCounters = 2048;
	uint32_t maxLimiters = 64;
	int nDed = 1;
	int nLim = 3;
	int drain = 0;
	uint64_t seed = 0;
	std::vector<OpSpec> ops;
};

struct ProgramSpec
{
	std::vector<EpochSpec> epochs;
};

std::string Print(const ProgramSpec& p)
{
	std::string s;
	for (const EpochSpec& e : p.epochs)
	{
		char buf[256];
		std::snprintf(buf, sizeof buf, "E%d,%u,%u,%u,%d,%d,%d,%llu:", e.workers, e.maxTasks, e.maxCounters, e.maxLimiters, e.nDed, e.nLim,
			e.drain, (unsigned long long)e.seed);
		s += buf;
		for (const OpSpec& o : e.ops)
		{
			std::snprintf(buf, sizeof buf, "%d.%llu.%d.%d/", o.kind, (unsigned long long)o.seed, o.size, o.join);
			s += buf;
		}
		s += ";";
	}
	return s;
}

bool Parse(const std::string& text, ProgramSpec& p)
{
	size_t at = 0;
	while (at < text.size())
	{
		const size_t end = text.find(';', at);
		if (end == std::string::npos)
			return false;
		const std::string ep = text.substr(at, end - at);
		at = end + 1;
		EpochSpec e;
		unsigned long long seed = 0;
		int consumed = 0;
		if (std::sscanf(ep.c_str(), "E%d,%u,%u,%u,%d,%d,%d,%llu:%n", &e.workers, &e.maxTasks, &e.maxCounters, &e.maxLimiters, &e.nDed,
				&e.nLim, &e.drain, &seed, &consumed) < 8)
			return false;
		e.seed = seed;
		size_t q = (size_t)consumed;
		while (q < ep.size())
		{
			const size_t slash = ep.find('/', q);
			if (slash == std::string::npos)
				return false;
			OpSpec o;
			unsigned long long os = 0;
			if (std::sscanf(ep.substr(q, slash - q).c_str(), "%d.%llu.%d.%d", &o.kind, &os, &o.size, &o.join) != 4 || o.kind < 0 || o.kind >= kOpCount)
				return false;
			o.seed = os;
			e.ops.push_back(o);
			q = slash + 1;
		}
		p.epochs.push_back(std::move(e));
	}
	return !p.epochs.empty();
}

ProgramSpec Generate(uint64_t seed, int size)
{
	Rng r(seed);
	ProgramSpec p;
	const int epochs = r.Range(1, 1 + std::min(3, size / 25));
	for (int i = 0; i < epochs; i++)
	{
		EpochSpec e;
		const int w = (int)r.Below(10);
		e.workers = w < 2 ? 0 : w < 4 ? 1 : w < 6 ? 2 : w < 8 ? r.Range(3, 4) : r.Range(5, 8);
		const uint32_t tasks[] = { 1, 100, 4096, 5000 };
		e.maxTasks = tasks[r.Below(4)];
		// Each pending operation holds up to ~60 counters, and each thread's
		// pile of nested waits a few dozen more: small tables for small programs.
		const uint32_t ctrs[] = { 256, 1024, 2048, 4096 };
		e.maxCounters = ctrs[size <= 20 ? r.Below(4) : 1 + r.Below(3)];
		e.nDed = r.Range(0, 3);
		e.nLim = r.Range(2, 5);
		e.maxLimiters = (uint32_t)(e.nLim + 1 + (int)r.Below(4));
		e.drain = r.Chance(0.2);
		e.seed = r.Fork();
		const int nops = r.Range(2, 3 + std::min(150, size));
		static const int weights[kOpCount] = { 16, 14, 8, 6, 4, 3, 3, 3, 4, 3, 2, 1, 1, 3 };
		int total = 0;
		for (int k : weights)
			total += k;
		for (int k = 0; k < nops; k++)
		{
			OpSpec o;
			int pick = (int)r.Below((uint32_t)total);
			for (o.kind = 0; pick >= weights[o.kind]; o.kind++)
				pick -= weights[o.kind];
			o.seed = r.Fork();
			o.size = r.Range(1, std::max(1, size));
			o.join = r.Chance(0.25) ? -1 : (int)r.Below(4);
			if (e.drain && o.kind == kOpDag && r.Chance(0.5))
				o.join = -2;
			e.ops.push_back(o);
		}
		p.epochs.push_back(std::move(e));
	}
	return p;
}

bool g_Verbose = false;
// --block-only: some waits are BlockOnly. The rules allow one only when
// another thread is free to run what it waits for, and nothing here can
// promise that: the free worker may itself be deep in waits, where its
// ParallelFor runs inline, or isolated, and then nothing runs the
// waited-for work.
// So these runs can hang without loom being wrong; they are off by default,
// and a hang in one is worth reading, not a failure.
bool g_BlockOnly = false;

void ProbeDedicated(Ded& d, int slot)
{
	loom::ScopedCounter c(1);
	loom::TaskDesc desc;
	desc.affinity = Affinity::Dedicated;
	desc.thread = d.id;
	desc.signals = c;
	struct P { Ded* d; int slot; };
	P* p = new P{ &d, slot };
	loom::Submit([p] {
		p->d->tid = std::this_thread::get_id();
		t_DedSlot = p->slot;
		delete p;
	}, desc);
	loom::WaitFor(c);
}

void AddDedicated(Env& env, Rng& r)
{
	auto d = std::make_unique<Ded>();
	const int k = (int)r.Below(3);
	if (k == 0)
	{
		d->lane = true;
		d->id = loom::AcquireLane("fz-lane");
	}
	else
	{
		d->steal = k == 2;
		d->id = loom::RegisterDedicatedThread({ .name = "fz-dedicated",
			.idle = d->steal ? loom::DedicatedConfig::Idle::StealFromPool : loom::DedicatedConfig::Idle::Sleep });
	}
	CHECK(env.dedIdsSeen.insert((uint32_t)d->id).second, "a dedicated id was handed out twice in one pool");
	// In the table before its probe: once the thread knows its slot it may run pool work that looks it up.
	env.ded.push_back(std::move(d));
	ProbeDedicated(*env.ded.back(), (int)env.ded.size() - 1);
}

// Unregisters slot `i` with a burst of work queued for it: everything must have run when it returns.
void RemoveDedicatedHere(Env& env, int i, Rng& r)
{
	Ded& d = *env.ded[(size_t)i];
	const int n = r.Range(0, 20);
	struct Burst
	{
		std::atomic<int> runs{ 0 };
		std::atomic<int> more{ 0 };
	};
	auto burst = std::make_unique<Burst>();
	Burst* b = burst.get();
	loom::ScopedCounter gate(1);
	const bool gated = r.Chance(0.5);
	loom::DedicatedId id = d.id;
	for (int k = 0; k < n; k++)
	{
		loom::TaskDesc desc;
		desc.affinity = Affinity::Dedicated;
		desc.thread = d.id;
		if (gated && r.Chance(0.5))
			desc.waitsFor = gate;
		const bool spawn = r.Chance(0.2);
		loom::Submit([b, spawn, id] {
			b->runs++;
			if (spawn)
			{
				// One of its tasks submits more to it while it is unregistered.
				loom::TaskDesc more;
				more.affinity = Affinity::Dedicated;
				more.thread = id;
				b->more++;
				loom::Submit([b] { b->runs++; }, more);
			}
		}, desc);
	}
	std::unique_ptr<Outside> opener;
	if (gated)
		opener = std::make_unique<Outside>([&gate] {
			std::this_thread::sleep_for(std::chrono::microseconds(500));
			loom::Signal(gate);
		});
	else
		loom::Signal(gate);
	if (d.lane)
		loom::ReleaseLane(d.id);
	else
		loom::UnregisterDedicatedThread(d.id);
	CHECK(b->runs.load() == n + b->more.load(), "UnregisterDedicatedThread returned with %d of %d tasks run", b->runs.load(), n + b->more.load());
	if (opener)
		opener->Join();
	d.alive = false;
}

void RemoveDedicated(Env& env, int i, Rng& r)
{
	// From the main thread, or from an outside thread while the main thread
	// goes on.
	if (r.Chance(0.5))
		return RemoveDedicatedHere(env, i, r);
	const uint64_t seed = r.Fork();
	Outside t([&env, i, seed] {
		Rng tr(seed);
		RemoveDedicatedHere(env, i, tr);
	});
	t.Join();
}

void DoPark(Rng& r)
{
	struct Participant
	{
		std::mutex mu;
		std::condition_variable cv;
		bool woken = false, stop = false;
		int id = 0;
		std::atomic<int> checkIns{ 0 };
		std::thread thread;
	};
	std::vector<std::unique_ptr<Participant>> parts;
	const int np = r.Range(0, 2);
	for (int i = 0; i < np; i++)
	{
		auto p = std::make_unique<Participant>();
		Participant* pp = p.get();
		p->thread = std::thread([pp] {
			t_Role = Role::Outside;
			std::unique_lock<std::mutex> lock(pp->mu);
			while (!pp->stop)
			{
				pp->cv.wait_for(lock, std::chrono::milliseconds(1), [&] { return pp->woken || pp->stop; });
				pp->woken = false;
				if (loom::IsParkRequested())
				{
					lock.unlock();
					pp->checkIns++;
					loom::ParkPoint("fz-participant");
					lock.lock();
				}
			}
		});
		p->id = loom::RegisterParkParticipant([](void* u) {
			auto* q = static_cast<Participant*>(u);
			std::lock_guard<std::mutex> lock(q->mu);
			q->woken = true;
			q->cv.notify_one();
		}, pp);
		parts.push_back(std::move(p));
	}
	const uint32_t timeout = (uint32_t)r.Range(5, 300);
	const bool ok = loom::ParkAll(timeout);
	if (ok)
	{
		for (int k = 0; k < 3; k++)
		{
			CHECK(g_PoolRunning.load() == 0, "%d task bodies running on pool threads while parked", g_PoolRunning.load());
			std::this_thread::sleep_for(std::chrono::microseconds(300));
		}
		std::vector<loom::ParkedThread> parked = loom::ParkedThreads();
		int alive = 0;
		for (auto& d : g_Env->ded)
			alive += d->alive;
		CHECK((int)parked.size() >= g_Env->workers + alive, "ParkedThreads lists %zu, want at least %d workers + %d dedicated", parked.size(),
			g_Env->workers, alive);
		loom::UnparkAll();
	}
	for (auto& p : parts)
	{
		loom::UnregisterParkParticipant(p->id);
		{
			std::lock_guard<std::mutex> lock(p->mu);
			p->stop = true;
			p->cv.notify_one();
		}
		p->thread.join();
	}
}

// A big fan of tasks waiting on one gate: grows the task-slot pool past maxTasks.
void DoBigFan(Rng& r, int op)
{
	const int n = 4000 + (int)r.Below(4000);
	loom::ScopedCounter gate(1), done(n);
	std::atomic<int> ran{ 0 }, early{ 0 };
	std::atomic<bool> open{ false };
	std::vector<loom::TaskDesc> batch;
	struct Arg { std::atomic<int>* ran; std::atomic<int>* early; std::atomic<bool>* open; };
	Arg arg{ &ran, &early, &open };
	for (int i = 0; i < n; i++)
	{
		loom::TaskDesc d;
		d.fn = [](void* p) {
			auto* a = static_cast<Arg*>(p);
			if (!a->open->load())
				(*a->early)++;
			(*a->ran)++;
		};
		d.userData = &arg;
		d.signals = done;
		d.waitsFor = gate;
		d.priority = (Priority)(i % 3);
		batch.push_back(d);
	}
	loom::Submit(std::span<const loom::TaskDesc>(batch));
	CHECK(loom::LiveTaskCount() >= n, "LiveTaskCount %d with %d tasks waiting", loom::LiveTaskCount(), n);
	open.store(true);
	loom::Signal(gate);
	loom::WaitFor(done);
	CHECK(ran.load() == n && early.load() == 0, "big fan: %d of %d ran, %d before the gate (op %d)", ran.load(), n, early.load(), op);
}

void RunEpoch(const EpochSpec& spec, int epochIndex)
{
	Rng er(spec.seed);
	loom::PoolConfig cfg;
	cfg.workerCount = spec.workers;
	cfg.maxTasks = spec.maxTasks;
	cfg.maxCounters = spec.maxCounters;
	cfg.maxLimiters = spec.maxLimiters;
	cfg.shutdownTimeoutMs = 30000;
	std::optional<loom::Pool> pool;
	pool.emplace(cfg);
	CHECK(loom::HasActivePool(), "no active pool after the Pool ctor");
	CHECK(loom::WorkerCount() == spec.workers, "WorkerCount %d, want %d", loom::WorkerCount(), spec.workers);
	Env env;
	env.workers = spec.workers;
	env.ded.reserve(64); // tasks read it while the main thread registers more: it never moves
	g_Env = &env;
	for (int i = 0; i < spec.nLim; i++)
	{
		auto l = std::make_unique<Lim>();
		l->leaf = i < 2;
		const uint32_t limit = (uint32_t)er.Range(1, 6);
		l->h = loom::CreateLimiter(limit);
		l->log.push_back({ Tick(), limit });
		l->maxEver = l->current = limit;
		env.lims.push_back(std::move(l));
	}
	if (spec.workers >= 2)
	{
		auto l = std::make_unique<Lim>();
		l->blockers = true;
		const uint32_t limit = (uint32_t)spec.workers - 1;
		l->h = loom::CreateLimiter(limit);
		l->log.push_back({ Tick(), limit });
		l->maxEver = l->current = limit;
		env.blockers = l.get();
		env.lims.push_back(std::move(l));
	}
	for (int i = 0; i < spec.nDed; i++)
		AddDedicated(env, er);

	std::vector<std::unique_ptr<Pending>> pending;
	auto joinOne = [&](std::unique_ptr<Pending>& p) {
		if (g_Verbose)
			std::fprintf(stderr, "  join op %d\n", p->op);
		p->Join();
		for (int u : p->users)
			env.ded[(size_t)u]->users--;
		p.reset();
	};
	for (size_t k = 0; k < spec.ops.size(); k++)
	{
		const OpSpec& o = spec.ops[k];
		const int opIndex = epochIndex * 1000 + (int)k;
		env.opIndex = opIndex;
		if (g_Verbose)
			std::fprintf(stderr, "epoch %d op %zu: %s seed %llu size %d join %d\n", epochIndex, k, kOpNames[o.kind], (unsigned long long)o.seed,
				o.size, o.join);
		Rng r(o.seed);
		std::unique_ptr<Pending> started;
		switch (o.kind)
		{
		case kOpDag:
		{
			auto d = std::make_unique<DagOp>();
			d->op = opIndex;
			d->Generate(r, o.size);
			d->Start();
			started = std::move(d);
			break;
		}
		case kOpWork:
		{
			WorkCtx ctx;
			ctx.op = opIndex;
			ctx.canBlock = env.workers >= 1 && g_BlockOnly;
			std::vector<int> alive;
			for (size_t i = 0; i < env.ded.size(); i++)
				if (env.ded[i]->alive)
					alive.push_back((int)i);
			ctx.ded = !alive.empty() && r.Chance(0.5) ? alive[r.Below((uint32_t)alive.size())] : -1;
			RunWork(r.Fork(), o.size, ctx);
			break;
		}
		case kOpFutures:
		{
			auto f = std::make_unique<FuturesOp>();
			f->op = opIndex;
			f->Generate(r, o.size);
			f->Start();
			started = std::move(f);
			break;
		}
		case kOpCoro:
		{
#if defined(LOOM_COROUTINES)
			auto c = std::make_unique<CoroOp>();
			c->op = opIndex;
			c->Start(r, o.size);
			started = std::move(c);
#endif
			break;
		}
		case kOpGraphAsync:
		{
			auto g = std::make_unique<GraphAsyncOp>();
			g->op = opIndex;
			g->Start(r, o.size);
			started = std::move(g);
			break;
		}
		case kOpRegister:
			if (env.ded.size() < 32)
				AddDedicated(env, r);
			break;
		case kOpUnregister:
		{
			std::vector<int> free;
			for (size_t i = 0; i < env.ded.size(); i++)
				if (env.ded[i]->alive && env.ded[i]->users == 0)
					free.push_back((int)i);
			if (!free.empty())
				RemoveDedicated(env, free[r.Below((uint32_t)free.size())], r);
			break;
		}
		case kOpPark: DoPark(r); break;
		case kOpOutside:
		{
			auto t = std::make_unique<OutsideOp>();
			t->op = opIndex;
			t->Start(r, o.size);
			started = std::move(t);
			break;
		}
		case kOpLimits:
		{
			const int n = r.Range(1, 4);
			for (int i = 0; i < n; i++)
			{
				Lim& l = *env.lims[r.Below((uint32_t)env.lims.size())];
				if (!l.blockers)
					l.Set((uint32_t)r.Range(1, 12));
			}
			break;
		}
		case kOpHelp:
		{
			const int n = r.Range(1, 200);
			for (int i = 0; i < n; i++)
			{
				if (r.Chance(0.5))
					loom::TryHelp();
				else
					loom::PumpMainThreadTasks();
			}
			break;
		}
		case kOpDebug:
		{
			std::string text = loom::DebugDump();
			CHECK(!text.empty(), "an empty DebugDump");
			// A worker shows up once it has started: give a new pool a moment.
			loom::PoolSnapshot s = loom::SnapshotPool();
			for (int i = 0; i < 1000 && (int)s.threads.size() < env.workers; i++)
			{
				std::this_thread::sleep_for(std::chrono::milliseconds(1));
				s = loom::SnapshotPool();
			}
			CHECK((int)s.threads.size() >= env.workers, "SnapshotPool lists %zu threads, %d workers", s.threads.size(), env.workers);
			break;
		}
		case kOpBigFan: DoBigFan(r, opIndex); break;
		case kOpOnDedicated:
		{
			// Work run inside a task on a dedicated thread (or a lane).
			std::vector<int> alive;
			for (size_t i = 0; i < env.ded.size(); i++)
				if (env.ded[i]->alive)
					alive.push_back((int)i);
			if (alive.empty())
				break;
			const int slot = alive[r.Below((uint32_t)alive.size())];
			loom::ScopedCounter c(1);
			loom::TaskDesc d;
			d.affinity = Affinity::Dedicated;
			d.thread = env.ded[(size_t)slot]->id;
			d.signals = c;
			const uint64_t seed = r.Fork();
			const int size = o.size;
			loom::Submit([seed, size, slot, opIndex] {
				WorkCtx ctx;
				ctx.op = opIndex;
				ctx.nested = true;
				ctx.ded = slot;
				RunWork(seed, size, ctx);
			}, d);
			std::optional<FuzzError> got = WaitCounter(c, WaitKind::Help, r, "on-dedicated");
			CHECK(!got, "on-dedicated: an error escaped the work");
			break;
		}
		}
		// At most kMostPending operations in flight (the counter budget): join the oldest.
		constexpr size_t kMostPending = 8;
		for (auto& p : pending)
			if (p && p->joinAt != -2 && pending.size() >= kMostPending)
			{
				joinOne(p);
				break;
			}
		std::erase_if(pending, [](auto& p) { return !p; });
		if (started)
		{
			for (int u : started->users)
				env.ded[(size_t)u]->users++;
			started->joinAt = o.join >= 0 ? (int)k + o.join : o.join == -2 && started->CanDrain() ? -2 : -1;
			if (started->joinAt == -2 && std::count_if(pending.begin(), pending.end(), [](auto& p) { return p->joinAt == -2; }) >= 3)
				started->joinAt = -1; // the counter budget again
			pending.push_back(std::move(started));
		}
		for (auto& p : pending)
			if (p && p->joinAt >= 0 && p->joinAt <= (int)k)
				joinOne(p);
		std::erase_if(pending, [](auto& p) { return !p; });
	}
	// The epoch's end: join the rest (or leave them to Shutdown's drain).
	for (auto& p : pending)
		if (p && p->joinAt != -2)
			joinOne(p);
	std::erase_if(pending, [](auto& p) { return !p; });
	const bool draining = !pending.empty();
	for (auto& p : pending)
	{
		// Manual signals still to come are given now; the drain runs the rest.
		auto* d = static_cast<DagOp*>(p.get());
		for (auto& c : d->ctrs)
			if (c->manual > 0 && c->manualWhen == 1)
			{
				c->finished.fetch_add(c->manual);
				loom::Signal(c->h, c->manual);
			}
		if (d->signaller)
			d->signaller->Join();
	}
	if (!draining)
	{
		// Nothing should be live now but the last bookkeeping, and the skipped
		// continuations of dropped futures (main-thread ones need this thread).
		const auto deadline = Clock::now() + std::chrono::seconds(5);
		HelpUntil([&] { return (loom::LiveTaskCount() == 0 && loom::PendingTaskCount() == 0) || Clock::now() > deadline; });
		CHECK(loom::LiveTaskCount() == 0, "LiveTaskCount %d after every operation was joined\n%s", loom::LiveTaskCount(),
			loom::LiveTaskCount() ? loom::DebugDump().c_str() : "");
		CHECK(loom::PendingTaskCount() == 0, "PendingTaskCount %d after every operation was joined", loom::PendingTaskCount());
		for (auto& l : env.lims)
		{
			CHECK(loom::HeldUnits(l->h) == 0, "limiter %d (%s) holds %u units with nothing live; limits %s", (int)(&l - &env.lims[0]),
				l->blockers ? "blockers" : l->leaf ? "leaf" : "outer", loom::HeldUnits(l->h), l->History().c_str());
			CHECK(loom::WaitingTasks(l->h) == 0, "limiter has %u waiting tasks with nothing live", loom::WaitingTasks(l->h));
			CHECK(loom::Limit(l->h) == l->current, "Limit %u, last set %u", loom::Limit(l->h), l->current);
			CHECK(l->running.load() == 0, "limiter model counts %d running", l->running.load());
			if (er.Chance(0.7))
				loom::DestroyLimiter(l->h);
		}
	}
	// Lanes must be released before Shutdown; the other dedicated threads are joined by it, or unregistered here.
	for (size_t i = 0; i < env.ded.size(); i++)
		if (env.ded[i]->alive && (env.ded[i]->lane || er.Chance(0.3)))
		{
			RemoveDedicated(env, (int)i, er);
		}
	if (er.Chance(0.5))
		pool->Shutdown();
	pool.reset();
	CHECK(!loom::HasActivePool(), "a pool is active after Shutdown");
	CHECK(loom::LiveTaskCount() == 0, "LiveTaskCount %d with no pool", loom::LiveTaskCount());
	for (auto& p : pending)
		p->AfterDrain();
	pending.clear();
	g_Env = nullptr;
}

void RunProgram(const ProgramSpec& p)
{
	for (size_t i = 0; i < p.epochs.size(); i++)
	{
		if (g_Verbose)
			std::fprintf(stderr, "epoch %zu: %d workers, maxTasks %u, maxCounters %u, %d dedicated, %d limiters, drain %d\n", i,
				p.epochs[i].workers, p.epochs[i].maxTasks, p.epochs[i].maxCounters, p.epochs[i].nDed, p.epochs[i].nLim, p.epochs[i].drain);
		RunEpoch(p.epochs[i], (int)i);
	}
}

} // namespace fz

int main(int argc, char** argv)
{
	using namespace fz;
	uint64_t seed = 1;
	int size = 10;
	double timeout = 120;
	bool print = false;
	std::string program;
	for (int i = 1; i < argc; i++)
	{
		auto value = [&]() -> const char* {
			if (i + 1 >= argc)
			{
				std::fprintf(stderr, "%s needs a value\n", argv[i]);
				std::exit(2);
			}
			return argv[++i];
		};
		if (!std::strcmp(argv[i], "--seed"))
			seed = std::strtoull(value(), nullptr, 10);
		else if (!std::strcmp(argv[i], "--size"))
			size = std::atoi(value());
		else if (!std::strcmp(argv[i], "--program"))
			program = value();
		else if (!std::strcmp(argv[i], "--timeout"))
			timeout = std::atof(value());
		else if (!std::strcmp(argv[i], "--print"))
			print = true;
		else if (!std::strcmp(argv[i], "--verbose"))
			g_Verbose = true;
		else if (!std::strcmp(argv[i], "--block-only"))
			g_BlockOnly = true;
		else
		{
			std::fprintf(stderr, "usage: %s [--seed N] [--size N] [--program TEXT] [--timeout S] [--print] [--verbose] [--block-only]\n", argv[0]);
			return 2;
		}
	}
	ProgramSpec spec;
	if (!program.empty())
	{
		if (!Parse(program, spec))
		{
			std::fprintf(stderr, "cannot parse the program\n");
			return 2;
		}
	}
	else
		spec = Generate(seed, size);
	g_ProgramText = Print(spec);
	if (print)
	{
		std::printf("%s\n", g_ProgramText.c_str());
		return 0;
	}
	t_Role = Role::Main;
	g_MainTid = std::this_thread::get_id();
#if defined(__linux__)
	prctl(PR_SET_PTRACER, PR_SET_PTRACER_ANY, 0, 0, 0); // lets FUZZ_GDB attach
	if (const char* g = std::getenv("FUZZ_GDB"); g && *g == '1')
	{
		// A crash: every thread's stack too.
		auto onCrash = [](int sig) {
			char cmd[256];
			std::snprintf(cmd, sizeof cmd, "gdb -p %d -batch -ex 'thread apply all bt 50' 1>&2 2>/dev/null", (int)getpid());
			(void)std::system(cmd);
			std::signal(sig, SIG_DFL);
			std::raise(sig);
		};
		std::signal(SIGSEGV, onCrash);
		std::signal(SIGBUS, onCrash);
	}
#endif

	// A hang aborts with the program and a debug dump.
	std::mutex wdMutex;
	std::condition_variable wdCv;
	bool finished = false;
	std::thread watchdog([&] {
		std::unique_lock<std::mutex> lock(wdMutex);
		if (!wdCv.wait_for(lock, std::chrono::duration<double>(timeout), [&] { return finished; }))
		{
			std::fprintf(stderr, "FUZZ HANG: still running after %.0f s\nprogram: %s\n", timeout, g_ProgramText.c_str());
			if (loom::HasActivePool())
				std::fprintf(stderr, "%s\n", loom::DebugDump().c_str());
			std::fflush(stderr);
#if defined(__linux__)
			// FUZZ_GDB=1: every thread's stack, from a gdb attached to this process.
			if (const char* g = std::getenv("FUZZ_GDB"); g && *g == '1')
			{
				char cmd[1024];
				std::snprintf(cmd, sizeof cmd, "gdb -p %d -batch -ex 'thread apply all bt 50' 1>&2 2>/dev/null", (int)getpid());
				(void)std::system(cmd);
			}
#endif
			std::abort();
		}
	});
	RunProgram(spec);
	{
		std::lock_guard<std::mutex> lock(wdMutex);
		finished = true;
	}
	wdCv.notify_all();
	watchdog.join();
	if (g_Failures.load() != 0)
	{
		std::fprintf(stderr, "FUZZ FAILED: %d checks; program: %s\n", g_Failures.load(), g_ProgramText.c_str());
		return 1;
	}
	return 0;
}
