// loom_stress: random mixes of loom's features on random pools, for as long
// as asked. Each round builds a pool of a random size, with dedicated threads
// some of the time, and runs a random set of scenarios at once: dependency
// graphs of tasks across every affinity, tree walks that add to their
// counter, cancellation, exceptions, limiters whose limit changes under
// load, future chains, coroutines, pipelines, task graphs, the parallel
// algorithms, isolated regions and timed waits. Each scenario checks what
// loom promises about it (every task once, in dependency order, never more
// than a limit at once, the right value or error), and its shape comes from
// its own seed, so a failure names the seed that made it.
//
//   loom_stress [--seconds N] [--rounds N] [--seed N] [--quiet]
//
// Runs until either limit is reached (10 seconds by default) and returns
// non-zero if a check failed. A round that takes over a minute is taken to
// be stuck: the debug dump is printed and the program aborts.
#include "loom/loom.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

using Rng = std::mt19937_64;
using Clock = std::chrono::steady_clock;

std::atomic<int> g_Failures{ 0 };
uint64_t g_RoundSeed = 0;
// More than a round ever has alive at once (16 workers, 2 dedicated threads,
// the watchdog, main and a few outside threads per scenario), and far fewer
// than a long run starts.
constexpr int kMostThreadsAlive = 256;

// What a scenario may use: the round's dedicated threads, if any.
struct Env
{
	std::vector<loom::DedicatedId> dedicated;
};

void Fail(const char* scenario, uint64_t seed, const char* what, long got = 0, long want = 0)
{
	g_Failures++;
	std::fprintf(stderr, "FAILED: %s (round seed %llu, scenario seed %llu): %s (got %ld, want %ld)\n", scenario,
		(unsigned long long)g_RoundSeed, (unsigned long long)seed, what, got, want);
}

#define STRESS_CHECK(cond, what, got, want) \
	do { if (!(cond)) Fail(kName, seed, what, (long)(got), (long)(want)); } while (0)

int Pick(Rng& rng, int lo, int hi) // inclusive
{
	return std::uniform_int_distribution<int>(lo, hi)(rng);
}

bool Chance(Rng& rng, double p)
{
	return std::uniform_real_distribution<double>(0.0, 1.0)(rng) < p;
}

uint64_t Mix(uint64_t x)
{
	x += 0x9e3779b97f4a7c15ull;
	x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
	x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
	return x ^ (x >> 31);
}

// A random affinity a task that does not wait may have.
loom::TaskDesc RandomPlacement(Rng& rng, const Env& env)
{
	loom::TaskDesc d;
	const int r = Pick(rng, 0, 19);
	if (r == 0)
		d.affinity = loom::Affinity::MainThread;
	else if (r == 1 && !env.dedicated.empty())
	{
		d.affinity = loom::Affinity::Dedicated;
		d.thread = env.dedicated[(size_t)Pick(rng, 0, (int)env.dedicated.size() - 1)];
	}
	const int p = Pick(rng, 0, 9);
	d.priority = p == 0 ? loom::Priority::High : p == 1 ? loom::Priority::Low : loom::Priority::Normal;
	return d;
}

// ---- Scenarios -------------------------------------------------------------------

// Tasks with random dependencies on earlier tasks, every affinity, some
// limited: each runs once, after all it waits for.
void TaskDag(Rng& rng, uint64_t seed, const Env& env)
{
	static constexpr const char* kName = "task dag";
	const int n = Pick(rng, 1, 250);
	struct Node
	{
		std::atomic<int> runs{ 0 };
		std::vector<int> preds;
		loom::CounterHandle done = loom::kInvalidCounter;
		std::vector<loom::CounterHandle> waits;
	};
	std::vector<Node> nodes((size_t)n);
	std::atomic<int> orderErrors{ 0 };
	loom::ScopedLimiter limiter((uint32_t)Pick(rng, 1, 4));
	loom::ScopedCounter all(n);
	std::vector<loom::TaskDesc> descs;
	for (int i = 0; i < n; i++)
	{
		Node& node = nodes[(size_t)i];
		node.done = loom::CreateCounter(1);
		const int preds = i == 0 ? 0 : Pick(rng, 0, std::min(i, 4));
		for (int k = 0; k < preds; k++)
		{
			const int p = Pick(rng, std::max(0, i - 40), i - 1);
			node.preds.push_back(p);
			node.waits.push_back(nodes[(size_t)p].done);
		}
	}
	// Submitted in one go or one by one, in a random order: a task may be
	// submitted before the tasks it waits for.
	std::vector<int> order((size_t)n);
	std::iota(order.begin(), order.end(), 0);
	std::shuffle(order.begin(), order.end(), rng);
	const bool batch = Chance(rng, 0.5);
	struct Run
	{
		Node* node;
		std::vector<Node>* nodes;
		std::atomic<int>* orderErrors;
	};
	std::vector<Run> runs((size_t)n);
	for (int i : order)
	{
		Node& node = nodes[(size_t)i];
		runs[(size_t)i] = Run{ &node, &nodes, &orderErrors };
		loom::TaskDesc d = RandomPlacement(rng, env);
		d.fn = [](void* p) {
			Run& r = *static_cast<Run*>(p);
			for (int pred : r.node->preds)
				if ((*r.nodes)[(size_t)pred].runs.load(std::memory_order_acquire) != 1)
					(*r.orderErrors)++;
			r.node->runs.fetch_add(1, std::memory_order_acq_rel);
			loom::Signal(r.node->done);
		};
		d.userData = &runs[(size_t)i];
		d.signals = all; // through the runner: the unit is back before the wait returns
		d.waitsForAll = node.waits;
		if (Chance(rng, 0.2))
			d.limiter = limiter;
		if (batch)
			descs.push_back(d);
		else
			loom::Submit(d);
	}
	if (batch)
		loom::Submit(std::span<const loom::TaskDesc>(descs));
	loom::WaitFor(all);
	for (int i = 0; i < n; i++)
		STRESS_CHECK(nodes[(size_t)i].runs.load() == 1, "a task ran other than once", nodes[(size_t)i].runs.load(), 1);
	STRESS_CHECK(orderErrors.load() == 0, "a task ran before one it waits for", orderErrors.load(), 0);
	STRESS_CHECK(loom::HeldUnits(limiter) == 0, "a limiter unit still held after the wait", loom::HeldUnits(limiter), 0);
	for (Node& node : nodes)
		loom::DestroyCounter(node.done);
}

// A tree walk that adds a count for each child before submitting it: one
// counter and one wait cover every node.
struct TreeWalk
{
	loom::CounterHandle done;
	std::atomic<long> visited{ 0 };
	uint64_t seed;
	int maxDepth;
};

long TreeSize(uint64_t node, int depth, const TreeWalk& w)
{
	long n = 1;
	const int children = depth >= w.maxDepth ? 0 : (int)(Mix(node ^ w.seed) % 4);
	for (int c = 0; c < children; c++)
		n += TreeSize(node * 4 + (uint64_t)c + 1, depth + 1, w);
	return n;
}

void VisitTree(TreeWalk& w, uint64_t node, int depth)
{
	w.visited++;
	const int children = depth >= w.maxDepth ? 0 : (int)(Mix(node ^ w.seed) % 4);
	for (int c = 0; c < children; c++)
	{
		if (!loom::AddToCounter(w.done))
			std::abort(); // a running task that signals the counter always adds
		const uint64_t child = node * 4 + (uint64_t)c + 1;
		loom::Submit([&w, child, depth] { VisitTree(w, child, depth + 1); }, { .signals = w.done });
	}
}

void Tree(Rng& rng, uint64_t seed, const Env&)
{
	static constexpr const char* kName = "tree walk";
	TreeWalk w;
	w.seed = seed;
	w.maxDepth = Pick(rng, 1, 7);
	loom::ScopedCounter done(1);
	w.done = done;
	loom::Submit([&w] { VisitTree(w, 0, 0); }, { .signals = done });
	loom::WaitFor(done);
	const long want = TreeSize(0, 0, w);
	STRESS_CHECK(w.visited.load() == want, "the walk did not visit every node once", w.visited.load(), want);
	STRESS_CHECK(!loom::AddToCounter(done), "a counter at zero took an add", 1, 0);
}

// Cancel at a random moment: no task runs twice, at most all run, and the
// wait returns. Without the cancel, every task runs.
void Cancellation(Rng& rng, uint64_t seed, const Env& env)
{
	static constexpr const char* kName = "cancellation";
	const int n = Pick(rng, 1, 300);
	const bool cancel = Chance(rng, 0.7);
	loom::ScopedCounter work(n);
	loom::ScopedCounter gate(1);
	std::vector<std::atomic<int>> runs((size_t)n);
	const int cancelAt = Pick(rng, 0, n);
	for (int i = 0; i < n; i++)
	{
		loom::TaskDesc d = RandomPlacement(rng, env);
		d.signals = work;
		if (Chance(rng, 0.5))
			d.waitsFor = gate;
		auto* slot = &runs[(size_t)i];
		loom::Submit([slot, &work, i, cancelAt, cancel] {
			slot->fetch_add(1);
			if (cancel && i == cancelAt)
				loom::Cancel(work);
			if (loom::IsCancelled(work))
				return; // a long task stops early
		}, d);
	}
	if (cancel && cancelAt == n)
		loom::Cancel(work);
	loom::Signal(gate);
	loom::WaitFor(work);
	int ran = 0;
	for (auto& r : runs)
	{
		STRESS_CHECK(r.load() <= 1, "a task ran twice", r.load(), 1);
		ran += r.load();
	}
	if (!cancel)
		STRESS_CHECK(ran == n, "an uncancelled counter's task was skipped", ran, n);
	STRESS_CHECK(loom::IsCancelled(work) == cancel, "IsCancelled is wrong", loom::IsCancelled(work), cancel);
}

#if defined(__cpp_exceptions)
// Some tasks throw: the wait rethrows one of their exceptions, or none.
void Exceptions(Rng& rng, uint64_t seed, const Env&)
{
	static constexpr const char* kName = "exceptions";
	const int n = Pick(rng, 1, 200);
	const int throwers = Chance(rng, 0.3) ? 0 : Pick(rng, 1, std::min(n, 5));
	std::vector<int> throws((size_t)n, 0);
	for (int k = 0; k < throwers; k++)
		throws[(size_t)Pick(rng, 0, n - 1)] = 1;
	const int thrown = (int)std::count(throws.begin(), throws.end(), 1);
	loom::ScopedCounter c(n);
	for (int i = 0; i < n; i++)
	{
		const bool t = throws[(size_t)i] != 0;
		loom::Submit([t, i] {
			if (t)
				throw std::runtime_error(std::to_string(i));
		}, { .signals = c });
	}
	bool caught = false;
	int which = -1;
	try
	{
		loom::WaitFor(c);
	}
	catch (const std::runtime_error& e)
	{
		caught = true;
		which = std::atoi(e.what());
	}
	STRESS_CHECK(caught == (thrown > 0), "WaitFor's rethrow does not match the throwers", caught, thrown > 0);
	if (caught)
		STRESS_CHECK(which >= 0 && which < n && throws[(size_t)which], "WaitFor rethrew an exception no task threw", which, -1);
	STRESS_CHECK(loom::IsCancelled(c) == (thrown > 0), "a throw did not cancel its counter", loom::IsCancelled(c), thrown > 0);
}
#endif

// Tasks of one limiter, whose limit changes while they run: never more of
// them at once than the highest limit set, each running one with a unit
// index no other running one has, and no unit held after the wait.
void Limits(Rng& rng, uint64_t seed, const Env& env)
{
	static constexpr const char* kName = "limiter";
	const int n = Pick(rng, 1, 300);
	const uint32_t first = (uint32_t)Pick(rng, 1, 6);
	loom::ScopedLimiter limiter(first);
	std::atomic<uint32_t> highest{ first };
	std::atomic<int> inside{ 0 }, most{ 0 }, unitErrors{ 0 };
	std::vector<std::atomic<int>> units(16);
	loom::ScopedCounter done(n);
	std::vector<uint32_t> changes((size_t)n, 0);
	for (int k = Pick(rng, 0, 4); k > 0; k--)
		changes[(size_t)Pick(rng, 0, n - 1)] = (uint32_t)Pick(rng, 1, 6);
	for (int i = 0; i < n; i++)
	{
		loom::TaskDesc d = RandomPlacement(rng, env);
		d.signals = done;
		d.limiter = limiter;
		const uint32_t change = changes[(size_t)i];
		const int spin = Pick(rng, 0, 2000);
		loom::Submit([&, change, spin] {
			const int now = ++inside;
			int seen = most.load();
			while (now > seen && !most.compare_exchange_weak(seen, now)) {}
			const uint32_t unit = loom::CurrentLimiterUnit();
			if (unit >= units.size() || units[unit].fetch_add(1) != 0)
				unitErrors++;
			for (volatile int s = 0; s < spin; s = s + 1) {}
			if (change)
			{
				uint32_t h = highest.load();
				while (change > h && !highest.compare_exchange_weak(h, change)) {}
				loom::SetLimit(limiter, change);
			}
			if (unit < units.size())
				units[unit].fetch_sub(1);
			inside--;
		}, d);
	}
	loom::WaitFor(done);
	STRESS_CHECK(most.load() <= (int)highest.load(), "more tasks at once than the limit", most.load(), highest.load());
	STRESS_CHECK(unitErrors.load() == 0, "two running tasks shared a unit index", unitErrors.load(), 0);
	STRESS_CHECK(loom::HeldUnits(limiter) == 0, "a unit still held after the wait", loom::HeldUnits(limiter), 0);
	STRESS_CHECK(loom::WaitingTasks(limiter) == 0, "a task still waiting for a unit", loom::WaitingTasks(limiter), 0);
}

// A chain of Async and Then steps, some on the main thread or limited,
// from a promise set by an outside thread some of the time, with an error
// at a random step: the end of the chain sees the value or the first error.
void FutureChain(Rng& rng, uint64_t seed, const Env& env)
{
	static constexpr const char* kName = "future chain";
	const int steps = Pick(rng, 1, 12);
#if defined(__cpp_exceptions)
	const int failAt = Chance(rng, 0.3) ? Pick(rng, 0, steps) : -1;
#else
	const int failAt = -1;
#endif
	loom::ScopedLimiter limiter((uint32_t)Pick(rng, 1, 3));
	const bool fromPromise = Chance(rng, 0.3);
	loom::Promise<long> promise;
	loom::Future<long> f;
	if (fromPromise)
		f = promise.GetFuture();
	else
		f = loom::Async([failAt] {
#if defined(__cpp_exceptions)
			if (failAt == 0)
				throw std::runtime_error("0");
#endif
			return 1L;
		});
	for (int s = 1; s <= steps; s++)
	{
		loom::TaskDesc d = RandomPlacement(rng, env);
		if (Chance(rng, 0.2))
			d.limiter = limiter;
		f = f.Then([s, failAt](long& v) {
#if defined(__cpp_exceptions)
			if (s == failAt)
				throw std::runtime_error(std::to_string(s));
#else
			(void)failAt;
#endif
			return v * 3 + s;
		}, d);
	}
	std::thread setter;
	if (fromPromise)
	{
		const int delayUs = Pick(rng, 0, 500);
		setter = std::thread([&promise, delayUs, failAt] {
			std::this_thread::sleep_for(std::chrono::microseconds(delayUs));
#if defined(__cpp_exceptions)
			if (failAt == 0)
			{
				promise.SetException(std::make_exception_ptr(std::runtime_error("0")));
				return;
			}
#endif
			promise.SetValue(1L);
		});
	}
	long want = 1;
	for (int s = 1; s <= steps; s++)
		want = want * 3 + s;
#if defined(__cpp_exceptions)
	try
	{
		const long got = f.Get();
		STRESS_CHECK(failAt < 0, "a chain with a failing step returned a value", got, -1);
		STRESS_CHECK(got == want, "the chain's value is wrong", got, want);
	}
	catch (const std::runtime_error& e)
	{
		STRESS_CHECK(std::atoi(e.what()) == failAt, "the chain's error came from the wrong step", std::atoi(e.what()), failAt);
	}
#else
	STRESS_CHECK(f.Get() == want, "the chain's value is wrong", f.Get(), want);
#endif
	if (setter.joinable())
		setter.join();
}

#if defined(LOOM_COROUTINES)
struct CoroutineEnv
{
	const Env* env;
	loom::LimiterHandle limiter;
	uint64_t seed;
	std::atomic<int> limitErrors{ 0 };
	std::atomic<int> inside{ 0 };
	uint32_t limit;
};

// A random tree of coroutines: each awaits its children, a task's value,
// and hops between the main thread, dedicated threads and a limiter.
loom::Future<long> CoroutineNode(CoroutineEnv& ce, uint64_t node, int depth)
{
	const uint64_t h = Mix(node ^ ce.seed);
	switch (h % 5)
	{
	case 0:
		co_await loom::ResumeOn();
		break;
	case 1:
		co_await loom::ResumeOn({ .affinity = loom::Affinity::MainThread });
		if (loom::CurrentThreadIndex() != 0)
			ce.limitErrors++;
		break;
	case 2:
		if (!ce.env->dedicated.empty())
			co_await loom::ResumeOn({ .affinity = loom::Affinity::Dedicated, .thread = ce.env->dedicated[(h >> 8) % ce.env->dedicated.size()] });
		break;
	case 3:
	{
		co_await loom::ResumeOn({ .limiter = ce.limiter });
		const int now = ++ce.inside;
		if (now > (int)ce.limit || loom::CurrentLimiterUnit() >= ce.limit)
			ce.limitErrors++;
		ce.inside--;
		co_await loom::ResumeOn(); // leave the limiter before waiting on children
		break;
	}
	default:
		break;
	}
	long sum = (long)(node % 7);
	sum += co_await loom::Async([node] { return (long)(node % 3); });
	const int children = depth >= 5 ? 0 : (int)((h >> 16) % 4);
	std::vector<loom::Future<long>> kids;
	for (int c = 0; c < children; c++)
		kids.push_back(CoroutineNode(ce, node * 4 + (uint64_t)c + 1, depth + 1));
	for (loom::Future<long>& k : kids)
		sum += co_await k;
	co_return sum;
}

long CoroutineExpected(uint64_t seed, uint64_t node, int depth)
{
	const uint64_t h = Mix(node ^ seed);
	long sum = (long)(node % 7) + (long)(node % 3);
	const int children = depth >= 5 ? 0 : (int)((h >> 16) % 4);
	for (int c = 0; c < children; c++)
		sum += CoroutineExpected(seed, node * 4 + (uint64_t)c + 1, depth + 1);
	return sum;
}

void Coroutines(Rng& rng, uint64_t seed, const Env& env)
{
	static constexpr const char* kName = "coroutines";
	CoroutineEnv ce;
	ce.env = &env;
	ce.seed = seed;
	ce.limit = (uint32_t)Pick(rng, 1, 4);
	loom::ScopedLimiter limiter(ce.limit);
	ce.limiter = limiter;
	const long got = CoroutineNode(ce, 0, 0).Get();
	const long want = CoroutineExpected(seed, 0, 0);
	STRESS_CHECK(got == want, "the coroutine tree's sum is wrong", got, want);
	STRESS_CHECK(ce.limitErrors.load() == 0, "a coroutine ran outside where it was sent", ce.limitErrors.load(), 0);
	STRESS_CHECK(loom::HeldUnits(limiter) == 0, "a unit still held after Get", loom::HeldUnits(limiter), 0);
}
#endif

// A pipeline of random stages: the serial ones see the items in order, and
// every item passes every stage once.
void Pipelines(Rng& rng, uint64_t seed, const Env& env)
{
	static constexpr const char* kName = "pipeline";
	struct Item
	{
		long index = -1;
		long value = 0;
	};
	const long count = Pick(rng, 0, 400);
	const int stages = Pick(rng, 1, 4);
	loom::Pipeline<Item> p((uint32_t)Pick(rng, 1, 16));
	long next = 0;
	p.Source([&next, count](Item& it) {
		if (next >= count)
			return false;
		it.index = next++;
		it.value = it.index;
		return true;
	});
	loom::ScopedLimiter limiter((uint32_t)Pick(rng, 1, 3));
	std::vector<long> seen((size_t)stages, 0);
	std::atomic<int> orderErrors{ 0 };
	for (int s = 0; s < stages; s++)
	{
		loom::TaskDesc d;
		if (Chance(rng, 0.2))
			d.limiter = limiter;
		if (Chance(rng, 0.15))
			d.affinity = loom::Affinity::MainThread;
		else if (Chance(rng, 0.1) && !env.dedicated.empty())
		{
			d.affinity = loom::Affinity::Dedicated;
			d.thread = env.dedicated[0];
		}
		if (Chance(rng, 0.5))
			p.Serial([&seen, &orderErrors, s](Item& it) {
				if (it.index != seen[(size_t)s]++)
					orderErrors++;
				it.value = it.value * 2 + 1;
			}, d);
		else
			p.Parallel([](Item& it) { it.value = it.value * 2 + 1; }, d);
	}
	long total = 0;
	p.Serial([&total](Item& it) { total += it.value; });
	p.Run();
	long want = 0;
	for (long i = 0; i < count; i++)
	{
		long v = i;
		for (int s = 0; s < stages; s++)
			v = v * 2 + 1;
		want += v;
	}
	STRESS_CHECK(total == want, "the pipeline's total is wrong", total, want);
	STRESS_CHECK(orderErrors.load() == 0, "a serial stage saw an item out of order", orderErrors.load(), 0);
	STRESS_CHECK(loom::HeldUnits(limiter) == 0, "a unit still held after Run", loom::HeldUnits(limiter), 0);
}

// A random graph run several times: each node, each run, after its
// predecessors' run of that same pass.
void Graphs(Rng& rng, uint64_t seed, const Env& env)
{
	static constexpr const char* kName = "task graph";
	const int n = Pick(rng, 1, 60);
	const int passes = Pick(rng, 1, 6);
	std::vector<std::atomic<int>> runs((size_t)n);
	std::vector<std::vector<int>> preds((size_t)n);
	std::atomic<int> orderErrors{ 0 };
	loom::ScopedLimiter limiter((uint32_t)Pick(rng, 1, 3));
	loom::TaskGraph g;
	std::vector<loom::TaskGraph::Node> ids;
	for (int i = 0; i < n; i++)
	{
		loom::TaskDesc d = RandomPlacement(rng, env);
		if (Chance(rng, 0.15))
			d.limiter = limiter;
		ids.push_back(g.Add([&runs, &preds, &orderErrors, i] {
			const int mine = runs[(size_t)i].load(std::memory_order_acquire);
			for (int p : preds[(size_t)i])
				if (runs[(size_t)p].load(std::memory_order_acquire) != mine + 1)
					orderErrors++;
			runs[(size_t)i].fetch_add(1, std::memory_order_acq_rel);
		}, d));
	}
	for (int i = 1; i < n; i++)
		for (int k = Pick(rng, 0, 3); k > 0; k--)
		{
			const int p = Pick(rng, 0, i - 1);
			preds[(size_t)i].push_back(p);
			g.Precede(ids[(size_t)p], ids[(size_t)i]);
		}
	STRESS_CHECK(g.Validate(), "an acyclic graph failed to validate", 0, 1);
	for (int pass = 0; pass < passes; pass++)
		loom::Run(g);
	for (int i = 0; i < n; i++)
		STRESS_CHECK(runs[(size_t)i].load() == passes, "a node ran other than once a pass", runs[(size_t)i].load(), passes);
	STRESS_CHECK(orderErrors.load() == 0, "a node ran before its predecessor", orderErrors.load(), 0);
}

// The parallel algorithms against their serial versions, inside an
// isolated region some of the time.
void Algorithms(Rng& rng, uint64_t seed, const Env&)
{
	static constexpr const char* kName = "algorithms";
	const size_t n = (size_t)Pick(rng, 0, Chance(rng, 0.1) ? 300000 : 20000);
	std::vector<long> v(n);
	for (size_t i = 0; i < n; i++)
		v[i] = (long)(Mix(seed + i) % 1000);
	auto body = [&] {
		const size_t chunk = Chance(rng, 0.5) ? 0 : (size_t)Pick(rng, 1, 5000);
		std::atomic<long> sum{ 0 };
		loom::ParallelFor(0, n, chunk, [&](size_t b, size_t e) {
			long s = 0;
			for (size_t i = b; i < e; i++)
				s += v[i];
			sum += s;
		});
		const long want = std::accumulate(v.begin(), v.end(), 0L);
		STRESS_CHECK(sum.load() == want, "ParallelFor's sum is wrong", sum.load(), want);
		const long reduced = loom::ParallelReduce(size_t(0), n, chunk, 0L,
			[&](size_t b, size_t e) { return std::accumulate(v.begin() + (long)b, v.begin() + (long)e, 0L); },
			[](long a, long b) { return a + b; });
		STRESS_CHECK(reduced == want, "ParallelReduce's sum is wrong", reduced, want);

		std::vector<long> scan(n), serial(n);
		loom::ParallelInclusiveScan(v.begin(), v.end(), scan.begin(), 0L);
		std::inclusive_scan(v.begin(), v.end(), serial.begin());
		STRESS_CHECK(scan == serial, "ParallelInclusiveScan differs from the serial scan", 0, 1);

		std::vector<long> sorted = v;
		loom::ParallelSort(sorted.begin(), sorted.end());
		std::vector<long> expect = v;
		std::sort(expect.begin(), expect.end());
		STRESS_CHECK(sorted == expect, "ParallelSort differs from std::sort", 0, 1);

		loom::PerThread<long> partial;
		std::atomic<int> topIndex{ 0 };
		loom::ParallelFor(0, n, [&](size_t b, size_t e) {
			const int me = loom::CurrentThreadIndex();
			int seen = topIndex.load();
			while (me > seen && !topIndex.compare_exchange_weak(seen, me)) {}
			for (size_t i = b; i < e; i++)
				partial.Local() += v[i];
		});
		long merged = 0;
		partial.ForEach([&](long& x) { merged += x; });
		STRESS_CHECK(merged == want, "PerThread partials do not add up", merged, want);
		// Indices are given back when threads exit, so however many rounds
		// have come and gone they stay below the threads alive at once.
		STRESS_CHECK(topIndex.load() < kMostThreadsAlive, "a thread index past the threads alive at once", topIndex.load(), kMostThreadsAlive);
	};
	if (Chance(rng, 0.3))
		loom::Isolated(body);
	else
		body();
}

// A gate opened by an outside thread after a random delay, waited for with
// short time limits until it opens.
void TimedWaits(Rng& rng, uint64_t seed, const Env&)
{
	static constexpr const char* kName = "timed waits";
	loom::ScopedCounter gate(1);
	const int delayUs = Pick(rng, 0, 3000);
	std::thread opener([&gate, delayUs] {
		std::this_thread::sleep_for(std::chrono::microseconds(delayUs));
		loom::Signal(gate);
	});
	const loom::Help help = Chance(rng, 0.5) ? loom::Help::WhileWaiting : loom::Help::BlockOnly;
	int tries = 0;
	while (!loom::TryWaitFor(gate, (uint32_t)Pick(rng, 0, 2), help))
		tries++;
	opener.join();
	STRESS_CHECK(loom::Count(gate) == 0, "TryWaitFor returned true before the count reached zero", loom::Count(gate), 0);
	(void)tries;
}

struct Scenario
{
	const char* name;
	void (*run)(Rng&, uint64_t, const Env&);
};

const Scenario kScenarios[] = {
	{ "task dag", &TaskDag },
	{ "tree walk", &Tree },
	{ "cancellation", &Cancellation },
#if defined(__cpp_exceptions)
	{ "exceptions", &Exceptions },
#endif
	{ "limiter", &Limits },
	{ "future chain", &FutureChain },
#if defined(LOOM_COROUTINES)
	{ "coroutines", &Coroutines },
#endif
	{ "pipeline", &Pipelines },
	{ "task graph", &Graphs },
	{ "algorithms", &Algorithms },
	{ "timed waits", &TimedWaits },
};
constexpr size_t kScenarioCount = sizeof(kScenarios) / sizeof(kScenarios[0]);

// ---- Rounds ----------------------------------------------------------------------

// Aborts with the debug dump if a round does not finish within a minute.
class Watchdog
{
public:
	Watchdog() : m_Thread([this] { Watch(); }) {}
	~Watchdog()
	{
		{
			std::lock_guard<std::mutex> lock(m_Mutex);
			m_Stop = true;
		}
		m_Wake.notify_all();
		m_Thread.join();
	}
	void RoundStarted()
	{
		std::lock_guard<std::mutex> lock(m_Mutex);
		m_Deadline = Clock::now() + std::chrono::seconds(60);
		m_Active = true;
	}
	void RoundEnded()
	{
		std::lock_guard<std::mutex> lock(m_Mutex);
		m_Active = false;
	}

private:
	void Watch()
	{
		std::unique_lock<std::mutex> lock(m_Mutex);
		while (!m_Stop)
		{
			m_Wake.wait_for(lock, std::chrono::milliseconds(500));
			if (m_Active && Clock::now() > m_Deadline)
			{
				std::fprintf(stderr, "STUCK: round seed %llu has run for over a minute\n%s\n",
					(unsigned long long)g_RoundSeed, loom::DebugDump().c_str());
				std::fflush(stderr);
				std::abort();
			}
		}
	}

	std::mutex m_Mutex;
	std::condition_variable m_Wake;
	bool m_Stop = false;
	bool m_Active = false;
	Clock::time_point m_Deadline;
	std::thread m_Thread;
};

std::vector<uint64_t> g_Ran(kScenarioCount, 0);

void RunRound(uint64_t roundSeed, bool quiet)
{
	g_RoundSeed = roundSeed;
	Rng rng(roundSeed);
	const int hw = (int)std::max(2u, std::thread::hardware_concurrency());
	const int workerChoice = Pick(rng, 0, 5);
	const int workers = workerChoice == 0 ? 0 : workerChoice == 1 ? 1 : workerChoice == 2 ? 2 : Pick(rng, 1, std::min(hw, 16));
	loom::Pool pool(loom::PoolConfig{ .workerCount = workers, .maxTasks = 65536, .maxCounters = 16384 });
	Env env;
	for (int d = Pick(rng, 0, 2); d > 0; d--)
		env.dedicated.push_back(loom::RegisterDedicatedThread({ .name = "stress-dedicated",
			.idle = Chance(rng, 0.5) ? loom::DedicatedConfig::Idle::Sleep : loom::DedicatedConfig::Idle::StealFromPool }));

	// A few scenarios at once, each as a task or two, the rest on this thread.
	const int count = Pick(rng, 1, 6);
	struct Planned
	{
		size_t which;
		uint64_t seed;
		bool onMain;
	};
	std::vector<Planned> plan;
	for (int i = 0; i < count; i++)
		plan.push_back({ (size_t)Pick(rng, 0, (int)kScenarioCount - 1), Mix(roundSeed * 131 + (uint64_t)i), Chance(rng, 0.25) });
	if (!quiet)
		std::printf("round %llu: %d workers, %zu dedicated, %d scenarios\n", (unsigned long long)roundSeed, workers,
			env.dedicated.size(), count);

	std::vector<loom::Future<void>> running;
	for (const Planned& p : plan)
	{
		g_Ran[p.which]++;
		if (p.onMain)
			continue;
		running.push_back(loom::Async([p, &env] {
			Rng r(p.seed);
			kScenarios[p.which].run(r, p.seed, env);
		}));
	}
	for (const Planned& p : plan)
		if (p.onMain)
		{
			Rng r(p.seed);
			kScenarios[p.which].run(r, p.seed, env);
		}
	for (loom::Future<void>& f : running)
		f.Get();
	running.clear();
	for (loom::DedicatedId id : env.dedicated)
		loom::UnregisterDedicatedThread(id);
}

} // namespace

int main(int argc, char** argv)
{
	double seconds = 10;
	uint64_t rounds = 0; // no limit
	uint64_t seed = (uint64_t)std::random_device{}() << 32 | std::random_device{}();
	bool quiet = false;
	for (int i = 1; i < argc; i++)
	{
		auto value = [&](const char* flag) -> const char* {
			if (i + 1 >= argc)
			{
				std::fprintf(stderr, "%s needs a value\n", flag);
				std::exit(2);
			}
			return argv[++i];
		};
		if (!std::strcmp(argv[i], "--seconds"))
			seconds = std::atof(value("--seconds"));
		else if (!std::strcmp(argv[i], "--rounds"))
			rounds = std::strtoull(value("--rounds"), nullptr, 10);
		else if (!std::strcmp(argv[i], "--seed"))
			seed = std::strtoull(value("--seed"), nullptr, 10);
		else if (!std::strcmp(argv[i], "--quiet"))
			quiet = true;
		else
		{
			std::fprintf(stderr, "usage: %s [--seconds N] [--rounds N] [--seed N] [--quiet]\n", argv[0]);
			return 2;
		}
	}
	std::printf("loom_stress: seed %llu, up to %.0f s%s\n", (unsigned long long)seed, seconds,
		rounds ? (", up to " + std::to_string(rounds) + " rounds").c_str() : "");
	std::fflush(stdout);

	Watchdog watchdog;
	const auto until = Clock::now() + std::chrono::duration<double>(seconds);
	uint64_t done = 0;
	// Round r uses seed + r: --seed with the failing round's seed repeats it first.
	while ((rounds == 0 || done < rounds) && Clock::now() < until && g_Failures.load() == 0)
	{
		watchdog.RoundStarted();
		RunRound(seed + done, quiet);
		watchdog.RoundEnded();
		done++;
	}

	std::printf("loom_stress: %llu rounds;", (unsigned long long)done);
	for (size_t i = 0; i < kScenarioCount; i++)
		std::printf(" %s %llu%s", kScenarios[i].name, (unsigned long long)g_Ran[i], i + 1 < kScenarioCount ? "," : "\n");
	if (g_Failures.load() != 0)
	{
		std::printf("loom_stress: %d checks failed; rerun the failing round with --seed %llu --rounds 1\n", g_Failures.load(),
			(unsigned long long)g_RoundSeed);
		return 1;
	}
	std::printf("loom_stress: all checks passed\n");
	return 0;
}
