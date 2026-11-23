// loom_bench: the per-task cost and the queue contention, at 1 worker, half
// the cores and every core. Each run fans out `tasks` tiny tasks (or a
// ParallelFor over the same range) on one counter and waits with help, the
// shape an engine frame's jobs have. It prints, per worker count:
//   submit tasks/s, ns/task  the whole round trip (create, steal, run,
//                            signal, wait) divided by the task count
//   create ns                the main thread's submit loop alone, per task,
//                            while the workers run (real contention)
//   create ns parked         the same loop with every worker parked: the
//                            bare cost of making a task, nothing competing
//   ParallelFor/s            items per second through ParallelFor
// Then a frame-shaped DAG (layers of tiny tasks, each waiting on a few of the
// layer before), per frame: as a TaskGraph built once, and rebuilt every
// frame from one counter per task. Then four shapes against the cheapest
// counter form of each (ratio below 1: the graph is faster): independent
// tasks against Submit(span), a fan out and in, layers each after the whole
// layer before (a counter per layer), and a chain.
// Then limiters: what a limited task costs (one Submit per task, with no
// limiter, a limiter that never makes a task wait, and one that holds most
// tasks back), and a frame of ParallelFor work beside long Low-priority
// background tasks, with the background unlimited or limited to 4 workers.
// Then futures: 4096 Async results read with Get, against tasks that write
// their results into an array behind one counter.
// Then continuations: chains of dependent steps written with Then, against
// the same chains with one counter per step and waitsFor.
// Usage: loom_bench [tasks] [repeats]

#include "loom/loom.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <random>
#include <span>
#include <thread>
#include <vector>

namespace {

	// Keeps a value observable without a shared write: a shared atomic sink
	// here made every task contend on one cache line, so the bench measured
	// that line instead of the scheduler.
	inline void KeepAlive(uint64_t x)
	{
#if defined(__GNUC__) || defined(__clang__)
		asm volatile("" : : "r"(x) : "memory");
#else
		static volatile uint64_t s_sink;
		s_sink = x;
#endif
	}

	void Work(void* user)
	{
		// A few hundred nanoseconds of work, so the queue, not the task, is measured.
		uint64_t x = (uint64_t)(uintptr_t)user;
		for (int i = 0; i < 64; i++)
			x = x * 6364136223846793005ull + 1442695040888963407ull;
		KeepAlive(x);
	}

	using Clock = std::chrono::steady_clock;

	double Seconds(Clock::time_point from, Clock::time_point to)
	{
		return std::chrono::duration<double>(to - from).count();
	}

	struct SubmitTimes
	{
		double total = 0;  // create + run + wait
		double create = 0; // the submit loop alone
	};

	// With `parked`, every worker is parked for the submit loop, so it times
	// task creation with nothing stealing, running or freeing slots.
	SubmitTimes SubmitRun(int tasks, bool parked)
	{
		SubmitTimes times;
		if (parked && !loom::ParkAll(5000))
		{
			std::fprintf(stderr, "loom_bench: ParkAll timed out\n");
			return { 1e9, 1e9 };
		}
		const auto start = Clock::now();
		loom::ScopedCounter done(tasks);
		std::vector<loom::TaskDesc> batch(256);
		for (int i = 0; i < tasks;)
		{
			const int n = std::min(256, tasks - i);
			for (int j = 0; j < n; j++)
				batch[j] = loom::TaskDesc{ .fn = &Work, .userData = (void*)(uintptr_t)(i + j), .signals = done };
			loom::Submit(std::span<const loom::TaskDesc>(batch.data(), n));
			i += n;
		}
		const auto created = Clock::now();
		if (parked)
			loom::UnparkAll();
		loom::WaitFor(done);
		times.total = Seconds(start, Clock::now());
		times.create = Seconds(start, created);
		return times;
	}

	double ParallelForRun(int tasks)
	{
		const auto start = Clock::now();
		const size_t chunk = std::max<size_t>(1, (size_t)tasks / (size_t)(loom::WorkerCount() * 4 + 4));
		loom::ParallelFor(0, (size_t)tasks, chunk, [](size_t begin, size_t end) {
			for (size_t i = begin; i < end; i++)
				Work((void*)(uintptr_t)i);
		});
		return Seconds(start, Clock::now());
	}

	// The frame DAG: kLayers layers of kWidth nodes, each node after kFanIn
	// random nodes of the layer before, then one sink after the last layer.
	constexpr int kLayers = 8, kWidth = 16, kFanIn = 3;
	constexpr int kNodes = kLayers * kWidth;

	struct FrameDag
	{
		std::vector<std::vector<int>> preds; // per node; the sink is node kNodes
		FrameDag()
		{
			std::mt19937 rng(7);
			preds.resize(kNodes + 1);
			for (int layer = 1; layer < kLayers; layer++)
				for (int i = 0; i < kWidth; i++)
					for (int k = 0; k < kFanIn; k++)
						preds[layer * kWidth + i].push_back((layer - 1) * kWidth + (int)(rng() % kWidth));
			for (int i = 0; i < kWidth; i++)
				preds[kNodes].push_back((kLayers - 1) * kWidth + i);
		}
	};

	double GraphFrames(const FrameDag& dag, int frames)
	{
		loom::TaskGraph graph;
		for (int i = 0; i <= kNodes; i++)
			graph.Add(loom::TaskDesc{ .fn = &Work, .userData = (void*)(uintptr_t)i });
		for (int i = 0; i <= kNodes; i++)
			for (int p : dag.preds[(size_t)i])
				graph.Precede((loom::TaskGraph::Node)p, (loom::TaskGraph::Node)i);
		const auto start = Clock::now();
		for (int f = 0; f < frames; f++)
			loom::Run(graph);
		return Seconds(start, Clock::now()) / frames;
	}

	double CounterFrames(const FrameDag& dag, int frames)
	{
		std::vector<loom::CounterHandle> done((size_t)kNodes + 1);
		std::vector<std::vector<loom::CounterHandle>> waits((size_t)kNodes + 1);
		std::vector<loom::TaskDesc> descs((size_t)kNodes + 1);
		const auto start = Clock::now();
		for (int f = 0; f < frames; f++)
		{
			for (auto& c : done)
				c = loom::CreateCounter(1);
			for (size_t i = 0; i <= (size_t)kNodes; i++)
			{
				waits[i].clear();
				for (int p : dag.preds[i])
					waits[i].push_back(done[(size_t)p]);
				descs[i] = loom::TaskDesc{ .fn = &Work, .userData = (void*)(uintptr_t)i, .signals = done[i], .waitsForAll = waits[i] };
			}
			loom::Submit(std::span<const loom::TaskDesc>(descs));
			loom::WaitFor(done[(size_t)kNodes]);
			for (size_t i = 0; i < (size_t)kNodes; i++)
				loom::WaitFor(done[i], loom::Help::BlockOnly); // settled, so they can be destroyed
			for (auto c : done)
				loom::DestroyCounter(c);
		}
		return Seconds(start, Clock::now()) / frames;
	}

	double BestPerFrame(int frames, int repeats, const std::function<void()>& frame)
	{
		double best = 1e9;
		for (int r = 0; r < repeats; r++)
		{
			const auto start = Clock::now();
			for (int i = 0; i < frames; i++)
				frame();
			best = std::min(best, Seconds(start, Clock::now()) / frames);
		}
		return best * 1e6;
	}

	loom::TaskDesc WorkDesc(int i, loom::CounterHandle signals = loom::kInvalidCounter, loom::CounterHandle waitsFor = loom::kInvalidCounter)
	{
		return loom::TaskDesc{ .fn = &Work, .userData = (void*)(uintptr_t)i, .signals = signals, .waitsFor = waitsFor };
	}

	void GraphShapes(int workers, int repeats)
	{
		auto row = [&](const char* name, double graph, double counters) {
			std::printf("%-30s  %7d  %10.1f  %11.1f  %6.2f\n", name, workers, graph, counters, graph / counters);
		};
		{
			const int n = 4096;
			loom::TaskGraph graph;
			for (int i = 0; i < n; i++)
				graph.Add(WorkDesc(i));
			std::vector<loom::TaskDesc> descs(n);
			row("4096 independent", BestPerFrame(100, repeats, [&] { loom::Run(graph); }), BestPerFrame(100, repeats, [&] {
				loom::ScopedCounter done(n);
				for (int i = 0; i < n; i++)
					descs[(size_t)i] = WorkDesc(i, done);
				loom::Submit(std::span<const loom::TaskDesc>(descs));
				loom::WaitFor(done);
			}));
		}
		{
			const int n = 1024;
			loom::TaskGraph graph;
			const auto root = graph.Add(WorkDesc(0)), sink = graph.Add(WorkDesc(1));
			for (int i = 0; i < n; i++)
			{
				const auto node = graph.Add(WorkDesc(i));
				graph.Precede(root, node);
				graph.Precede(node, sink);
			}
			std::vector<loom::TaskDesc> descs(n);
			row("1 -> 1024 -> 1", BestPerFrame(500, repeats, [&] { loom::Run(graph); }), BestPerFrame(500, repeats, [&] {
				loom::ScopedCounter rootDone(1), middleDone(n), done(1);
				loom::Submit(WorkDesc(0, rootDone));
				for (int i = 0; i < n; i++)
					descs[(size_t)i] = WorkDesc(i, middleDone, rootDone);
				loom::Submit(std::span<const loom::TaskDesc>(descs));
				loom::Submit(WorkDesc(1, done, middleDone));
				loom::WaitFor(done);
			}));
		}
		{
			constexpr int layers = 8, width = 16;
			loom::TaskGraph graph;
			for (int i = 0; i < layers * width; i++)
				graph.Add(WorkDesc(i));
			for (int l = 1; l < layers; l++)
				for (int a = 0; a < width; a++)
					for (int b = 0; b < width; b++)
						graph.Precede((loom::TaskGraph::Node)((l - 1) * width + a), (loom::TaskGraph::Node)(l * width + b));
			std::vector<loom::TaskDesc> descs(width);
			row("8x16, layer after layer", BestPerFrame(500, repeats, [&] { loom::Run(graph); }), BestPerFrame(500, repeats, [&] {
				loom::CounterHandle layerDone[layers];
				for (auto& c : layerDone)
					c = loom::CreateCounter(width);
				for (int l = 0; l < layers; l++)
				{
					for (int i = 0; i < width; i++)
						descs[(size_t)i] = WorkDesc(i, layerDone[l], l ? layerDone[l - 1] : loom::kInvalidCounter);
					loom::Submit(std::span<const loom::TaskDesc>(descs));
				}
				loom::WaitFor(layerDone[layers - 1]);
				for (auto c : layerDone)
				{
					loom::WaitFor(c, loom::Help::BlockOnly); // settled, so it can be destroyed
					loom::DestroyCounter(c);
				}
			}));
		}
		{
			const int n = 256;
			loom::TaskGraph graph;
			for (int i = 0; i < n; i++)
			{
				const auto node = graph.Add(WorkDesc(i));
				if (i)
					graph.Precede(node - 1, node);
			}
			std::vector<loom::TaskDesc> descs(n);
			std::vector<loom::CounterHandle> done(n);
			row("chain of 256", BestPerFrame(100, repeats, [&] { loom::Run(graph); }), BestPerFrame(100, repeats, [&] {
				for (auto& c : done)
					c = loom::CreateCounter(1);
				for (int i = 0; i < n; i++)
					descs[(size_t)i] = WorkDesc(i, done[(size_t)i], i ? done[(size_t)i - 1] : loom::kInvalidCounter);
				loom::Submit(std::span<const loom::TaskDesc>(descs));
				loom::WaitFor(done[(size_t)n - 1]);
				for (auto c : done)
				{
					loom::WaitFor(c, loom::Help::BlockOnly);
					loom::DestroyCounter(c);
				}
			}));
		}
	}

	// What a limiter adds per task: `tasks` tiny tasks, one Submit each.
	double LimitedRun(int tasks, loom::LimiterHandle limiter)
	{
		const auto start = Clock::now();
		loom::ScopedCounter done(tasks);
		for (int i = 0; i < tasks; i++)
			loom::Submit(loom::TaskDesc{ .fn = &Work, .userData = (void*)(uintptr_t)i, .signals = done, .limiter = limiter });
		loom::WaitFor(done);
		return Seconds(start, Clock::now());
	}

	// About a nanosecond an iteration.
	void Burn(int iterations)
	{
		uint64_t x = 1;
		for (int i = 0; i < iterations; i++)
			x = x * 6364136223846793005ull + 1442695040888963407ull;
		KeepAlive(x);
	}

	// Background work that never runs out: Low tasks of about a millisecond,
	// each submitting the next until `stop`.
	struct Background
	{
		std::atomic<bool> stop{ false };
		std::atomic<int>  live{ 0 };
		loom::LimiterHandle limiter = loom::kInvalidLimiter;

		static void Task(void* p)
		{
			Background& b = *static_cast<Background*>(p);
			Burn(1'000'000);
			if (!b.stop.load(std::memory_order_relaxed))
				b.Start();
			b.live.fetch_sub(1, std::memory_order_release);
		}
		void Start()
		{
			live.fetch_add(1, std::memory_order_relaxed);
			loom::Submit(loom::TaskDesc{ .fn = &Task, .userData = this, .priority = loom::Priority::Low, .limiter = limiter });
		}
		void Finish()
		{
			stop = true;
			while (live.load(std::memory_order_acquire) > 0)
				std::this_thread::sleep_for(std::chrono::milliseconds(1));
		}
	};

	// A frame of 256 items of about 5 us each (1.3 ms of work), started every
	// 3 ms; returns the median and the slowest 1% of frame times, in us. With
	// `isolated`, the frame waits in an isolated region, so the waiting thread
	// never helps with a background task.
	std::pair<double, double> FramesBeside(int backgroundTasks, loom::LimiterHandle limiter, bool isolated)
	{
		Background bg;
		bg.limiter = limiter;
		for (int i = 0; i < backgroundTasks; i++)
			bg.Start();
		std::vector<double> times;
		for (int f = 0; f < 300; f++)
		{
			std::this_thread::sleep_for(std::chrono::milliseconds(3));
			const auto start = Clock::now();
			auto frame = [] { loom::ParallelFor(0, 256, 1, [](size_t, size_t) { Burn(5'000); }); };
			if (isolated)
				loom::Isolated(frame);
			else
				frame();
			times.push_back(Seconds(start, Clock::now()) * 1e6);
		}
		bg.Finish();
		std::sort(times.begin(), times.end());
		return { times[times.size() / 2], times[times.size() * 99 / 100] };
	}

	// `n` results through futures, or through one counter and an array.
	double FutureRun(int n)
	{
		std::vector<loom::Future<uint64_t>> futures;
		futures.reserve((size_t)n);
		const auto start = Clock::now();
		for (int i = 0; i < n; i++)
			futures.push_back(loom::Async([i] {
				Work((void*)(uintptr_t)i);
				return (uint64_t)i;
			}));
		uint64_t sum = 0;
		for (auto& f : futures)
			sum += f.Get();
		futures.clear(); // freeing them is part of the cost
		const double seconds = Seconds(start, Clock::now());
		KeepAlive(sum);
		return seconds;
	}

	double ArrayRun(int n)
	{
		std::vector<uint64_t> results((size_t)n);
		const auto start = Clock::now();
		loom::ScopedCounter done(n);
		for (int i = 0; i < n; i++)
			loom::Submit([i, out = results.data()] {
				Work((void*)(uintptr_t)i);
				out[i] = (uint64_t)i;
			}, loom::TaskDesc{ .signals = done });
		loom::WaitFor(done);
		uint64_t sum = 0;
		for (uint64_t r : results)
			sum += r;
		const double seconds = Seconds(start, Clock::now());
		KeepAlive(sum);
		return seconds;
	}

	// Chains of `steps` dependent tasks, each step reading the one before:
	// written with Then, and by hand with one counter per step and waitsFor.
	double ThenChainRun(int chains, int steps)
	{
		std::vector<loom::Future<uint64_t>> ends;
		ends.reserve((size_t)chains);
		const auto start = Clock::now();
		for (int c = 0; c < chains; c++)
		{
			loom::Future<uint64_t> f = loom::Async([c] {
				Work((void*)(uintptr_t)c);
				return (uint64_t)c;
			});
			for (int s = 1; s < steps; s++)
				f = f.Then([](uint64_t& v) {
					Work((void*)(uintptr_t)v);
					return v + 1;
				});
			ends.push_back(std::move(f));
		}
		uint64_t sum = 0;
		for (auto& f : ends)
			sum += f.Get();
		ends.clear();
		const double seconds = Seconds(start, Clock::now());
		KeepAlive(sum);
		return seconds;
	}

	double CounterChainRun(int chains, int steps)
	{
		std::vector<uint64_t> values((size_t)chains * (size_t)steps);
		std::vector<loom::CounterHandle> counters((size_t)chains * (size_t)steps);
		const auto start = Clock::now();
		for (int c = 0; c < chains; c++)
		{
			loom::CounterHandle prev = loom::kInvalidCounter;
			for (int s = 0; s < steps; s++)
			{
				const size_t at = (size_t)c * (size_t)steps + (size_t)s;
				counters[at] = loom::CreateCounter(1);
				const uint64_t* in = s > 0 ? &values[at - 1] : nullptr;
				loom::Submit([out = &values[at], in, c] {
					const uint64_t v = in ? *in + 1 : (uint64_t)c;
					Work((void*)(uintptr_t)v);
					*out = v;
				}, loom::TaskDesc{ .signals = counters[at], .waitsFor = prev });
				prev = counters[at];
			}
		}
		uint64_t sum = 0;
		for (int c = 0; c < chains; c++)
		{
			const size_t last = (size_t)c * (size_t)steps + (size_t)steps - 1;
			loom::WaitFor(counters[last]);
			sum += values[last];
		}
		for (loom::CounterHandle h : counters)
			loom::DestroyCounter(h);
		const double seconds = Seconds(start, Clock::now());
		KeepAlive(sum);
		return seconds;
	}

} // namespace

int main(int argc, char** argv)
{
	const int tasks = argc > 1 ? std::atoi(argv[1]) : 200000;
	const int repeats = argc > 2 ? std::atoi(argv[2]) : 5;
	const int cores = (int)std::max(1u, std::thread::hardware_concurrency());
	std::printf("loom_bench: %d tasks, best of %d runs\n", tasks, repeats);
	std::printf("%8s  %14s  %10s  %10s  %17s  %14s\n", "workers", "submit tasks/s", "ns/task", "create ns", "create ns parked", "ParallelFor/s");
	for (int workers : { 1, std::max(1, cores / 2), std::max(1, cores - 1) })
	{
		// Room for every task queued at once (the parked run queues them all).
		loom::Pool pool(loom::LoomConfig{ .workerCount = workers, .maxTasks = (uint32_t)std::max(tasks + 1024, 1 << 18) });
		double bestTotal = 1e9, bestCreate = 1e9, bestParked = 1e9, bestFor = 1e9;
		for (int r = 0; r < repeats; r++)
		{
			const SubmitTimes live = SubmitRun(tasks, false);
			bestTotal = std::min(bestTotal, live.total);
			bestCreate = std::min(bestCreate, live.create);
			bestParked = std::min(bestParked, SubmitRun(tasks, true).create);
			bestFor = std::min(bestFor, ParallelForRun(tasks));
		}
		std::printf("%8d  %14.0f  %10.1f  %10.1f  %17.1f  %14.0f\n", workers, tasks / bestTotal, bestTotal * 1e9 / tasks,
			bestCreate * 1e9 / tasks, bestParked * 1e9 / tasks, tasks / bestFor);
	}
	std::printf("\nframe DAG: %d tiny tasks in %d layers, %d predecessors each, best of %d runs\n", kNodes + 1, kLayers, kFanIn, repeats);
	std::printf("%8s  %16s  %18s\n", "workers", "TaskGraph us", "counters us");
	const FrameDag dag;
	for (int workers : { 1, std::max(1, cores / 2), std::max(1, cores - 1) })
	{
		loom::Pool pool(loom::LoomConfig{ .workerCount = workers });
		double bestGraph = 1e9, bestCounters = 1e9;
		for (int r = 0; r < repeats; r++)
		{
			bestGraph = std::min(bestGraph, GraphFrames(dag, 2000));
			bestCounters = std::min(bestCounters, CounterFrames(dag, 2000));
		}
		std::printf("%8d  %16.2f  %18.2f\n", workers, bestGraph * 1e6, bestCounters * 1e6);
	}
	std::printf("\ngraph shapes against their cheapest counter form, us per run\n");
	std::printf("%-30s  %7s  %10s  %11s  %6s\n", "shape", "workers", "TaskGraph", "counters", "ratio");
	for (int workers : { 1, std::max(1, cores / 2), std::max(1, cores - 1) })
	{
		loom::Pool pool(loom::LoomConfig{ .workerCount = workers, .maxTasks = 1 << 16 });
		GraphShapes(workers, repeats);
	}

	std::printf("\nlimiter cost: 4096 tiny tasks, one Submit each, ns per task\n");
	std::printf("%8s  %12s  %16s  %20s\n", "workers", "no limiter", "limit never hit", "limit workers / 4");
	for (int workers : { 1, std::max(1, cores / 2), std::max(1, cores - 1) })
	{
		loom::Pool pool(loom::LoomConfig{ .workerCount = workers, .maxTasks = 1 << 16 });
		loom::ScopedLimiter wide(1u << 20), narrow((uint32_t)std::max(1, workers / 4));
		const int n = 4096;
		double none = 1e9, never = 1e9, held = 1e9;
		for (int r = 0; r < repeats * 20; r++)
		{
			none = std::min(none, LimitedRun(n, loom::kInvalidLimiter));
			never = std::min(never, LimitedRun(n, wide));
			held = std::min(held, LimitedRun(n, narrow));
		}
		std::printf("%8d  %12.1f  %16.1f  %20.1f\n", workers, none * 1e9 / n, never * 1e9 / n, held * 1e9 / n);
	}

	{
		const int workers = std::max(1, cores - 1);
		loom::Pool pool(loom::LoomConfig{ .workerCount = workers });
		std::printf("\na frame (256 x 5 us ParallelFor) beside 1 ms Low background tasks, %d workers, us\n", workers);
		std::printf("%-34s  %8s  %8s  %15s  %12s\n", "background", "median", "p99", "isolated median", "isolated p99");
		auto row = [&](const char* name, int tasks, loom::LimiterHandle limiter) {
			const auto plain = FramesBeside(tasks, limiter, false), isolated = FramesBeside(tasks, limiter, true);
			std::printf("%-34s  %8.0f  %8.0f  %15.0f  %12.0f\n", name, plain.first, plain.second, isolated.first, isolated.second);
		};
		row("none", 0, loom::kInvalidLimiter);
		row("one task per worker, no limiter", workers, loom::kInvalidLimiter);
		loom::ScopedLimiter four(4);
		row("one task per worker, limited to 4", workers, four);
	}

	std::printf("\nfutures: 4096 results, ns per result\n");
	std::printf("%8s  %10s  %22s  %6s\n", "workers", "futures", "one counter + array", "ratio");
	for (int workers : { 1, std::max(1, cores / 2), std::max(1, cores - 1) })
	{
		loom::Pool pool(loom::LoomConfig{ .workerCount = workers, .maxTasks = 1 << 16, .maxCounters = 8192 });
		const int n = 4096;
		double futures = 1e9, array = 1e9;
		for (int r = 0; r < repeats * 20; r++)
		{
			futures = std::min(futures, FutureRun(n));
			array = std::min(array, ArrayRun(n));
		}
		std::printf("%8d  %10.1f  %22.1f  %6.2f\n", workers, futures * 1e9 / n, array * 1e9 / n, futures / array);
	}

	const int chains = 512, steps = 8;
	std::printf("\ncontinuations: %d chains of %d steps, ns per step\n", chains, steps);
	std::printf("%8s  %10s  %22s  %6s\n", "workers", "Then", "counter per step", "ratio");
	for (int workers : { 1, std::max(1, cores / 2), std::max(1, cores - 1) })
	{
		loom::Pool pool(loom::LoomConfig{ .workerCount = workers, .maxTasks = 1 << 16, .maxCounters = 8192 });
		double then = 1e9, counters = 1e9;
		for (int r = 0; r < repeats * 20; r++)
		{
			then = std::min(then, ThenChainRun(chains, steps));
			counters = std::min(counters, CounterChainRun(chains, steps));
		}
		const double perStep = 1e9 / (chains * steps);
		std::printf("%8d  %10.1f  %22.1f  %6.2f\n", workers, then * perStep, counters * perStep, then / counters);
	}
	return 0;
}
