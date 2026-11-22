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
// frame from one counter per task.
// Usage: loom_bench [tasks] [repeats]

#include "loom/loom.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
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
	return 0;
}
