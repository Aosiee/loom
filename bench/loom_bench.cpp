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
// Usage: loom_bench [tasks] [repeats]

#include "loom/loom.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
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
	return 0;
}
