// loom_bench: the per-task cost and the queue contention, at 1 worker, half
// the cores and every core. Each run fans out `tasks` tiny tasks (or a
// ParallelFor over the same range) on one counter and waits with help, the
// shape an engine frame's jobs have; it prints tasks per second and the cost
// per task. Usage: loom_bench [tasks] [repeats]

#include "loom/loom.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

namespace {

	std::atomic<uint64_t> g_Sink{ 0 };

	void Work(void* user)
	{
		// A few hundred nanoseconds of work, so the queue, not the task, is measured.
		uint64_t x = (uint64_t)(uintptr_t)user;
		for (int i = 0; i < 64; i++)
			x = x * 6364136223846793005ull + 1442695040888963407ull;
		g_Sink.fetch_add(x & 1, std::memory_order_relaxed);
	}

	double SubmitRun(int tasks)
	{
		const auto start = std::chrono::steady_clock::now();
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
		loom::WaitFor(done);
		return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
	}

	double ParallelForRun(int tasks)
	{
		const auto start = std::chrono::steady_clock::now();
		const size_t chunk = std::max<size_t>(1, (size_t)tasks / (size_t)(loom::WorkerCount() * 4 + 4));
		loom::ParallelFor(0, (size_t)tasks, chunk, [](size_t begin, size_t end) {
			for (size_t i = begin; i < end; i++)
				Work((void*)(uintptr_t)i);
		});
		return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
	}

} // namespace

int main(int argc, char** argv)
{
	const int tasks = argc > 1 ? std::atoi(argv[1]) : 200000;
	const int repeats = argc > 2 ? std::atoi(argv[2]) : 5;
	const int cores = (int)std::max(1u, std::thread::hardware_concurrency());
	std::printf("loom_bench: %d tasks, best of %d runs\n", tasks, repeats);
	std::printf("%8s  %14s  %10s  %14s\n", "workers", "submit tasks/s", "ns/task", "ParallelFor/s");
	for (int workers : { 1, std::max(1, cores / 2), std::max(1, cores - 1) })
	{
		loom::Pool pool(loom::LoomConfig{ .workerCount = workers, .maxTasks = 1u << 18 });
		double bestSubmit = 1e9, bestFor = 1e9;
		for (int r = 0; r < repeats; r++)
		{
			bestSubmit = std::min(bestSubmit, SubmitRun(tasks));
			bestFor = std::min(bestFor, ParallelForRun(tasks));
		}
		std::printf("%8d  %14.0f  %10.1f  %14.0f\n", workers, tasks / bestSubmit, bestSubmit * 1e9 / tasks, tasks / bestFor);
	}
	return g_Sink.load() == ~0ull ? 1 : 0; // keeps the work observable
}
