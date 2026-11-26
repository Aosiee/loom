#pragma once

// Loom's headless suite: seeded stress, DAGs, affinity pinning,
// help-while-waiting, dedicated lifecycle, shutdown under load. One doctest
// TEST_CASE per named check; every case owns its own loom::Pool (no lazy
// auto-init - the pool is created explicitly). Runs from
// tinytest and from the minimal-link loom_tests app (TSAN-clean).

#include "loom/loom.h"
#include "loom/loom_profiler_shim.h"
#include <doctest/doctest.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <memory>
#include <random>
#include <set>
#include <cstring>
#include <ctime>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#include <csignal>
#include <fcntl.h>
#include <sys/wait.h>
#include <unistd.h>

// Runs fn in a child process: true when it ended by abort(), with what it
// wrote to stderr in `err`. Call it with no pool active: the child starts
// from this thread alone.
static bool EndsInAbort(void (*fn)(), std::string& err)
{
	int fds[2];
	if (pipe(fds) != 0)
		return false;
	const pid_t pid = fork();
	if (pid == 0)
	{
		// The test runner's crash handler would report the abort as its own.
		std::signal(SIGABRT, SIG_DFL);
		dup2(fds[1], 2);
		const int devNull = open("/dev/null", O_WRONLY);
		dup2(devNull, 1);
		close(fds[0]);
		fn();
		_exit(0);
	}
	close(fds[1]);
	char buf[512];
	ssize_t n;
	while ((n = read(fds[0], buf, sizeof(buf))) > 0)
		err.append(buf, (size_t)n);
	close(fds[0]);
	int status = 0;
	waitpid(pid, &status, 0);
	return WIFSIGNALED(status) && WTERMSIG(status) == SIGABRT;
}

// Runs fn in a child process: true when it returned within `ms` (a hang is
// killed and counts as false). Call it with no pool active.
static bool EndsWithin(void (*fn)(), int ms)
{
	const pid_t pid = fork();
	if (pid == 0)
	{
		std::signal(SIGABRT, SIG_DFL);
		const int devNull = open("/dev/null", O_WRONLY);
		dup2(devNull, 1);
		fn();
		_exit(0);
	}
	const auto until = std::chrono::steady_clock::now() + std::chrono::milliseconds(ms);
	int status = 0;
	while (waitpid(pid, &status, WNOHANG) != pid)
	{
		if (std::chrono::steady_clock::now() > until)
		{
			kill(pid, SIGKILL);
			waitpid(pid, &status, 0);
			return false;
		}
		std::this_thread::sleep_for(std::chrono::milliseconds(2));
	}
	return WIFEXITED(status) && WEXITSTATUS(status) == 0;
}
#endif

#if defined(LOOM_TEST_HOOKS)
// Only loom's own test build has these (see LOOM_TEST_POINT in loom.cpp).
namespace loom::detail { using TestHookFn = void (*)(int point); void SetTestHook(TestHookFn fn); }
#endif

// Each kind of handle is its own type: none converts to another, to an
// integer or from one, so passing the wrong one does not compile.
static_assert(!std::is_convertible_v<loom::LimiterHandle, loom::CounterHandle>);
static_assert(!std::is_convertible_v<loom::CounterHandle, loom::LimiterHandle>);
static_assert(!std::is_convertible_v<loom::DedicatedId, loom::CounterHandle>);
static_assert(!std::is_convertible_v<loom::CounterHandle, loom::DedicatedId>);
static_assert(!std::is_convertible_v<uint32_t, loom::CounterHandle>);
static_assert(!std::is_convertible_v<loom::CounterHandle, uint32_t>);
static_assert(std::is_same_v<loom::LaneId, loom::DedicatedId>); // a lane is a dedicated thread

namespace loom { namespace tests {

namespace {

void SpinUntil(int target, CounterHandle c, int timeoutMs)
{
	auto start = std::chrono::steady_clock::now();
	while (loom::Count(c) != target)
	{
		if (std::chrono::duration_cast<std::chrono::milliseconds>(
			std::chrono::steady_clock::now() - start).count() > timeoutMs)
			break;
		std::this_thread::sleep_for(std::chrono::microseconds(50));
	}
}

inline void Noop(void*) {}

// ---- 1. Basic: N AnyWorker tasks, counter fan-in, exactly-once ----

struct BasicResults
{
	bool exactlyOnce = false;
	bool drained = false;
	bool quiesced = false;
};

BasicResults RunBasicScenario()
{
	BasicResults r;
	loom::Pool pool; // the suite owns its pool explicitly
	const int N = 1000;
	std::atomic<int> runs{ 0 };
	CounterHandle done = loom::CreateCounter(N);
	for (int i = 0; i < N; i++)
	{
		loom::Submit([&runs]() { runs++; },
			loom::TaskDesc{ nullptr, nullptr, loom::Affinity::AnyWorker, {}, done, loom::kInvalidCounter });
	}
	loom::WaitFor(done);
	r.exactlyOnce = (runs.load() == N);
	r.drained = (loom::Count(done) == 0);
	loom::DestroyCounter(done);
	pool.Shutdown();
	r.quiesced = !loom::HasActivePool();
	return r;
}

TEST_CASE("loom: basic — every task ran exactly once")
{
	CHECK_MESSAGE(RunBasicScenario().exactlyOnce, "basic: every task ran exactly once");
}

TEST_CASE("loom: basic — counter drained to zero")
{
	CHECK_MESSAGE(RunBasicScenario().drained, "basic: counter drained to zero");
}

TEST_CASE("loom: shutdown quiesced")
{
	CHECK_MESSAGE(RunBasicScenario().quiesced, "shutdown: quiesced");
}

// ---- 2. 10k seeded stress (mixed plain + inline submission) ----

TEST_CASE("loom: stress — 10k tasks ran exactly once")
{
	loom::Pool pool; // the suite owns its pool explicitly
	const int N = 10000;
	std::atomic<int> runs{ 0 };
	CounterHandle done = loom::CreateCounter(N);
	std::mt19937 rng(12345);
	for (int i = 0; i < N; i++)
	{
		if (rng() % 2)
			loom::Submit([&runs]() { runs++; },
				loom::TaskDesc{ nullptr, nullptr, loom::Affinity::AnyWorker, {}, done, loom::kInvalidCounter });
		else
			loom::Submit(loom::TaskDesc{ [](void* p) { (*(std::atomic<int>*)p)++; },
				&runs, loom::Affinity::AnyWorker, {}, done, loom::kInvalidCounter });
	}
	loom::WaitFor(done);
	CHECK_MESSAGE(runs.load() == N, "stress: 10k tasks ran exactly once");
	loom::DestroyCounter(done);
	pool.Shutdown();
}

// ---- 3. Random DAG: fan-in/fan-out via counters ----

struct DagResults
{
	bool everyTaskRan = false;
	bool leafZero = false;
};

DagResults RunDagScenario()
{
	DagResults r;
	loom::Pool pool; // the suite owns its pool explicitly
	std::mt19937 rng(999);
	const int N = 2000;
	std::vector<CounterHandle> counters;
	std::atomic<int> ran{ 0 };
	CounterHandle leaf = loom::CreateCounter(N); // fan-in sink
	for (int i = 0; i < N; i++)
	{
		CounterHandle waitOn = loom::kInvalidCounter;
		if (i > 0 && (rng() % 3) == 0)
			waitOn = counters[rng() % (int)counters.size()];
		int initial = ((rng() % 4) == 0) ? (int)(rng() % 3) + 1 : 0;
		CounterHandle mine = loom::CreateCounter(initial);
		counters.push_back(mine);
		loom::Submit(loom::TaskDesc{ [](void* p) { (*(std::atomic<int>*)p)++; },
			&ran, loom::Affinity::AnyWorker, {}, leaf, waitOn });
	}
	// Release the initial counts so the whole DAG unblocks.
	for (CounterHandle c : counters)
		while (loom::Count(c) > 0)
			loom::Signal(c);
	SpinUntil(0, leaf, 10000);
	r.everyTaskRan = (ran.load() == N);
	r.leafZero = (loom::Count(leaf) == 0);
	for (CounterHandle c : counters) loom::DestroyCounter(c);
	loom::DestroyCounter(leaf);
	pool.Shutdown();
	return r;
}

TEST_CASE("loom: dag — every task ran exactly once")
{
	CHECK_MESSAGE(RunDagScenario().everyTaskRan, "dag: every task ran exactly once");
}

TEST_CASE("loom: dag — leaf counter zero")
{
	CHECK_MESSAGE(RunDagScenario().leafZero, "dag: leaf counter zero");
}

// ---- 4. Affinity: main + dedicated pinning ----

struct AffinityResults
{
	bool mainRanOnMain = false;
	bool dedicatedRanOnTheirThread = false;
};

AffinityResults RunAffinityScenario()
{
	AffinityResults r;
	loom::Pool pool; // the suite owns its pool explicitly
	std::thread::id mainId = std::this_thread::get_id();
	CounterHandle mainDone = loom::CreateCounter(2);
	std::atomic<bool> mainOnMain{ true }, mainRan{ false };
	for (int i = 0; i < 2; i++)
	{
		loom::Submit(
			[&]() {
				if (std::this_thread::get_id() != mainId) mainOnMain = false;
				mainRan = true;
			},
			loom::TaskDesc{ nullptr, nullptr, loom::Affinity::MainThread, {}, mainDone, loom::kInvalidCounter });
	}
	loom::PumpMainThreadTasks();
	r.mainRanOnMain = mainRan && mainOnMain;

	// Dedicated pinning: capture the dedicated thread's id, then verify
	// every pinned task runs on exactly that thread.
	loom::DedicatedConfig dc;
	dc.name = "loom-test-ded";
	loom::DedicatedId ded = loom::RegisterDedicatedThread(dc);

	std::thread::id dedId;
	CounterHandle idDone = loom::CreateCounter(1);
	loom::Submit(
		[&dedId]() { dedId = std::this_thread::get_id(); },
		loom::TaskDesc{ nullptr, nullptr, loom::Affinity::Dedicated, ded, idDone, loom::kInvalidCounter });
	loom::WaitFor(idDone);
	loom::DestroyCounter(idDone);

	CounterHandle dedDone = loom::CreateCounter(3);
	std::atomic<int> wrongThread{ 0 };
	for (int i = 0; i < 3; i++)
	{
		loom::Submit(
			[&dedId, &wrongThread]() {
				if (std::this_thread::get_id() != dedId) wrongThread++;
			},
			loom::TaskDesc{ nullptr, nullptr, loom::Affinity::Dedicated, ded, dedDone, loom::kInvalidCounter });
	}
	loom::WaitFor(dedDone);
	r.dedicatedRanOnTheirThread = (wrongThread.load() == 0);

	loom::UnregisterDedicatedThread(ded);
	loom::DestroyCounter(mainDone);
	loom::DestroyCounter(dedDone);
	pool.Shutdown();
	return r;
}

TEST_CASE("loom: affinity — MainThread tasks ran on the main thread")
{
	CHECK_MESSAGE(RunAffinityScenario().mainRanOnMain, "affinity: MainThread tasks ran on the main thread");
}

TEST_CASE("loom: affinity — Dedicated tasks ran on their thread")
{
	CHECK_MESSAGE(RunAffinityScenario().dedicatedRanOnTheirThread, "affinity: Dedicated tasks ran on their thread");
}

// ---- 5. Help-while-waiting: the waiting MAIN thread completes
// main-affinity work (the deadlock-free main-only property) ----

struct HelpResults
{
	bool mainTaskRan = false;
	bool ranOnMain = false;
};

HelpResults RunHelpScenario()
{
	HelpResults r;
	loom::Pool pool; // the suite owns its pool explicitly
	std::thread::id mainId = std::this_thread::get_id();
	CounterHandle gate = loom::CreateCounter(1);
	std::atomic<bool> mainTaskRan{ false }, onMain{ true };

	// The ONLY way gate hits zero is this MainThread task completing —
	// submitted while the main thread is about to wait on gate. The
	// main's WaitFor must pump the main queue (help) to run it.
	loom::Submit(
		[&]() {
			mainTaskRan = true;
			if (std::this_thread::get_id() != mainId) onMain = false;
		},
		loom::TaskDesc{ nullptr, nullptr, loom::Affinity::MainThread, {}, gate, loom::kInvalidCounter });

	loom::WaitFor(gate);
	r.mainTaskRan = mainTaskRan;
	r.ranOnMain = onMain;
	loom::DestroyCounter(gate);
	pool.Shutdown();
	return r;
}

TEST_CASE("loom: help — main-affinity work completed while main waited")
{
	CHECK_MESSAGE(RunHelpScenario().mainTaskRan, "help: main-affinity work completed while main waited");
}

TEST_CASE("loom: help — it ran on the main thread")
{
	CHECK_MESSAGE(RunHelpScenario().ranOnMain, "help: it ran on the main thread");
}

// ---- 6. Dedicated register/unregister under load (crash canary) ----

TEST_CASE("loom: dedicated — 5 register/unregister rounds under load")
{
	loom::Pool pool; // the suite owns its pool explicitly
	for (int round = 0; round < 5; round++)
	{
		loom::DedicatedConfig dc;
		dc.name = "loom-load-ded";
		loom::DedicatedId ded = loom::RegisterDedicatedThread(dc);
		CounterHandle done = loom::CreateCounter(200);
		for (int i = 0; i < 200; i++)
		{
			loom::Submit(loom::TaskDesc{ &Noop, nullptr,
				loom::Affinity::Dedicated, ded, done, loom::kInvalidCounter });
		}
		loom::WaitFor(done);
		loom::DestroyCounter(done);
		loom::UnregisterDedicatedThread(ded);
	}
	pool.Shutdown();
}

// ---- 7. Inline Submit(F) + BlockOnly wait ----

TEST_CASE("loom: inline Submit(F) ran via a BlockOnly wait")
{
	loom::Pool pool; // the suite owns its pool explicitly
	std::atomic<int> n{ 0 };
	CounterHandle done = loom::CreateCounter(2);
	loom::Submit([&n]() { n += 5; },
		loom::TaskDesc{ nullptr, nullptr, loom::Affinity::AnyWorker, {}, done, loom::kInvalidCounter });
	loom::Submit([&n]() { n += 7; },
		loom::TaskDesc{ nullptr, nullptr, loom::Affinity::AnyWorker, {}, done, loom::kInvalidCounter });
	loom::WaitFor(done, loom::Help::BlockOnly);
	CHECK_MESSAGE(n.load() == 12, "inline Submit(F) ran via a BlockOnly wait");
	loom::DestroyCounter(done);
	pool.Shutdown();
}

// ---- 8. Shutdown under load (drain-or-assert) ----

TEST_CASE("loom: shutdown under load — all tasks completed")
{
	loom::Pool pool; // the suite owns its pool explicitly
	const int N = 5000;
	std::atomic<int> runs{ 0 };
	CounterHandle done = loom::CreateCounter(N);
	for (int i = 0; i < N; i++)
	{
		loom::Submit([&runs]() { runs++; },
			loom::TaskDesc{ nullptr, nullptr, loom::Affinity::AnyWorker, {}, done, loom::kInvalidCounter });
	}
	loom::WaitFor(done);
	loom::DestroyCounter(done);
	pool.Shutdown(); // drains + joins; asserts queues empty
	CHECK_MESSAGE(runs.load() == N, "shutdown under load: all tasks completed");
}

// ---- ParallelFor, ScopedCounter, lanes, profiler zones ----

TEST_CASE("loom: ParallelFor — every index covered exactly once (pointer form)")
{
	loom::Pool pool; // the suite owns its pool explicitly
	const size_t N = 1000;
	std::vector<std::atomic<int>> hits(N);
	for (auto& h : hits) h = 0;

	auto fn = [](size_t begin, size_t end, void* userData) {
		auto& counters = *static_cast<std::vector<std::atomic<int>>*>(userData);
		for (size_t i = begin; i < end; i++)
			counters[i]++;
	};

	for (size_t chunkSize : { size_t(1), size_t(7), size_t(128) })
	{
		loom::ParallelFor(0, N, chunkSize, fn, &hits);
		bool allOnce = true;
		for (size_t i = 0; i < N; i++)
			if (hits[i].load() != 1) { allOnce = false; break; }
		CHECK_MESSAGE(allOnce, "ParallelFor chunkSize ", chunkSize, ": every index hit exactly once");
		for (auto& h : hits) h = 0;
	}
}

TEST_CASE("loom: ParallelFor — the lambda convenience covers the range")
{
	loom::Pool pool; // the suite owns its pool explicitly
	const size_t N = 500;
	std::vector<std::atomic<int>> hits(N);
	for (auto& h : hits) h = 0;

	loom::ParallelFor(0, N, 37, [&hits](size_t begin, size_t end) {
		for (size_t i = begin; i < end; i++)
			hits[i]++;
	});

	bool allOnce = true;
	for (size_t i = 0; i < N; i++)
		if (hits[i].load() != 1) { allOnce = false; break; }
	CHECK_MESSAGE(allOnce, "lambda ParallelFor: every index hit exactly once");
}

TEST_CASE("loom: ParallelFor — nesting fills a product matrix exactly once")
{
	loom::Pool pool; // the suite owns its pool explicitly
	const size_t Rows = 64, Cols = 96;
	std::vector<std::atomic<int>> cells(Rows * Cols);
	for (auto& c : cells) c = 0;

	loom::ParallelFor(0, Rows, 8, [&cells](size_t rowBegin, size_t rowEnd) {
		loom::ParallelFor(0, Cols, 16, [&cells, rowBegin, rowEnd](size_t colBegin, size_t colEnd) {
			for (size_t r = rowBegin; r < rowEnd; r++)
				for (size_t c = colBegin; c < colEnd; c++)
					cells[r * Cols + c]++;
		});
	});

	bool allOnce = true;
	for (size_t i = 0; i < Rows * Cols; i++)
		if (cells[i].load() != 1) { allOnce = false; break; }
	CHECK_MESSAGE(allOnce, "nested ParallelFor: every cell hit exactly once");
}

TEST_CASE("loom: ScopedCounter — RAII fan-in and generation-safe re-acquire")
{
	loom::Pool pool; // the suite owns its pool explicitly
	const int N = 300;

	CounterHandle reused;
	{
		loom::ScopedCounter done(N);
		reused = done.Get();
		for (int i = 0; i < N; i++)
		{
			loom::Submit(loom::TaskDesc{ Noop, nullptr, loom::Affinity::AnyWorker,
				{}, done.Get(), loom::kInvalidCounter });
		}
		loom::WaitFor(done);
		CHECK(loom::Count(done) == 0);
	} // destroyed here (no assert)

	// The next counter gets a different handle even if the slot is reused
	// (the generation bumped on destroy). Initial 0: a ScopedCounter must
	// drain to 0 before its destructor (the destroy-time assert).
	loom::ScopedCounter next(0);
	CHECK(next.Get() != reused);
}

TEST_CASE("loom: counters — a concurrent create never gets a handle the destroy invalidates")
{
	// The sandbox's 1/2000 abort: DestroyCounter cleared `alive`, then bumped
	// the generation, unlocked. A create landing between the two claimed the
	// slot with the OLD generation, so the bump killed the fresh handle and
	// the first Signal/WaitFor asserted on a live counter.
	loom::Pool pool; // the suite owns its pool explicitly
	const int Threads = 4;
	const int Iterations = 200000;
	std::atomic<int> completed{ 0 };
	std::vector<std::thread> threads;
	for (int t = 0; t < Threads; t++)
	{
		threads.emplace_back([&]
		{
			for (int i = 0; i < Iterations; i++)
			{
				const CounterHandle handle = loom::CreateCounter(1);
				loom::Signal(handle, 1); // asserts on a stale handle
				loom::DestroyCounter(handle);
				completed.fetch_add(1, std::memory_order_relaxed);
			}
		});
	}
	for (std::thread& thread : threads)
		thread.join();
	CHECK(completed.load() == Threads * Iterations);
}

TEST_CASE("loom: lanes — exclusive reservation, lifecycle, shutdown wait")
{
	loom::Pool pool; // the suite owns its pool explicitly

	loom::LaneId lane = loom::AcquireLane("test-lane");
	CHECK(lane != loom::kInvalidLane);

	std::atomic<int> laneThread{ -1 };
	CounterHandle done = loom::CreateCounter(64);
	for (int i = 0; i < 64; i++)
	{
		loom::Submit([&laneThread]() { laneThread = loom::CurrentThreadIndex(); },
			loom::TaskDesc{ nullptr, nullptr, loom::Affinity::Dedicated, lane,
				done, loom::kInvalidCounter });
	}
	loom::WaitFor(done);
	loom::DestroyCounter(done);
	CHECK(laneThread.load() >= 0);

	// The lane thread never runs pool work (Idle::Sleep): a burst of
	// AnyWorker tasks completes while the lane's recorded index is
	// unchanged... (the index is per-thread; the lane's index stays the
	// same regardless - the exclusivity guarantee is that the lane never
	// STEALS, exercised by the shutdown assert below staying clean).
	loom::ReleaseLane(lane);

	// Re-acquire/release cycles cleanly.
	for (int i = 0; i < 3; i++)
	{
		loom::LaneId l2 = loom::AcquireLane("cycle-lane");
		CHECK(l2 != loom::kInvalidLane);
		loom::ReleaseLane(l2);
	}

	pool.Shutdown(); // asserts no lanes still held
}

TEST_CASE("loom: park — a thread registered mid-park cannot stand in for a busy one")
{
	using namespace std::chrono_literals;
	loom::Pool pool(loom::PoolConfig{ .workerCount = 1 });
	std::atomic<bool> release{ false }, inTask{ false };
	loom::ScopedCounter busy(1);
	loom::Submit([&] {
		inTask = true;
		while (!release) std::this_thread::sleep_for(1ms);
	}, loom::TaskDesc{ .signals = busy });
	while (!inTask) std::this_thread::sleep_for(1ms);
	std::atomic<int> result{ -1 };
	std::thread requester([&] { result = loom::ParkAll(300) ? 1 : 0; });
	std::this_thread::sleep_for(50ms);
	const loom::DedicatedId late = loom::RegisterDedicatedThread({ .name = "late" }); // parks at once
	requester.join();
	CHECK(result.load() == 0); // the worker never parked: its task still runs
	release = true;
	loom::WaitFor(busy);
	loom::UnregisterDedicatedThread(late);
}

TEST_CASE("loom: park — a parked dedicated thread can be unregistered, and Shutdown ends a park")
{
	using namespace std::chrono_literals;
	{
		loom::Pool pool(loom::PoolConfig{ .workerCount = 2 });
		const loom::DedicatedId d = loom::RegisterDedicatedThread({ .name = "parked" });
		REQUIRE(loom::ParkAll(2000));
		std::atomic<bool> gone{ false };
		std::thread unregister([&] { loom::UnregisterDedicatedThread(d); gone = true; });
		for (int i = 0; i < 200 && !gone; i++)
			std::this_thread::sleep_for(10ms);
		CHECK(gone.load()); // it left the park to exit, rather than wait for UnparkAll
		loom::UnparkAll();
		unregister.join();
	}
	{
		loom::Pool pool(loom::PoolConfig{ .workerCount = 2 });
		const loom::DedicatedId d = loom::RegisterDedicatedThread({ .name = "parked" });
		(void)d;
		REQUIRE(loom::ParkAll(2000));
		// No UnparkAll (an exception skipped it, say): ~Pool must not hang.
	}
}

TEST_CASE("loom: park — unregistering a participant waits for a wake in flight")
{
	using namespace std::chrono_literals;
	loom::Pool pool(loom::PoolConfig{ .workerCount = 1 });
	struct Slow
	{
		std::atomic<bool> inWake{ false }, wakeDone{ false };
	} slow;
	const int id = loom::RegisterParkParticipant([](void* u) {
		auto* p = static_cast<Slow*>(u);
		p->inWake = true;
		std::this_thread::sleep_for(100ms);
		p->wakeDone = true;
	}, &slow);
	std::thread requester([] { (void)loom::ParkAll(200); }); // the participant never checks in: it times out
	while (!slow.inWake) std::this_thread::sleep_for(1ms);
	loom::UnregisterParkParticipant(id);
	CHECK(slow.wakeDone.load()); // `slow` could be freed now
	requester.join();
	if (loom::IsParkRequested())
		loom::UnparkAll();
}

#if defined(NDEBUG)
TEST_CASE("loom: park — a second ParkAll while parked is refused and leaves the first in effect")
{
	loom::Pool pool(loom::PoolConfig{ .workerCount = 2 });
	REQUIRE(loom::ParkAll(2000));
	CHECK_FALSE(loom::ParkAll(300)); // asserts in a debug build
	CHECK(loom::IsParkRequested());
	CHECK(loom::ParkedThreads().size() == 2);
	loom::UnparkAll();
}
#endif

// The park stops every pool thread (workers,
// dedicated threads, lanes, participants) at its loop top; queued work waits.
TEST_CASE("loom: park — every thread parks with its stack, queued work waits, unpark resumes")
{
	loom::Pool pool(loom::PoolConfig{ .workerCount = 3 });
	loom::LaneId lane = loom::AcquireLane("park-lane");

	// A participant outside the pool: a thread that waits on its own cv.
	struct Participant
	{
		std::mutex m;
		std::condition_variable cv;
		std::atomic<bool> stop{ false };
		std::atomic<int> wakes{ 0 };
	} part;
	const int pid = loom::RegisterParkParticipant([](void* u) {
		auto* p = static_cast<Participant*>(u);
		std::lock_guard<std::mutex> lock(p->m);
		p->wakes++;
		p->cv.notify_all();
	}, &part);
	std::thread outsider([&part]() {
		while (!part.stop.load())
		{
			{
				std::unique_lock<std::mutex> lock(part.m);
				part.cv.wait_for(lock, std::chrono::milliseconds(200), [&] { return part.stop.load() || loom::IsParkRequested(); });
			}
			loom::ParkPoint("park-outsider");
		}
	});
	// A failed REQUIRE throws; the outsider must be joined on that path too
	// (a joinable std::thread's destructor is std::terminate).
	struct StopOutsider
	{
		Participant& part;
		std::thread& thread;
		~StopOutsider()
		{
			if (loom::IsParkRequested())
				loom::UnparkAll();
			part.stop = true;
			{
				std::lock_guard<std::mutex> lock(part.m);
				part.cv.notify_all();
			}
			if (thread.joinable())
				thread.join();
		}
	} stopOutsider{ part, outsider };

	REQUIRE(loom::ParkAll(2000));
	const auto parked = loom::ParkedThreads();
	CHECK(parked.size() == 3 + 1 + 1); // the workers, the lane, the participant
	CHECK(part.wakes.load() >= 1);
	for (const auto& t : parked)
	{
		CHECK(t.osThreadId != 0);
		CHECK(!t.returnAddresses.empty());
	}

	std::atomic<int> ran{ 0 };
	CounterHandle done = loom::CreateCounter(16);
	for (int i = 0; i < 16; i++)
		loom::Submit([&ran]() { ran++; }, loom::TaskDesc{ nullptr, nullptr, loom::Affinity::AnyWorker, {}, done, loom::kInvalidCounter });
	std::this_thread::sleep_for(std::chrono::milliseconds(50));
	CHECK(ran.load() == 0); // nothing runs while parked

	loom::UnparkAll();
	loom::WaitFor(done, loom::Help::BlockOnly);
	CHECK(ran.load() == 16);
	loom::DestroyCounter(done);

	// A second park/unpark round works the same.
	REQUIRE(loom::ParkAll(2000));
	CHECK(loom::ParkedThreads().size() == 5);
	loom::UnparkAll();

	loom::UnregisterParkParticipant(pid);
	part.stop = true;
	{
		std::lock_guard<std::mutex> lock(part.m);
		part.cv.notify_all();
	}
	outsider.join();
	loom::ReleaseLane(lane);
	pool.Shutdown();
}

// The back-to-back rounds: UnparkAll followed at once by ParkAll used to leave a
// thread that had not yet woken from the first round asleep in it (it saw the
// request set again), uncounted by the second - a 2 s timeout about 1 run in 40.
TEST_CASE("loom: park — back-to-back park/unpark rounds never strand a thread")
{
	loom::Pool pool(loom::PoolConfig{ .workerCount = 4 });
	loom::LaneId lane = loom::AcquireLane("park-rounds-lane");
	int failedRound = -1;
	for (int round = 0; round < 500 && failedRound < 0; round++)
	{
		if (!loom::ParkAll(2000))
			failedRound = round;
		else
			loom::UnparkAll();
	}
	CHECK(failedRound == -1);
	loom::ReleaseLane(lane);
	pool.Shutdown();
}

// Submit(span) chains runnable AnyWorker tasks into groups handed over as one
// queue entry; whoever takes a group spreads the rest. Every path: the main
// thread's deque, an outside thread's injection queue (taken by a stealing
// dedicated thread too), a worker's own deque, and a span mixing in tasks that
// must NOT be grouped (main, dedicated, waitsFor).
TEST_CASE("loom: batch submit — grouped tasks run exactly once from any thread")
{
	loom::Pool pool; // the suite owns its pool explicitly
	loom::DedicatedId stealer = loom::RegisterDedicatedThread(
		loom::DedicatedConfig{ .name = "batch-stealer", .idle = loom::DedicatedConfig::Idle::StealFromPool });

	auto bump = [](void* p) { static_cast<std::atomic<int>*>(p)->fetch_add(1, std::memory_order_relaxed); };
	auto allOnce = [](const std::vector<std::atomic<int>>& hits) {
		for (const auto& h : hits)
			if (h.load() != 1)
				return false;
		return true;
	};

	// 1. A mixed span from the main thread.
	const int N = 10000;
	std::vector<std::atomic<int>> mixed(N);
	loom::ScopedCounter gate(1);
	loom::ScopedCounter mixedDone(N);
	{
		std::vector<loom::TaskDesc> descs(N);
		for (int i = 0; i < N; i++)
		{
			loom::TaskDesc d{ .fn = bump, .userData = &mixed[i], .signals = mixedDone };
			if (i % 7 == 0)
				d.affinity = loom::Affinity::MainThread;
			else if (i % 11 == 0)
				d.waitsFor = gate;
			else if (i % 13 == 0)
			{
				d.affinity = loom::Affinity::Dedicated;
				d.thread = stealer;
			}
			descs[i] = d;
		}
		loom::Submit(std::span<const loom::TaskDesc>(descs));
	}
	loom::Signal(gate);

	// 2. A span from a thread loom did not start (the injection queue).
	const int M = 5000;
	std::vector<std::atomic<int>> outside(M);
	loom::ScopedCounter outsideDone(M);
	std::thread outsider([&] {
		std::vector<loom::TaskDesc> descs(M);
		for (int i = 0; i < M; i++)
			descs[i] = loom::TaskDesc{ .fn = bump, .userData = &outside[i], .signals = outsideDone };
		loom::Submit(std::span<const loom::TaskDesc>(descs));
	});

	// 3. Spans submitted from inside tasks (a worker's own deque).
	const int Parents = 100, Children = 100;
	std::vector<std::atomic<int>> nested(Parents * Children);
	loom::ScopedCounter nestedDone(Parents * Children);
	loom::ScopedCounter parentsDone(Parents);
	struct Parent { std::atomic<int>* hits; loom::CounterHandle done; void (*bump)(void*); };
	std::vector<Parent> parents(Parents);
	for (int p = 0; p < Parents; p++)
	{
		parents[p] = Parent{ &nested[(size_t)p * Children], nestedDone, +bump };
		loom::Submit(loom::TaskDesc{ .fn = [](void* user) {
			Parent& parent = *static_cast<Parent*>(user);
			loom::TaskDesc kids[100];
			for (int c = 0; c < 100; c++)
				kids[c] = loom::TaskDesc{ .fn = parent.bump, .userData = &parent.hits[c], .signals = parent.done };
			loom::Submit(std::span<const loom::TaskDesc>(kids, 100));
		}, .userData = &parents[p], .signals = parentsDone });
	}

	loom::WaitFor(mixedDone); // pumps the main-thread tasks too
	outsider.join();
	loom::WaitFor(outsideDone);
	loom::WaitFor(parentsDone);
	loom::WaitFor(nestedDone);

	CHECK_MESSAGE(allOnce(mixed), "mixed span: every task ran exactly once");
	CHECK_MESSAGE(allOnce(outside), "outside-thread span: every task ran exactly once");
	CHECK_MESSAGE(allOnce(nested), "spans from tasks: every task ran exactly once");
	loom::UnregisterDedicatedThread(stealer);
}

// A thread applies its finished tasks' signals to a counter in one go (one
// shared-counter write per run of same-counter tasks). These pin the rules
// that keep that invisible: nothing returns to a caller, waits or runs other
// work while holding signals.

TEST_CASE("loom: deferred signals — TryHelp never returns holding signals another thread waits for")
{
	loom::Pool pool(loom::PoolConfig{ .workerCount = 0 }); // only the main thread runs tasks
	loom::ScopedCounter done(8);
	for (int i = 0; i < 8; i++)
		loom::Submit(loom::TaskDesc{ .fn = &Noop, .signals = done });

	std::atomic<bool> finished{ false };
	std::thread waiter([&] {
		loom::WaitFor(done, loom::Help::BlockOnly);
		finished = true;
	});
	for (int i = 0; i < 8; i++)
		CHECK(loom::TryHelp());

	// The waiter must see zero with the main thread doing nothing more.
	const auto start = std::chrono::steady_clock::now();
	while (!finished && std::chrono::steady_clock::now() - start < std::chrono::seconds(2))
		std::this_thread::sleep_for(std::chrono::milliseconds(1));
	CHECK_MESSAGE(finished.load(), "the BlockOnly waiter saw zero after the last TryHelp");
	if (!finished)
		(void)loom::Count(done); // apply anything held back, so the join cannot hang
	waiter.join();
}

TEST_CASE("loom: deferred signals — a long task of another counter does not delay a finished one")
{
	loom::Pool pool(loom::PoolConfig{ .workerCount = 1 });
	loom::ScopedCounter quick(1), slow(1);
	// The worker steals oldest-first: the quick task, then the long one.
	loom::Submit(loom::TaskDesc{ .fn = &Noop, .signals = quick });
	loom::Submit(loom::TaskDesc{ .fn = [](void*) { std::this_thread::sleep_for(std::chrono::milliseconds(400)); }, .signals = slow });

	const auto start = std::chrono::steady_clock::now();
	loom::WaitFor(quick, loom::Help::BlockOnly); // the main thread runs nothing itself
	const auto waited = std::chrono::steady_clock::now() - start;
	CHECK_MESSAGE(waited < std::chrono::milliseconds(200), "the quick counter reached zero before the long task finished");
	loom::WaitFor(slow, loom::Help::BlockOnly);
}

TEST_CASE("loom: deferred signals — a task polling its own counter sees its predecessors' signals")
{
	loom::Pool pool(loom::PoolConfig{ .workerCount = 1 });
	const int Quick = 16;
	loom::ScopedCounter done(Quick + 1);
	std::atomic<bool> sawOnlyItself{ false };
	struct Poll { loom::CounterHandle done; std::atomic<bool>* saw; } poll{ done, &sawOnlyItself };
	for (int i = 0; i < Quick; i++)
		loom::Submit(loom::TaskDesc{ .fn = &Noop, .signals = done });
	// Runs after the quick ones on the same worker, signalling the same counter:
	// it waits until only its own signal is left.
	loom::Submit(loom::TaskDesc{ .fn = [](void* user) {
		Poll& p = *static_cast<Poll*>(user);
		const auto start = std::chrono::steady_clock::now();
		while (std::chrono::steady_clock::now() - start < std::chrono::seconds(2))
		{
			if (loom::Count(p.done) == 1)
			{
				*p.saw = true;
				return;
			}
			std::this_thread::yield();
		}
	}, .userData = &poll, .signals = done });
	loom::WaitFor(done, loom::Help::BlockOnly);
	CHECK_MESSAGE(sawOnlyItself.load(), "Load counted the finished tasks the polling thread still held");
}

TEST_CASE("loom: deferred signals — interleaved counters in grouped batches all reach zero")
{
	loom::Pool pool; // the suite owns its pool explicitly
	const int Counters = 4, PerCounter = 2000;
	std::vector<loom::CounterHandle> counters;
	for (int k = 0; k < Counters; k++)
		counters.push_back(loom::CreateCounter(PerCounter));
	std::atomic<int> ran{ 0 };
	std::vector<loom::TaskDesc> descs;
	for (int i = 0; i < Counters * PerCounter; i++)
		descs.push_back(loom::TaskDesc{ .fn = [](void* p) { static_cast<std::atomic<int>*>(p)->fetch_add(1, std::memory_order_relaxed); },
			.userData = &ran, .signals = counters[(size_t)(i % Counters)] });
	loom::Submit(std::span<const loom::TaskDesc>(descs));
	for (int k = Counters; k-- > 0;)
	{
		loom::WaitFor(counters[(size_t)k]);
		CHECK(loom::Count(counters[(size_t)k]) == 0);
	}
	CHECK(ran.load() == Counters * PerCounter);
	for (loom::CounterHandle c : counters)
		loom::DestroyCounter(c);
}

// Task slots recycle through per-thread caches and a shared stack of batches.
// Every allocation path at once (the main thread's cache, the workers' caches,
// outside threads without one) while the pool grows past its first chunk.
TEST_CASE("loom: slots — recycling and growth under every submitter keep every task exactly once")
{
	loom::Pool pool(loom::PoolConfig{ .maxTasks = 1 }); // one chunk, so the bursts below grow it
	auto bump = [](void* p) { static_cast<std::atomic<int>*>(p)->fetch_add(1, std::memory_order_relaxed); };
	const int Rounds = 3, PerSource = 12000;
	for (int round = 0; round < Rounds; round++)
	{
		std::vector<std::atomic<int>> fromMain(PerSource), fromOutside(2 * PerSource), fromTasks(PerSource);
		loom::ScopedCounter mainDone(PerSource), outsideDone(2 * PerSource), taskDone(PerSource), spawnersDone(PerSource / 100);

		// The workers parked: nothing runs, so the main thread's burst (and the
		// outside threads' first submits) hold more slots than the pool has.
		REQUIRE(loom::ParkAll(2000));
		std::vector<loom::TaskDesc> descs((size_t)PerSource);
		for (int i = 0; i < PerSource; i++)
			descs[(size_t)i] = loom::TaskDesc{ .fn = bump, .userData = &fromMain[(size_t)i], .signals = mainDone };
		loom::Submit(std::span<const loom::TaskDesc>(descs));

		// Outside threads: no slot cache, one slot per allocation.
		std::vector<std::thread> outsiders;
		for (int t = 0; t < 2; t++)
		{
			outsiders.emplace_back([&, t] {
				for (int i = 0; i < PerSource; i++)
					loom::Submit(loom::TaskDesc{ .fn = bump, .userData = &fromOutside[(size_t)(t * PerSource + i)], .signals = outsideDone });
			});
		}
		std::this_thread::sleep_for(std::chrono::milliseconds(5));
		loom::UnparkAll();
		// Spans submitted from tasks: the workers' caches.
		struct Spawn { std::atomic<int>* hits; loom::CounterHandle done; void (*bump)(void*); };
		std::vector<Spawn> spawns((size_t)PerSource / 100);
		for (size_t s = 0; s < spawns.size(); s++)
		{
			spawns[s] = Spawn{ &fromTasks[s * 100], taskDone, +bump };
			loom::Submit(loom::TaskDesc{ .fn = [](void* user) {
				Spawn& sp = *static_cast<Spawn*>(user);
				loom::TaskDesc kids[100];
				for (int c = 0; c < 100; c++)
					kids[c] = loom::TaskDesc{ .fn = sp.bump, .userData = &sp.hits[c], .signals = sp.done };
				loom::Submit(std::span<const loom::TaskDesc>(kids, 100));
			}, .userData = &spawns[s], .signals = spawnersDone });
		}
		for (auto& t : outsiders)
			t.join();
		loom::WaitFor(mainDone);
		loom::WaitFor(outsideDone);
		loom::WaitFor(spawnersDone);
		loom::WaitFor(taskDone);

		auto allOnce = [](const std::vector<std::atomic<int>>& hits) {
			for (const auto& h : hits)
				if (h.load() != 1)
					return false;
			return true;
		};
		CHECK_MESSAGE(allOnce(fromMain), "round ", round, ": main-thread tasks ran exactly once");
		CHECK_MESSAGE(allOnce(fromOutside), "round ", round, ": outside-thread tasks ran exactly once");
		CHECK_MESSAGE(allOnce(fromTasks), "round ", round, ": tasks submitted from tasks ran exactly once");
	}
	// Every slot was handed back: nothing is live once all the counters are done
	// (the spawner tasks finish before their counter's final signal).
	const auto start = std::chrono::steady_clock::now();
	while (loom::LiveTaskCount() != 0 && std::chrono::steady_clock::now() - start < std::chrono::seconds(2))
		std::this_thread::yield();
	CHECK(loom::LiveTaskCount() == 0);
}

#if defined(__linux__)
TEST_CASE("loom: thread names — a long name is cut to the OS limit, loom keeps it whole")
{
	loom::Pool pool(loom::PoolConfig{ .workerCount = 1 });
	const char* longName = "a-dedicated-thread-name-26";
	loom::DedicatedId ded = loom::RegisterDedicatedThread(loom::DedicatedConfig{ .name = longName });
	char osName[64] = {};
	loom::ScopedCounter done(1);
	loom::Submit(loom::TaskDesc{ .fn = [](void* out) { pthread_getname_np(pthread_self(), static_cast<char*>(out), 64); },
		.userData = osName, .affinity = loom::Affinity::Dedicated, .thread = ded, .signals = done });
	loom::WaitFor(done);
	CHECK(std::string(osName) == std::string(longName).substr(0, 15));

	bool statsHaveFullName = false;
	for (const auto& t : loom::SnapshotPool().threads)
		statsHaveFullName |= t.name == longName;
	CHECK_MESSAGE(statsHaveFullName, "the pool's thread stats keep the full name");
	loom::UnregisterDedicatedThread(ded);
}
#endif

TEST_CASE("loom: thread stats — helping is counted and shown, rows outlive their threads")
{
	loom::Pool pool(loom::PoolConfig{ .workerCount = 1 });
	struct Probe { std::atomic<int> sawHelping{ 0 }; loom::CounterHandle kids = loom::kInvalidCounter; } probe;
	loom::ScopedCounter outerDone(1);
	// One worker runs the outer task, which waits for 8 children on its own
	// deque: it runs every child while helping (the main thread only blocks).
	loom::Submit(loom::TaskDesc{ .fn = [](void* user) {
		Probe& p = *static_cast<Probe*>(user);
		p.kids = loom::CreateCounter(8);
		loom::TaskDesc kids[8];
		for (auto& k : kids)
			k = loom::TaskDesc{ .fn = [](void* u) {
				Probe& pr = *static_cast<Probe*>(u);
				const int me = loom::CurrentThreadIndex();
				for (const auto& t : loom::SnapshotPool().threads)
					if (t.index == me && t.state == loom::ThreadState::Helping)
						pr.sawHelping++;
			}, .userData = &p, .signals = p.kids };
		loom::Submit(std::span<const loom::TaskDesc>(kids, 8));
		loom::WaitFor(p.kids);
		loom::DestroyCounter(p.kids);
	}, .userData = &probe, .signals = outerDone });
	loom::WaitFor(outerDone, loom::Help::BlockOnly);

	CHECK_MESSAGE(probe.sawHelping.load() == 8, "every child saw its thread as Helping");
	bool found = false;
	for (const auto& t : loom::SnapshotPool().threads)
	{
		if (t.name.rfind("loom-worker", 0) != 0)
			continue;
		found = true;
		CHECK(t.tasksHelped == 8);
		CHECK(t.tasksRun == 9);
		CHECK(t.state != loom::ThreadState::Helping);
	}
	CHECK(found);

	// A dedicated thread's row stays readable after the thread is gone.
	loom::DedicatedId ded = loom::RegisterDedicatedThread(loom::DedicatedConfig{ .name = "stats-ded" });
	loom::ScopedCounter dedDone(1);
	loom::Submit(loom::TaskDesc{ .fn = &Noop, .affinity = loom::Affinity::Dedicated, .thread = ded, .signals = dedDone });
	loom::WaitFor(dedDone);
	loom::UnregisterDedicatedThread(ded);
	bool rowKept = false;
	for (const auto& t : loom::SnapshotPool().threads)
		rowKept |= t.name == "stats-ded";
	CHECK(rowKept);
}

TEST_CASE("loom: thread stats — two dedicated threads with one name keep their own queue depths")
{
	loom::Pool pool(loom::PoolConfig{ .workerCount = 1 });
	const loom::DedicatedId busy = loom::RegisterDedicatedThread(loom::DedicatedConfig{ .name = "twin" });
	const loom::DedicatedId quiet = loom::RegisterDedicatedThread(loom::DedicatedConfig{ .name = "twin" });
	struct Gate { std::atomic<bool> open{ false }; } gate;
	loom::ScopedCounter done(4);
	loom::Submit(loom::TaskDesc{ .fn = [](void* g) {
		while (!static_cast<Gate*>(g)->open)
			std::this_thread::yield();
	}, .userData = &gate, .affinity = loom::Affinity::Dedicated, .thread = busy, .signals = done });
	for (int i = 0; i < 3; i++)
		loom::Submit(loom::TaskDesc{ .fn = &Noop, .affinity = loom::Affinity::Dedicated, .thread = busy, .signals = done });

	// Both rows exist once both threads have started.
	std::vector<loom::ThreadStatsView> twins;
	for (int tries = 0; tries < 2000 && twins.size() != 2; tries++)
	{
		twins.clear();
		for (const auto& t : loom::SnapshotPool().threads)
			if (t.name == "twin")
				twins.push_back(t);
		if (twins.size() != 2)
			std::this_thread::sleep_for(std::chrono::milliseconds(1));
	}
	REQUIRE(twins.size() == 2);
	const int deep = std::max(twins[0].queueDepth, twins[1].queueDepth);
	const int shallow = std::min(twins[0].queueDepth, twins[1].queueDepth);
	CHECK(deep >= 3); // 3, or 4 if the busy thread has not taken the gate task yet
	CHECK(shallow == 0);

	gate.open = true;
	loom::WaitFor(done);
	loom::UnregisterDedicatedThread(busy);
	loom::UnregisterDedicatedThread(quiet);
}

TEST_CASE("loom: TryWaitFor — times out, returns early on zero, and rethrows")
{
	using Clock = std::chrono::steady_clock;
	loom::Pool pool(loom::PoolConfig{ .workerCount = 2 });
	for (loom::Help help : { loom::Help::WhileWaiting, loom::Help::BlockOnly })
	{
		loom::ScopedCounter never(1);
		const auto start = Clock::now();
		CHECK_FALSE(loom::TryWaitFor(never, 30, help));
		const auto waited = Clock::now() - start;
		CHECK(waited >= std::chrono::milliseconds(30));
		CHECK(waited < std::chrono::seconds(2));
		CHECK_FALSE(loom::TryWaitFor(never, 0, help));
		loom::Signal(never);
		CHECK(loom::TryWaitFor(never, 0, help));

		// Woken by the signal, long before the limit.
		loom::ScopedCounter soon(1);
		loom::Submit([&soon]() {
			std::this_thread::sleep_for(std::chrono::milliseconds(20));
			loom::Signal(soon);
		});
		const auto before = Clock::now();
		CHECK(loom::TryWaitFor(soon, 10000, help));
		CHECK(Clock::now() - before < std::chrono::seconds(5));
	}

#if defined(__cpp_exceptions)
	loom::ScopedCounter failed(1);
	loom::Submit([]() { throw std::runtime_error("timed"); }, loom::TaskDesc{ .signals = failed });
	CHECK_THROWS_AS((void)loom::TryWaitFor(failed, 10000), std::runtime_error);
#endif
}

TEST_CASE("loom: TryWaitFor — a zero limit never helps; a longer one does")
{
	loom::Pool pool(loom::PoolConfig{ .workerCount = 0 }); // only a helping wait runs AnyWorker work
	std::atomic<int> ran{ 0 };
	loom::ScopedCounter done(1);
	loom::Submit([&ran]() { ran++; }, loom::TaskDesc{ .signals = done });
	CHECK_FALSE(loom::TryWaitFor(done, 0));
	CHECK(ran.load() == 0);
	CHECK(loom::TryWaitFor(done, 10000));
	CHECK(ran.load() == 1);
}

TEST_CASE("loom: dedicated — submits stay safe while other dedicated threads come and go")
{
	loom::Pool pool(loom::PoolConfig{ .workerCount = 2 });
	loom::DedicatedId steady = loom::RegisterDedicatedThread(loom::DedicatedConfig{ .name = "steady" });

	// Ids are never reused: a stale id cannot reach a newer thread.
	loom::DedicatedId first = loom::RegisterDedicatedThread(loom::DedicatedConfig{ .name = "first" });
	loom::UnregisterDedicatedThread(first);
	loom::DedicatedId second = loom::RegisterDedicatedThread(loom::DedicatedConfig{ .name = "second" });
	CHECK(second != first);
	loom::UnregisterDedicatedThread(second);

	// One thread churns the registry (its vector reallocates) while the main
	// thread keeps submitting to the steady thread.
	std::atomic<bool> stop{ false };
	std::thread churn([&] {
		while (!stop)
		{
			loom::DedicatedId d = loom::RegisterDedicatedThread(loom::DedicatedConfig{ .name = "churn" });
			loom::CounterHandle c = loom::CreateCounter(4);
			for (int i = 0; i < 4; i++)
				loom::Submit(loom::TaskDesc{ .fn = &Noop, .affinity = loom::Affinity::Dedicated, .thread = d, .signals = c });
			loom::WaitFor(c, loom::Help::BlockOnly);
			loom::DestroyCounter(c);
			loom::UnregisterDedicatedThread(d);
		}
	});
	std::atomic<int> ran{ 0 };
	const int N = 20000;
	loom::ScopedCounter done(N);
	for (int i = 0; i < N; i++)
		loom::Submit(loom::TaskDesc{ .fn = [](void* p) { static_cast<std::atomic<int>*>(p)->fetch_add(1); },
			.userData = &ran, .affinity = loom::Affinity::Dedicated, .thread = steady, .signals = done });
	loom::WaitFor(done);
	stop = true;
	churn.join();
	CHECK(ran.load() == N);
	loom::UnregisterDedicatedThread(steady);
}

TEST_CASE("loom: shutdown — drains first: a task's wait completes and late work runs")
{
	struct State
	{
		std::atomic<bool> childRan{ false };
		std::atomic<int>  countAfterWait{ -1 };
		std::atomic<bool> mainWorkRan{ false };
		std::atomic<bool> mainWorkOnMain{ false };
		std::thread::id   mainId;
	} st;
	st.mainId = std::this_thread::get_id();
	{
		loom::Pool pool(loom::PoolConfig{ .workerCount = 2 });
		// A task that waits for its child, blocking (no helping), while
		// Shutdown runs: the wait must end at zero, not when stopping starts.
		loom::Submit([&st]() {
			loom::CounterHandle child = loom::CreateCounter(1);
			loom::Submit([&st]() {
				std::this_thread::sleep_for(std::chrono::milliseconds(50));
				st.childRan = true;
			}, loom::TaskDesc{ .signals = child });
			loom::WaitFor(child, loom::Help::BlockOnly);
			st.countAfterWait = loom::Count(child);
			loom::DestroyCounter(child);
		});
		// A task that submits main-thread work after Shutdown has begun.
		loom::Submit([&st]() {
			std::this_thread::sleep_for(std::chrono::milliseconds(30));
			loom::Submit([&st]() {
				st.mainWorkOnMain = std::this_thread::get_id() == st.mainId;
				st.mainWorkRan = true;
			}, loom::TaskDesc{ .affinity = loom::Affinity::MainThread });
		});
		pool.Shutdown(); // at once: both tasks are still running
	}
	CHECK(st.childRan.load());
	CHECK_MESSAGE(st.countAfterWait.load() == 0, "the task's WaitFor returned only at zero");
	CHECK_MESSAGE(st.mainWorkRan.load(), "main-thread work submitted during Shutdown ran");
	CHECK(st.mainWorkOnMain.load());
}

#if defined(__cpp_exceptions)
TEST_CASE("loom: exceptions — WaitFor rethrows a task's exception, for every waiter")
{
	loom::Pool pool(loom::PoolConfig{ .workerCount = 2 });
	loom::CounterHandle c = loom::CreateCounter(3);
	std::atomic<int> ran{ 0 };
	loom::Submit([&ran]() { ran++; }, loom::TaskDesc{ .signals = c });
	loom::Submit([&ran]() { ran++; throw std::runtime_error("boom"); }, loom::TaskDesc{ .signals = c });
	loom::Submit([&ran]() { ran++; }, loom::TaskDesc{ .signals = c });

	std::string otherSaw;
	std::thread other([&] {
		try { loom::WaitFor(c, loom::Help::BlockOnly); }
		catch (const std::runtime_error& e) { otherSaw = e.what(); }
	});
	std::string mainSaw;
	try { loom::WaitFor(c); }
	catch (const std::runtime_error& e) { mainSaw = e.what(); }
	other.join();
	CHECK(mainSaw == "boom");
	CHECK(otherSaw == "boom");
	CHECK(ran.load() >= 1); // the throw cancelled the counter: tasks not yet started were skipped
	CHECK(ran.load() <= 3);
	CHECK(loom::IsCancelled(c));
	CHECK(loom::Count(c) == 0);
	loom::DestroyCounter(c);

	// A re-used counter starts clean.
	loom::CounterHandle fresh = loom::CreateCounter(1);
	loom::Submit(loom::TaskDesc{ .fn = &Noop, .signals = fresh });
	CHECK_NOTHROW(loom::WaitFor(fresh));
	loom::DestroyCounter(fresh);
}

TEST_CASE("loom: exceptions — a helper keeps other counters' exceptions out of its own wait")
{
	loom::Pool pool(loom::PoolConfig{ .workerCount = 0 }); // the main thread runs everything
	loom::ScopedCounter good(1), bad(1);
	loom::Submit(loom::TaskDesc{ .fn = &Noop, .signals = good });
	loom::Submit([]() { throw std::logic_error("bad"); }, loom::TaskDesc{ .signals = bad }); // runs first (newest)
	CHECK_NOTHROW(loom::WaitFor(good));
	CHECK_THROWS_AS(loom::WaitFor(bad), std::logic_error);
}

TEST_CASE("loom: exceptions — ParallelFor, nested or not, rethrows; chunks not yet started are skipped")
{
	loom::Pool pool; // the suite owns its pool explicitly
	const size_t N = 1000;
	std::vector<std::atomic<int>> hits(N);
	CHECK_THROWS_WITH_AS(loom::ParallelFor(0, N, 10, [&hits](size_t b, size_t e) {
		for (size_t i = b; i < e; i++)
			hits[i]++;
		if (b == 500)
			throw std::runtime_error("chunk 50");
	}), "chunk 50", std::runtime_error);
	bool atMostOnce = true;
	for (auto& h : hits)
		atMostOnce &= h.load() <= 1;
	CHECK_MESSAGE(atMostOnce, "no index ran twice");
	CHECK_MESSAGE(hits[500].load() == 1, "the chunk that threw ran");

	CHECK_THROWS_AS(loom::ParallelFor(0, 64, 8, [](size_t b, size_t) {
		loom::ParallelFor(0, 64, 8, [b](size_t ib, size_t) {
			if (b == 32 && ib == 16)
				throw std::out_of_range("inner");
		});
	}), std::out_of_range);
	CHECK(loom::LiveTaskCount() == 0);
}
#endif

TEST_CASE("loom: dependencies — a task waits for every counter it lists")
{
	loom::Pool pool; // the suite owns its pool explicitly
	for (int round = 0; round < 50; round++)
	{
		struct State { std::atomic<bool> released[3]; std::atomic<int> runs{ 0 }; std::atomic<bool> early{ false }; } st;
		for (auto& r : st.released)
			r = false;
		loom::CounterHandle a = loom::CreateCounter(1), b = loom::CreateCounter(1), c = loom::CreateCounter(1);
		loom::ScopedCounter done(1);
		const loom::CounterHandle more[] = { b, c };
		loom::Submit(loom::TaskDesc{ .fn = [](void* p) {
			State& s = *static_cast<State*>(p);
			for (auto& r : s.released)
				if (!r)
					s.early = true;
			s.runs++;
		}, .userData = &st, .signals = done, .waitsFor = a, .waitsForAll = more });
		// Release the three from three threads, in a different order each round.
		const loom::CounterHandle order[3] = { a, b, c };
		std::vector<std::thread> releasers;
		for (int k = 0; k < 3; k++)
		{
			const int which = (k + round) % 3;
			releasers.emplace_back([&, which] {
				std::this_thread::sleep_for(std::chrono::microseconds(50 * ((which + round) % 4)));
				st.released[which] = true;
				loom::Signal(order[which]);
			});
		}
		for (auto& t : releasers)
			t.join();
		loom::WaitFor(done);
		CHECK_MESSAGE(!st.early.load(), "round ", round, ": the task ran only after all three counters reached zero");
		CHECK(st.runs.load() == 1);
		loom::DestroyCounter(a);
		loom::DestroyCounter(b);
		loom::DestroyCounter(c);
	}
}

TEST_CASE("loom: dependencies — counters already at zero, and one listed twice")
{
	loom::Pool pool; // the suite owns its pool explicitly
	loom::ScopedCounter zero(0), gate(1), done(2);
	std::atomic<int> runs{ 0 };
	auto bump = [](void* p) { static_cast<std::atomic<int>*>(p)->fetch_add(1); };
	const loom::CounterHandle allZero[] = { zero, zero };
	loom::Submit(loom::TaskDesc{ .fn = bump, .userData = &runs, .signals = done, .waitsFor = zero, .waitsForAll = allZero });
	const loom::CounterHandle twice[] = { gate, zero, gate };
	loom::Submit(loom::TaskDesc{ .fn = bump, .userData = &runs, .signals = done, .waitsForAll = twice });
	SpinUntil(1, done, 2000); // the first ran at once; the second still waits on the gate
	CHECK(runs.load() == 1);
	loom::Signal(gate);
	loom::WaitFor(done);
	CHECK(runs.load() == 2);
}

TEST_CASE("loom: dependencies — a random DAG with several counters per task runs each task once")
{
	loom::Pool pool; // the suite owns its pool explicitly
	std::mt19937 rng(4242);
	const int Counters = 64, Tasks = 3000;
	std::vector<loom::CounterHandle> gates;
	for (int i = 0; i < Counters; i++)
		gates.push_back(loom::CreateCounter(1 + (int)(rng() % 3)));
	std::vector<std::atomic<int>> runs(Tasks);
	loom::ScopedCounter done(Tasks);
	for (int t = 0; t < Tasks; t++)
	{
		loom::CounterHandle deps[4];
		const int n = 1 + (int)(rng() % 4);
		for (int k = 0; k < n; k++)
			deps[k] = gates[rng() % Counters];
		loom::Submit(loom::TaskDesc{ .fn = [](void* p) { static_cast<std::atomic<int>*>(p)->fetch_add(1); },
			.userData = &runs[(size_t)t], .signals = done, .waitsForAll = std::span<const loom::CounterHandle>(deps, (size_t)n) });
	}
	// Two threads release every gate, interleaved.
	auto release = [&](int parity) {
		for (int i = parity; i < Counters; i += 2)
			while (loom::Count(gates[(size_t)i]) > 0)
				loom::Signal(gates[(size_t)i]);
	};
	std::thread other(release, 1);
	release(0);
	other.join();
	loom::WaitFor(done);
	bool allOnce = true;
	for (auto& r : runs)
		allOnce &= r.load() == 1;
	CHECK(allOnce);
	for (loom::CounterHandle g : gates)
		loom::DestroyCounter(g);
}

namespace {
	// One worker, held by a gate task until released: what it runs next, and
	// in which order, is up to the priorities alone (the main thread only
	// blocks).
	struct OrderLog
	{
		std::mutex m;
		std::vector<char> order;
		std::atomic<bool> open{ false };
		void Add(char c) { std::lock_guard<std::mutex> l(m); order.push_back(c); }
	};
	struct Tagged { OrderLog* log; char tag; };
	void LogTag(void* p) { auto* t = static_cast<Tagged*>(p); t->log->Add(t->tag); }
	void HoldUntilOpen(void* p)
	{
		auto* log = static_cast<OrderLog*>(p);
		while (!log->open)
			std::this_thread::yield();
	}
}

TEST_CASE("loom: priorities — High before Normal before Low")
{
	loom::Pool pool(loom::PoolConfig{ .workerCount = 1 });
	OrderLog log;
	loom::ScopedCounter gate(1);
	loom::Submit(loom::TaskDesc{ .fn = &HoldUntilOpen, .userData = &log, .signals = gate });
	std::this_thread::sleep_for(std::chrono::milliseconds(20)); // (let the worker take it; if not, it runs first and logs nothing)

	const int PerLevel = 6;
	std::vector<Tagged> tags;
	tags.reserve(3 * PerLevel);
	loom::ScopedCounter done(3 * PerLevel);
	const loom::Priority levels[3] = { loom::Priority::Low, loom::Priority::Normal, loom::Priority::High };
	const char names[3] = { 'L', 'N', 'H' };
	for (int i = 0; i < PerLevel; i++)
		for (int l = 0; l < 3; l++) // interleaved: L, N, H, L, N, H, ...
		{
			tags.push_back(Tagged{ &log, names[l] });
			loom::Submit(loom::TaskDesc{ .fn = &LogTag, .userData = &tags.back(), .signals = done, .priority = levels[l] });
		}
	log.open = true;
	loom::WaitFor(gate, loom::Help::BlockOnly);
	loom::WaitFor(done, loom::Help::BlockOnly);
	std::string order(log.order.begin(), log.order.end());
	CHECK(order == std::string(PerLevel, 'H') + std::string(PerLevel, 'N') + std::string(PerLevel, 'L'));
}

TEST_CASE("loom: priorities — a High task released later still runs before queued Normal work")
{
	loom::Pool pool(loom::PoolConfig{ .workerCount = 1 });
	OrderLog log;
	loom::ScopedCounter gate(1), release(1);
	loom::Submit(loom::TaskDesc{ .fn = &HoldUntilOpen, .userData = &log, .signals = gate });
	std::this_thread::sleep_for(std::chrono::milliseconds(20));
	std::vector<Tagged> tags(5, Tagged{ &log, 'N' });
	Tagged high{ &log, 'H' };
	loom::ScopedCounter done(6);
	for (auto& t : tags)
		loom::Submit(loom::TaskDesc{ .fn = &LogTag, .userData = &t, .signals = done });
	loom::Submit(loom::TaskDesc{ .fn = &LogTag, .userData = &high, .signals = done, .waitsFor = release, .priority = loom::Priority::High });
	loom::Signal(release);
	log.open = true;
	loom::WaitFor(done, loom::Help::BlockOnly);
	loom::WaitFor(gate, loom::Help::BlockOnly);
	CHECK(std::string(log.order.begin(), log.order.end()) == "HNNNNN");
}

TEST_CASE("loom: priorities — High goes to the front of the main thread's queue")
{
	loom::Pool pool(loom::PoolConfig{ .workerCount = 1 });
	OrderLog log;
	Tagged n1{ &log, '1' }, n2{ &log, '2' }, h{ &log, 'H' };
	loom::ScopedCounter done(3);
	loom::Submit(loom::TaskDesc{ .fn = &LogTag, .userData = &n1, .affinity = loom::Affinity::MainThread, .signals = done });
	loom::Submit(loom::TaskDesc{ .fn = &LogTag, .userData = &n2, .affinity = loom::Affinity::MainThread, .signals = done });
	loom::Submit(loom::TaskDesc{ .fn = &LogTag, .userData = &h, .affinity = loom::Affinity::MainThread, .signals = done, .priority = loom::Priority::High });
	loom::PumpMainThreadTasks();
	CHECK(std::string(log.order.begin(), log.order.end()) == "H12");
}

TEST_CASE("loom: cancellation — queued tasks are skipped, still signal, and a fresh counter is clean")
{
	loom::Pool pool(loom::PoolConfig{ .workerCount = 1 });
	OrderLog hold;
	loom::ScopedCounter gate(1);
	loom::Submit(loom::TaskDesc{ .fn = &HoldUntilOpen, .userData = &hold, .signals = gate });
	std::atomic<int> ran{ 0 };
	loom::CounterHandle c = loom::CreateCounter(10);
	for (int i = 0; i < 10; i++)
		loom::Submit([&ran]() { ran++; }, loom::TaskDesc{ .signals = c });
	CHECK_FALSE(loom::IsCancelled(c));
	loom::Cancel(c);
	CHECK(loom::IsCancelled(c));
	hold.open = true;
	loom::WaitFor(c, loom::Help::BlockOnly); // returns: every skipped task still signalled
	CHECK(ran.load() == 0);
	CHECK(loom::Count(c) == 0);
	loom::DestroyCounter(c);
	loom::WaitFor(gate, loom::Help::BlockOnly);

	loom::CounterHandle fresh = loom::CreateCounter(1);
	CHECK_FALSE(loom::IsCancelled(fresh));
	loom::Submit([&ran]() { ran++; }, loom::TaskDesc{ .signals = fresh });
	loom::WaitFor(fresh);
	CHECK(ran.load() == 1);
	loom::DestroyCounter(fresh);
}

TEST_CASE("loom: cancellation — a running task can see it and stop early")
{
	loom::Pool pool(loom::PoolConfig{ .workerCount = 1 });
	struct Long { loom::CounterHandle self; std::atomic<bool> started{ false }, sawCancel{ false }; } job;
	job.self = loom::CreateCounter(1);
	loom::Submit(loom::TaskDesc{ .fn = [](void* p) {
		Long& j = *static_cast<Long*>(p);
		j.started = true;
		const auto start = std::chrono::steady_clock::now();
		while (std::chrono::steady_clock::now() - start < std::chrono::seconds(2))
		{
			if (loom::IsCancelled(j.self))
			{
				j.sawCancel = true;
				return;
			}
			std::this_thread::yield();
		}
	}, .userData = &job, .signals = job.self });
	while (!job.started)
		std::this_thread::yield();
	const auto start = std::chrono::steady_clock::now();
	loom::Cancel(job.self);
	loom::WaitFor(job.self, loom::Help::BlockOnly);
	CHECK(job.sawCancel.load());
	CHECK(std::chrono::steady_clock::now() - start < std::chrono::milliseconds(500));
	loom::DestroyCounter(job.self);
}

#if defined(__cpp_exceptions)
TEST_CASE("loom: cancellation — a task's exception skips the rest of its counter's queued work")
{
	loom::Pool pool(loom::PoolConfig{ .workerCount = 1 });
	OrderLog hold;
	loom::ScopedCounter gate(1);
	loom::Submit(loom::TaskDesc{ .fn = &HoldUntilOpen, .userData = &hold, .signals = gate });
	std::atomic<int> after{ 0 };
	loom::CounterHandle c = loom::CreateCounter(6);
	// The one worker takes these oldest first: the throw, then the rest.
	loom::Submit([]() { throw std::runtime_error("first"); }, loom::TaskDesc{ .signals = c });
	for (int i = 0; i < 5; i++)
		loom::Submit([&after]() { after++; }, loom::TaskDesc{ .signals = c });
	hold.open = true;
	CHECK_THROWS_AS(loom::WaitFor(c, loom::Help::BlockOnly), std::runtime_error);
	CHECK(after.load() == 0);
	loom::DestroyCounter(c);
	loom::WaitFor(gate, loom::Help::BlockOnly);
}
#endif

namespace {
	// Counts its live instances, so a test can see every copy destroyed.
	struct Tracked
	{
		static inline std::atomic<int> live{ 0 };
		Tracked() { live++; }
		Tracked(const Tracked&) { live++; }
		Tracked(Tracked&&) noexcept { live++; }
		Tracked& operator=(const Tracked&) = default; // assigning makes no new object
		Tracked& operator=(Tracked&&) noexcept = default;
		~Tracked() { live--; }
	};
}

TEST_CASE("loom: Submit(F) — non-trivial, large and over-aligned callables run once and are destroyed")
{
	loom::Pool pool(loom::PoolConfig{ .workerCount = 2 });
	Tracked::live = 0;
	auto shared = std::make_shared<int>(0);
	std::atomic<int> sum{ 0 };
	loom::ScopedCounter done(3);
	{
		// Moved into the slot: a shared_ptr and a Tracked, well under 40 bytes.
		Tracked tracked;
		loom::Submit([shared, tracked, &sum]() { (void)tracked; sum += 1 + *shared; }, loom::TaskDesc{ .signals = done });
		// Too big for the slot: on the heap.
		std::array<int, 64> big{};
		for (int i = 0; i < 64; i++)
			big[(size_t)i] = i;
		loom::Submit([big, tracked, &sum]() {
			int s = 0;
			for (int v : big)
				s += v;
			sum += s;
		}, loom::TaskDesc{ .signals = done });
		// Aligned past the slot's 16 bytes: on the heap, and still aligned.
		struct alignas(64) Wide
		{
			std::atomic<int>* sum;
			void operator()() const { *sum += ((uintptr_t)this % 64 == 0) ? 100 : -1000; }
		};
		loom::Submit(Wide{ &sum }, loom::TaskDesc{ .signals = done });
	}
	loom::WaitFor(done);
	CHECK(sum.load() == 1 + 2016 + 100);
	CHECK_MESSAGE(shared.use_count() == 1, "the task's copy of the shared_ptr is gone by the time WaitFor returns");
	CHECK(Tracked::live.load() == 0);
}

TEST_CASE("loom: Submit(F) — an owned callable is destroyed when cancelled or when it throws")
{
	loom::Pool pool(loom::PoolConfig{ .workerCount = 1 });
	Tracked::live = 0;
	OrderLog hold;
	loom::ScopedCounter gate(1);
	loom::Submit(loom::TaskDesc{ .fn = &HoldUntilOpen, .userData = &hold, .signals = gate });
	std::atomic<int> ran{ 0 };
	loom::CounterHandle c = loom::CreateCounter(6);
	{
		Tracked tracked;
		std::array<char, 100> big{};
		for (int i = 0; i < 3; i++)
		{
			loom::Submit([tracked, &ran]() { (void)tracked; ran++; }, loom::TaskDesc{ .signals = c });
			loom::Submit([tracked, big, &ran]() { (void)tracked; (void)big; ran++; }, loom::TaskDesc{ .signals = c });
		}
	}
	CHECK(Tracked::live.load() == 6); // one per queued task
	loom::Cancel(c);
	hold.open = true;
	loom::WaitFor(c);
	CHECK(ran.load() == 0);
	CHECK_MESSAGE(Tracked::live.load() == 0, "skipped tasks still destroyed their callables");
	loom::DestroyCounter(c);
	loom::WaitFor(gate);

#if defined(__cpp_exceptions)
	loom::ScopedCounter thrown(1);
	{
		Tracked tracked;
		loom::Submit([tracked]() { (void)tracked; throw std::runtime_error("owned"); }, loom::TaskDesc{ .signals = thrown });
	}
	CHECK_THROWS_AS(loom::WaitFor(thrown), std::runtime_error);
	CHECK(Tracked::live.load() == 0);
#endif
}

TEST_CASE("loom: isolation — an isolated wait skips a foreign task on its own deque")
{
	loom::Pool pool(loom::PoolConfig{ .workerCount = 1 });
	OrderLog hold; // keeps the one worker busy until the checks are done
	loom::ScopedCounter gate(1);
	loom::Submit(loom::TaskDesc{ .fn = &HoldUntilOpen, .userData = &hold, .signals = gate });
	std::this_thread::sleep_for(std::chrono::milliseconds(20));

	struct Foreign { std::thread::id mainId; std::atomic<bool> ranOnMain{ false }, ran{ false }; } foreign;
	foreign.mainId = std::this_thread::get_id();
	// A long task from outside the region, held until `release`: signalling
	// it from inside the region puts it on top of the main thread's deque,
	// above the region's own task.
	loom::ScopedCounter release(1), foreignDone(1), mine(1);
	loom::Submit(loom::TaskDesc{ .fn = [](void* p) {
		Foreign& f = *static_cast<Foreign*>(p);
		f.ranOnMain = std::this_thread::get_id() == f.mainId;
		std::this_thread::sleep_for(std::chrono::milliseconds(300));
		f.ran = true;
	}, .userData = &foreign, .signals = foreignDone, .waitsFor = release });

	std::chrono::steady_clock::duration waited{};
	loom::Isolated([&] {
		loom::Submit(loom::TaskDesc{ .fn = &Noop, .signals = mine });
		loom::Signal(release);
		const auto start = std::chrono::steady_clock::now();
		loom::WaitFor(mine); // helps: must run its own task, not the long one above it
		waited = std::chrono::steady_clock::now() - start;
	});
	CHECK_MESSAGE(waited < std::chrono::milliseconds(150), "the isolated wait did not run the foreign 300 ms task");
	CHECK_FALSE(foreign.ran.load());
	hold.open = true; // now the worker takes the foreign task
	loom::WaitFor(foreignDone, loom::Help::BlockOnly);
	CHECK(foreign.ran.load());
	CHECK_FALSE(foreign.ranOnMain.load());
	loom::WaitFor(gate, loom::Help::BlockOnly);
}

TEST_CASE("loom: isolation — a region's tasks, and their own waits, stay in the region")
{
	loom::Pool pool(loom::PoolConfig{ .workerCount = 1 });
	OrderLog hold; // the worker is held: only the main thread runs anything
	loom::ScopedCounter gate(1);
	loom::Submit(loom::TaskDesc{ .fn = &HoldUntilOpen, .userData = &hold, .signals = gate });
	std::this_thread::sleep_for(std::chrono::milliseconds(20));

	struct State
	{
		loom::CounterHandle release = loom::kInvalidCounter;
		std::atomic<bool> foreignRan{ false };
		std::chrono::steady_clock::duration childWait{};
	} st;
	loom::ScopedCounter release(1), foreignDone(1), parentDone(1);
	st.release = release;
	loom::Submit(loom::TaskDesc{ .fn = [](void* p) {
		std::this_thread::sleep_for(std::chrono::milliseconds(300));
		static_cast<State*>(p)->foreignRan = true;
	}, .userData = &st, .signals = foreignDone, .waitsFor = release }); // outside the region

	loom::Isolated([&] {
		// The parent runs on the main thread (its wait helps). Inside it, a
		// child, then the foreign task released on top of it: the parent's own
		// wait must be isolated to the region (skip the foreign task) and must
		// count the child as the region's (or nobody could run it).
		loom::Submit(loom::TaskDesc{ .fn = [](void* p) {
			State& s = *static_cast<State*>(p);
			loom::CounterHandle child = loom::CreateCounter(1);
			loom::Submit(loom::TaskDesc{ .fn = &Noop, .signals = child });
			loom::Signal(s.release);
			const auto start = std::chrono::steady_clock::now();
			loom::WaitFor(child);
			s.childWait = std::chrono::steady_clock::now() - start;
			loom::DestroyCounter(child);
		}, .userData = &st, .signals = parentDone });
		loom::WaitFor(parentDone);

		std::atomic<int> cells{ 0 }; // nested ParallelFor inside a region
		loom::ParallelFor(0, 4, 1, [&cells](size_t, size_t) {
			loom::ParallelFor(0, 4, 1, [&cells](size_t, size_t) { cells++; });
		});
		CHECK(cells.load() == 16);
	});
	CHECK_MESSAGE(st.childWait < std::chrono::milliseconds(150), "the task's own wait skipped the foreign task");
	CHECK_FALSE(st.foreignRan.load());
	hold.open = true;
	loom::WaitFor(foreignDone, loom::Help::BlockOnly);
	loom::WaitFor(gate, loom::Help::BlockOnly);
}

TEST_CASE("loom: isolation — without workers a region still waits on outside work")
{
	loom::Pool pool(loom::PoolConfig{ .workerCount = 0 });
	loom::ScopedCounter outside(1);
	loom::Submit(loom::TaskDesc{ .fn = &Noop, .signals = outside }); // not in the region
	loom::Isolated([&] { loom::WaitFor(outside); }); // nobody else could run it: no isolation
	CHECK(loom::Count(outside) == 0);
}

TEST_CASE("loom: ParallelFor — automatic chunking covers every index once, any size, any pool")
{
	for (int workers : { 0, 1, 3, -1 })
	{
		loom::Pool pool(loom::PoolConfig{ .workerCount = workers });
		for (size_t n : { size_t(1), size_t(2), size_t(7), size_t(1000), size_t(100000), size_t(1000000) })
		{
			std::vector<std::atomic<uint8_t>> hits(n);
			loom::ParallelFor(0, n, [&hits](size_t b, size_t e) {
				for (size_t i = b; i < e; i++)
					hits[i].fetch_add(1, std::memory_order_relaxed);
			});
			bool allOnce = true;
			for (auto& h : hits)
				allOnce &= h.load() == 1;
			CHECK_MESSAGE(allOnce, "workers ", workers, ", n ", n, ": every index exactly once");
		}
		// The pointer form with chunkSize 0, over an offset range.
		std::vector<std::atomic<int>> hits(500);
		loom::ParallelFor(100, 600, 0, [](size_t b, size_t e, void* user) {
			auto& h = *static_cast<std::vector<std::atomic<int>>*>(user);
			for (size_t i = b; i < e; i++)
				h[i - 100]++;
		}, &hits);
		bool allOnce = true;
		for (auto& h : hits)
			allOnce &= h.load() == 1;
		CHECK(allOnce);
	}
}

TEST_CASE("loom: ParallelFor — automatic chunking spreads uneven work and nests")
{
	loom::Pool pool(loom::PoolConfig{ .workerCount = 3 });
	// The last few items are expensive: a fixed one-piece-per-thread split
	// would leave them on one thread.
	std::mutex m;
	std::set<int> heavyThreads;
	loom::ParallelFor(0, 4096, [&](size_t b, size_t e) {
		for (size_t i = b; i < e; i++)
			if (i >= 4096 - 64)
			{
				std::this_thread::sleep_for(std::chrono::milliseconds(2));
				std::lock_guard<std::mutex> l(m);
				heavyThreads.insert(loom::CurrentThreadIndex());
			}
	});
	CHECK_MESSAGE(heavyThreads.size() >= 2, "the expensive tail ran on ", heavyThreads.size(), " thread(s)");

	std::atomic<int> cells{ 0 };
	loom::ParallelFor(0, 64, [&cells](size_t b, size_t e) {
		for (size_t i = b; i < e; i++)
			loom::ParallelFor(0, 64, [&cells](size_t ib, size_t ie) { cells += (int)(ie - ib); });
	});
	CHECK(cells.load() == 64 * 64);
}

#if defined(__cpp_exceptions)
TEST_CASE("loom: ParallelFor — automatic chunking rethrows and runs no index twice")
{
	loom::Pool pool; // the suite owns its pool explicitly
	std::vector<std::atomic<int>> hits(100000);
	CHECK_THROWS_AS(loom::ParallelFor(0, hits.size(), [&hits](size_t b, size_t e) {
		for (size_t i = b; i < e; i++)
		{
			hits[i]++;
			if (i == 54321)
				throw std::runtime_error("auto");
		}
	}), std::runtime_error);
	bool atMostOnce = true;
	for (auto& h : hits)
		atMostOnce &= h.load() <= 1;
	CHECK(atMostOnce);
	// Automatic chunking counts items, so the task that signalled the last
	// ones may still be finishing its own bookkeeping as ParallelFor returns.
	const auto start = std::chrono::steady_clock::now();
	while (loom::LiveTaskCount() != 0 && std::chrono::steady_clock::now() - start < std::chrono::seconds(2))
		std::this_thread::yield();
	CHECK(loom::LiveTaskCount() == 0);
}
#endif

TEST_CASE("loom: ParallelReduce — the same bits on every run and with any number of workers")
{
	const size_t N = 1000000;
	auto map = [](size_t b, size_t e) {
		double sum = 0.0;
		for (size_t i = b; i < e; i++)
			sum += 1.0 / (double)(i + 1);
		return sum;
	};
	auto add = [](double a, double b) { return a + b; };
	// The reference: the same chunks folded in order, serially.
	const size_t chunk = (N - 1) / 256 + 1;
	double expected = 0.0;
	for (size_t b = 0; b < N; b += chunk)
		expected = add(expected, map(b, std::min(N, b + chunk)));

	for (int workers : { 0, 1, 4, -1 })
	{
		loom::Pool pool(loom::PoolConfig{ .workerCount = workers });
		for (int run = 0; run < 3; run++)
		{
			const double got = loom::ParallelReduce(0, N, 0, 0.0, map, add);
			CHECK_MESSAGE(std::memcmp(&got, &expected, sizeof got) == 0, "workers ", workers, ", run ", run, ": bit-identical sum");
		}
	}

	// The combine runs in index order: a non-commutative one shows it.
	loom::Pool pool; // the suite owns its pool explicitly
	const std::string joined = loom::ParallelReduce(0, 40, 3, std::string(),
		[](size_t b, size_t e) { std::string s; for (size_t i = b; i < e; i++) s += (char)('A' + (i % 26)); return s; },
		[](std::string a, std::string b) { return a + b; });
	std::string serial;
	for (size_t i = 0; i < 40; i++)
		serial += (char)('A' + (i % 26));
	CHECK(joined == serial);
	CHECK(loom::ParallelReduce(5, 5, 0, 42, [](size_t, size_t) { return 0; }, [](int a, int b) { return a + b; }) == 42);
}

// ---- Task graphs ----

namespace {
	// A random DAG over `n` nodes: edges only go forward in a shuffled order,
	// so there is never a cycle, and the shuffle keeps node ids out of order.
	struct RandomDag
	{
		std::vector<std::pair<uint32_t, uint32_t>> edges;
		RandomDag(uint32_t n, uint32_t edgesPerNode, uint32_t seed)
		{
			std::mt19937 rng(seed);
			std::vector<uint32_t> order(n);
			for (uint32_t i = 0; i < n; i++)
				order[i] = i;
			std::shuffle(order.begin(), order.end(), rng);
			for (uint32_t i = 1; i < n; i++)
				for (uint32_t k = 0; k < edgesPerNode; k++)
					edges.emplace_back(order[rng() % i], order[i]);
		}
	};
}

TEST_CASE("loom: graphs — every node runs once per run, after its predecessors, run after run")
{
	for (int workers : { 0, 1, -1 })
	{
		loom::Pool pool(loom::PoolConfig{ .workerCount = workers });
		const uint32_t N = 300;
		const RandomDag dag(N, 3, 77);
		std::atomic<uint32_t> ticket{ 0 };
		std::vector<std::atomic<uint32_t>> stamp(N), runs(N);
		loom::TaskGraph graph;
		for (uint32_t i = 0; i < N; i++)
			CHECK(graph.Add([&, i] { stamp[i] = ticket++; runs[i]++; }) == i);
		for (auto [a, b] : dag.edges)
			graph.Precede(a, b);
		CHECK(graph.Validate());
		for (int run = 0; run < 40; run++)
		{
			loom::Run(graph);
			CHECK(!graph.IsRunning());
			bool ordered = true;
			for (auto [a, b] : dag.edges)
				ordered = ordered && stamp[a] < stamp[b];
			CHECK_MESSAGE(ordered, "workers ", workers, " run ", run, ": a node started before a predecessor finished");
		}
		bool once = true;
		for (auto& r : runs)
			once = once && r == 40;
		CHECK_MESSAGE(once, "every node ran exactly once per run");
		CHECK(loom::LiveTaskCount() >= 0);
	}
}

TEST_CASE("loom: graphs — dense layers share a wait; mixed affinities and repeated edges in one set")
{
	loom::Pool pool(loom::PoolConfig{ .workerCount = 3 });
	// Layer by layer, every node after every node of the layer before: each
	// layer is one wait set. One member per layer runs on the main thread and
	// one is High, so a set is queued partly in groups and partly one by one.
	const int L = 6, W = 40;
	std::atomic<int> finished[L] = {};
	std::atomic<bool> early{ false }, offMain{ false };
	loom::TaskGraph graph;
	std::vector<loom::TaskGraph::Node> layer[L];
	for (int l = 0; l < L; l++)
		for (int i = 0; i < W; i++)
		{
			loom::TaskDesc opts;
			if (i == 0)
				opts.affinity = loom::Affinity::MainThread;
			if (i == 1)
				opts.priority = loom::Priority::High;
			layer[l].push_back(graph.Add([&, l, i] {
				if (l > 0 && finished[l - 1].load() != W)
					early = true;
				if (i == 0 && loom::CurrentThreadIndex() != 0)
					offMain = true;
				finished[l]++;
			}, opts));
		}
	for (int l = 1; l < L; l++)
		for (auto a : layer[l - 1])
			for (auto b : layer[l])
			{
				graph.Precede(a, b);
				if (b == layer[l][2])
					graph.Precede(a, b); // listed twice: still one wait per predecessor
			}
	// A tail after the last layer's first node only: a one-predecessor set.
	std::atomic<int> tail{ 0 };
	graph.Precede(layer[L - 1][5], graph.Add([&] {
		if (finished[L - 1].load() == 0)
			early = true;
		tail++;
	}));
	for (int run = 0; run < 30; run++)
	{
		for (auto& f : finished)
			f = 0;
		loom::Run(graph);
		for (int l = 0; l < L; l++)
			CHECK(finished[l].load() == W);
	}
	CHECK_MESSAGE(!early.load(), "a node started before its whole previous layer finished");
	CHECK(!offMain.load());
	CHECK(tail.load() == 30);
}

TEST_CASE("loom: graphs — nodes keep their affinity, and one graph runs inside another's node")
{
	loom::Pool pool(loom::PoolConfig{ .workerCount = 2 });
	const loom::DedicatedId io = loom::RegisterDedicatedThread(loom::DedicatedConfig{ .name = "graph-io" });
	std::atomic<int> mainIndex{ -1 }, dedicatedIndex{ -1 };
	std::atomic<int> innerRuns{ 0 };

	loom::TaskGraph inner;
	const auto i0 = inner.Add([&] { innerRuns++; });
	const auto i1 = inner.Add([&] { innerRuns++; });
	inner.Precede(i0, i1);

	loom::TaskGraph outer;
	const auto first = outer.Add([&] { loom::Run(inner); }); // a nested run: the worker helps
	const auto onMain = outer.Add([&] { mainIndex = loom::CurrentThreadIndex(); }, loom::TaskDesc{ .affinity = loom::Affinity::MainThread });
	const auto onIo = outer.Add([&] { dedicatedIndex = loom::CurrentThreadIndex(); },
		loom::TaskDesc{ .affinity = loom::Affinity::Dedicated, .thread = io, .priority = loom::Priority::High });
	outer.Precede(first, onMain);
	outer.Precede(first, onIo);
	for (int run = 0; run < 20; run++)
		loom::Isolated([&] { loom::Run(outer); });
	CHECK(innerRuns.load() == 40);
	CHECK(mainIndex.load() == 0);
	CHECK(dedicatedIndex.load() > 0);
	loom::UnregisterDedicatedThread(io);
}

TEST_CASE("loom: graphs — Submit waits for its counter and signals one it may share")
{
	loom::Pool pool(loom::PoolConfig{ .workerCount = 2 });
	std::atomic<int> ran{ 0 };
	loom::TaskGraph graph;
	const auto a = graph.Add([&] { ran++; });
	const auto b = graph.Add([&] { ran++; });
	graph.Precede(a, b);

	loom::ScopedCounter gate(1), done(2); // the run and one plain task
	loom::Submit(graph, done, gate);
	loom::Submit([&] { ran += 10; }, loom::TaskDesc{ .signals = done });
	std::this_thread::sleep_for(std::chrono::milliseconds(20));
	CHECK_MESSAGE(ran.load() == 10, "no node runs before the gate opens");
	CHECK(graph.IsRunning());
	loom::Signal(gate);
	loom::WaitFor(done);
	CHECK(ran.load() == 12);
	CHECK(!graph.IsRunning());

	// An empty graph completes at once, and still signals.
	loom::TaskGraph empty;
	loom::ScopedCounter emptyDone(1);
	loom::Submit(empty, emptyDone);
	CHECK(loom::TryWaitFor(emptyDone, 1000));
	loom::Run(empty);
}

#if defined(__cpp_exceptions)
TEST_CASE("loom: graphs — an exception or a Cancel skips the nodes not yet started; the graph runs again")
{
	loom::Pool pool(loom::PoolConfig{ .workerCount = 2 });
	std::atomic<int> c{ 0 }, d{ 0 };
	bool throwNow = true;
	loom::TaskGraph graph;
	const auto a = graph.Add([] {});
	const auto b = graph.Add([&] { if (throwNow) throw std::runtime_error("b"); });
	const auto nc = graph.Add([&] { c++; });
	const auto nd = graph.Add([&] { d++; });
	graph.Precede(a, b);
	graph.Precede(b, nc);
	graph.Precede(nc, nd);
	CHECK_THROWS_AS(loom::Run(graph), std::runtime_error);
	CHECK(c.load() == 0);
	CHECK(d.load() == 0);
	CHECK(!graph.IsRunning());
	throwNow = false;
	loom::Run(graph);
	CHECK(c.load() == 1);
	CHECK(d.load() == 1);

	// Cancel the run's counter while its first node runs.
	std::atomic<bool> started{ false }, cancelled{ false };
	loom::TaskGraph slow;
	const auto s0 = slow.Add([&] {
		started = true;
		while (!cancelled)
			std::this_thread::yield();
	});
	std::atomic<int> after{ 0 };
	for (int i = 0; i < 5; i++)
		slow.Precede(s0, slow.Add([&] { after++; }));
	loom::ScopedCounter done(1);
	loom::Submit(slow, done);
	while (!started)
		std::this_thread::yield();
	loom::Cancel(done);
	cancelled = true;
	loom::WaitFor(done);
	CHECK(after.load() == 0);
	CHECK(!slow.IsRunning());
}
#endif

TEST_CASE("loom: graphs — cycles are found, and callables live as long as the graph")
{
	Tracked::live = 0;
	{
		loom::TaskGraph graph; // built before any pool: building needs none
		Tracked tracked;
		const auto a = graph.Add([tracked] { (void)tracked; });
		const auto b = graph.Add([tracked] { (void)tracked; });
		const auto c = graph.Add([tracked] { (void)tracked; });
		graph.Precede(a, b);
		graph.Precede(b, c);
		CHECK(graph.Validate());
		graph.Precede(c, a);
		CHECK(!graph.Validate());
		CHECK(Tracked::live.load() == 4);
		graph.Clear();
		CHECK(Tracked::live.load() == 1);
		CHECK(graph.NodeCount() == 0);
		CHECK(graph.Validate());
		graph.Add([tracked] { (void)tracked; });
		loom::TaskGraph moved(std::move(graph));
		CHECK(moved.NodeCount() == 1);
		CHECK(graph.NodeCount() == 0);
		graph.Add([] {}); // a moved-from graph is empty and usable
		CHECK(graph.NodeCount() == 1);

		loom::Pool pool(loom::PoolConfig{ .workerCount = 1 });
		loom::Run(moved);
		loom::Run(moved);
		CHECK(Tracked::live.load() == 2);
	}
	CHECK(Tracked::live.load() == 0);
}

namespace {
	// How many tasks of a limiter run at once, and the most seen.
	struct Concurrency
	{
		std::atomic<int> now{ 0 }, most{ 0 }, ran{ 0 };
		void Enter()
		{
			const int n = ++now;
			int seen = most.load();
			while (n > seen && !most.compare_exchange_weak(seen, n)) {}
		}
		void Leave() { now--; ran++; }
	};

	void SpinFor(std::chrono::microseconds d)
	{
		const auto until = std::chrono::steady_clock::now() + d;
		while (std::chrono::steady_clock::now() < until) {}
	}
}

TEST_CASE("loom: limiters — never more than the limit at once, every task once, any pool")
{
	for (int workers : { 0, 1, -1 })
	{
		for (uint32_t limit : { 1u, 3u })
		{
			loom::Pool pool(loom::PoolConfig{ .workerCount = workers });
			loom::ScopedLimiter limiter(limit);
			Concurrency c;
			constexpr int kTasks = 300;
			std::vector<std::atomic<int>> runs(kTasks);
			struct Job { Concurrency* c; std::atomic<int>* runs; };
			std::vector<Job> jobs(kTasks);
			auto body = [](void* p) {
				Job& j = *static_cast<Job*>(p);
				j.c->Enter();
				SpinFor(std::chrono::microseconds(30));
				j.runs->fetch_add(1);
				j.c->Leave();
			};
			loom::ScopedCounter done(kTasks);
			// A third one by one, a third as one batch, a third through Submit(F).
			std::vector<loom::TaskDesc> batch;
			for (int i = 0; i < kTasks; i++)
			{
				jobs[(size_t)i] = Job{ &c, &runs[(size_t)i] };
				const loom::TaskDesc d{ .fn = body, .userData = &jobs[(size_t)i], .signals = done, .limiter = limiter };
				if (i % 3 == 0)
					loom::Submit(d);
				else if (i % 3 == 1)
					batch.push_back(d);
				else
					loom::Submit([&job = jobs[(size_t)i], body] { body(&job); }, loom::TaskDesc{ .signals = done, .limiter = limiter });
			}
			loom::Submit(std::span<const loom::TaskDesc>(batch));
			loom::WaitFor(done);
			for (auto& r : runs)
				CHECK(r.load() == 1);
			CHECK(c.ran.load() == kTasks);
			CHECK(c.most.load() <= (int)limit);
			if (workers < 0 && limit > 1)
				CHECK_MESSAGE(c.most.load() >= 2, "a limit above one still runs tasks side by side");
			CHECK(loom::HeldUnits(limiter) == 0);
			CHECK(loom::WaitingTasks(limiter) == 0);
		}
	}
}

TEST_CASE("loom: limiters — a limit of one is a serial queue, High first, then in order")
{
	loom::Pool pool(loom::PoolConfig{ .workerCount = 4 });
	loom::ScopedLimiter serial(1);
	std::atomic<bool> release{ false }, blocking{ false };
	loom::ScopedCounter done(1);
	loom::Submit([&] {
		blocking = true;
		while (!release)
			std::this_thread::yield();
	}, loom::TaskDesc{ .signals = done, .limiter = serial });
	while (!blocking)
		std::this_thread::yield();

	// Serial: no lock needed, each task finishes before the next is queued.
	std::vector<int> order;
	constexpr int kTasks = 50;
	loom::ScopedCounter queued(kTasks + 1);
	for (int i = 0; i < kTasks; i++)
		loom::Submit([&order, i] { order.push_back(i); }, loom::TaskDesc{ .signals = queued, .limiter = serial });
	loom::Submit([&order] { order.push_back(-1); }, loom::TaskDesc{ .signals = queued, .priority = loom::Priority::High, .limiter = serial });
	CHECK(loom::HeldUnits(serial) == 1);
	CHECK(loom::WaitingTasks(serial) == kTasks + 1);
	release = true;
	loom::WaitFor(queued);
	loom::WaitFor(done);
	REQUIRE(order.size() == kTasks + 1);
	CHECK(order[0] == -1);
	for (int i = 0; i < kTasks; i++)
		CHECK(order[(size_t)i + 1] == i);
}

TEST_CASE("loom: limiters — a task takes its unit only once its counters reach zero")
{
	loom::Pool pool(loom::PoolConfig{ .workerCount = 2 });
	loom::ScopedLimiter limiter(1);
	loom::ScopedCounter gate(1), other(1), gatedDone(1), freeDone(1);
	std::atomic<int> gatedRan{ 0 };
	const loom::CounterHandle all[] = { other };
	loom::Submit([&] { gatedRan++; }, loom::TaskDesc{ .signals = gatedDone, .waitsFor = gate, .waitsForAll = all, .limiter = limiter });
	CHECK(loom::HeldUnits(limiter) == 0);
	// The gated task holds no unit, so this one runs.
	loom::Submit([] {}, loom::TaskDesc{ .signals = freeDone, .limiter = limiter });
	CHECK(loom::TryWaitFor(freeDone, 5000));
	CHECK(gatedRan.load() == 0);
	loom::Signal(gate);
	loom::Signal(other);
	loom::WaitFor(gatedDone);
	CHECK(gatedRan.load() == 1);
	CHECK(loom::HeldUnits(limiter) == 0);
}

TEST_CASE("loom: limiters — many submitters at once; tasks of every affinity and graph nodes")
{
	loom::Pool pool(loom::PoolConfig{ .workerCount = 4 });
	const loom::DedicatedId io = loom::RegisterDedicatedThread(loom::DedicatedConfig{ .name = "limit-io" });
	loom::ScopedLimiter limiter(2);
	Concurrency c;

	// Tiny tasks submitted from eight tasks at once: units are taken and
	// handed over from every thread together.
	constexpr int kSpawners = 8, kEach = 400;
	loom::ScopedCounter done(kSpawners * kEach);
	loom::ScopedCounter spawned(kSpawners);
	for (int s = 0; s < kSpawners; s++)
		loom::Submit([&] {
			for (int i = 0; i < kEach; i++)
			{
				const loom::Affinity affinity = i % 50 == 0 ? loom::Affinity::MainThread : loom::Affinity::AnyWorker;
				if (i % 50 == 25)
					loom::Submit([&c] { c.Enter(); c.Leave(); }, loom::TaskDesc{ .affinity = loom::Affinity::Dedicated, .thread = io, .signals = done, .limiter = limiter });
				else
					loom::Submit([&c] { c.Enter(); c.Leave(); }, loom::TaskDesc{ .affinity = affinity, .signals = done, .limiter = limiter });
			}
		}, loom::TaskDesc{ .signals = spawned });
	loom::WaitFor(spawned);
	loom::WaitFor(done);
	CHECK(c.ran.load() == kSpawners * kEach);
	CHECK(c.most.load() <= 2);
	CHECK(loom::HeldUnits(limiter) == 0);

	// Graph nodes: a wide layer under the limit, and nodes without one.
	Concurrency g;
	std::atomic<int> unlimited{ 0 };
	loom::TaskGraph graph;
	const auto root = graph.Add([] {});
	for (int i = 0; i < 64; i++)
	{
		const auto n = graph.Add([&g] { g.Enter(); SpinFor(std::chrono::microseconds(10)); g.Leave(); }, loom::TaskDesc{ .limiter = limiter });
		graph.Precede(root, n);
		graph.Precede(root, graph.Add([&unlimited] { unlimited++; }));
	}
	for (int run = 0; run < 10; run++)
		loom::Run(graph);
	CHECK(g.ran.load() == 640);
	CHECK(unlimited.load() == 640);
	CHECK(g.most.load() <= 2);
	CHECK(loom::HeldUnits(limiter) == 0);
	loom::UnregisterDedicatedThread(io);
}

TEST_CASE("loom: limiters — a cancelled or throwing task gives its unit back")
{
	loom::Pool pool(loom::PoolConfig{ .workerCount = 2 });
	loom::ScopedLimiter limiter(1);
	std::atomic<bool> started{ false }, release{ false };
	std::atomic<int> after{ 0 };
	loom::ScopedCounter done(11);
	loom::Submit([&] {
		started = true;
		while (!release)
			std::this_thread::yield();
	}, loom::TaskDesc{ .signals = done, .limiter = limiter });
	for (int i = 0; i < 10; i++)
		loom::Submit([&] { after++; }, loom::TaskDesc{ .signals = done, .limiter = limiter });
	while (!started)
		std::this_thread::yield();
	loom::Cancel(done);
	release = true;
	loom::WaitFor(done);
	CHECK(after.load() == 0);
	CHECK(loom::HeldUnits(limiter) == 0);
	CHECK(loom::WaitingTasks(limiter) == 0);

#if defined(__cpp_exceptions)
	loom::ScopedCounter failing(3);
	loom::Submit([] { throw std::runtime_error("limited"); }, loom::TaskDesc{ .signals = failing, .limiter = limiter });
	loom::Submit([] {}, loom::TaskDesc{ .signals = failing, .limiter = limiter });
	loom::Submit([] {}, loom::TaskDesc{ .signals = failing, .limiter = limiter });
	CHECK_THROWS_AS(loom::WaitFor(failing), std::runtime_error);
	CHECK(loom::HeldUnits(limiter) == 0);
	// The unit is free for the next task.
	loom::ScopedCounter next(1);
	std::atomic<int> nextRan{ 0 };
	loom::Submit([&] { nextRan++; }, loom::TaskDesc{ .signals = next, .limiter = limiter });
	loom::WaitFor(next);
	CHECK(nextRan.load() == 1);
#endif
}

TEST_CASE("loom: limiters — handles are reused safely and the table size is configurable")
{
	loom::Pool pool(loom::PoolConfig{ .workerCount = 1, .maxLimiters = 3 });
	for (int round = 0; round < 100; round++)
	{
		loom::ScopedLimiter a(1), b(2), c(5);
		CHECK(a.Get() != b.Get());
		CHECK(b.Get() != c.Get());
		CHECK(loom::HeldUnits(c) == 0);
		loom::ScopedCounter done(3);
		for (loom::LimiterHandle h : { a.Get(), b.Get(), c.Get() })
			loom::Submit([] {}, loom::TaskDesc{ .signals = done, .limiter = h });
		loom::WaitFor(done);
	}
}

namespace {
	int LiveCounters() { return loom::SnapshotPool().counterCount; }
	// A future's task drops its hold on the future's state just after the
	// future's value is ready, so a future dropped right after Get may leave
	// its counter to that task to free, a moment later.
	bool CountersReturnTo(int baseline)
	{
		const auto until = std::chrono::steady_clock::now() + std::chrono::seconds(10);
		while (LiveCounters() != baseline && std::chrono::steady_clock::now() < until)
			std::this_thread::yield();
		return LiveCounters() == baseline;
	}
}

TEST_CASE("loom: counters — a task released by a counter already sees it at zero")
{
	// The final signal releases many waiting tasks one by one; the first ones
	// can start on other workers while it is still releasing the rest.
	loom::Pool pool(loom::PoolConfig{ .workerCount = 4 });
	std::atomic<int> notZero{ 0 };
	for (int round = 0; round < 200; round++)
	{
		loom::ScopedCounter gate(1);
		constexpr int kWaiting = 64;
		loom::ScopedCounter done(kWaiting);
		for (int i = 0; i < kWaiting; i++)
			loom::Submit([&notZero, g = gate.Get()] {
				if (loom::Count(g) != 0)
					notZero++;
			}, loom::TaskDesc{ .signals = done, .waitsFor = gate });
		loom::Signal(gate);
		loom::WaitFor(done);
	}
	CHECK(notZero.load() == 0);
}

namespace {
	// A tree walked by tasks that each add their children to one counter: node
	// i has children 4i+1 .. 4i+4 below `nodes`.
	struct TreeWalk
	{
		loom::CounterHandle done = loom::kInvalidCounter;
		size_t nodes = 0;
		std::atomic<size_t> visited{ 0 };
		std::atomic<int> refused{ 0 };
		std::atomic<int> peakCounters{ 0 };
		size_t throwAt = SIZE_MAX;

		static void Visit(TreeWalk* w, size_t i)
		{
			w->visited++;
			if (i == w->throwAt)
				throw std::runtime_error("bad node");
			for (size_t c = 4 * i + 1; c <= 4 * i + 4 && c < w->nodes; c++)
			{
				if (!loom::AddToCounter(w->done))
					w->refused++;
				loom::Submit([w, c] { Visit(w, c); }, loom::TaskDesc{ .signals = w->done });
			}
			if (i % 997 == 0)
			{
				const int live = loom::SnapshotPool().counterCount;
				int seen = w->peakCounters.load();
				while (live > seen && !w->peakCounters.compare_exchange_weak(seen, live)) {}
			}
		}
	};
}

TEST_CASE("loom: counters — AddToCounter: a whole tree of tasks on one counter, no task waiting")
{
	for (int workers : { 0, 1, -1 })
	{
		loom::Pool pool(loom::PoolConfig{ .workerCount = workers, .maxTasks = 1 << 16 });
		const int baseline = LiveCounters();
		TreeWalk walk;
		walk.nodes = 200000;
		loom::ScopedCounter done(1);
		walk.done = done.Get();
		loom::Submit([w = &walk] { TreeWalk::Visit(w, 0); }, loom::TaskDesc{ .signals = done });
		loom::WaitFor(done);
		CHECK(walk.visited.load() == walk.nodes);
		CHECK(walk.refused.load() == 0);
		CHECK(walk.peakCounters.load() <= baseline + 1); // the one counter, however deep the tree
		CHECK(!loom::AddToCounter(done)); // reached zero: refused
		CHECK(loom::Count(done) == 0);
	}
}

TEST_CASE("loom: counters — AddToCounter keeps waiters asleep, refuses once zero, works with Cancel and exceptions")
{
	loom::Pool pool(loom::PoolConfig{ .workerCount = 3 });
	{
		// A thread asleep on the counter stays asleep while adds keep it up.
		loom::ScopedCounter c(1);
		std::atomic<bool> returned{ false };
		std::thread waiter([&] { loom::WaitFor(c.Get(), loom::Help::BlockOnly); returned = true; });
		CHECK(loom::AddToCounter(c, 5));
		CHECK(loom::Count(c) == 6);
		loom::Signal(c, 5);
		CHECK(loom::Count(c) == 1);
		CHECK(loom::AddToCounter(c, 0)); // still open
		std::this_thread::sleep_for(std::chrono::milliseconds(20));
		CHECK(!returned.load());
		loom::Signal(c);
		waiter.join();
		CHECK(returned.load());
		CHECK(!loom::AddToCounter(c, 1));
		CHECK(!loom::AddToCounter(c, 0));
		CHECK(loom::Count(c) == 0);
	}
	{
		// Added tasks of a cancelled counter are skipped and still signal it.
		loom::ScopedCounter c(1);
		std::atomic<int> ran{ 0 };
		loom::Cancel(c);
		REQUIRE(loom::AddToCounter(c, 50));
		for (int i = 0; i < 50; i++)
			loom::Submit([&] { ran++; }, loom::TaskDesc{ .signals = c });
		loom::Signal(c);
		loom::WaitFor(c);
		CHECK(ran.load() == 0);
	}
#if defined(__cpp_exceptions)
	{
		// A node that throws cancels the counter: the nodes not yet started are
		// skipped (adding no children), so the count still comes down to zero.
		TreeWalk walk;
		walk.nodes = 100000;
		walk.throwAt = 40;
		loom::ScopedCounter done(1);
		walk.done = done.Get();
		loom::Submit([w = &walk] { TreeWalk::Visit(w, 0); }, loom::TaskDesc{ .signals = done });
		CHECK_THROWS_AS(loom::WaitFor(done), std::runtime_error);
		CHECK(walk.visited.load() < walk.nodes);
		CHECK(walk.refused.load() == 0);
	}
#endif
	{
		// Adds and signals racing from many threads: every add is matched.
		loom::ScopedCounter c(1);
		constexpr int kThreads = 8, kRounds = 20000;
		std::vector<std::thread> threads;
		std::atomic<int> refused{ 0 };
		for (int t = 0; t < kThreads; t++)
			threads.emplace_back([&] {
				for (int r = 0; r < kRounds; r++)
				{
					if (!loom::AddToCounter(c, 2))
						refused++;
					loom::Signal(c);
					loom::Signal(c);
				}
			});
		for (auto& t : threads)
			t.join();
		CHECK(refused.load() == 0); // the initial 1 holds it open throughout
		CHECK(loom::Count(c) == 1);
		loom::Signal(c);
		loom::WaitFor(c);
	}
}

TEST_CASE("loom: limiters — each running task holds its own unit index")
{
	for (int workers : { 0, 1, -1 })
	{
		loom::Pool pool(loom::PoolConfig{ .workerCount = workers });
		CHECK(loom::CurrentLimiterUnit() == loom::kNoUnit); // not in a task
		loom::ScopedLimiter four(4), other(2);
		CHECK(loom::Limit(four) == 4);
		std::atomic<bool> inUse[4] = {};
		std::atomic<int> errors{ 0 }, ran{ 0 };
		constexpr int kTasks = 400;
		loom::ScopedCounter done(kTasks + 1);
		for (int i = 0; i < kTasks; i++)
			loom::Submit([&, i] {
				const uint32_t u = loom::CurrentLimiterUnit();
				if (u >= 4 || inUse[u].exchange(true))
				{
					errors++;
					return;
				}
				if (i % 50 == 0)
				{
					// Another limiter's task run inside this one (helping) has its own
					// index, and this task's comes back afterwards.
					loom::ScopedCounter nested(1);
					loom::Submit([&] { if (loom::CurrentLimiterUnit() >= 2) errors++; }, loom::TaskDesc{ .signals = nested, .limiter = other });
					loom::WaitFor(nested);
				}
				SpinFor(std::chrono::microseconds(5));
				if (loom::CurrentLimiterUnit() != u)
					errors++;
				inUse[u] = false;
				ran++;
			}, loom::TaskDesc{ .signals = done, .limiter = four });
		loom::Submit([&] { if (loom::CurrentLimiterUnit() != loom::kNoUnit) errors++; }, loom::TaskDesc{ .signals = done });
		loom::WaitFor(done);
		CHECK(errors.load() == 0);
		CHECK(ran.load() == kTasks);
		CHECK(loom::HeldUnits(four) == 0);
	}
}

#if defined(LOOM_TEST_HOOKS)
namespace {
	// Held at the moment a task's unit becomes free: a second task takes that
	// unit and asks for its index before the first task's run has ended.
	struct UnitHandover
	{
		static inline std::atomic<bool> armed{ false };
		static inline std::atomic<uint32_t> secondUnit{ 0 };
		static inline std::atomic<bool> secondAsked{ false };
		static inline loom::LimiterHandle limiter = loom::kInvalidLimiter;
		static inline loom::CounterHandle secondDone = loom::kInvalidCounter;

		static void Hook(int point)
		{
			if (point != 1 || !armed.exchange(false))
				return;
			loom::Submit([] {
				secondUnit = loom::CurrentLimiterUnit();
				secondAsked = true;
			}, loom::TaskDesc{ .signals = secondDone, .limiter = limiter });
			const auto until = std::chrono::steady_clock::now() + std::chrono::seconds(5);
			while (!secondAsked && std::chrono::steady_clock::now() < until)
				std::this_thread::yield();
		}
	};
}

TEST_CASE("loom: limiters — a unit's index is free before the unit is")
{
	loom::Pool pool(loom::PoolConfig{ .workerCount = 2 });
	// A plain task gives its unit back when it ends; a future's task when its
	// value is ready, from inside its run.
	for (bool future : { false, true })
	{
		loom::ScopedLimiter one(1);
		loom::ScopedCounter secondDone(1);
		UnitHandover::limiter = one.Get();
		UnitHandover::secondDone = secondDone.Get();
		UnitHandover::secondAsked = false;
		std::atomic<uint32_t> firstUnit{ 99 };
		loom::detail::SetTestHook(&UnitHandover::Hook);
		UnitHandover::armed = true;
		if (future)
		{
			loom::Future<uint32_t> f = loom::Async([] { return loom::CurrentLimiterUnit(); }, loom::TaskDesc{ .limiter = one });
			firstUnit = f.Get();
		}
		else
		{
			loom::ScopedCounter firstDone(1);
			loom::Submit([&] { firstUnit = loom::CurrentLimiterUnit(); }, loom::TaskDesc{ .signals = firstDone, .limiter = one });
			loom::WaitFor(firstDone);
		}
		loom::WaitFor(secondDone);
		loom::detail::SetTestHook(nullptr);
		while (loom::LiveTaskCount() > 0)
			std::this_thread::yield();
		CHECK(UnitHandover::secondAsked.load());
		CHECK(firstUnit.load() == 0);
		CHECK(UnitHandover::secondUnit.load() == 0); // the only index below the limit, already back
	}
}
#endif

TEST_CASE("loom: limiters — raising a limit starts waiting tasks, lowering it waits for holders")
{
	loom::Pool pool(loom::PoolConfig{ .workerCount = 6 });
	loom::ScopedLimiter limiter(1);
	std::atomic<int> active{ 0 }, started{ 0 };
	std::atomic<int> release{ 0 }; // the long tasks numbered below this may finish
	loom::ScopedCounter longDone(4);
	for (int i = 0; i < 4; i++)
		loom::Submit([&, i] {
			active++;
			started++;
			while (release.load() <= i)
				std::this_thread::yield();
			active--;
		}, loom::TaskDesc{ .signals = longDone, .limiter = limiter });
	auto waitUntil = [](auto&& ready) {
		const auto until = std::chrono::steady_clock::now() + std::chrono::seconds(10);
		while (!ready() && std::chrono::steady_clock::now() < until)
			std::this_thread::yield();
		return ready();
	};
	REQUIRE(waitUntil([&] { return started.load() == 1; }));
	CHECK(loom::WaitingTasks(limiter) == 3);

	loom::SetLimit(limiter, 4); // the three waiting start now, beside the first
	REQUIRE(waitUntil([&] { return started.load() == 4; }));
	CHECK(loom::HeldUnits(limiter) == 4);
	CHECK(loom::WaitingTasks(limiter) == 0);

	loom::SetLimit(limiter, 2); // four hold units: two must finish, and one more, before a new task starts
	CHECK(loom::Limit(limiter) == 2);
	CHECK(loom::HeldUnits(limiter) == 4);
	std::atomic<int> newStarted{ 0 }, maxActive{ 0 }, badUnit{ 0 };
	constexpr int kNew = 20;
	loom::ScopedCounter newDone(kNew);
	for (int i = 0; i < kNew; i++)
		loom::Submit([&] {
			newStarted++;
			const int now = ++active;
			int seen = maxActive.load();
			while (now > seen && !maxActive.compare_exchange_weak(seen, now)) {}
			if (loom::CurrentLimiterUnit() >= 2)
				badUnit++;
			SpinFor(std::chrono::microseconds(20));
			active--;
		}, loom::TaskDesc{ .signals = newDone, .limiter = limiter });
	CHECK(loom::WaitingTasks(limiter) == kNew);
	for (int finished = 1; finished <= 2; finished++)
	{
		release = finished;
		REQUIRE(waitUntil([&] { return loom::HeldUnits(limiter) == (uint32_t)(4 - finished); }));
		CHECK(newStarted.load() == 0); // still at or over the new limit
	}
	release = 4; // the rest finish; the new tasks then run two at a time
	loom::WaitFor(newDone);
	loom::WaitFor(longDone);
	CHECK(newStarted.load() == kNew);
	CHECK(maxActive.load() <= 2);
	CHECK(badUnit.load() == 0);
	CHECK(loom::HeldUnits(limiter) == 0);
	CHECK(loom::WaitingTasks(limiter) == 0);
}

TEST_CASE("loom: limiters — a limit changed at random under load keeps every count straight")
{
	loom::Pool pool(loom::PoolConfig{});
	loom::ScopedLimiter limiter(3);
	constexpr uint32_t kMaxLimit = 8;
	std::atomic<bool> inUse[64] = {};
	std::atomic<int> errors{ 0 };
	std::atomic<bool> stop{ false };
	std::thread changer([&] {
		uint32_t x = 12345;
		while (!stop)
		{
			x = x * 1664525u + 1013904223u;
			loom::SetLimit(limiter, 1 + (x >> 16) % kMaxLimit);
			std::this_thread::yield();
		}
	});
	constexpr int kTasks = 4000;
	loom::ScopedCounter done(kTasks);
	for (int i = 0; i < kTasks; i++)
		loom::Submit([&] {
			const uint32_t u = loom::CurrentLimiterUnit();
			if (u >= 64 || inUse[u].exchange(true)) // distinct even across changes
			{
				errors++;
				return;
			}
			SpinFor(std::chrono::microseconds(2));
			inUse[u] = false;
		}, loom::TaskDesc{ .signals = done, .limiter = limiter });
	loom::WaitFor(done);
	stop = true;
	changer.join();
	CHECK(errors.load() == 0);
	CHECK(loom::HeldUnits(limiter) == 0);
	CHECK(loom::WaitingTasks(limiter) == 0);
	loom::SetLimit(limiter, 2);
	CHECK(loom::Limit(limiter) == 2);
	// Destroying it asserts every unit came back and no debt is left.
}

TEST_CASE("loom: per-thread storage — one per thread, merged after the work")
{
	loom::Pool pool(loom::PoolConfig{});
	// A histogram per thread, merged once the loop is done.
	constexpr size_t kN = 200000, kBins = 16;
	loom::PerThread<std::vector<uint64_t>> bins(std::vector<uint64_t>(kBins, 0));
	loom::ParallelFor(0, kN, [&](size_t b, size_t e) {
		std::vector<uint64_t>& mine = bins.Local();
		REQUIRE(mine.size() == kBins); // a copy of the exemplar
		for (size_t i = b; i < e; i++)
			mine[(i * 2654435761u) % kBins]++;
	});
	std::vector<uint64_t> merged(kBins, 0), serial(kBins, 0);
	bins.ForEach([&](std::vector<uint64_t>& v) {
		for (size_t k = 0; k < kBins; k++)
			merged[k] += v[k];
	});
	for (size_t i = 0; i < kN; i++)
		serial[(i * 2654435761u) % kBins]++;
	CHECK(merged == serial);
	CHECK(bins.Size() >= 1);
	CHECK(bins.Size() <= (size_t)loom::WorkerCount() + 1);

	// Each thread sees its own, on its own cache line, and the same one each time.
	struct Slot { int owner = -1; int uses = 0; };
	loom::PerThread<Slot> slots;
	std::atomic<int> wrongOwner{ 0 };
	loom::ParallelFor(0, 2000, size_t(1), [&](size_t, size_t) {
		Slot& s = slots.Local();
		if (s.owner == -1)
			s.owner = loom::CurrentThreadIndex();
		else if (s.owner != loom::CurrentThreadIndex())
			wrongOwner++;
		s.uses++;
		if (&slots.Local() != &s)
			wrongOwner++;
	});
	CHECK(wrongOwner.load() == 0);
	int uses = 0;
	std::vector<uintptr_t> addresses;
	slots.ForEach([&](Slot& s) { uses += s.uses; addresses.push_back((uintptr_t)&s); });
	CHECK(uses == 2000);
	std::sort(addresses.begin(), addresses.end());
	for (size_t k = 1; k < addresses.size(); k++)
		CHECK(addresses[k] - addresses[k - 1] >= 64);
}

TEST_CASE("loom: per-thread storage — move-only, over-aligned, threads outside the pool, Clear")
{
	loom::Pool pool(loom::PoolConfig{ .workerCount = 2 });
	{
		loom::PerThread<std::unique_ptr<int>> owned;
		owned.Local() = std::make_unique<int>(5);
		CHECK(*owned.Local() == 5);
		std::thread outside([&] { owned.Local() = std::make_unique<int>(7); });
		outside.join();
		int sum = 0;
		owned.ForEach([&](std::unique_ptr<int>& p) { sum += p ? *p : 0; });
		CHECK(sum == 12); // the outside thread's stays after it ended
	}
	{
		struct alignas(128) Wide { double v[4] = {}; };
		loom::PerThread<Wide> wide;
		loom::ScopedCounter done(8);
		for (int i = 0; i < 8; i++)
			loom::Submit([&] {
				if ((uintptr_t)&wide.Local() % 128 != 0)
					std::terminate();
			}, loom::TaskDesc{ .signals = done });
		loom::WaitFor(done);
		CHECK((uintptr_t)&wide.Local() % 128 == 0);
	}
	{
		Tracked::live = 0;
		{
			loom::PerThread<Tracked> tracked;
			loom::ScopedCounter done(50);
			for (int i = 0; i < 50; i++)
				loom::Submit([&] { (void)tracked.Local(); }, loom::TaskDesc{ .signals = done });
			loom::WaitFor(done);
			(void)tracked.Local();
			const int made = Tracked::live.load();
			CHECK(made == (int)tracked.Size());
			CHECK(made >= 1);
			tracked.Clear();
			CHECK(Tracked::live.load() == 0);
			CHECK(tracked.Size() == 0);
			(void)tracked.Local(); // made again after Clear
			CHECK(Tracked::live.load() == 1);
		}
		CHECK(Tracked::live.load() == 0); // destroyed with the PerThread
	}
}

#if defined(__linux__)
TEST_CASE("loom: running out of counters or limiters ends the program, saying which setting to raise")
{
	REQUIRE(!loom::HasActivePool());
	std::string err;
	CHECK(EndsInAbort([] {
		loom::Pool pool(loom::PoolConfig{ .workerCount = 1, .maxCounters = 4 });
		for (int i = 0; i < 5; i++)
			(void)loom::CreateCounter(1);
	}, err));
	CHECK(err.find("PoolConfig::maxCounters") != std::string::npos);
	err.clear();
	CHECK(EndsInAbort([] {
		loom::Pool pool(loom::PoolConfig{ .workerCount = 1, .maxLimiters = 2 });
		for (int i = 0; i < 3; i++)
			(void)loom::CreateLimiter(1);
	}, err));
	CHECK(err.find("PoolConfig::maxLimiters") != std::string::npos);
}
#endif

TEST_CASE("loom: Shutdown from another thread — the main thread's pump runs main-thread work through the drain")
{
	using namespace std::chrono_literals;
	std::atomic<bool> ran{ false }, done{ false };
	{
		loom::Pool pool(loom::PoolConfig{ .workerCount = 2, .shutdownTimeoutMs = 2000 });
		// Still running when the drain starts, it then queues main-thread work.
		loom::Submit([&] {
			std::this_thread::sleep_for(50ms);
			loom::Submit([&] { ran = true; }, { .affinity = loom::Affinity::MainThread });
		});
		std::thread closer([&] { pool.Shutdown(); done = true; });
		for (int i = 0; i < 400 && !done; i++)
		{
			loom::PumpMainThreadTasks();
			std::this_thread::sleep_for(5ms);
		}
		closer.join();
	}
	CHECK(ran.load());
}

TEST_CASE("loom: Pool — a thread that sees the pool active sees it built")
{
	for (int round = 0; round < 50; round++)
	{
		std::atomic<int> workers{ -1 };
		std::thread watcher([&] {
			while (!loom::HasActivePool()) {}
			workers = loom::WorkerCount();
		});
		{
			loom::Pool pool(loom::PoolConfig{ .workerCount = 3 });
			watcher.join();
		}
		CHECK(workers.load() == 3);
	}
}

#if defined(NDEBUG)
TEST_CASE("loom: PumpMainThreadTasks — off the main thread it runs nothing")
{
	loom::Pool pool(loom::PoolConfig{ .workerCount = 1 });
	std::atomic<bool> ran{ false };
	loom::ScopedCounter done(1);
	loom::Submit([&] { ran = true; }, { .affinity = loom::Affinity::MainThread, .signals = done });
	std::thread([] { loom::PumpMainThreadTasks(); }).join(); // asserts in a debug build
	CHECK_FALSE(ran.load());
	loom::PumpMainThreadTasks();
	CHECK(ran.load());
}
#endif

#if defined(__linux__)
TEST_CASE("loom: Shutdown from inside a task ends the program, saying why")
{
	REQUIRE(!loom::HasActivePool());
	std::string err;
	CHECK(EndsInAbort([] {
		loom::Pool pool(loom::PoolConfig{ .workerCount = 1 });
		loom::ScopedCounter done(1);
		loom::Submit([&pool] { pool.Shutdown(); }, { .signals = done });
		loom::WaitFor(done, loom::Help::BlockOnly);
	}, err));
	CHECK(err.find("from inside a task") != std::string::npos);
}
#endif

namespace profiler_marks {
	std::atomic<int> begins{ 0 }, ends{ 0 }, unpaired{ 0 };
	uint64_t Zone(int, uint32_t, uint8_t, const char*, bool begin, uint64_t handle)
	{
		if (begin)
		{
			begins++;
			return 0x5EED;
		}
		if (handle != 0x5EED)
			unpaired++; // an end whose begin this hook never saw
		ends++;
		return 0;
	}
	std::mutex m;
	std::vector<int> states;
	std::atomic<int> mainIndex{ -1 };
	void State(int threadIndex, uint8_t state)
	{
		if (threadIndex != mainIndex.load())
			return; // the main thread's marks only
		std::lock_guard<std::mutex> lock(m);
		states.push_back(state);
	}
}

TEST_CASE("loom: profiler hooks — a zone ends on the hook it began on, and helping is marked as it starts")
{
	using namespace std::chrono_literals;
	using namespace profiler_marks;
	loom::Pool pool(loom::PoolConfig{ .workerCount = 1 });
	// Installed while a task runs: that task's end must not reach it.
	{
		begins = ends = unpaired = 0;
		std::atomic<bool> inside{ false }, go{ false };
		loom::ScopedCounter done(1);
		loom::Submit([&] { inside = true; while (!go) std::this_thread::sleep_for(1ms); }, { .signals = done });
		while (!inside) std::this_thread::sleep_for(1ms);
		loom::SetProfilerZoneHook(&Zone);
		go = true;
		loom::WaitFor(done, loom::Help::BlockOnly);
		CHECK(unpaired.load() == 0);
		CHECK(ends.load() == begins.load());
	}
	// Removed while a task runs: the begin it saw still gets its end.
	{
		begins = ends = unpaired = 0;
		std::atomic<bool> inside{ false }, go{ false };
		loom::ScopedCounter done(1);
		loom::Submit([&] { inside = true; while (!go) std::this_thread::sleep_for(1ms); }, { .signals = done });
		while (!inside) std::this_thread::sleep_for(1ms);
		loom::SetProfilerZoneHook(nullptr);
		go = true;
		loom::WaitFor(done, loom::Help::BlockOnly);
		CHECK(begins.load() == 1);
		CHECK(ends.load() == 1);
	}
	// A wait on the main thread that runs one task, then sleeps: 0 as it
	// enters, 1 as the task starts, 0 once it is done, 2 at the end.
	{
		{
			std::lock_guard<std::mutex> lock(m);
			states.clear();
		}
		mainIndex = loom::CurrentThreadIndex();
		loom::ScopedCounter gate(1), helped(1);
		std::atomic<bool> blocker{ false };
		loom::Submit([&] { blocker = true; loom::WaitFor(gate, loom::Help::BlockOnly); }); // keeps the worker busy
		while (!blocker) std::this_thread::sleep_for(1ms);
		loom::Submit([] { std::this_thread::sleep_for(20ms); }, { .signals = helped });
		std::thread opener([&] { std::this_thread::sleep_for(150ms); loom::Signal(gate); });
		loom::SetProfilerStateHook(&State);
		loom::WaitFor(gate);
		loom::SetProfilerStateHook(nullptr);
		opener.join();
		std::vector<int> seen;
		{
			std::lock_guard<std::mutex> lock(m);
			seen = states;
		}
		CHECK(seen == std::vector<int>{ 0, 1, 0, 2 });
	}
}

TEST_CASE("loom: TryWaitFor — true when the tasks it ran itself brought the count to zero")
{
	loom::Pool pool(loom::PoolConfig{ .workerCount = 0 }); // the wait runs every task
	loom::ScopedCounter c(3);
	for (int i = 0; i < 3; i++)
		loom::Submit([i] { std::this_thread::sleep_for(std::chrono::milliseconds(i == 0 ? 30 : 1)); }, loom::TaskDesc{ .signals = c });
	// The deadline passes during the first task; the other two follow, and
	// their signals were still held back when the deadline was checked.
	CHECK(loom::TryWaitFor(c, 20));
	CHECK(loom::Count(c) == 0);
}

TEST_CASE("loom: handles — a destroyed counter's handle stays stale through many reuses")
{
	loom::Pool pool(loom::PoolConfig{ .workerCount = 1, .maxCounters = 64 });
	// From an entry's second use: the first generation is never handed out again.
	loom::DestroyCounter(loom::CreateCounter(0));
	const loom::CounterHandle stale = loom::CreateCounter(0);
	loom::DestroyCounter(stale);
	// 4095 reuses of one entry wrap its generation. Entries are reused oldest
	// first, so with 64 of them that takes 64 times as many cycles.
	int aliased = 0;
	for (int i = 0; i < 20000; i++)
	{
		const loom::CounterHandle h = loom::CreateCounter(0);
		aliased += h == stale ? 1 : 0;
		loom::DestroyCounter(h);
	}
	CHECK(aliased == 0);
}

TEST_CASE("loom: SetLimit — a large limit costs nothing until that many run")
{
	loom::Pool pool(loom::PoolConfig{ .workerCount = 2 });
	const loom::LimiterHandle l = loom::CreateLimiter(2);
	const auto t0 = std::chrono::steady_clock::now();
	loom::SetLimit(l, 200'000'000);
	const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
	CHECK(ms < 100.0); // building 200M unit indices up front took seconds and a gigabyte
	CHECK(loom::Limit(l) == 200'000'000u);
	std::atomic<int> ran{ 0 };
	std::set<uint32_t> units;
	std::mutex m;
	loom::ScopedCounter done(16);
	for (int i = 0; i < 16; i++)
		loom::Submit([&] {
			const uint32_t u = loom::CurrentLimiterUnit();
			std::lock_guard<std::mutex> lock(m);
			units.insert(u);
			ran++;
		}, loom::TaskDesc{ .signals = done, .limiter = l });
	loom::WaitFor(done);
	CHECK(ran.load() == 16);
	CHECK(*units.rbegin() < 3); // at most three at once here: indices from 0 up
	loom::SetLimit(l, 1);
	loom::DestroyLimiter(l);
}

#if defined(__linux__) && !defined(NDEBUG)
TEST_CASE("loom: DestroyLimiter — asserts while a task of it still waits for its counter")
{
	REQUIRE(!loom::HasActivePool());
	std::string err;
	CHECK(EndsInAbort([] {
		loom::Pool pool(loom::PoolConfig{ .workerCount = 1 });
		const loom::LimiterHandle l = loom::CreateLimiter(1);
		loom::ScopedCounter gate(1);
		loom::Submit([] {}, loom::TaskDesc{ .waitsFor = gate, .limiter = l });
		loom::DestroyLimiter(l); // the task holds no unit yet, but it is live
	}, err));
	CHECK(err.find("destroying a limiter whose tasks are live") != std::string::npos);
}
#endif

#if defined(__linux__) && defined(__cpp_exceptions)
TEST_CASE("loom: Promise — a moved-from promise ends the program rather than crash")
{
	REQUIRE(!loom::HasActivePool());
	for (int which = 0; which < 3; which++)
	{
		static int s_which;
		s_which = which;
		std::string err;
		CHECK(EndsInAbort([] {
			loom::Pool pool(loom::PoolConfig{ .workerCount = 1 });
			loom::Promise<int> p;
			loom::Promise<int> q(std::move(p));
			if (s_which == 0)
				p.SetValue(1);
			else if (s_which == 1)
				p.SetException(std::make_exception_ptr(std::runtime_error("x")));
			else
				(void)p.GetFuture();
		}, err));
	}
}
#endif

#if defined(__linux__)
TEST_CASE("loom: Pipeline — a stage before the source ends the program")
{
	REQUIRE(!loom::HasActivePool());
	std::string err;
	CHECK(EndsInAbort([] {
		loom::Pool pool(loom::PoolConfig{ .workerCount = 1 });
		loom::Pipeline<int> p(4);
		p.Serial([](int&) {}); // without a source first, Run would never end
	}, err));
}
#endif

namespace thread_start {
	std::atomic<int> started{ 0 };
	void OnStart() { started++; }
}

TEST_CASE("loom: Shutdown — dedicated threads still registered, with work queued on them")
{
	for (int round = 0; round < 30; round++)
	{
		std::atomic<int> ran{ 0 };
		int queued = 0;
		{
			loom::Pool pool(loom::PoolConfig{ .workerCount = round % 3 });
			const loom::DedicatedId sleepy = loom::RegisterDedicatedThread({ .name = "sleepy" });
			const loom::DedicatedId stealer = loom::RegisterDedicatedThread({ .name = "stealer", .idle = loom::DedicatedConfig::Idle::StealFromPool });
			for (int i = 0; i < 20; i++, queued += 3)
			{
				loom::Submit([&] { ran++; std::this_thread::yield(); }, { .affinity = loom::Affinity::Dedicated, .thread = sleepy });
				loom::Submit([&] { ran++; }, { .affinity = loom::Affinity::Dedicated, .thread = stealer });
				loom::Submit([&] { ran++; });
			}
			// No unregister: Shutdown drains them, then stops and joins them.
		}
		CHECK(ran.load() == queued);
	}
}

TEST_CASE("loom: PoolConfig::onThreadStart runs first on every worker and dedicated thread")
{
	thread_start::started = 0;
	{
		loom::Pool pool(loom::PoolConfig{ .workerCount = 3, .onThreadStart = &thread_start::OnStart });
		const loom::DedicatedId d = loom::RegisterDedicatedThread({ .name = "d" });
		loom::ScopedCounter done(1);
		loom::Submit([] {}, { .affinity = loom::Affinity::Dedicated, .thread = d, .signals = done });
		loom::WaitFor(done);
		loom::UnregisterDedicatedThread(d);
	}
	CHECK(thread_start::started.load() == 4);
}

TEST_CASE("loom: debug dump — a dedicated thread's queue, and a long name in full")
{
	using namespace std::chrono_literals;
	loom::Pool pool(loom::PoolConfig{ .workerCount = 1 });
	const std::string longName(600, 'n');
	const loom::DedicatedId d = loom::RegisterDedicatedThread({ .name = longName.c_str() });
	loom::ScopedCounter gate(1), done(6);
	std::atomic<bool> inside{ false };
	loom::Submit([&] { inside = true; loom::WaitFor(gate, loom::Help::BlockOnly); }, { .affinity = loom::Affinity::Dedicated, .thread = d, .signals = done });
	for (int i = 0; i < 5; i++)
		loom::Submit([] {}, { .affinity = loom::Affinity::Dedicated, .thread = d, .signals = done });
	while (!inside)
		std::this_thread::sleep_for(1ms);
	const loom::DebugState s = loom::CaptureDebugState();
	int depth = -1;
	for (const auto& t : s.threads)
		if (t.name == longName)
			depth = t.queueDepth;
	CHECK(depth == 5);
	const std::string text = loom::FormatDebugState(s);
	CHECK(text.find("5 queued for it") != std::string::npos);
	// The thread's line is not cut short at a fixed buffer's length.
	const size_t at = text.find(longName);
	REQUIRE(at != std::string::npos);
	const size_t lineEnd = text.find('\n', at);
	CHECK(text.substr(at, lineEnd - at).find("; ran ") != std::string::npos);
	loom::Signal(gate);
	loom::WaitFor(done);
	loom::UnregisterDedicatedThread(d);
}

namespace shutdown_drain {
	std::atomic<long> mainSubmitted{ 0 }, mainRan{ 0 };
	void MainFinal(void*) { mainRan++; }
	void Noop(void*) {}
	// A chain of tasks, each submitting a few quick siblings (whose live counts
	// move between the threads' shards) and the next step; the last one queues
	// work for the main thread.
	void Step(void* p)
	{
		const uintptr_t k = (uintptr_t)p;
		if (k == 0)
		{
			mainSubmitted++;
			loom::Submit(loom::TaskDesc{ .fn = &MainFinal, .affinity = loom::Affinity::MainThread });
			return;
		}
		for (int i = 0; i < 3; i++)
			loom::Submit(loom::TaskDesc{ .fn = &Noop });
		loom::Submit(loom::TaskDesc{ .fn = &Step, .userData = (void*)(k - 1) });
	}
}

TEST_CASE("loom: Shutdown — the drain never sees zero while a task is live")
{
	shutdown_drain::mainSubmitted = 0;
	shutdown_drain::mainRan = 0;
	for (int round = 0; round < 300; round++)
	{
		loom::Pool pool(loom::PoolConfig{ .workerCount = 8 });
		for (int c = 0; c < 4; c++)
			loom::Submit(loom::TaskDesc{ .fn = &shutdown_drain::Step, .userData = (void*)(uintptr_t)(200 + round % 50) });
		// ~Pool drains: every chain's main-thread task must run.
	}
	CHECK(shutdown_drain::mainRan.load() == shutdown_drain::mainSubmitted.load());
	CHECK(shutdown_drain::mainSubmitted.load() == 1200);
}

TEST_CASE("loom: UnregisterDedicatedThread — every task submitted to the thread runs first")
{
	loom::Pool pool(loom::PoolConfig{ .workerCount = 4 });
	SUBCASE("a task queued just before it")
	{
		// The thread looks at its queue, then at the stop flag: a task queued
		// between the two used to be left behind.
		int lost = 0;
		for (int i = 0; i < 1000; i++)
		{
			const loom::DedicatedId d = loom::RegisterDedicatedThread({ .name = "r", .idle = loom::DedicatedConfig::Idle::StealFromPool });
			std::atomic<bool> started{ false }, ran{ false };
			loom::Submit([&] { started = true; for (volatile int k = 0; k < 1000; k = k + 1) {} }, { .affinity = loom::Affinity::Dedicated, .thread = d });
			while (!started) {}
			for (volatile int k = 0; k < (i % 128) * 40; k = k + 1) {}
			loom::Submit([&] { ran = true; }, { .affinity = loom::Affinity::Dedicated, .thread = d });
			loom::UnregisterDedicatedThread(d);
			lost += ran ? 0 : 1;
		}
		CHECK(lost == 0);
	}
	SUBCASE("a task still waiting for its counter, and one a task of the thread submits")
	{
		const loom::DedicatedId d = loom::RegisterDedicatedThread({ .name = "io" });
		loom::ScopedCounter gate(1), done(3);
		std::atomic<int> ran{ 0 };
		loom::Submit([&] { ran++; }, { .affinity = loom::Affinity::Dedicated, .thread = d, .signals = done, .waitsFor = gate });
		loom::Submit([&] {
			std::this_thread::sleep_for(std::chrono::milliseconds(20));
			ran++;
			loom::Submit([&] { ran++; }, { .affinity = loom::Affinity::Dedicated, .thread = d, .signals = done }); // a follow-up
		}, { .affinity = loom::Affinity::Dedicated, .thread = d, .signals = done });
		std::thread opener([&] {
			std::this_thread::sleep_for(std::chrono::milliseconds(50));
			loom::Signal(gate);
		});
		loom::UnregisterDedicatedThread(d); // returns once all three have run
		CHECK(ran.load() == 3);
		CHECK(loom::Count(done) == 0);
		opener.join();
	}
}

TEST_CASE("loom: ReleaseLane — only a lane, and only once")
{
	loom::Pool pool(loom::PoolConfig{ .workerCount = 1 });
	const loom::LaneId lane = loom::AcquireLane("lane");
	const loom::DedicatedId plain = loom::RegisterDedicatedThread({ .name = "plain" });
#if defined(NDEBUG)
	// Misuse asserts in a debug build; in a release one it changes nothing:
	// the plain thread is still registered and still runs its tasks.
	loom::ReleaseLane(plain);
	loom::ScopedCounter done(1);
	loom::Submit([] {}, { .affinity = loom::Affinity::Dedicated, .thread = plain, .signals = done });
	CHECK(loom::TryWaitFor(done, 5000));
#endif
	loom::ReleaseLane(lane);
#if defined(NDEBUG)
	loom::ReleaseLane(lane);
#endif
	loom::UnregisterDedicatedThread(plain);
	// ~Pool asserts no lane is still held, and that the count is not below zero.
}

namespace helper_wake {
	loom::CounterHandle c1, c2, cX;
}

TEST_CASE("loom: wakes — a wake for pool work that a waiting thread does not use is passed on")
{
	using namespace std::chrono_literals;
	using namespace helper_wake;
	loom::Pool pool(loom::PoolConfig{ .workerCount = 1 });
	c1 = loom::CreateCounter(1);
	c2 = loom::CreateCounter(1);
	cX = loom::CreateCounter(1);
	// X on the worker waits (helping) for c1, which only B signals.
	loom::Submit([] { loom::WaitFor(c1); }, loom::TaskDesc{ .signals = cX });
	std::this_thread::sleep_for(100ms); // the worker is asleep inside its wait
	std::thread outside([] {
		std::this_thread::sleep_for(100ms); // main is asleep in WaitFor(c2)
		loom::TaskDesc d[2];
		d[0].fn = [](void*) { loom::Signal(c2); }; // A ends main's wait itself
		d[1].fn = [](void*) {};                    // B is what the worker waits for
		d[1].signals = c1;
		loom::Submit(std::span<const loom::TaskDesc>(d, 2));
	});
	loom::WaitFor(c2); // woken for the pair, runs A and returns
	outside.join();
	// The worker must have been woken for B: main does not help here.
	CHECK(loom::TryWaitFor(cX, 3000, loom::Help::BlockOnly));
	loom::WaitFor(cX);
	loom::DestroyCounter(c1);
	loom::DestroyCounter(c2);
	loom::DestroyCounter(cX);
}

#if defined(__linux__)
static double ThreadCpuMs()
{
	timespec ts{};
	clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts);
	return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}

TEST_CASE("loom: Isolated — a wait sleeps while another region's task sits in its own queue")
{
	loom::Pool pool(loom::PoolConfig{ .workerCount = 2 });
	const loom::DedicatedId io = loom::RegisterDedicatedThread({ .name = "io" });
	loom::ScopedCounter foreignDone(1);
	// A main-thread task from outside the region; nothing runs it until later.
	loom::Submit([] {}, loom::TaskDesc{ .affinity = loom::Affinity::MainThread, .signals = foreignDone });
	double cpu = 0;
	loom::Isolated([&] {
		loom::ScopedCounter mine(1);
		loom::Submit([] { std::this_thread::sleep_for(std::chrono::milliseconds(300)); },
			loom::TaskDesc{ .affinity = loom::Affinity::Dedicated, .thread = io, .signals = mine });
		const double c0 = ThreadCpuMs();
		loom::WaitFor(mine);
		cpu = ThreadCpuMs() - c0;
	});
	CHECK(loom::Count(foreignDone) == 1); // the region's wait left it alone
	CHECK(cpu < 100.0);                   // and slept, rather than spinning for 300 ms
	loom::WaitFor(foreignDone);
	loom::UnregisterDedicatedThread(io);
}
#endif

TEST_CASE("loom: no workers — a dedicated thread's fork-join runs its own chunks")
{
	loom::Pool pool(loom::PoolConfig{ .workerCount = 0 });
	const loom::DedicatedId io = loom::RegisterDedicatedThread({ .name = "io" });
	std::atomic<long> sum{ 0 };
	std::vector<int> v(5000);
	std::iota(v.begin(), v.end(), 0);
	loom::ScopedCounter done(1);
	loom::Submit([&] {
		loom::ParallelFor(0, 1000, [&](size_t b, size_t e) {
			for (size_t i = b; i < e; i++)
				sum += (long)i;
		});
		loom::ParallelSort(v.begin(), v.end(), std::greater<int>());
	}, loom::TaskDesc{ .affinity = loom::Affinity::Dedicated, .thread = io, .signals = done });
	// Without helping: the main thread runs none of the chunks.
	CHECK(loom::TryWaitFor(done, 5000, loom::Help::BlockOnly));
	loom::WaitFor(done);
	CHECK(sum.load() == 499500);
	CHECK(std::is_sorted(v.begin(), v.end(), std::greater<int>()));
	loom::UnregisterDedicatedThread(io);
}

TEST_CASE("loom: thread indices — given back when a thread exits, and taken again")
{
	loom::Pool pool(loom::PoolConfig{ .workerCount = 2 });

	// Threads alive at the same time never share one.
	{
		constexpr int kThreads = 8;
		std::vector<int> seen(kThreads, -1);
		std::atomic<int> arrived{ 0 };
		std::vector<std::thread> threads;
		for (int i = 0; i < kThreads; i++)
			threads.emplace_back([&, i] {
				seen[i] = loom::CurrentThreadIndex();
				arrived++;
				while (arrived.load() < kThreads)
					std::this_thread::yield();
			});
		for (auto& t : threads)
			t.join();
		std::sort(seen.begin(), seen.end());
		CHECK(std::adjacent_find(seen.begin(), seen.end()) == seen.end());
	}

	// The next thread takes the index of the one that exited, with the
	// PerThread value it left.
	loom::PerThread<int> partial;
	int first = -1, second = -1;
	std::thread([&] { first = loom::CurrentThreadIndex(); partial.Local() = 41; }).join();
	std::thread([&] { second = loom::CurrentThreadIndex(); partial.Local() += 1; }).join();
	CHECK(second == first);
	CHECK(partial.Size() == 1);
	partial.ForEach([](int& v) { CHECK(v == 42); });

	// A thread-local made before the thread's first call is destroyed after
	// the index went back; asking from its destructor still gets an index.
	struct Late
	{
		std::atomic<int>* out = nullptr;
		~Late() { if (out) out->store(loom::CurrentThreadIndex()); }
	};
	std::atomic<int> late{ -100 };
	std::thread([&] {
		static thread_local Late t_Late;
		t_Late.out = &late;
		(void)loom::CurrentThreadIndex();
	}).join();
	CHECK(late.load() >= 0);

	// A dedicated thread that takes an index takes over its stats row too.
	const loom::DedicatedId a = loom::RegisterDedicatedThread({ .name = "index-first" });
	int aIndex = -1;
	{
		loom::ScopedCounter done(1);
		loom::Submit([&] { aIndex = loom::CurrentThreadIndex(); }, { .affinity = loom::Affinity::Dedicated, .thread = a, .signals = done });
		loom::WaitFor(done);
	}
	loom::UnregisterDedicatedThread(a);
	const loom::DedicatedId b = loom::RegisterDedicatedThread({ .name = "index-second" });
	int bIndex = -1;
	{
		loom::ScopedCounter done(1);
		loom::Submit([&] { bIndex = loom::CurrentThreadIndex(); }, { .affinity = loom::Affinity::Dedicated, .thread = b, .signals = done });
		loom::WaitFor(done);
	}
	CHECK(bIndex == aIndex);
	int rows = 0;
	for (const loom::ThreadStatsView& t : loom::SnapshotPool().threads)
		if (t.index == bIndex)
		{
			rows++;
			CHECK(t.name == "index-second");
		}
	CHECK(rows == 1);
	loom::UnregisterDedicatedThread(b);
}

TEST_CASE("loom: thread indices — stay small however many pools come and go")
{
	int first = -1, highest = 0;
	loom::PerThread<long> partial; // outlives every pool
	long want = 0;
	for (int round = 0; round < 100; round++)
	{
		loom::Pool pool(loom::PoolConfig{ .workerCount = 4 });
		const loom::DedicatedId io = loom::RegisterDedicatedThread({ .name = "index-round" });
		std::atomic<int> top{ 0 };
		auto note = [&] {
			const int me = loom::CurrentThreadIndex();
			int seen = top.load();
			while (me > seen && !top.compare_exchange_weak(seen, me)) {}
		};
		loom::ParallelFor(0, 64, size_t(1), [&](size_t, size_t) { note(); partial.Local() += 1; });
		loom::ScopedCounter done(1);
		loom::Submit([&] { note(); partial.Local() += 1; }, { .affinity = loom::Affinity::Dedicated, .thread = io, .signals = done });
		loom::WaitFor(done);
		std::thread([&] { note(); partial.Local() += 1; }).join();
		want += 66;
		if (first < 0)
			first = top.load();
		highest = std::max(highest, top.load());
		loom::UnregisterDedicatedThread(io);
	}
	// 100 pools started 600 threads, with at most seven alive at once: every
	// round takes the same few indices again.
	CHECK(highest <= first + 6);
	long sum = 0;
	partial.ForEach([&](long& v) { sum += v; });
	CHECK(sum == want);
	CHECK(partial.Size() <= (size_t)highest + 1);
}

TEST_CASE("loom: ParallelFor inside ParallelFor — the nesting stays shallow, every body runs")
{
	// Each wait runs other chunks while it waits, and each of those starts
	// another inner ParallelFor: unchecked, one thread nests a thousand deep,
	// a counter each, and runs out of them.
	for (size_t chunk : { size_t(1), size_t(0) })
	{
		loom::Pool pool(loom::PoolConfig{ .workerCount = 7, .maxCounters = 256 });
		std::atomic<long> inner{ 0 };
		std::atomic<int> deepest{ 0 };
		static thread_local int t_depth = 0;
		loom::ParallelFor(0, 20000, chunk, [&](size_t b, size_t e) {
			for (size_t i = b; i < e; i++)
			{
				const int d = ++t_depth;
				int seen = deepest.load();
				while (d > seen && !deepest.compare_exchange_weak(seen, d)) {}
				loom::ParallelFor(0, 8, size_t(1), [&](size_t, size_t) { inner++; });
				t_depth--;
			}
		});
		CHECK(inner.load() == 160000);
		CHECK(deepest.load() <= 20);
	}
}

namespace self_move {
	std::atomic<int> selfMoves{ 0 };
	struct Key
	{
		int v = 0;
		Key() = default;
		explicit Key(int x) : v(x) {}
		Key(Key&&) noexcept = default;
		Key& operator=(Key&& o) noexcept
		{
			if (this == &o)
				selfMoves++;
			v = o.v;
			return *this;
		}
		bool operator<(const Key& o) const { return v < o.v; }
	};
}

TEST_CASE("loom: ParallelSort — never moves an element onto itself, as std::sort does not")
{
	loom::Pool pool(loom::PoolConfig{});
	for (int mod : { 3, 1000000 })
	{
		std::vector<self_move::Key> v;
		v.reserve(400000);
		uint32_t x = 2463534242u;
		for (int i = 0; i < 400000; i++)
		{
			x ^= x << 13; x ^= x >> 17; x ^= x << 5;
			v.emplace_back((int)(x % (uint32_t)mod));
		}
		self_move::selfMoves = 0;
		loom::ParallelSort(v.begin(), v.end());
		CHECK(self_move::selfMoves.load() == 0);
		CHECK(std::is_sorted(v.begin(), v.end()));
	}
}

TEST_CASE("loom: ParallelInclusiveScan / ParallelExclusiveScan — match std, in place, reproducible")
{
	for (int workers : { 0, 1, -1 })
	{
		loom::Pool pool(loom::PoolConfig{ .workerCount = workers });
		for (size_t n : { size_t(0), size_t(1), size_t(100), size_t(8191), size_t(8192), size_t(100000), size_t(1000003) })
		{
			std::vector<int64_t> in(n);
			for (size_t i = 0; i < n; i++)
				in[i] = (int64_t)((i * 2654435761u) % 1000) - 500;
			std::vector<int64_t> want(n), got(n, -1);
			std::inclusive_scan(in.begin(), in.end(), want.begin(), std::plus<>{}, int64_t(7));
			loom::ParallelInclusiveScan(in.begin(), in.end(), got.begin(), int64_t(7));
			CHECK(got == want);
			std::exclusive_scan(in.begin(), in.end(), want.begin(), int64_t(7));
			loom::ParallelExclusiveScan(in.begin(), in.end(), got.begin(), int64_t(7));
			CHECK(got == want);
			std::vector<int64_t> inPlace = in; // out == first
			loom::ParallelExclusiveScan(inPlace.begin(), inPlace.end(), inPlace.begin(), int64_t(7));
			CHECK(inPlace == want);
		}
		// Associative but not commutative: composing affine maps x -> a*x + b (mod a prime).
		struct Affine { uint64_t a = 1, b = 0; bool operator==(const Affine&) const = default; };
		auto compose = [](Affine f, Affine g) { // f, then g
			constexpr uint64_t p = 1000000007;
			return Affine{ g.a * f.a % p, (g.a * f.b + g.b) % p };
		};
		std::vector<Affine> maps(300000);
		for (size_t i = 0; i < maps.size(); i++)
			maps[i] = Affine{ 1 + i % 13, i % 17 };
		std::vector<Affine> want(maps.size()), got(maps.size());
		std::inclusive_scan(maps.begin(), maps.end(), want.begin(), compose, Affine{});
		loom::ParallelInclusiveScan(maps.begin(), maps.end(), got.begin(), Affine{}, compose);
		CHECK(got == want);
	}
	// Floating point: the same bits whatever the pool.
	std::vector<double> in(500000);
	for (size_t i = 0; i < in.size(); i++)
		in[i] = 1.0 / (double)(i + 1) * ((i % 3) ? 1.0 : -1e-3);
	std::vector<std::vector<double>> results;
	for (int workers : { 0, 1, 3, -1 })
	{
		loom::Pool pool(loom::PoolConfig{ .workerCount = workers });
		std::vector<double> out(in.size());
		loom::ParallelInclusiveScan(in.begin(), in.end(), out.begin(), 0.0);
		results.push_back(std::move(out));
	}
	for (size_t r = 1; r < results.size(); r++)
		CHECK(std::memcmp(results[r].data(), results[0].data(), in.size() * sizeof(double)) == 0);
#if defined(__cpp_exceptions)
	{
		loom::Pool pool(loom::PoolConfig{});
		std::vector<int> v(200000, 1);
		v[150000] = -1;
		std::vector<int> out(v.size());
		auto checked = [](int acc, int x) { if (x < 0) throw std::runtime_error("bad value"); return acc + x; };
		CHECK_THROWS_AS(loom::ParallelInclusiveScan(v.begin(), v.end(), out.begin(), 0, checked), std::runtime_error);
	}
#endif
}

TEST_CASE("loom: ParallelSort — matches std::sort on awkward inputs, any pool, same order every time")
{
	auto shapes = [](size_t n) {
		std::vector<std::vector<int>> inputs;
		std::vector<int> random(n), sorted(n), reversed(n), equal(n, 42), few(n), organ(n);
		uint32_t x = 2463534242u;
		for (size_t i = 0; i < n; i++)
		{
			x ^= x << 13; x ^= x >> 17; x ^= x << 5;
			random[i] = (int)(x % 1000000);
			sorted[i] = (int)i;
			reversed[i] = (int)(n - i);
			few[i] = (int)(x % 4);
			organ[i] = (int)(i < n / 2 ? i : n - i);
		}
		for (auto* v : { &random, &sorted, &reversed, &equal, &few, &organ })
			inputs.push_back(*v);
		return inputs;
	};
	for (int workers : { 0, 1, -1 })
	{
		loom::Pool pool(loom::PoolConfig{ .workerCount = workers });
		for (size_t n : { size_t(0), size_t(1), size_t(2), size_t(2047), size_t(2049), size_t(100000), size_t(1000000) })
			for (std::vector<int>& v : shapes(n))
			{
				std::vector<int> want = v;
				std::sort(want.begin(), want.end());
				loom::ParallelSort(v.begin(), v.end());
				CHECK(v == want);
			}
		std::vector<int> down = shapes(300000)[0];
		std::vector<int> want = down;
		std::sort(want.begin(), want.end(), std::greater<>{});
		loom::ParallelSort(down.begin(), down.end(), std::greater<>{});
		CHECK(down == want);

		// Move-only elements, and strings.
		std::vector<std::unique_ptr<int>> owned;
		for (int i = 0; i < 50000; i++)
			owned.push_back(std::make_unique<int>((i * 7919) % 50000));
		loom::ParallelSort(owned.begin(), owned.end(), [](const auto& a, const auto& b) { return *a < *b; });
		bool inOrder = true;
		for (size_t i = 0; i < owned.size(); i++)
			inOrder = inOrder && *owned[i] == (int)i;
		CHECK(inOrder);
		std::vector<std::string> words;
		for (int i = 0; i < 40000; i++)
			words.push_back(std::to_string((int64_t)i * 104729 % 40000));
		std::vector<std::string> wantWords = words;
		std::sort(wantWords.begin(), wantWords.end());
		loom::ParallelSort(words.begin(), words.end());
		CHECK(words == wantWords);
	}
	// Not stable, but the same order of equal keys whatever the pool: sort by
	// key alone and compare where each original index ended up.
	std::vector<std::vector<std::pair<int, int>>> orders;
	for (int workers : { 0, 1, 3, -1 })
	{
		loom::Pool pool(loom::PoolConfig{ .workerCount = workers });
		std::vector<std::pair<int, int>> v(400000);
		for (int i = 0; i < (int)v.size(); i++)
			v[(size_t)i] = { (i * 31) % 97, i };
		loom::ParallelSort(v.begin(), v.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
		orders.push_back(std::move(v));
	}
	for (size_t r = 1; r < orders.size(); r++)
		CHECK(orders[r] == orders[0]);
#if defined(__cpp_exceptions)
	{
		loom::Pool pool(loom::PoolConfig{});
		// Failing while counting into buckets, while sorting a bucket, and in a
		// range too small for buckets. As after std::sort the order is then
		// unspecified, but every element is still one live object: none leaked
		// from the buffer, none destroyed twice.
		struct Keyed { int key = 0; Tracked tracked; };
		for (auto [n, failAt] : { std::pair<size_t, int>{ 400000, 100000 }, { 400000, 3000000 }, { 100000, 500000 } })
		{
			const int liveBefore = Tracked::live.load();
			{
				const std::vector<int> keys = shapes(n)[0];
				std::vector<Keyed> v(n);
				for (size_t i = 0; i < n; i++)
					v[i].key = keys[i];
				std::atomic<int> calls{ 0 };
				auto failing = [&](const Keyed& a, const Keyed& b) {
					if (++calls == failAt)
						throw std::runtime_error("comparison failed");
					return a.key < b.key;
				};
				CHECK_THROWS_AS(loom::ParallelSort(v.begin(), v.end(), failing), std::runtime_error);
				CHECK(v.size() == n);
				CHECK(Tracked::live.load() == liveBefore + (int)n);
			}
			CHECK(Tracked::live.load() == liveBefore);
		}
	}
#endif
}

TEST_CASE("loom: pipelines — serial stages keep the source's order, at most inFlight items at once")
{
	for (int workers : { 0, 1, -1 })
	{
		loom::Pool pool(loom::PoolConfig{ .workerCount = workers });
		struct Item { int index = -1; int64_t value = 0; };
		for (uint32_t inFlight : { 1u, 4u, 16u })
		{
			constexpr int kItems = 600;
			int produced = 0;
			std::vector<int64_t> collected;
			std::atomic<int> inParallel{ 0 }, maxParallel{ 0 }, serialBusy{ 0 }, errors{ 0 }, live{ 0 }, maxLive{ 0 };
			std::set<const Item*> seenItems;
			std::mutex seenMutex;
			auto noteMax = [](std::atomic<int>& m, int v) { int s = m.load(); while (v > s && !m.compare_exchange_weak(s, v)) {} };
			loom::Pipeline<Item> p(inFlight);
			p.Source([&](Item& it) {
				if (produced == kItems)
					return false;
				noteMax(maxLive, ++live);
				it.index = produced++;
				{
					std::lock_guard<std::mutex> lock(seenMutex);
					seenItems.insert(&it);
				}
				return true;
			})
			.Parallel([&](Item& it) {
				noteMax(maxParallel, ++inParallel);
				if (it.index % 7 == 0)
					SpinFor(std::chrono::microseconds(30)); // overtake each other
				it.value = (int64_t)it.index * it.index;
				inParallel--;
			})
			.Serial([&](Item& it) {
				if (serialBusy.exchange(1) != 0)
					errors++;
				if (it.value != (int64_t)it.index * it.index)
					errors++;
				collected.push_back(it.value); // serial: no lock needed
				serialBusy = 0;
				live--;
			});
			p.Run();
			REQUIRE(collected.size() == (size_t)kItems);
			bool inOrder = true;
			for (int i = 0; i < kItems; i++)
				inOrder = inOrder && collected[(size_t)i] == (int64_t)i * i;
			CHECK(inOrder);
			CHECK(errors.load() == 0);
			CHECK(maxLive.load() <= (int)inFlight);
			CHECK(maxParallel.load() <= (int)inFlight);
			CHECK(seenItems.size() <= inFlight); // the items are reused
		}
	}
}

TEST_CASE("loom: pipelines — stage TaskDescs, a run inside a task, running again, errors")
{
	loom::Pool pool(loom::PoolConfig{ .workerCount = 4 });
	struct Item { int index = 0; };
	{
		// A main-thread last stage and a limited middle one.
		loom::ScopedLimiter two(2);
		std::atomic<int> limited{ 0 }, maxLimited{ 0 }, offMain{ 0 };
		int produced = 0, uploaded = 0;
		loom::Pipeline<Item> p(8);
		p.Source([&](Item& it) { it.index = produced; return produced++ < 200; })
		 .Parallel([&](Item&) {
			 const int now = ++limited;
			 int seen = maxLimited.load();
			 while (now > seen && !maxLimited.compare_exchange_weak(seen, now)) {}
			 SpinFor(std::chrono::microseconds(20));
			 limited--;
		 }, loom::TaskDesc{ .limiter = two })
		 .Serial([&](Item&) {
			 if (loom::CurrentThreadIndex() != 0)
				 offMain++;
			 uploaded++;
		 }, loom::TaskDesc{ .affinity = loom::Affinity::MainThread });
		p.Run();
		CHECK(uploaded == 200);
		CHECK(offMain.load() == 0);
		CHECK(maxLimited.load() <= 2);

		// The same pipeline again: the stages and items stay.
		produced = 0;
		uploaded = 0;
		p.Run();
		CHECK(uploaded == 200);
	}
	{
		// Run from inside a task: its wait helps.
		std::atomic<int> sum{ 0 };
		loom::ScopedCounter done(1);
		loom::Submit([&] {
			int n = 0;
			loom::Pipeline<Item> p(3);
			p.Source([&](Item& it) { it.index = n; return n++ < 100; })
			 .Parallel([&](Item& it) { sum += it.index; });
			p.Run();
		}, loom::TaskDesc{ .signals = done });
		loom::WaitFor(done);
		CHECK(sum.load() == 99 * 100 / 2);
	}
	{
		// A source alone, and an empty stream.
		int n = 0;
		loom::Pipeline<Item> sourceOnly(2);
		sourceOnly.Source([&](Item&) { return ++n < 10; });
		sourceOnly.Run();
		CHECK(n == 10);
		int stages = 0;
		loom::Pipeline<Item> empty(2);
		empty.Source([](Item&) { return false; }).Serial([&](Item&) { stages++; });
		empty.Run();
		CHECK(stages == 0);
	}
#if defined(__cpp_exceptions)
	{
		std::atomic<int> produced{ 0 }, reachedEnd{ 0 };
		loom::Pipeline<Item> p(4);
		p.Source([&](Item& it) { it.index = produced++; return it.index < 100000; })
		 .Parallel([](Item& it) { if (it.index == 50) throw std::runtime_error("decode failed"); })
		 .Serial([&](Item& it) { if (it.index > 50) reachedEnd++; });
		CHECK_THROWS_AS(p.Run(), std::runtime_error);
		CHECK(produced.load() < 1000); // the source stopped
		CHECK(reachedEnd.load() <= 4); // at most the items already past the failing stage
	}
#endif
}

#if defined(LOOM_TEST_HOOKS)
namespace {
	// Holds every task at the end of its run, before its runner gives back
	// its unit: a unit a task still holds after the work's wait shows.
	struct SlowTaskEnds
	{
		static inline std::atomic<bool> armed{ false };

		static void Hook(int point)
		{
			if (point == 2 && armed.load())
				std::this_thread::sleep_for(std::chrono::milliseconds(3));
		}
	};
}

TEST_CASE("loom: pipelines — no limited stage holds a unit once Run returns")
{
	// The caller may destroy the limiter as soon as Run returns.
	loom::Pool pool(loom::PoolConfig{ .workerCount = 3 });
	struct Item { int index = 0; };
	for (int limitedStage = 0; limitedStage < 3; limitedStage++)
	{
		CAPTURE(limitedStage);
		loom::ScopedLimiter two(2);
		auto desc = [&](int stage) { return stage == limitedStage ? loom::TaskDesc{ .limiter = two } : loom::TaskDesc{}; };
		int produced = 0;
		loom::Pipeline<Item> p(4);
		p.Source([&](Item& it) { it.index = produced; return produced++ < 8; }, desc(0))
		 .Parallel([](Item&) {}, desc(1))
		 .Serial([](Item&) {}, desc(2));
		loom::detail::SetTestHook(&SlowTaskEnds::Hook);
		SlowTaskEnds::armed = true;
		int stillHeld = 0;
		for (int run = 0; run < 5; run++)
		{
			produced = 0;
			p.Run();
			if (loom::HeldUnits(two) != 0)
				stillHeld++;
		}
		SlowTaskEnds::armed = false;
		loom::detail::SetTestHook(nullptr);
		while (loom::LiveTaskCount() > 0)
			std::this_thread::yield();
		CHECK(stillHeld == 0);
	}
}
#endif

TEST_CASE("loom: futures — values, void, move-only results and repeated Get, any pool")
{
	for (int workers : { 0, 1, -1 })
	{
		loom::Pool pool(loom::PoolConfig{ .workerCount = workers });
		const int baseline = LiveCounters();
		{
			std::vector<loom::Future<int>> squares;
			for (int i = 0; i < 200; i++)
				squares.push_back(loom::Async([i] { return i * i; }));
			std::atomic<int> sideEffect{ 0 };
			loom::Future<void> done = loom::Async([&] { sideEffect = 7; });
			loom::Future<std::vector<int>> big = loom::Async([] { return std::vector<int>(1000, 3); });
			loom::Future<std::unique_ptr<int>> owned = loom::Async([] { return std::make_unique<int>(42); });
			loom::Future<std::string> text = loom::Async([s = std::string(100, 'x')] { return s + "!"; }); // a non-trivial callable

			for (int i = 0; i < 200; i++)
				CHECK(squares[(size_t)i].Get() == i * i);
			CHECK(squares[5].IsReady());
			CHECK(squares[5].Get() == 25); // again: the same value
			done.Get();
			CHECK(sideEffect.load() == 7);
			CHECK(big.Get().size() == 1000);
			REQUIRE(owned.Get() != nullptr);
			CHECK(*owned.Get() == 42);
			std::unique_ptr<int> taken = std::move(owned.Get()); // move out to take it
			CHECK(*taken == 42);
			CHECK(text.Get().size() == 101);

			loom::Future<int> moved = std::move(squares[3]);
			CHECK(!squares[3].IsValid());
			CHECK(squares[3].Counter() == loom::kInvalidCounter);
			CHECK(moved.Get() == 9);
			moved = std::move(squares[4]); // drops the old one
			CHECK(moved.Get() == 16);
		}
		CHECK_MESSAGE(CountersReturnTo(baseline), "every future's counter is freed");
	}
}

TEST_CASE("loom: futures — a dropped future's task still runs, and frees its state")
{
	loom::Pool pool(loom::PoolConfig{ .workerCount = 2 });
	const int baseline = LiveCounters();
	Tracked::live = 0;
	std::atomic<int> ran{ 0 };
	{
		loom::ScopedCounter gate(1);
		for (int i = 0; i < 50; i++)
		{
			Tracked tracked;
			loom::Future<int> f = loom::Async([tracked, &ran] { (void)tracked; ran++; return 1; }, loom::TaskDesc{ .waitsFor = gate });
			// Dropped here, before its task can start.
		}
		CHECK(ran.load() == 0);
		CHECK(Tracked::live.load() == 50); // the callables wait with their tasks
		loom::Signal(gate);
		while (loom::LiveTaskCount() > 0)
			std::this_thread::yield();
	}
	CHECK(ran.load() == 50);
	CHECK(Tracked::live.load() == 0);
	CHECK(LiveCounters() == baseline);

	// Dropped while its task runs, and a Reset after it finished.
	std::atomic<bool> started{ false }, release{ false };
	{
		loom::Future<int> f = loom::Async([&] {
			started = true;
			while (!release)
				std::this_thread::yield();
			return 3;
		});
		while (!started)
			std::this_thread::yield();
	}
	release = true;
	loom::Future<int> g = loom::Async([] { return 4; });
	CHECK(g.Get() == 4);
	g.Reset();
	CHECK(!g.IsValid());
	while (loom::LiveTaskCount() > 0)
		std::this_thread::yield();
	CHECK(LiveCounters() == baseline);
}

TEST_CASE("loom: futures — a task can wait for futures and read them without blocking")
{
	loom::Pool pool(loom::PoolConfig{ .workerCount = 3 });
	const loom::DedicatedId io = loom::RegisterDedicatedThread(loom::DedicatedConfig{ .name = "future-io" });
	loom::ScopedCounter gate(1);
	loom::Future<int> a = loom::Async([] { return 20; }, loom::TaskDesc{ .waitsFor = gate });
	loom::Future<int> b = loom::Async([] { return 22; }, loom::TaskDesc{ .affinity = loom::Affinity::Dedicated, .thread = io });
	const loom::CounterHandle both[] = { a.Counter(), b.Counter() };
	// A future of futures: runs once both are ready, reads them, never waits.
	std::atomic<bool> bothReady{ false };
	loom::Future<int> sum = loom::Async([&] {
		bothReady = a.IsReady() && b.IsReady();
		return a.Get() + b.Get();
	}, loom::TaskDesc{ .waitsForAll = both });
	// A plain task can wait for one too.
	std::atomic<int> seen{ 0 };
	loom::ScopedCounter plainDone(1);
	loom::Submit([&] { seen = a.Get(); }, loom::TaskDesc{ .signals = plainDone, .waitsFor = a.Counter() });

	CHECK(!sum.TryWait(20));
	CHECK(!a.IsReady());
	loom::Signal(gate);
	CHECK(sum.Get() == 42);
	CHECK(bothReady.load());
	loom::WaitFor(plainDone);
	CHECK(seen.load() == 20);

	// Main-thread affinity, a priority and a limiter come from the TaskDesc.
	loom::ScopedLimiter one(1);
	loom::Future<int> onMain = loom::Async([] { return loom::CurrentThreadIndex(); }, loom::TaskDesc{ .affinity = loom::Affinity::MainThread });
	loom::Future<int> limited = loom::Async([] { return 5; }, loom::TaskDesc{ .priority = loom::Priority::High, .limiter = one });
	CHECK(onMain.Get() == 0);
	CHECK(limited.Get() == 5);
	CHECK(loom::HeldUnits(one) == 0);
	loom::UnregisterDedicatedThread(io);
}

TEST_CASE("loom: futures — once Get returns, the task holds no limiter unit")
{
	loom::Pool pool(loom::PoolConfig{ .workerCount = 4 });
	int held = 0;
	for (int i = 0; i < 500; i++)
	{
		loom::ScopedLimiter limiter(1 + i % 2); // destroyed right after Get: it must be idle
		loom::Future<int> f = loom::Async([i] { return i; }, loom::TaskDesc{ .limiter = limiter });
		loom::Future<void> g = loom::Async([] {}, loom::TaskDesc{ .limiter = limiter });
		CHECK(f.Get() == i);
		g.Get();
		held += (int)loom::HeldUnits(limiter);
	}
	CHECK(held == 0);
}

#if defined(__cpp_exceptions)
TEST_CASE("loom: futures — Get rethrows the callable's exception, every time")
{
	loom::Pool pool(loom::PoolConfig{ .workerCount = 2 });
	const int baseline = LiveCounters();
	Tracked::live = 0;
	{
		Tracked tracked;
		loom::Future<int> bad = loom::Async([tracked]() -> int { (void)tracked; throw std::runtime_error("parse failed"); });
		loom::Future<void> badVoid = loom::Async([] { throw std::logic_error("no"); });
		loom::Future<int> good = loom::Async([] { return 1; });
		CHECK_THROWS_AS(bad.Get(), std::runtime_error);
		CHECK_THROWS_AS(bad.Get(), std::runtime_error);
		CHECK(bad.IsReady());
		bad.Wait(); // waiting does not throw
		CHECK(bad.TryWait(0));
		CHECK_THROWS_AS(badVoid.Get(), std::logic_error);
		CHECK(good.Get() == 1);
		CHECK(Tracked::live.load() == 1); // the callable's copy is gone once it has run
	}
	CHECK(Tracked::live.load() == 0);
	CHECK(CountersReturnTo(baseline));
}
#endif

TEST_CASE("loom: promises — set from a thread outside loom, before or after GetFuture")
{
	loom::Pool pool(loom::PoolConfig{ .workerCount = 2 });
	const int baseline = LiveCounters();
	{
		// An IO callback on a thread loom did not start.
		loom::Promise<std::string> read;
		loom::Future<std::string> text = read.GetFuture();
		std::thread io([p = std::move(read)]() mutable {
			std::this_thread::sleep_for(std::chrono::milliseconds(10));
			p.SetValue(std::string(64, 'r'));
		});
		CHECK(text.Get().size() == 64);
		io.join();

		// Set first, the future taken afterwards; void; a value built in place.
		loom::Promise<int> early;
		early.SetValue(9);
		CHECK(early.IsSet());
		loom::Future<int> late = early.GetFuture();
		CHECK(late.IsReady());
		CHECK(late.Get() == 9);
		loom::Promise<void> signal;
		loom::Future<void> fired = signal.GetFuture();
		CHECK(!fired.IsReady());
		signal.SetValue();
		fired.Get();
		loom::Promise<std::vector<int>> built;
		built.SetValue(5, 7);
		CHECK(built.GetFuture().Get() == std::vector<int>(5, 7));

		// A task can wait for a promise's future like any other.
		loom::Promise<int> gate;
		loom::Future<int> gated = gate.GetFuture();
		std::atomic<int> seen{ 0 };
		loom::ScopedCounter done(1);
		loom::Submit([&] { seen = gated.Get(); }, loom::TaskDesc{ .signals = done, .waitsFor = gated.Counter() });
		CHECK(!loom::TryWaitFor(done, 10));
		gate.SetValue(11);
		loom::WaitFor(done);
		CHECK(seen.load() == 11);

		// Neither side need outlive the other.
		{
			loom::Promise<int> unread;
			(void)unread.GetFuture(); // the future dropped at once
			unread.SetValue(1);
		}
		{
			loom::Promise<int> neverTaken; // no future, never set
		}
	}
	CHECK(LiveCounters() == baseline);
}

#if defined(__cpp_exceptions)
TEST_CASE("loom: promises — an unset promise breaks its future; SetException")
{
	loom::Pool pool(loom::PoolConfig{ .workerCount = 2 });
	const int baseline = LiveCounters();
	{
		loom::Future<int> orphan;
		{
			loom::Promise<int> p;
			orphan = p.GetFuture();
		}
		CHECK(orphan.IsReady());
		CHECK_THROWS_AS(orphan.Get(), loom::BrokenPromise);

		loom::Promise<int> moved;
		loom::Future<int> f = moved.GetFuture();
		loom::Promise<int> owner(std::move(moved)); // the moved-from one breaks nothing
		CHECK(!f.IsReady());
		owner.SetValue(3);
		CHECK(f.Get() == 3);

		loom::Promise<void> failing;
		loom::Future<void> failed = failing.GetFuture();
		failing.SetException(std::make_exception_ptr(std::runtime_error("io failed")));
		CHECK_THROWS_AS(failed.Get(), std::runtime_error);
	}
	CHECK(LiveCounters() == baseline);
}
#endif

TEST_CASE("loom: promises — set inside a limited task, the task keeps its unit")
{
	loom::Pool pool(loom::PoolConfig{ .workerCount = 2 });
	loom::ScopedLimiter one(1);
	loom::Promise<int> p;
	loom::Future<int> f = p.GetFuture();
	std::atomic<bool> release{ false };
	loom::ScopedCounter done(1);
	loom::Submit([&] {
		p.SetValue(5);
		while (!release)
			std::this_thread::yield();
	}, loom::TaskDesc{ .signals = done, .limiter = one });
	// Without helping: a helping Get could run the task here, and it spins
	// until this thread sets `release`.
	CHECK(f.Get(loom::Help::BlockOnly) == 5);
	CHECK_MESSAGE(loom::HeldUnits(one) == 1, "the setting task is still running and still holds its unit");
	release = true;
	loom::WaitFor(done);
	CHECK(loom::HeldUnits(one) == 0);
}

TEST_CASE("loom: continuations — chains run in order and outlive the futures they read")
{
	for (int workers : { 0, 1, -1 })
	{
		loom::Pool pool(loom::PoolConfig{ .workerCount = workers });
		const int baseline = LiveCounters();
		Tracked::live = 0;
		{
			loom::Future<std::string> chain = loom::Async([] { return 20; })
				.Then([](int& v) { return v + 1; })
				.Then([](int& v) { return std::vector<int>((size_t)v, 2); })
				.Then([](std::vector<int>& v) { return std::to_string(v.size() * (size_t)v[0]); });
			CHECK(chain.Get() == "42");

			// Void at either end; Then on an already ready future; several on one.
			std::atomic<int> steps{ 0 };
			loom::Future<int> fromVoid = loom::Async([&] { steps++; }).Then([&] { steps++; return 7; });
			loom::Future<void> toVoid = fromVoid.Then([&](int& v) { steps += v; });
			toVoid.Get();
			CHECK(steps.load() == 9);
			CHECK(fromVoid.IsReady());
			loom::Future<int> again = fromVoid.Then([](const int& v) { return v * 2; });
			loom::Future<int> branch = fromVoid.Then([](int v) { return v * 3; });
			CHECK(again.Get() == 14);
			CHECK(branch.Get() == 21);

			// Only the last future kept: the chain still completes, and the
			// callables are destroyed as each step runs.
			loom::ScopedCounter gate(1);
			loom::Future<int> last;
			{
				Tracked tracked;
				loom::Future<int> first = loom::Async([tracked] { (void)tracked; return 1; }, loom::TaskDesc{ .waitsFor = gate });
				loom::Future<int> second = first.Then([tracked](int& v) { (void)tracked; return v + 1; });
				last = second.Then([tracked](int& v) { (void)tracked; return v + 1; });
			}
			CHECK(Tracked::live.load() == 3);
			loom::Signal(gate);
			CHECK(last.Get() == 3);
			CHECK(Tracked::live.load() == 0);

			// A chain from a promise, every step on the main thread.
			loom::Promise<int> p;
			loom::Future<int> onMain = p.GetFuture()
				.Then([](int& v) { return v + loom::CurrentThreadIndex(); }, loom::TaskDesc{ .affinity = loom::Affinity::MainThread });
			std::thread setter([&p] { p.SetValue(100); });
			CHECK(onMain.Get() == 100);
			setter.join();
		}
		while (loom::LiveTaskCount() > 0)
			std::this_thread::yield();
		CHECK(LiveCounters() == baseline);
	}
}

TEST_CASE("loom: continuations — a limited step, and a step waiting on more counters")
{
	loom::Pool pool(loom::PoolConfig{ .workerCount = 3 });
	loom::ScopedLimiter one(1);
	for (int i = 0; i < 200; i++)
	{
		loom::Future<int> f = loom::Async([i] { return i; }).Then([](int& v) { return v * 2; }, loom::TaskDesc{ .limiter = one });
		CHECK(f.Get() == 2 * i);
		CHECK(loom::HeldUnits(one) == 0);
	}
	loom::ScopedCounter extra(1);
	const loom::CounterHandle more[] = { extra };
	loom::Future<int> both = loom::Async([] { return 1; }).Then([](int& v) { return v + 1; }, loom::TaskDesc{ .waitsForAll = more });
	CHECK(!both.TryWait(20));
	loom::Signal(extra);
	CHECK(both.Get() == 2);
}

#if defined(__cpp_exceptions)
TEST_CASE("loom: continuations — an error skips the rest of the chain and reaches its end")
{
	loom::Pool pool(loom::PoolConfig{ .workerCount = 2 });
	const int baseline = LiveCounters();
	{
		std::atomic<int> ran{ 0 };
		loom::Future<int> end = loom::Async([]() -> int { throw std::runtime_error("read failed"); })
			.Then([&](int& v) { ran++; return v; })
			.Then([&](int& v) { ran++; return v; });
		CHECK_THROWS_AS(end.Get(), std::runtime_error);
		CHECK(ran.load() == 0);

		// A step that throws; the steps before it ran.
		loom::Future<void> mid = loom::Async([&] { ran++; return 1; })
			.Then([](int&) -> int { throw std::logic_error("parse failed"); })
			.Then([&](int&) { ran += 100; });
		CHECK_THROWS_AS(mid.Get(), std::logic_error);
		CHECK(ran.load() == 1);

		// A broken promise reaches a chain's end too.
		loom::Future<int> fromBroken;
		{
			loom::Promise<int> p;
			fromBroken = p.GetFuture().Then([](int& v) { return v; });
		}
		CHECK_THROWS_AS(fromBroken.Get(), loom::BrokenPromise);
	}
	while (loom::LiveTaskCount() > 0)
		std::this_thread::yield();
	CHECK(LiveCounters() == baseline);
}
#endif

#if defined(LOOM_COROUTINES)
namespace {
	loom::Future<int> AddLater(int a, int b)
	{
		const int x = co_await loom::Async([a] { return a; });
		const int y = co_await loom::Async([b] { return b; });
		co_return x + y;
	}

	loom::Future<void> Bump(std::atomic<int>& n, int times)
	{
		for (int i = 0; i < times; i++)
			co_await loom::Async([&n] { n++; });
	}

	loom::Future<std::unique_ptr<int>> MakeOwned(int v)
	{
		std::unique_ptr<int> p = co_await loom::Async([v] { return std::make_unique<int>(v); });
		co_return p;
	}

	// The halves start as tasks (above a size), so the tree spreads over the pool.
	loom::Future<long> Fib(int n)
	{
		if (n >= 14)
			co_await loom::ResumeOn();
		if (n < 2)
			co_return n;
		loom::Future<long> a = Fib(n - 1);
		loom::Future<long> b = Fib(n - 2);
		co_return co_await a + co_await b;
	}
}

TEST_CASE("loom: coroutines — co_await futures, co_return values, void and move-only, any pool")
{
	for (int workers : { 0, 1, -1 })
	{
		loom::Pool pool(loom::PoolConfig{ .workerCount = workers });
		const int baseline = LiveCounters();
		{
			std::vector<loom::Future<int>> sums;
			for (int i = 0; i < 300; i++)
				sums.push_back(AddLater(i, 2 * i));
			for (int i = 0; i < 300; i++)
				CHECK(sums[(size_t)i].Get() == 3 * i);

			std::atomic<int> n{ 0 };
			Bump(n, 50).Get();
			CHECK(n.load() == 50);

			loom::Future<std::unique_ptr<int>> owned = MakeOwned(9);
			REQUIRE(owned.Get() != nullptr);
			CHECK(*owned.Get() == 9);

			CHECK(Fib(22).Get() == 17711);

			// Awaiting a future that is kept: a reference into it, which stays valid.
			loom::Future<std::vector<int>> kept = loom::Async([] { return std::vector<int>(10, 4); });
			auto reader = [](loom::Future<std::vector<int>>& f) -> loom::Future<int> {
				std::vector<int>& v = co_await f;
				v.push_back(1); // the future's own value
				co_return (int)v.size();
			};
			CHECK(reader(kept).Get() == 11);
			CHECK(kept.Get().size() == 11);

			// A ready future does not suspend: the coroutine carries on where it is.
			loom::Future<int> ready = loom::Async([] { return 5; });
			ready.Wait();
			auto same = [](loom::Future<int>& f) -> loom::Future<bool> {
				const int before = loom::CurrentThreadIndex();
				(void)co_await f;
				co_return loom::CurrentThreadIndex() == before;
			};
			loom::Future<bool> stayed = same(ready);
			CHECK(stayed.IsReady()); // it never suspended, so it is done already
			CHECK(stayed.Get());

			// A coroutine awaiting a promise set from outside loom.
			loom::Promise<int> promise;
			auto waiter = [](loom::Future<int> f) -> loom::Future<int> { co_return co_await f + 1; };
			loom::Future<int> fromPromise = waiter(promise.GetFuture());
			CHECK(!fromPromise.IsReady());
			std::thread setter([&promise] { promise.SetValue(41); });
			CHECK(fromPromise.Get() == 42);
			setter.join();
		}
		CHECK_MESSAGE(CountersReturnTo(baseline), "every coroutine's counter is freed");
		while (loom::LiveTaskCount() > 0)
			std::this_thread::yield();
	}
}

TEST_CASE("loom: coroutines — ResumeOn and After(future, opts) move the rest to the main thread, a dedicated thread, a priority")
{
	loom::Pool pool(loom::PoolConfig{ .workerCount = 3 });
	const loom::DedicatedId io = loom::RegisterDedicatedThread({ .name = "coroutine-io" });
	int ioIndex = -1;
	{
		loom::ScopedCounter found(1);
		loom::Submit([&] { ioIndex = loom::CurrentThreadIndex(); }, { .affinity = loom::Affinity::Dedicated, .thread = io, .signals = found });
		loom::WaitFor(found);
	}
	REQUIRE(ioIndex > 0);

	auto hop = [](loom::DedicatedId io) -> loom::Future<std::vector<int>> {
		std::vector<int> where;
		co_await loom::ResumeOn();
		where.push_back(loom::CurrentThreadIndex());
		co_await loom::ResumeOn({ .affinity = loom::Affinity::MainThread });
		where.push_back(loom::CurrentThreadIndex());
		co_await loom::ResumeOn({ .affinity = loom::Affinity::Dedicated, .thread = io });
		where.push_back(loom::CurrentThreadIndex());
		// Ready already, but with opts it still moves: to the main thread.
		loom::Future<int> done = loom::Async([] { return 1; });
		done.Wait();
		const int v = co_await loom::After(done, { .affinity = loom::Affinity::MainThread });
		where.push_back(v == 1 ? loom::CurrentThreadIndex() : -1);
		co_await loom::ResumeOn({ .priority = loom::Priority::High });
		co_return where;
	};
	for (int round = 0; round < 50; round++)
	{
		std::vector<int> where = hop(io).Get(); // Get on the main thread runs its main-thread parts
		REQUIRE(where.size() == 4);
		CHECK(where[1] == 0);
		CHECK(where[2] == ioIndex);
		CHECK(where[3] == 0);
	}
	loom::UnregisterDedicatedThread(io);
}

TEST_CASE("loom: coroutines — a limited stretch holds a unit until its next suspension, and none once Get returns")
{
	loom::Pool pool(loom::PoolConfig{ .workerCount = 6 });
	loom::ScopedLimiter limiter(3);
	std::atomic<int> inside{ 0 }, most{ 0 }, badUnit{ 0 };
	std::array<std::atomic<int>, 3> unitUsers{};
	auto limited = [&](int i) -> loom::Future<int> {
		co_await loom::ResumeOn({ .limiter = limiter });
		const int now = ++inside;
		int seen = most.load();
		while (now > seen && !most.compare_exchange_weak(seen, now)) {}
		const uint32_t unit = loom::CurrentLimiterUnit();
		if (unit >= 3 || unitUsers[unit]++ != 0)
			badUnit++;
		std::this_thread::sleep_for(std::chrono::microseconds(200));
		if (unit < 3)
			unitUsers[unit]--;
		inside--;
		// Leaving the limiter: the rest runs as a task without one.
		co_await loom::ResumeOn();
		if (loom::CurrentLimiterUnit() != loom::kNoUnit)
			badUnit++;
		co_return co_await loom::Async([i] { return i; });
	};
	std::vector<loom::Future<int>> all;
	for (int i = 0; i < 200; i++)
		all.push_back(limited(i));
	for (int i = 0; i < 200; i++)
		CHECK(all[(size_t)i].Get() == i);
	CHECK(most.load() <= 3);
	CHECK(most.load() >= 1);
	CHECK(badUnit.load() == 0);

	// A coroutine that ends inside a limited stretch gives its unit back
	// before its future is ready, so the limiter is idle once Get returns.
	int held = 0;
	for (int i = 0; i < 500; i++)
	{
		loom::ScopedLimiter once(1 + i % 2); // destroyed right after Get: it must be idle
		auto ends = [](loom::LimiterHandle l, int v) -> loom::Future<int> {
			co_await loom::ResumeOn({ .limiter = l });
			co_return v;
		};
		CHECK(ends(once, i).Get() == i);
		held += (int)loom::HeldUnits(once);
	}
	CHECK(held == 0);
}

TEST_CASE("loom: coroutines — After(counter) and After({ ... }) resume once every counter is zero")
{
	for (int workers : { 0, 2 })
	{
		loom::Pool pool(loom::PoolConfig{ .workerCount = workers });
		loom::ScopedCounter a(1), b(2), c(1);
		std::atomic<int> stage{ 0 };
		auto waits = [&]() -> loom::Future<void> {
			co_await loom::After(a);
			stage = 1;
			co_await loom::After({ a.Get(), b.Get(), c.Get() });
			stage = 2;
			co_await loom::After(c, { .affinity = loom::Affinity::MainThread }); // zero already, but opts: it moves
			stage = loom::CurrentThreadIndex() == 0 ? 3 : -1;
		};
		loom::Future<void> f = waits();
		CHECK(stage.load() == 0);
		CHECK(!f.IsReady());
		loom::Signal(a);
		for (int i = 0; i < 1000 && stage.load() != 1; i++)
			(void)loom::TryHelp(), std::this_thread::yield();
		CHECK(stage.load() == 1);
		loom::Signal(b);
		loom::Signal(c);
		for (int i = 0; i < 100; i++)
			(void)loom::TryHelp(), std::this_thread::yield();
		CHECK(stage.load() == 1); // b is still 1
		loom::Signal(b);
		f.Get();
		CHECK(stage.load() == 3);

		// A satisfied wait without opts does not suspend.
		const loom::CounterHandle zeros[] = { a.Get(), c.Get() };
		auto nowait = [&]() -> loom::Future<int> {
			co_await loom::After(a);
			co_await loom::After(std::span<const loom::CounterHandle>(zeros));
			co_return loom::CurrentThreadIndex();
		};
		loom::Future<int> g = nowait();
		CHECK(g.IsReady());
		CHECK(g.Get() == 0);
	}
}

TEST_CASE("loom: coroutines — locals are gone before the future is ready, parameters once it is dropped")
{
	loom::Pool pool(loom::PoolConfig{ .workerCount = 4 });
	Tracked::live = 0;
	auto keeps = [](Tracked param, int v) -> loom::Future<int> {
		(void)param;
		Tracked local;
		(void)local;
		co_await loom::ResumeOn();
		const int r = co_await loom::Async([v] { return v; });
		co_return r;
	};
	for (int i = 0; i < 300; i++)
	{
		loom::Future<int> f = keeps(Tracked{}, i);
		CHECK(f.Get() == i);
		CHECK(Tracked::live.load() == 1); // the parameter, in the frame the future holds
		f.Reset();
		// The coroutine drops its own hold just after the value is ready.
		const auto until = std::chrono::steady_clock::now() + std::chrono::seconds(10);
		while (Tracked::live.load() != 0 && std::chrono::steady_clock::now() < until)
			std::this_thread::yield();
		CHECK(Tracked::live.load() == 0);
	}
	// Dropped before it finishes: the coroutine frees its frame as it ends.
	for (int i = 0; i < 300; i++)
	{
		loom::ScopedCounter gate(1);
		{
			auto waits = [](Tracked param, loom::CounterHandle g) -> loom::Future<void> {
				(void)param;
				co_await loom::After(g);
			};
			(void)waits(Tracked{}, gate);
		}
		CHECK(Tracked::live.load() == 1);
		loom::Signal(gate);
		while (loom::LiveTaskCount() > 0)
			std::this_thread::yield();
		CHECK(Tracked::live.load() == 0);
	}
}

#if defined(__cpp_exceptions)
TEST_CASE("loom: coroutines — errors reach the coroutine at its co_await, and its future if uncaught")
{
	for (int workers : { 0, 3 })
	{
		loom::Pool pool(loom::PoolConfig{ .workerCount = workers });
		auto throws = []() -> loom::Future<int> {
			co_await loom::ResumeOn();
			throw std::runtime_error("inside");
		};
		loom::Future<int> f = throws();
		CHECK_THROWS_WITH_AS(f.Get(), "inside", std::runtime_error);
		CHECK_THROWS_WITH_AS(f.Get(), "inside", std::runtime_error); // every time

		auto catches = []() -> loom::Future<std::string> {
			try
			{
				(void)co_await loom::Async([]() -> int { throw std::runtime_error("task"); });
			}
			catch (const std::runtime_error& e)
			{
				co_return std::string("caught ") + e.what();
			}
			co_return "not thrown";
		};
		CHECK(catches().Get() == "caught task");

		// A counter whose task threw: After rethrows, as WaitFor would.
		loom::ScopedCounter failed(1);
		loom::Submit([] { throw std::runtime_error("counted"); }, { .signals = failed });
		auto afterFailed = [&]() -> loom::Future<void> { co_await loom::After(failed); };
		CHECK_THROWS_WITH_AS(afterFailed().Get(), "counted", std::runtime_error);

		// A broken promise.
		loom::Future<int> broken;
		{
			loom::Promise<int> p;
			broken = p.GetFuture();
		}
		auto readsBroken = [](loom::Future<int>& b) -> loom::Future<int> { co_return co_await b; };
		CHECK_THROWS_AS(readsBroken(broken).Get(), loom::BrokenPromise);

		// An error before the first suspension still goes to the future.
		auto early = []() -> loom::Future<void> {
			throw std::logic_error("early");
			co_return;
		};
		loom::Future<void> e = early();
		CHECK(e.IsReady());
		CHECK_THROWS_AS(e.Get(), std::logic_error);
	}
}
#endif

TEST_CASE("loom: coroutines — thousands at once, chained on each other and on tasks")
{
	for (int workers : { 0, 1, -1 })
	{
		loom::Pool pool(loom::PoolConfig{ .workerCount = workers });
		const int baseline = LiveCounters();
		{
			loom::Promise<int> start;
			loom::Future<int> go = start.GetFuture();
			std::atomic<long> total{ 0 };
			auto leaf = [&](int i) -> loom::Future<int> {
				const int s = co_await go; // every leaf waits for the same promise
				int sum = s;
				for (int k = 0; k < 3; k++)
					sum += co_await loom::Async([i, k] { return i + k; });
				total += sum;
				co_return sum;
			};
			auto branch = [&](int i) -> loom::Future<long> {
				loom::Future<int> x = leaf(2 * i);
				loom::Future<int> y = leaf(2 * i + 1);
				co_return (long)co_await x + co_await y;
			};
			const int n = 600;
			std::vector<loom::Future<long>> branches;
			for (int i = 0; i < n; i++)
				branches.push_back(branch(i));
			std::thread setter([&start] { start.SetValue(1); });
			long expected = 0;
			for (int i = 0; i < n; i++)
			{
				const long want = 2 + (2L * i * 3 + 3) + ((2L * i + 1) * 3 + 3);
				expected += want;
				CHECK(branches[(size_t)i].Get() == want);
			}
			setter.join();
			CHECK(total.load() == expected);
		}
		CHECK_MESSAGE(CountersReturnTo(baseline), "every coroutine's counter is freed");
	}
}

namespace coroutine_moves {
	loom::Future<size_t> TakeString(loom::CounterHandle gate)
	{
		loom::Future<std::string> f = loom::Async([] { return std::string(1000, 'x'); }, { .waitsFor = gate });
		loom::Future<size_t> length = f.Then([](std::string& s) { return s.size(); });
		std::string s = co_await std::move(f); // moves out of the future its continuation reads
		co_return co_await length + s.size();
	}
	loom::Future<int> TakeUnique(loom::CounterHandle gate)
	{
		loom::Future<std::unique_ptr<int>> f = loom::Async([] { return std::make_unique<int>(7); }, { .waitsFor = gate });
		loom::Future<int> read = f.Then([](std::unique_ptr<int>& p) { return p ? *p : -1; });
		std::unique_ptr<int> p = co_await std::move(f); // cannot copy: waits for the continuation first
		co_return co_await read + (p ? *p : -100);
	}
	loom::Future<int> Step(loom::Future<int> previous) { co_return co_await previous + 1; }
}

TEST_CASE("loom: coroutines — moving a value out of a future its continuations still read")
{
	loom::Pool pool(loom::PoolConfig{ .workerCount = 3 });
	int wrong = 0;
	for (int i = 0; i < 300; i++)
	{
		loom::ScopedCounter gate(1), gate2(1);
		loom::Future<size_t> a = coroutine_moves::TakeString(gate);
		loom::Future<int> b = coroutine_moves::TakeUnique(gate2);
		loom::Signal(gate);
		loom::Signal(gate2);
		wrong += a.Get() == 2000 ? 0 : 1; // the continuation saw all 1000, and so did the move
		wrong += b.Get() == 14 ? 0 : 1;
		a.Reset();
		b.Reset();
	}
	CHECK(wrong == 0);
}

TEST_CASE("loom: coroutines — dropping a long chain of awaiting coroutines does not recurse")
{
	// Each frame keeps the future it was given until the frame goes, so the
	// last future holds the whole chain: dropping it used to destroy the
	// frames one inside the other, as deep as the chain.
	constexpr int kChain = 200000;
	loom::Pool pool(loom::PoolConfig{ .workerCount = 2, .maxCounters = kChain + 64 });
	loom::Future<int> f = loom::Async([] { return 0; });
	for (int i = 0; i < kChain; i++)
		f = coroutine_moves::Step(std::move(f));
	CHECK(f.Get() == kChain);
	f.Reset(); // destroys every frame, without a stack as deep as the chain
}
#endif // LOOM_COROUTINES

namespace {
	const char* DumpTestName(uint32_t id)
	{
		switch (id)
		{
		case 11: return "draw";
		case 22: return "physics";
		case 33: return "decode";
		default: return nullptr;
		}
	}

	// Captures until `ready` holds for a capture (up to 10 s), returning the last.
	template <typename F>
	loom::DebugState CaptureWhen(F ready)
	{
		const auto until = std::chrono::steady_clock::now() + std::chrono::seconds(10);
		loom::DebugState s = loom::CaptureDebugState();
		while (!ready(s) && std::chrono::steady_clock::now() < until)
		{
			std::this_thread::yield();
			s = loom::CaptureDebugState();
		}
		return s;
	}

	const loom::DebugCounter* FindCounter(const loom::DebugState& s, loom::CounterHandle h)
	{
		for (const loom::DebugCounter& c : s.counters)
			if (c.handle == h)
				return &c;
		return nullptr;
	}

	const loom::DebugThread* FindRunning(const loom::DebugState& s, uint32_t nameId)
	{
		for (const loom::DebugThread& t : s.threads)
			if (t.runningDepth > 0 && t.runningNameId == nameId)
				return &t;
		return nullptr;
	}
}

TEST_CASE("loom: debug dump — a stuck wait, the tasks waiting for it and who sleeps on it")
{
	loom::Pool pool(loom::PoolConfig{ .workerCount = 2 });
	CHECK(loom::CaptureDebugState().threads.front().isMain);

	loom::ScopedCounter gate(1), done(3), inner(1);
	for (int i = 0; i < 3; i++)
		loom::Submit([] {}, loom::TaskDesc{ .signals = done, .waitsFor = gate, .nameId = 11 });
	// A task that waits inside itself, and a thread outside the pool waiting.
	loom::Submit([g = gate.Get()] { loom::WaitFor(g, loom::Help::BlockOnly); }, loom::TaskDesc{ .signals = inner, .nameId = 22 });
	std::atomic<int> outside{ -1 };
	std::thread waiter([&] {
		outside = loom::CurrentThreadIndex();
		loom::WaitFor(done.Get(), loom::Help::BlockOnly);
	});

	const loom::DebugState s = CaptureWhen([&](const loom::DebugState& st) {
		const loom::DebugCounter* d = FindCounter(st, done.Get());
		const loom::DebugThread* t = FindRunning(st, 22);
		const loom::DebugCounter* gc = FindCounter(st, gate.Get());
		// The thread's record names the counter before it goes to sleep on it.
		return d && !d->sleepers.empty() && t && t->waitingOn == gate.Get() && gc
			&& std::find(gc->sleepers.begin(), gc->sleepers.end(), t->index) != gc->sleepers.end();
	});
	const loom::DebugCounter* g = FindCounter(s, gate.Get());
	const loom::DebugCounter* d = FindCounter(s, done.Get());
	const loom::DebugThread* t = FindRunning(s, 22);
	REQUIRE(g);
	REQUIRE(d);
	REQUIRE(t);
	CHECK(g->count == 1);
	CHECK(g->waitingTotal == 3);
	REQUIRE(g->waiting.size() == 3);
	CHECK(g->waiting[0].nameId == 11);
	CHECK(std::find(g->sleepers.begin(), g->sleepers.end(), t->index) != g->sleepers.end()); // the task's own wait sleeps
	CHECK(d->count == 3);
	CHECK(d->sleepers == std::vector<int>{ outside.load() });
	CHECK(t->runningSignals == inner.Get());
	CHECK(t->waitingSeconds >= 0.0);
	CHECK(!t->isMain);

	const std::string text = loom::FormatDebugState(s, &DumpTestName);
	CHECK(text.find("3 waiting: 'draw' x3") != std::string::npos);
	CHECK(text.find("running 'physics'") != std::string::npos);
	CHECK(text.find("asleep on it: [" + std::to_string(outside.load()) + "]") != std::string::npos);
	CHECK(loom::FormatDebugState(s).find("task#11 x3") != std::string::npos); // no names: ids

	loom::Signal(gate);
	waiter.join();
	loom::WaitFor(inner);
	const loom::DebugState after = CaptureWhen([](const loom::DebugState& st) {
		for (const loom::DebugThread& th : st.threads)
			if (th.runningDepth > 0)
				return false;
		return true;
	});
	CHECK(FindRunning(after, 22) == nullptr);
	CHECK(FindCounter(after, gate.Get())->count == 0);
	CHECK(FindCounter(after, gate.Get())->waitingTotal == 0);
}

TEST_CASE("loom: debug dump — hints for a missing signal, unpumped main work and a limiter waiting on itself")
{
	loom::Pool pool(loom::PoolConfig{ .workerCount = 2 });
	{
		// Nobody will signal this: only a thread outside the pool waits.
		loom::ScopedCounter never(1);
		std::thread waiter([&] { loom::WaitFor(never.Get(), loom::Help::BlockOnly); });
		CaptureWhen([&](const loom::DebugState& st) {
			const loom::DebugCounter* c = FindCounter(st, never.Get());
			return c && !c->sleepers.empty();
		});
		CHECK(loom::DebugDump().find("hint: nothing is running or queued") != std::string::npos);
		loom::Signal(never);
		waiter.join();
	}
	{
		loom::ScopedCounter onMain(1);
		loom::Submit([] {}, loom::TaskDesc{ .affinity = loom::Affinity::MainThread, .signals = onMain });
		const std::string text = loom::DebugDump();
		CHECK(text.find("for the main thread 1") != std::string::npos);
		CHECK(text.find("hint: main-thread tasks are queued") != std::string::npos);
		loom::WaitFor(onMain);
	}
	{
		// A limited task waiting for work of its own limiter: stuck until released.
		loom::ScopedLimiter one(1);
		loom::ScopedCounter outerDone(1), innerDone(1);
		std::atomic<bool> release{ false };
		loom::Submit([&] {
			loom::Submit([] {}, loom::TaskDesc{ .signals = innerDone, .nameId = 33, .limiter = one });
			while (!release && !loom::TryWaitFor(innerDone, 1, loom::Help::BlockOnly))
			{
			}
		}, loom::TaskDesc{ .signals = outerDone, .nameId = 22, .limiter = one });
		const loom::DebugState s = CaptureWhen([&](const loom::DebugState& st) {
			const loom::DebugThread* t = FindRunning(st, 22);
			return t && t->waitingOn == innerDone.Get() && !st.limiters.empty() && st.limiters[0].waitingTotal == 1;
		});
		REQUIRE(s.limiters.size() == 1);
		CHECK(s.limiters[0].handle == one.Get());
		CHECK(s.limiters[0].held == 1);
		REQUIRE(s.limiters[0].waiting.size() == 1);
		CHECK(s.limiters[0].waiting[0].nameId == 33);
		CHECK(s.limiters[0].waiting[0].limiter == one.Get());
		const loom::DebugThread* t = FindRunning(s, 22);
		REQUIRE(t);
		CHECK(t->runningLimiter == one.Get());
		const std::string text = loom::FormatDebugState(s, &DumpTestName);
		CHECK(text.find("1 waiting: 'decode'") != std::string::npos);
		CHECK(text.find("a limited task must not wait for its own limiter's work") != std::string::npos);
		release = true;
		loom::WaitFor(outerDone);
		loom::WaitFor(innerDone);
	}
}

TEST_CASE("loom: debug dump — main thread waits and nested tasks, no pool, capture under load")
{
	CHECK(loom::CaptureDebugState().threads.empty()); // no pool
	CHECK(loom::DebugDump().find("0 tasks live") != std::string::npos);
	{
		// No workers: the main thread runs everything, inside its own wait.
		loom::Pool pool(loom::PoolConfig{ .workerCount = 0 });
		loom::DebugState seen, back;
		loom::ScopedCounter outer(1);
		loom::Submit([&] {
			loom::ScopedCounter nested(1);
			loom::Submit([&] { seen = loom::CaptureDebugState(); }, loom::TaskDesc{ .signals = nested, .nameId = 11 });
			loom::WaitFor(nested);
			back = loom::CaptureDebugState(); // the nested task and the wait are over
		}, loom::TaskDesc{ .signals = outer, .nameId = 22 });
		loom::WaitFor(outer);
		REQUIRE(!seen.threads.empty());
		const loom::DebugThread& m = seen.threads.front();
		CHECK(m.isMain);
		CHECK(m.runningDepth == 2);       // the outer task, and the one its wait ran
		CHECK(m.runningNameId == 11);  // the innermost
		CHECK(m.waitingOn != loom::kInvalidCounter);
		CHECK(loom::FormatDebugState(seen, &DumpTestName).find("running 'draw' (signals") != std::string::npos);
		REQUIRE(!back.threads.empty());
		CHECK(back.threads.front().runningDepth == 1);
		CHECK(back.threads.front().runningNameId == 22);
		CHECK(back.threads.front().waitingOn == outer.Get()); // the main thread's own wait, which ran the task
		const loom::DebugState after = loom::CaptureDebugState();
		CHECK(after.threads.front().runningDepth == 0);
		CHECK(after.threads.front().waitingOn == loom::kInvalidCounter);
		CHECK(after.threads.front().tasksRun >= 2);
	}
	{
		// Captures racing a busy pool: counters, limiters, futures and waits
		// come and go while another thread reads them.
		loom::Pool pool(loom::PoolConfig{});
		std::atomic<bool> stop{ false };
		std::atomic<int> captures{ 0 };
		std::thread reader([&] {
			while (!stop)
			{
				(void)loom::DebugDump(&DumpTestName);
				captures++;
			}
		});
		loom::ScopedLimiter four(4);
		for (int round = 0; round < 200; round++)
		{
			loom::ScopedCounter gate(1), done(64);
			for (int i = 0; i < 64; i++)
				loom::Submit([&] { (void)loom::Count(gate); }, loom::TaskDesc{ .signals = done, .waitsFor = gate, .nameId = 11, .limiter = i % 2 ? four.Get() : loom::kInvalidLimiter });
			loom::Future<int> f = loom::Async([] { return 1; }).Then([](int& v) { return v + 1; });
			loom::Signal(gate);
			loom::WaitFor(done);
			CHECK(f.Get() == 2);
		}
		while (captures.load() < 10)
			std::this_thread::yield();
		stop = true;
		reader.join();
	}
}

TEST_CASE("loom: profiler zone hook — begin/end pairs carry the executing thread index")
{
	loom::Pool pool; // the suite owns its pool explicitly

	struct ZoneRecord { int threadIndex = -1; std::string name; bool begin = false; uint64_t handle = 0; };
	static std::mutex s_recordMutex;
	static std::vector<ZoneRecord> s_records;
	s_recordMutex.lock();
	s_records.clear();
	s_recordMutex.unlock();

	loom::SetProfilerZoneHook([](int threadIndex, uint32_t, uint8_t, const char* name, bool begin, uint64_t handle) -> uint64_t {
		uint64_t issued = 0;
		std::lock_guard<std::mutex> lock(s_recordMutex);
		if (begin)
		{
			issued = s_records.size() + 1;
			s_records.push_back({ threadIndex, name, true, issued });
		}
		else
		{
			s_records.push_back({ threadIndex, name, false, handle });
		}
		return issued;
	});

	loom::ParallelFor(0, 200, 25, [](size_t begin, size_t end) {
		for (size_t i = begin; i < end; i++) { (void)i; }
	});

	loom::SetProfilerZoneHook(nullptr);

	{
		std::lock_guard<std::mutex> lock(s_recordMutex);
		size_t begins = 0, ends = 0;
		for (const ZoneRecord& r : s_records)
		{
			CHECK(r.name == "Task");
			CHECK(r.threadIndex >= 0);
			if (r.begin) begins++;
			else ends++;
		}
		CHECK(begins == ends);
		CHECK(begins >= 8); // 200 / 25 = 8 chunks
	}
}

// ---- Helping loops apply what they hold back ---------------------------------------

TEST_CASE("loom: dedicated threads — unregistering applies the signals of the tasks it ran while helping")
{
	// The unregister helps while it waits: the task it runs defers its
	// signal, and the dedicated thread's task waits for that signal.
	REQUIRE(!loom::HasActivePool());
	CHECK(EndsWithin([] {
		loom::Pool pool(loom::PoolConfig{ .workerCount = 0, .shutdownTimeoutMs = 500 });
		const loom::DedicatedId d = loom::RegisterDedicatedThread({ .name = "io" });
		loom::ScopedCounter gate(1);
		loom::Submit([] {}, { .affinity = loom::Affinity::Dedicated, .thread = d, .waitsFor = gate });
		loom::Submit([] {}, { .signals = gate }); // on this thread's deque: the unregister runs it
		loom::UnregisterDedicatedThread(d);
	}, 5000));
}

#if defined(LOOM_COROUTINES)
static loom::Future<int> TakeMoveOnly(loom::Future<std::unique_ptr<int>> f)
{
	std::unique_ptr<int> v = co_await std::move(f); // waits for the continuation reading it
	co_return *v;
}

TEST_CASE("loom: coroutines — a moving co_await of a move-only value waits for a gated continuation")
{
	// The continuation waits for a gate whose signaller the co_await's own
	// thread runs: the wait must apply that signal, not hold it back.
	REQUIRE(!loom::HasActivePool());
	CHECK(EndsWithin([] {
		loom::Pool pool(loom::PoolConfig{ .workerCount = 0, .shutdownTimeoutMs = 500 });
		loom::Promise<std::unique_ptr<int>> p;
		loom::Future<std::unique_ptr<int>> f = p.GetFuture();
		p.SetValue(std::make_unique<int>(7));
		loom::ScopedCounter gate(1);
		const loom::CounterHandle gates[1] = { gate };
		loom::Future<int> g = f.Then([](std::unique_ptr<int>& v) { return *v; }, { .waitsForAll = gates });
		loom::Submit([] {}, { .signals = gate });
		loom::Future<int> t = TakeMoveOnly(std::move(f));
		if (t.Get() != 7 || g.Get() != 7)
			_exit(3);
	}, 5000));
}

TEST_CASE("loom: coroutines — a moving co_await suspends while continuations have yet to read, without spinning")
{
	loom::Pool pool(loom::PoolConfig{ .workerCount = 2 });
	loom::Promise<std::unique_ptr<int>> p;
	loom::Future<std::unique_ptr<int>> f = p.GetFuture();
	loom::ScopedCounter gate(1);
	const loom::CounterHandle gates[1] = { gate };
	std::atomic<int> read{ 0 };
	loom::Future<int> g = f.Then([&](std::unique_ptr<int>& v) { read++; return *v; }, { .waitsForAll = gates });
	p.SetValue(std::make_unique<int>(5));
	loom::Future<int> t = TakeMoveOnly(std::move(f)); // returns at its suspension
	const auto processCpuMs = [] {
		timespec ts{};
		clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ts);
		return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
	};
	const double c0 = processCpuMs();
	std::this_thread::sleep_for(std::chrono::milliseconds(200));
	const double spent = processCpuMs() - c0;
	CHECK(!t.IsReady()); // suspended until the continuation has read the value
	CHECK(read.load() == 0);
	CHECK(spent < 50.0); // nobody spins while it waits
	loom::Signal(gate);
	CHECK(t.Get() == 5);
	CHECK(g.Get() == 5);
	CHECK(read.load() == 1);
}
#endif

// ---- A wake for pool work reaches a thread that runs it ------------------------------

static loom::CounterHandle s_WakeX, s_WakeA;

TEST_CASE("loom: wakes — a main thread woken for pool work that runs main-thread work first passes the wake on")
{
	// The main thread, asleep in a helping wait, is the newest helper on the
	// idle list; a StealFromPool dedicated thread is the other. One batch
	// queues a main-thread task that blocks on X, then X: the wake for X goes
	// to the main thread, whose help takes the main-thread task first.
	REQUIRE(!loom::HasActivePool());
	CHECK(EndsWithin([] {
		for (int round = 0; round < 5; round++)
		{
			loom::Pool pool(loom::PoolConfig{ .workerCount = 0, .shutdownTimeoutMs = 200 });
			const loom::DedicatedId d = loom::RegisterDedicatedThread({ .name = "D", .idle = loom::DedicatedConfig::Idle::StealFromPool });
			std::this_thread::sleep_for(std::chrono::milliseconds(20));
			s_WakeX = loom::CreateCounter(1);
			s_WakeA = loom::CreateCounter(1);
			std::thread outside([] {
				std::this_thread::sleep_for(std::chrono::milliseconds(20));
				loom::TaskDesc both[2];
				both[0] = loom::TaskDesc{ .fn = [](void*) { loom::WaitFor(s_WakeX, loom::Help::BlockOnly); },
					.affinity = loom::Affinity::MainThread, .signals = s_WakeA };
				both[1] = loom::TaskDesc{ .fn = [](void*) {}, .signals = s_WakeX };
				loom::Submit(std::span<const loom::TaskDesc>(both, 2));
			});
			loom::WaitFor(s_WakeA);
			outside.join();
			loom::UnregisterDedicatedThread(d);
			loom::DestroyCounter(s_WakeX);
			loom::DestroyCounter(s_WakeA);
		}
	}, 10000));
}

TEST_CASE("loom: wakes — a dedicated thread woken for pool work that runs its own task first passes the wake on")
{
	// The same through a StealFromPool dedicated thread: it is the newest
	// helper, the wake for X goes to it, and it runs a task of its own queue
	// that blocks on X. The main thread, asleep helping, could run X.
	REQUIRE(!loom::HasActivePool());
	CHECK(EndsWithin([] {
		for (int round = 0; round < 5; round++)
		{
			loom::Pool pool(loom::PoolConfig{ .workerCount = 0, .shutdownTimeoutMs = 200 });
			const loom::DedicatedId d = loom::RegisterDedicatedThread({ .name = "D", .idle = loom::DedicatedConfig::Idle::StealFromPool });
			s_WakeX = loom::CreateCounter(1);
			std::atomic<bool> yDone{ false };
			std::thread outside([d, &yDone] {
				std::this_thread::sleep_for(std::chrono::milliseconds(20));
				loom::ScopedCounter z(1);
				loom::Submit([] {}, loom::TaskDesc{ .affinity = loom::Affinity::Dedicated, .thread = d, .signals = z });
				loom::WaitFor(z, loom::Help::BlockOnly); // D slept again after the main thread
				std::this_thread::sleep_for(std::chrono::milliseconds(20));
				loom::TaskDesc both[2];
				both[0] = loom::TaskDesc{ .fn = [](void* done) {
					loom::WaitFor(s_WakeX, loom::Help::BlockOnly);
					static_cast<std::atomic<bool>*>(done)->store(true);
				}, .userData = &yDone, .affinity = loom::Affinity::Dedicated, .thread = d };
				both[1] = loom::TaskDesc{ .fn = [](void*) {}, .signals = s_WakeX };
				loom::Submit(std::span<const loom::TaskDesc>(both, 2));
			});
			loom::WaitFor(s_WakeX);
			outside.join();
			while (!yDone)
				loom::TryHelp();
			loom::UnregisterDedicatedThread(d);
			loom::DestroyCounter(s_WakeX);
		}
	}, 10000));
}

// ---- Isolated waits find their region's tasks ---------------------------------------

static loom::CounterHandle s_IsoIo, s_IsoMine;
static std::atomic<bool> s_IsoRan{ false };

TEST_CASE("loom: Isolated — a region's task released by a thread outside the region reaches its wait")
{
	// The only worker waits inside Isolated for a task gated on an IO
	// counter; the main thread signals it, pumps a frame loop and never
	// helps. The released task must not sit on the main thread's deque,
	// where the isolated wait cannot take it.
	REQUIRE(!loom::HasActivePool());
	CHECK(EndsWithin([] {
		loom::Pool pool(loom::PoolConfig{ .workerCount = 1, .shutdownTimeoutMs = 500 });
		s_IsoIo = loom::CreateCounter(1);
		loom::ScopedCounter done(1);
		loom::Submit([] {
			loom::Isolated([] {
				loom::ScopedCounter mine(1);
				s_IsoMine = mine;
				loom::Submit(loom::TaskDesc{ .fn = [](void*) {}, .signals = s_IsoMine, .waitsFor = s_IsoIo });
				loom::WaitFor(mine);
			});
		}, { .signals = done });
		std::this_thread::sleep_for(std::chrono::milliseconds(50)); // the worker is in its wait
		loom::Signal(s_IsoIo);
		while (loom::Count(done) != 0)
		{
			loom::PumpMainThreadTasks();
			std::this_thread::sleep_for(std::chrono::milliseconds(5));
		}
		loom::DestroyCounter(s_IsoIo);
	}, 5000));
}

TEST_CASE("loom: Isolated — a wait finds its region's task below many foreign entries in its own deque")
{
	// An inner region queues 100 fire-and-forget tasks above the outer
	// region's task in the only worker's deque; the outer wait sets them all
	// aside and runs its own.
	loom::Pool pool(loom::PoolConfig{ .workerCount = 1 });
	s_IsoRan = false;
	loom::ScopedCounter outer(1);
	loom::Submit([] {
		loom::Isolated([] {
			loom::ScopedCounter mine(1);
			s_IsoMine = mine;
			loom::Submit([] { s_IsoRan = true; }, { .signals = s_IsoMine });
			loom::Isolated([] { for (int i = 0; i < 100; i++) loom::Submit([] {}); });
			loom::WaitFor(mine);
		});
	}, { .signals = outer });
	CHECK(loom::TryWaitFor(outer, 4000, loom::Help::BlockOnly));
	CHECK(s_IsoRan.load());
	loom::WaitFor(outer);
}

TEST_CASE("loom: Isolated — main-thread work of another region does not wake an isolated main-thread wait")
{
	// A worker outside the region streams main-thread tasks while the main
	// thread waits in Isolated: each used to wake it to scan the whole queue.
	loom::Pool pool(loom::PoolConfig{ .workerCount = 2, .maxTasks = 1u << 18 });
	static std::atomic<bool> s_flooded, s_started;
	s_flooded = false;
	s_started = false;
	loom::ScopedCounter flood(1);
	loom::Submit([] {
		while (!s_started) std::this_thread::yield();
		for (int i = 0; i < 40000; i++)
		{
			loom::Submit([] {}, { .affinity = loom::Affinity::MainThread });
			if (i % 16 == 0) std::this_thread::sleep_for(std::chrono::microseconds(20));
		}
		s_flooded = true;
	}, { .signals = flood });
	double cpu = 0;
	loom::Isolated([&] {
		loom::ScopedCounter regionDone(1);
		loom::Submit([] { while (!s_flooded) std::this_thread::sleep_for(std::chrono::microseconds(100)); }, { .signals = regionDone });
		std::this_thread::sleep_for(std::chrono::milliseconds(20));
		s_started = true;
		const double c0 = ThreadCpuMs();
		loom::WaitFor(regionDone);
		cpu = ThreadCpuMs() - c0;
	});
	CHECK(cpu < 40.0); // 100 ms and more when every submit woke it
	loom::WaitFor(flood);
	loom::PumpMainThreadTasks();
}

// ---- Priorities ----------------------------------------------------------------------

static std::atomic<bool> s_ChainStop{ false };
static std::atomic<int> s_Chains{ 0 };

static void SpinChain(void*)
{
	const auto t = std::chrono::steady_clock::now();
	while (std::chrono::steady_clock::now() - t < std::chrono::microseconds(20)) {}
	if (!s_ChainStop)
		loom::Submit(loom::TaskDesc{ .fn = &SpinChain }); // its successor, onto this worker's deque
	else
		s_Chains--;
}

TEST_CASE("loom: priorities — High queued elsewhere runs before a worker's own Normal work")
{
	// Every worker feeds its own deque a chain of Normal tasks, so none ever
	// runs out of work of its own; a High task from the main thread and one
	// from an outside thread must still run at once.
	loom::Pool pool(loom::PoolConfig{ .workerCount = 4 });
	s_ChainStop = false;
	s_Chains = 4;
	for (int i = 0; i < 4; i++)
		loom::Submit(loom::TaskDesc{ .fn = &SpinChain });
	std::this_thread::sleep_for(std::chrono::milliseconds(50));
	loom::ScopedCounter fromMain(1), fromOutside(1);
	loom::Submit([] {}, { .signals = fromMain, .priority = loom::Priority::High });
	std::thread([&] { loom::Submit([] {}, { .signals = fromOutside, .priority = loom::Priority::High }); }).join();
	CHECK(loom::TryWaitFor(fromMain, 1000, loom::Help::BlockOnly));
	CHECK(loom::TryWaitFor(fromOutside, 1000, loom::Help::BlockOnly));
	s_ChainStop = true;
	while (s_Chains > 0)
		std::this_thread::sleep_for(std::chrono::milliseconds(1));
}

TEST_CASE("loom: priorities — a value outside the enum is refused, or runs as Normal in a release build")
{
	REQUIRE(!loom::HasActivePool());
#if defined(NDEBUG)
	// From a thread without deques it used to index a fourth injection queue.
	loom::Pool pool(loom::PoolConfig{ .workerCount = 2 });
	std::thread([] {
		loom::ScopedCounter done(1);
		loom::Submit([] {}, { .signals = done, .priority = loom::Priority(3) });
		CHECK(loom::TryWaitFor(done, 2000));
	}).join();
#else
	std::string err;
	CHECK(EndsInAbort([] {
		loom::Pool pool(loom::PoolConfig{ .workerCount = 1 });
		loom::Submit([] {}, { .priority = loom::Priority(3) });
	}, err));
	CHECK(err.find("priority out of range") != std::string::npos);
#endif
}

// ---- Counters and limiters at their edges ---------------------------------------------

TEST_CASE("loom: counters — a wait woken by the last signal returns though the counter is destroyed and its entry reused")
{
	loom::Pool pool(loom::PoolConfig{ .workerCount = 1, .maxCounters = 1 });
	int stuck = 0;
	for (int round = 0; round < 50; round++)
	{
		loom::CounterHandle c = loom::CreateCounter(1);
		std::atomic<int> phase{ 0 };
		std::thread waiter([&] {
			phase = 1;
			loom::WaitFor(c, loom::Help::BlockOnly);
			phase = 2;
		});
		while (phase.load() != 1)
			std::this_thread::yield();
		std::this_thread::sleep_for(std::chrono::microseconds(200)); // asleep on c
		loom::Signal(c);
		loom::DestroyCounter(c);
		loom::CounterHandle next = loom::CreateCounter(1); // the same entry (maxCounters 1)
		const auto until = std::chrono::steady_clock::now() + std::chrono::milliseconds(500);
		while (phase.load() != 2 && std::chrono::steady_clock::now() < until)
			std::this_thread::sleep_for(std::chrono::microseconds(100));
		if (phase.load() != 2)
			stuck++;
		loom::Signal(next);
		waiter.join();
		loom::DestroyCounter(next);
	}
	CHECK(stuck == 0);
}

TEST_CASE("loom: limiters — lowering a huge limit is one step, and the lower limit holds")
{
	loom::Pool pool(loom::PoolConfig{ .workerCount = 2 });
	loom::LimiterHandle l = loom::CreateLimiter(2147483646u);
	const auto t0 = std::chrono::steady_clock::now();
	loom::SetLimit(l, 1);
	const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
	CHECK(ms < 100.0); // one atomic per unit took about 9 s
	CHECK(loom::Limit(l) == 1);
	// The lowered limit holds: one task at a time.
	std::atomic<int> running{ 0 }, peak{ 0 };
	loom::ScopedCounter done(16);
	for (int i = 0; i < 16; i++)
		loom::Submit([&] {
			const int now = ++running;
			int p = peak.load();
			while (now > p && !peak.compare_exchange_weak(p, now)) {}
			std::this_thread::sleep_for(std::chrono::microseconds(200));
			running--;
		}, { .signals = done, .limiter = l });
	loom::WaitFor(done);
	CHECK(peak.load() == 1);
	loom::SetLimit(l, 4);
	CHECK(loom::Limit(l) == 4);
	loom::DestroyLimiter(l);
}

// ---- Shutdown at the edges --------------------------------------------------------------

TEST_CASE("loom: Shutdown — a long task is not a stuck one: the main-thread work it submits as it ends runs")
{
	REQUIRE(!loom::HasActivePool());
	CHECK(EndsWithin([] {
		static std::atomic<bool> ran{ false };
		{
			loom::Pool pool(loom::PoolConfig{ .workerCount = 2, .shutdownTimeoutMs = 100 });
			loom::Submit([] {
				std::this_thread::sleep_for(std::chrono::milliseconds(400));
				loom::Submit([] { ran = true; }, { .affinity = loom::Affinity::MainThread });
			});
			std::this_thread::sleep_for(std::chrono::milliseconds(10));
		}
		if (!ran)
			_exit(3);
	}, 5000));
}

TEST_CASE("loom: Shutdown — a Pool destroyed on another thread while the main thread pumps")
{
	// The documented off-thread shutdown: the main thread's frame loop only
	// pumps, and the Pool must not be freed while a pump still reads it.
	for (int trial = 0; trial < 200; trial++)
	{
		auto pool = std::make_unique<loom::Pool>(loom::PoolConfig{ .workerCount = 2 });
		std::atomic<bool> done{ false };
		std::atomic<int> ran{ 0 };
		std::thread quitter([&] {
			{
				loom::ScopedCounter c(4);
				for (int i = 0; i < 4; i++)
					loom::Submit([&] { ran++; }, { .affinity = loom::Affinity::MainThread, .signals = c });
				loom::WaitFor(c, loom::Help::BlockOnly);
			}
			loom::Submit([&] { ran++; }, { .affinity = loom::Affinity::MainThread });
			pool.reset();
			done = true;
		});
		while (!done)
			loom::PumpMainThreadTasks();
		quitter.join();
		REQUIRE(ran.load() == 5);
	}
}

struct SubmitsOnExit
{
	bool used = false;
	~SubmitsOnExit()
	{
		if (used)
			loom::Submit([] {});
	}
};

TEST_CASE("loom: Shutdown — a pool thread's thread-local destructor that submits ends the program, saying why")
{
	// Its destructor runs as the worker exits, after the drain: the task
	// would never run.
	REQUIRE(!loom::HasActivePool());
	std::string err;
	CHECK(EndsInAbort([] {
		loom::Pool pool(loom::PoolConfig{ .workerCount = 2 });
		// One task on each worker: each waits until both have started, and the
		// main thread does not help, so on any number of CPUs both workers
		// make the thread-local.
		static std::atomic<int> s_started{ 0 };
		loom::ScopedCounter done(2);
		for (int i = 0; i < 2; i++)
			loom::Submit([] {
				static thread_local SubmitsOnExit t_flush;
				t_flush.used = true;
				s_started++;
				while (s_started.load() < 2)
					std::this_thread::yield();
			}, { .signals = done });
		loom::WaitFor(done, loom::Help::BlockOnly);
	}, err));
	CHECK(err.find("submitted after Shutdown stopped") != std::string::npos);
}

// ---- Threads that come and go ------------------------------------------------------------

struct WaitsOnExit
{
	bool armed = false;
	~WaitsOnExit()
	{
		if (!armed)
			return;
		loom::ScopedCounter done(1);
		loom::Submit([] {}, { .signals = done });
		loom::WaitFor(done);
	}
};

TEST_CASE("loom: thread indices — a thread-local destructor that calls loom does not keep an index of its own")
{
	// Outside threads, one alive at a time, each flushing work from a
	// thread-local destructor made before its first loom call: the index
	// goes back once that destructor is done, so the next thread reuses it.
	loom::Pool pool(loom::PoolConfig{ .workerCount = 2 });
	int highest = 0;
	for (int i = 0; i < 300; i++)
	{
		std::thread([&highest] {
			static thread_local WaitsOnExit t_flush; // constructed before the index is taken
			t_flush.armed = true;
			highest = std::max(highest, loom::CurrentThreadIndex());
		}).join();
	}
	CHECK(highest < 16); // each used to take a new index, never given back
}

TEST_CASE("loom: PoolConfig — the default worker count follows the CPUs this process may use")
{
	REQUIRE(!loom::HasActivePool());
	CHECK(EndsWithin([] {
		cpu_set_t set;
		CPU_ZERO(&set);
		CPU_SET(0, &set);
		CPU_SET(1, &set);
		if (sched_setaffinity(0, sizeof(set), &set) != 0)
			return; // not allowed here: nothing to check
		loom::Pool pool; // max(1, usable CPUs - 1)
		if (loom::WorkerCount() != 1)
			_exit(3);
	}, 5000));
}

} // namespace

}} // namespace loom::tests
