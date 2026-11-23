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
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#if defined(__linux__)
#include <pthread.h>
#endif

namespace loom { namespace tests {

namespace {

void SpinUntil(int target, CounterHandle c, int timeoutMs)
{
	auto start = std::chrono::steady_clock::now();
	while (loom::Load(c) != target)
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
			loom::TaskDesc{ nullptr, nullptr, loom::Affinity::AnyWorker, 0, done, loom::kInvalidCounter });
	}
	loom::WaitFor(done);
	r.exactlyOnce = (runs.load() == N);
	r.drained = (loom::Load(done) == 0);
	loom::DestroyCounter(done);
	pool.Shutdown();
	r.quiesced = !loom::IsInitialised();
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
				loom::TaskDesc{ nullptr, nullptr, loom::Affinity::AnyWorker, 0, done, loom::kInvalidCounter });
		else
			loom::Submit(loom::TaskDesc{ [](void* p) { (*(std::atomic<int>*)p)++; },
				&runs, loom::Affinity::AnyWorker, 0, done, loom::kInvalidCounter });
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
			&ran, loom::Affinity::AnyWorker, 0, leaf, waitOn });
	}
	// Release the initial counts so the whole DAG unblocks.
	for (CounterHandle c : counters)
		while (loom::Load(c) > 0)
			loom::Signal(c);
	SpinUntil(0, leaf, 10000);
	r.everyTaskRan = (ran.load() == N);
	r.leafZero = (loom::Load(leaf) == 0);
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
			loom::TaskDesc{ nullptr, nullptr, loom::Affinity::MainThread, 0, mainDone, loom::kInvalidCounter });
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
		loom::TaskDesc{ nullptr, nullptr, loom::Affinity::MainThread, 0, gate, loom::kInvalidCounter });

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
		loom::TaskDesc{ nullptr, nullptr, loom::Affinity::AnyWorker, 0, done, loom::kInvalidCounter });
	loom::Submit([&n]() { n += 7; },
		loom::TaskDesc{ nullptr, nullptr, loom::Affinity::AnyWorker, 0, done, loom::kInvalidCounter });
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
			loom::TaskDesc{ nullptr, nullptr, loom::Affinity::AnyWorker, 0, done, loom::kInvalidCounter });
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
				0, done.Get(), loom::kInvalidCounter });
		}
		loom::WaitFor(done);
		CHECK(loom::Load(done) == 0);
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

// The park stops every pool thread (workers,
// dedicated threads, lanes, participants) at its loop top; queued work waits.
TEST_CASE("loom: park — every thread parks with its stack, queued work waits, unpark resumes")
{
	loom::Pool pool(loom::LoomConfig{ .workerCount = 3 });
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
				part.cv.wait_for(lock, std::chrono::milliseconds(200), [&] { return part.stop.load() || loom::ParkRequested(); });
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
			if (loom::ParkRequested())
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
		loom::Submit([&ran]() { ran++; }, loom::TaskDesc{ nullptr, nullptr, loom::Affinity::AnyWorker, 0, done, loom::kInvalidCounter });
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
	loom::Pool pool(loom::LoomConfig{ .workerCount = 4 });
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
	loom::Pool pool(loom::LoomConfig{ .workerCount = 0 }); // only the main thread runs tasks
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
		(void)loom::Load(done); // apply anything held back, so the join cannot hang
	waiter.join();
}

TEST_CASE("loom: deferred signals — a long task of another counter does not delay a finished one")
{
	loom::Pool pool(loom::LoomConfig{ .workerCount = 1 });
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
	loom::Pool pool(loom::LoomConfig{ .workerCount = 1 });
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
			if (loom::Load(p.done) == 1)
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
		CHECK(loom::Load(counters[(size_t)k]) == 0);
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
	loom::Pool pool(loom::LoomConfig{ .maxTasks = 1 }); // one chunk, so the bursts below grow it
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
	loom::Pool pool(loom::LoomConfig{ .workerCount = 1 });
	const char* longName = "a-dedicated-thread-name-26";
	loom::DedicatedId ded = loom::RegisterDedicatedThread(loom::DedicatedConfig{ .name = longName });
	char osName[64] = {};
	loom::ScopedCounter done(1);
	loom::Submit(loom::TaskDesc{ .fn = [](void* out) { pthread_getname_np(pthread_self(), static_cast<char*>(out), 64); },
		.userData = osName, .affinity = loom::Affinity::Dedicated, .thread = ded, .signals = done });
	loom::WaitFor(done);
	CHECK(std::string(osName) == std::string(longName).substr(0, 15));

	bool statsHaveFullName = false;
	for (const auto& t : loom::SnapshotThreadStats().threads)
		statsHaveFullName |= t.name == longName;
	CHECK_MESSAGE(statsHaveFullName, "the pool's thread stats keep the full name");
	loom::UnregisterDedicatedThread(ded);
}
#endif

TEST_CASE("loom: thread stats — helping is counted and shown, rows outlive their threads")
{
	loom::Pool pool(loom::LoomConfig{ .workerCount = 1 });
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
				for (const auto& t : loom::SnapshotThreadStats().threads)
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
	for (const auto& t : loom::SnapshotThreadStats().threads)
	{
		if (t.name.rfind("loom-worker", 0) != 0)
			continue;
		found = true;
		CHECK(t.helpedRun == 8);
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
	for (const auto& t : loom::SnapshotThreadStats().threads)
		rowKept |= t.name == "stats-ded";
	CHECK(rowKept);
}

TEST_CASE("loom: thread stats — two dedicated threads with one name keep their own queue depths")
{
	loom::Pool pool(loom::LoomConfig{ .workerCount = 1 });
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
		for (const auto& t : loom::SnapshotThreadStats().threads)
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
	loom::Pool pool(loom::LoomConfig{ .workerCount = 2 });
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
	loom::Pool pool(loom::LoomConfig{ .workerCount = 0 }); // only a helping wait runs AnyWorker work
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
	loom::Pool pool(loom::LoomConfig{ .workerCount = 2 });
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
		loom::Pool pool(loom::LoomConfig{ .workerCount = 2 });
		// A task that waits for its child, blocking (no helping), while
		// Shutdown runs: the wait must end at zero, not when stopping starts.
		loom::Submit([&st]() {
			loom::CounterHandle child = loom::CreateCounter(1);
			loom::Submit([&st]() {
				std::this_thread::sleep_for(std::chrono::milliseconds(50));
				st.childRan = true;
			}, loom::TaskDesc{ .signals = child });
			loom::WaitFor(child, loom::Help::BlockOnly);
			st.countAfterWait = loom::Load(child);
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
	loom::Pool pool(loom::LoomConfig{ .workerCount = 2 });
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
	CHECK(loom::Load(c) == 0);
	loom::DestroyCounter(c);

	// A re-used counter starts clean.
	loom::CounterHandle fresh = loom::CreateCounter(1);
	loom::Submit(loom::TaskDesc{ .fn = &Noop, .signals = fresh });
	CHECK_NOTHROW(loom::WaitFor(fresh));
	loom::DestroyCounter(fresh);
}

TEST_CASE("loom: exceptions — a helper keeps other counters' exceptions out of its own wait")
{
	loom::Pool pool(loom::LoomConfig{ .workerCount = 0 }); // the main thread runs everything
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
			while (loom::Load(gates[(size_t)i]) > 0)
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
	loom::Pool pool(loom::LoomConfig{ .workerCount = 1 });
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
	loom::Pool pool(loom::LoomConfig{ .workerCount = 1 });
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
	loom::Pool pool(loom::LoomConfig{ .workerCount = 1 });
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
	loom::Pool pool(loom::LoomConfig{ .workerCount = 1 });
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
	CHECK(loom::Load(c) == 0);
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
	loom::Pool pool(loom::LoomConfig{ .workerCount = 1 });
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
	loom::Pool pool(loom::LoomConfig{ .workerCount = 1 });
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
		~Tracked() { live--; }
	};
}

TEST_CASE("loom: Submit(F) — non-trivial, large and over-aligned callables run once and are destroyed")
{
	loom::Pool pool(loom::LoomConfig{ .workerCount = 2 });
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
	loom::Pool pool(loom::LoomConfig{ .workerCount = 1 });
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
	loom::Pool pool(loom::LoomConfig{ .workerCount = 1 });
	OrderLog hold; // keeps the one worker busy until the checks are done
	loom::ScopedCounter gate(1);
	loom::Submit(loom::TaskDesc{ .fn = &HoldUntilOpen, .userData = &hold, .signals = gate });
	std::this_thread::sleep_for(std::chrono::milliseconds(20));

	struct Foreign { std::thread::id mainId; std::atomic<bool> ranOnMain{ false }, ran{ false }; } foreign;
	foreign.mainId = std::this_thread::get_id();
	// A long task from outside the region, parked until `release`: signalling
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
	loom::Pool pool(loom::LoomConfig{ .workerCount = 1 });
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
	loom::Pool pool(loom::LoomConfig{ .workerCount = 0 });
	loom::ScopedCounter outside(1);
	loom::Submit(loom::TaskDesc{ .fn = &Noop, .signals = outside }); // not in the region
	loom::Isolated([&] { loom::WaitFor(outside); }); // nobody else could run it: no isolation
	CHECK(loom::Load(outside) == 0);
}

TEST_CASE("loom: ParallelFor — automatic chunking covers every index once, any size, any pool")
{
	for (int workers : { 0, 1, 3, -1 })
	{
		loom::Pool pool(loom::LoomConfig{ .workerCount = workers });
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
	loom::Pool pool(loom::LoomConfig{ .workerCount = 3 });
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
		loom::Pool pool(loom::LoomConfig{ .workerCount = workers });
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
		loom::Pool pool(loom::LoomConfig{ .workerCount = workers });
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
			CHECK(!graph.Running());
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
	loom::Pool pool(loom::LoomConfig{ .workerCount = 3 });
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
	loom::Pool pool(loom::LoomConfig{ .workerCount = 2 });
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
	loom::Pool pool(loom::LoomConfig{ .workerCount = 2 });
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
	CHECK(graph.Running());
	loom::Signal(gate);
	loom::WaitFor(done);
	CHECK(ran.load() == 12);
	CHECK(!graph.Running());

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
	loom::Pool pool(loom::LoomConfig{ .workerCount = 2 });
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
	CHECK(!graph.Running());
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
	CHECK(!slow.Running());
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

		loom::Pool pool(loom::LoomConfig{ .workerCount = 1 });
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
			loom::Pool pool(loom::LoomConfig{ .workerCount = workers });
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
			CHECK(loom::LimiterHeld(limiter) == 0);
			CHECK(loom::LimiterWaiting(limiter) == 0);
		}
	}
}

TEST_CASE("loom: limiters — a limit of one is a serial queue, High first, then in order")
{
	loom::Pool pool(loom::LoomConfig{ .workerCount = 4 });
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
	CHECK(loom::LimiterHeld(serial) == 1);
	CHECK(loom::LimiterWaiting(serial) == kTasks + 1);
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
	loom::Pool pool(loom::LoomConfig{ .workerCount = 2 });
	loom::ScopedLimiter limiter(1);
	loom::ScopedCounter gate(1), other(1), gatedDone(1), freeDone(1);
	std::atomic<int> gatedRan{ 0 };
	const loom::CounterHandle all[] = { other };
	loom::Submit([&] { gatedRan++; }, loom::TaskDesc{ .signals = gatedDone, .waitsFor = gate, .waitsForAll = all, .limiter = limiter });
	CHECK(loom::LimiterHeld(limiter) == 0);
	// The gated task holds no unit, so this one runs.
	loom::Submit([] {}, loom::TaskDesc{ .signals = freeDone, .limiter = limiter });
	CHECK(loom::TryWaitFor(freeDone, 5000));
	CHECK(gatedRan.load() == 0);
	loom::Signal(gate);
	loom::Signal(other);
	loom::WaitFor(gatedDone);
	CHECK(gatedRan.load() == 1);
	CHECK(loom::LimiterHeld(limiter) == 0);
}

TEST_CASE("loom: limiters — many submitters at once; tasks of every affinity and graph nodes")
{
	loom::Pool pool(loom::LoomConfig{ .workerCount = 4 });
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
	CHECK(loom::LimiterHeld(limiter) == 0);

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
	CHECK(loom::LimiterHeld(limiter) == 0);
	loom::UnregisterDedicatedThread(io);
}

TEST_CASE("loom: limiters — a cancelled or throwing task gives its unit back")
{
	loom::Pool pool(loom::LoomConfig{ .workerCount = 2 });
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
	CHECK(loom::LimiterHeld(limiter) == 0);
	CHECK(loom::LimiterWaiting(limiter) == 0);

#if defined(__cpp_exceptions)
	loom::ScopedCounter failing(3);
	loom::Submit([] { throw std::runtime_error("limited"); }, loom::TaskDesc{ .signals = failing, .limiter = limiter });
	loom::Submit([] {}, loom::TaskDesc{ .signals = failing, .limiter = limiter });
	loom::Submit([] {}, loom::TaskDesc{ .signals = failing, .limiter = limiter });
	CHECK_THROWS_AS(loom::WaitFor(failing), std::runtime_error);
	CHECK(loom::LimiterHeld(limiter) == 0);
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
	loom::Pool pool(loom::LoomConfig{ .workerCount = 1, .maxLimiters = 3 });
	for (int round = 0; round < 100; round++)
	{
		loom::ScopedLimiter a(1), b(2), c(5);
		CHECK(a.Get() != b.Get());
		CHECK(b.Get() != c.Get());
		CHECK(loom::LimiterHeld(c) == 0);
		loom::ScopedCounter done(3);
		for (loom::LimiterHandle h : { a.Get(), b.Get(), c.Get() })
			loom::Submit([] {}, loom::TaskDesc{ .signals = done, .limiter = h });
		loom::WaitFor(done);
	}
}

namespace {
	int LiveCounters() { return loom::SnapshotThreadStats().counterCount; }
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
	// The final signal releases many parked tasks one by one; the first ones
	// can start on other workers while it is still releasing the rest.
	loom::Pool pool(loom::LoomConfig{ .workerCount = 4 });
	std::atomic<int> notZero{ 0 };
	for (int round = 0; round < 200; round++)
	{
		loom::ScopedCounter gate(1);
		constexpr int kParked = 64;
		loom::ScopedCounter done(kParked);
		for (int i = 0; i < kParked; i++)
			loom::Submit([&notZero, g = gate.Get()] {
				if (loom::Load(g) != 0)
					notZero++;
			}, loom::TaskDesc{ .signals = done, .waitsFor = gate });
		loom::Signal(gate);
		loom::WaitFor(done);
	}
	CHECK(notZero.load() == 0);
}

TEST_CASE("loom: futures — values, void, move-only results and repeated Get, any pool")
{
	for (int workers : { 0, 1, -1 })
	{
		loom::Pool pool(loom::LoomConfig{ .workerCount = workers });
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
			CHECK(squares[5].Ready());
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
			CHECK(!squares[3].Valid());
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
	loom::Pool pool(loom::LoomConfig{ .workerCount = 2 });
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
	CHECK(!g.Valid());
	while (loom::LiveTaskCount() > 0)
		std::this_thread::yield();
	CHECK(LiveCounters() == baseline);
}

TEST_CASE("loom: futures — a task can wait for futures and read them without blocking")
{
	loom::Pool pool(loom::LoomConfig{ .workerCount = 3 });
	const loom::DedicatedId io = loom::RegisterDedicatedThread(loom::DedicatedConfig{ .name = "future-io" });
	loom::ScopedCounter gate(1);
	loom::Future<int> a = loom::Async([] { return 20; }, loom::TaskDesc{ .waitsFor = gate });
	loom::Future<int> b = loom::Async([] { return 22; }, loom::TaskDesc{ .affinity = loom::Affinity::Dedicated, .thread = io });
	const loom::CounterHandle both[] = { a.Counter(), b.Counter() };
	// A future of futures: runs once both are ready, reads them, never waits.
	std::atomic<bool> bothReady{ false };
	loom::Future<int> sum = loom::Async([&] {
		bothReady = a.Ready() && b.Ready();
		return a.Get() + b.Get();
	}, loom::TaskDesc{ .waitsForAll = both });
	// A plain task can wait for one too.
	std::atomic<int> seen{ 0 };
	loom::ScopedCounter plainDone(1);
	loom::Submit([&] { seen = a.Get(); }, loom::TaskDesc{ .signals = plainDone, .waitsFor = a.Counter() });

	CHECK(!sum.TryWait(20));
	CHECK(!a.Ready());
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
	CHECK(loom::LimiterHeld(one) == 0);
	loom::UnregisterDedicatedThread(io);
}

TEST_CASE("loom: futures — once Get returns, the task holds no limiter unit")
{
	loom::Pool pool(loom::LoomConfig{ .workerCount = 4 });
	int held = 0;
	for (int i = 0; i < 500; i++)
	{
		loom::ScopedLimiter limiter(1 + i % 2); // destroyed right after Get: it must be idle
		loom::Future<int> f = loom::Async([i] { return i; }, loom::TaskDesc{ .limiter = limiter });
		loom::Future<void> g = loom::Async([] {}, loom::TaskDesc{ .limiter = limiter });
		CHECK(f.Get() == i);
		g.Get();
		held += (int)loom::LimiterHeld(limiter);
	}
	CHECK(held == 0);
}

#if defined(__cpp_exceptions)
TEST_CASE("loom: futures — Get rethrows the callable's exception, every time")
{
	loom::Pool pool(loom::LoomConfig{ .workerCount = 2 });
	const int baseline = LiveCounters();
	Tracked::live = 0;
	{
		Tracked tracked;
		loom::Future<int> bad = loom::Async([tracked]() -> int { (void)tracked; throw std::runtime_error("parse failed"); });
		loom::Future<void> badVoid = loom::Async([] { throw std::logic_error("no"); });
		loom::Future<int> good = loom::Async([] { return 1; });
		CHECK_THROWS_AS(bad.Get(), std::runtime_error);
		CHECK_THROWS_AS(bad.Get(), std::runtime_error);
		CHECK(bad.Ready());
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
	loom::Pool pool(loom::LoomConfig{ .workerCount = 2 });
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
		CHECK(late.Ready());
		CHECK(late.Get() == 9);
		loom::Promise<void> signal;
		loom::Future<void> fired = signal.GetFuture();
		CHECK(!fired.Ready());
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
	loom::Pool pool(loom::LoomConfig{ .workerCount = 2 });
	const int baseline = LiveCounters();
	{
		loom::Future<int> orphan;
		{
			loom::Promise<int> p;
			orphan = p.GetFuture();
		}
		CHECK(orphan.Ready());
		CHECK_THROWS_AS(orphan.Get(), loom::BrokenPromise);

		loom::Promise<int> moved;
		loom::Future<int> f = moved.GetFuture();
		loom::Promise<int> owner(std::move(moved)); // the moved-from one breaks nothing
		CHECK(!f.Ready());
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
	loom::Pool pool(loom::LoomConfig{ .workerCount = 2 });
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
	CHECK_MESSAGE(loom::LimiterHeld(one) == 1, "the setting task is still running and still holds its unit");
	release = true;
	loom::WaitFor(done);
	CHECK(loom::LimiterHeld(one) == 0);
}

TEST_CASE("loom: continuations — chains run in order and outlive the futures they read")
{
	for (int workers : { 0, 1, -1 })
	{
		loom::Pool pool(loom::LoomConfig{ .workerCount = workers });
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
			CHECK(fromVoid.Ready());
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
	loom::Pool pool(loom::LoomConfig{ .workerCount = 3 });
	loom::ScopedLimiter one(1);
	for (int i = 0; i < 200; i++)
	{
		loom::Future<int> f = loom::Async([i] { return i; }).Then([](int& v) { return v * 2; }, loom::TaskDesc{ .limiter = one });
		CHECK(f.Get() == 2 * i);
		CHECK(loom::LimiterHeld(one) == 0);
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
	loom::Pool pool(loom::LoomConfig{ .workerCount = 2 });
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

} // namespace

}} // namespace loom::tests
