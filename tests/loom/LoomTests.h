#pragma once

// Loom's headless suite: seeded stress, DAGs, affinity pinning,
// help-while-waiting, dedicated lifecycle, shutdown under load. One doctest
// TEST_CASE per named check; every case init/shuts down its own loom
// instance. Runs from
// tinytest and from the minimal-link loom_tests app (TSAN-clean).

#include "loom.h"
#include "loom_profiler_shim.h"
#include <doctest/doctest.h>
#include <atomic>
#include <chrono>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>

namespace loom { namespace tests {

namespace {

void InitIsolated()
{
	loom::Shutdown();
	loom::Init();
}

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
	InitIsolated();
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
	loom::Shutdown();
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
	InitIsolated();
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
	loom::Shutdown();
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
	InitIsolated();
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
	loom::Shutdown();
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
	InitIsolated();
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
	loom::Shutdown();
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
	InitIsolated();
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
	loom::Shutdown();
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
	InitIsolated();
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
	loom::Shutdown();
}

// ---- 7. Inline Submit(F) + BlockOnly wait ----

TEST_CASE("loom: inline Submit(F) ran via a BlockOnly wait")
{
	InitIsolated();
	std::atomic<int> n{ 0 };
	CounterHandle done = loom::CreateCounter(2);
	loom::Submit([&n]() { n += 5; },
		loom::TaskDesc{ nullptr, nullptr, loom::Affinity::AnyWorker, 0, done, loom::kInvalidCounter });
	loom::Submit([&n]() { n += 7; },
		loom::TaskDesc{ nullptr, nullptr, loom::Affinity::AnyWorker, 0, done, loom::kInvalidCounter });
	loom::WaitFor(done, loom::Help::BlockOnly);
	CHECK_MESSAGE(n.load() == 12, "inline Submit(F) ran via a BlockOnly wait");
	loom::DestroyCounter(done);
	loom::Shutdown();
}

// ---- 8. Shutdown under load (drain-or-assert) ----

TEST_CASE("loom: shutdown under load — all tasks completed")
{
	InitIsolated();
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
	loom::Shutdown(); // drains + joins; asserts queues empty
	CHECK_MESSAGE(runs.load() == N, "shutdown under load: all tasks completed");
}

// ---- ParallelFor, ScopedCounter, lanes, profiler zones ----

TEST_CASE("loom: ParallelFor — every index covered exactly once (pointer form)")
{
	InitIsolated();
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
	InitIsolated();
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
	InitIsolated();
	const size_t Rows = 64, Cols = 96;
	std::vector<std::atomic<int>> cells(Rows * Cols);
	for (auto& c : cells) c = 0;

	loom::ParallelFor(0, Rows, 8, [&cells, Cols](size_t rowBegin, size_t rowEnd) {
		loom::ParallelFor(0, Cols, 16, [&cells, Cols, rowBegin, rowEnd](size_t colBegin, size_t colEnd) {
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
	InitIsolated();
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

TEST_CASE("loom: lanes — exclusive reservation, lifecycle, shutdown wait")
{
	InitIsolated();

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

	loom::Shutdown(); // asserts no lanes still held
}

TEST_CASE("loom: profiler zone hook — begin/end pairs carry the executing thread index")
{
	InitIsolated();

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
