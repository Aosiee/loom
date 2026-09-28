#pragma once

// Loom's headless suite: seeded stress, DAGs, affinity pinning,
// help-while-waiting, dedicated lifecycle, shutdown under load. One doctest
// TEST_CASE per named check; every case init/shuts down its own loom
// instance. Runs from
// tinytest and from the minimal-link loom_tests app (TSAN-clean).

#include "loom.h"
#include <doctest/doctest.h>
#include <atomic>
#include <chrono>
#include <random>
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

} // namespace

}} // namespace loom::tests
