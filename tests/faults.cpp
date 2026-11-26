// loom_faults: allocation failure at every point of loom's public calls.
//
// This program replaces the global operator new, so that the N-th allocation
// throws std::bad_alloc: the N-th made on the thread that calls loom, or, in
// the second sweep, the N-th made on any thread (the pool's own threads
// included). For each scenario it runs a child process for N = 1, 2, ...
// until a run no longer reaches the fault, and each run must end one of two
// ways:
//   - the call threw bad_alloc, and the pool is still whole: more work runs,
//     no task is left live, and Shutdown is clean;
//   - loom ended the program, saying it ran out of memory where it could not
//     back out (docs/rules.md, "Out of memory").
// A crash, a hang, a lost element or any other abort fails the scenario.
// Its own executable: the replacement operator new would change every test
// of the main suite.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "loom/loom.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <new>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include <execinfo.h>
#include <fcntl.h>
#include <sys/wait.h>
#include <unistd.h>

// ---- The failing allocator ------------------------------------------------------------

namespace {
// Allocations left before one fails (0: none fails): on this thread, or on
// any thread when g_AnyThread.
thread_local int t_FailIn = 0;
std::atomic<int> g_FailIn{ 0 };
std::atomic<bool> g_Failed{ false };
bool g_AnyThread = false;

void* Allocate(std::size_t size, std::size_t align)
{
	if (t_FailIn > 0 && --t_FailIn == 0)
	{
		g_Failed = true;
		throw std::bad_alloc();
	}
	if (g_FailIn.load(std::memory_order_relaxed) > 0 && g_FailIn.fetch_sub(1) == 1)
	{
		g_Failed = true;
		throw std::bad_alloc();
	}
	if (size == 0)
		size = 1;
	void* p = align > alignof(std::max_align_t) ? std::aligned_alloc(align, (size + align - 1) / align * align) : std::malloc(size);
	if (!p)
		throw std::bad_alloc();
	return p;
}

void Arm(int n)
{
	g_Failed = false;
	if (g_AnyThread)
		g_FailIn = n;
	else
		t_FailIn = n;
}

// Whether the fault fired, and no more after it.
bool Disarm()
{
	t_FailIn = 0;
	g_FailIn = 0;
	return g_Failed;
}
} // namespace

void* operator new(std::size_t size) { return Allocate(size, 0); }
void* operator new[](std::size_t size) { return Allocate(size, 0); }
void* operator new(std::size_t size, std::align_val_t a) { return Allocate(size, (std::size_t)a); }
void* operator new[](std::size_t size, std::align_val_t a) { return Allocate(size, (std::size_t)a); }
void* operator new(std::size_t size, const std::nothrow_t&) noexcept
{
	try { return Allocate(size, 0); } catch (...) { return nullptr; }
}
void* operator new[](std::size_t size, const std::nothrow_t&) noexcept
{
	try { return Allocate(size, 0); } catch (...) { return nullptr; }
}
void operator delete(void* p) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t) noexcept { std::free(p); }
void operator delete(void* p, std::align_val_t) noexcept { std::free(p); }
void operator delete[](void* p, std::align_val_t) noexcept { std::free(p); }
void operator delete(void* p, std::size_t, std::align_val_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t, std::align_val_t) noexcept { std::free(p); }

// The thread sanitizer sleeps a second as a process exits, to catch races
// with threads still running then. Every child here exits, and most of
// them cleanly: without this, each costs that second.
#if defined(__SANITIZE_THREAD__)
extern "C" const char* __tsan_default_options() { return "atexit_sleep_ms=0"; }
#endif

// ---- Running a scenario in a child ----------------------------------------------------

namespace {

enum class End { Clean, NotReached, OutOfMemory, Other };

// What the child's exit codes mean.
constexpr int kNotReached = 10;
constexpr int kBroken = 11;

// Checks that a pool that just threw from a call is still whole, then ends
// the child: kNotReached if the fault never fired.
[[noreturn]] void Finish(bool failed)
{
	if (!failed)
		_exit(kNotReached);
	loom::ScopedCounter more(64);
	for (int i = 0; i < 64; i++)
		loom::Submit([] {}, { .signals = more });
	if (!loom::TryWaitFor(more, 5000))
		_exit(kBroken);
	_exit(0);
}

End RunChild(void (*scenario)(int), int n, bool anyThread, std::string& err)
{
	int fds[2];
	if (pipe(fds) != 0)
		return End::Other;
	const pid_t pid = fork();
	if (pid == 0)
	{
		std::signal(SIGABRT, SIG_DFL);
		dup2(fds[1], 2);
		close(fds[0]);
		// An exception that escapes: where from, for the report.
		std::set_terminate([] {
			void* frames[48];
			backtrace_symbols_fd(frames, backtrace(frames, 48), 2);
			std::abort();
		});
		g_AnyThread = anyThread;
		scenario(n);
		_exit(kBroken); // a scenario ends in Finish
	}
	close(fds[1]);
	fcntl(fds[0], F_SETFL, O_NONBLOCK);
	const auto until = std::chrono::steady_clock::now() + std::chrono::seconds(20);
	int status = 0;
	bool ended = false;
	while (!ended)
	{
		char buf[512];
		ssize_t got;
		while ((got = read(fds[0], buf, sizeof(buf))) > 0)
			err.append(buf, (size_t)got);
		if (waitpid(pid, &status, WNOHANG) == pid)
			ended = true;
		else if (std::chrono::steady_clock::now() > until)
		{
			kill(pid, SIGKILL);
			waitpid(pid, &status, 0);
			close(fds[0]);
			err += "(hung)";
			return End::Other;
		}
		else
			std::this_thread::sleep_for(std::chrono::milliseconds(1));
	}
	char buf[512];
	ssize_t got;
	while ((got = read(fds[0], buf, sizeof(buf))) > 0)
		err.append(buf, (size_t)got);
	close(fds[0]);
	if (WIFEXITED(status) && WEXITSTATUS(status) == 0)
		return End::Clean;
	if (WIFEXITED(status) && WEXITSTATUS(status) == kNotReached)
		return End::NotReached;
	if (WIFSIGNALED(status) && WTERMSIG(status) == SIGABRT && err.find("loom: out of memory") != std::string::npos)
		return End::OutOfMemory;
	return End::Other;
}

// Every N until the fault is no longer reached, on the calling thread and
// then on any thread; each run must end well. Limited to `most` runs each.
void Sweep(void (*scenario)(int), int most = 400)
{
	REQUIRE(!loom::HasActivePool());
	for (const bool anyThread : { false, true })
	{
		int clean = 0, fatal = 0;
		for (int n = 1; n <= most; n++)
		{
			std::string err;
			const End end = RunChild(scenario, n, anyThread, err);
			if (end == End::NotReached)
			{
				MESSAGE((std::string(anyThread ? "any thread: " : "calling thread: ") + std::to_string(n - 1) + " faults, "
					+ std::to_string(clean) + " threw cleanly, " + std::to_string(fatal) + " ended the program"));
				break;
			}
			INFO((std::string(anyThread ? "any thread, " : "calling thread, ") + "fault at allocation " + std::to_string(n) + ": " + err));
			REQUIRE((end == End::Clean || end == End::OutOfMemory));
			(end == End::Clean ? clean : fatal)++;
		}
	}
}

loom::PoolConfig Config()
{
	return loom::PoolConfig{ .workerCount = 2, .shutdownTimeoutMs = 300 };
}

} // namespace

// ---- Scenarios ------------------------------------------------------------------------

TEST_CASE("faults: a Pool whose threads cannot all start")
{
	Sweep([](int n) {
		Arm(n);
		bool threw = false;
		std::optional<loom::Pool> pool;
		try
		{
			pool.emplace(loom::PoolConfig{ .workerCount = 4, .shutdownTimeoutMs = 300 });
		}
		catch (const std::bad_alloc&) { threw = true; }
		const bool failed = Disarm() || threw;
		// A constructor that threw installed nothing, and stopped what it started.
		if (threw)
		{
			if (loom::HasActivePool())
				_exit(kBroken);
			pool.emplace(Config());
		}
		Finish(failed);
	});
}

TEST_CASE("faults: ParallelFor with more chunks than a deque holds")
{
	Sweep([](int n) {
		loom::Pool pool(Config());
		std::vector<int> hits(4000, 0);
		Arm(n);
		bool failed = false;
		try
		{
			loom::ParallelFor(0, hits.size(), 1, [&](size_t b, size_t e) {
				for (size_t i = b; i < e; i++)
					hits[i]++;
			});
		}
		catch (const std::bad_alloc&) { failed = true; }
		failed = Disarm() || failed;
		if (!failed)
			for (int h : hits)
				if (h != 1)
					_exit(kBroken);
		Finish(failed);
	});
}

TEST_CASE("faults: ParallelSort of strings keeps every element")
{
	Sweep([](int n) {
		loom::Pool pool(Config());
		std::vector<std::string> v(200000);
		for (size_t i = 0; i < v.size(); i++)
			v[i] = "element-" + std::to_string((i * 7919) % v.size()) + "-with-a-long-tail";
		std::vector<std::string> before = v;
		std::sort(before.begin(), before.end());
		Arm(n);
		bool threw = false;
		try { loom::ParallelSort(v.begin(), v.end()); }
		catch (const std::bad_alloc&) { threw = true; }
		const bool failed = Disarm() || threw;
		std::sort(v.begin(), v.end());
		if (v != before)
			_exit(kBroken); // an element lost or duplicated
		Finish(failed);
	});
}

TEST_CASE("faults: futures, continuations and a large callable")
{
	Sweep([](int n) {
		loom::Pool pool(Config());
		Arm(n);
		bool threw = false;
		try
		{
			loom::Future<int> a = loom::Async([] { return 1; });
			loom::Future<int> b = a.Then([](int& v) { return v + 1; });
			std::array<char, 200> big{};
			big[0] = 3;
			loom::ScopedCounter done(1);
			try
			{
				loom::Submit([big] { (void)big; }, { .signals = done }); // a heap callable
			}
			catch (...)
			{
				loom::Signal(done); // nothing was queued: the count is ours to bring down
				throw;
			}
			loom::WaitFor(done);
			if (b.Get() != 2)
				_exit(kBroken);
		}
		catch (const std::bad_alloc&) { threw = true; }
		Finish(Disarm() || threw);
	});
}

TEST_CASE("faults: a counter's last signal releasing many waiting tasks")
{
	Sweep([](int n) {
		loom::Pool pool(Config());
		std::atomic<int> ran{ 0 };
		{
			loom::ScopedCounter gate(1);
			loom::ScopedCounter done(600);
			for (int i = 0; i < 600; i++)
				loom::Submit([&] { ran++; }, { .affinity = i % 2 ? loom::Affinity::MainThread : loom::Affinity::AnyWorker,
					.signals = done, .waitsFor = gate });
			Arm(n);
			loom::Signal(gate); // releases all 600 onto the queues
			const bool failed = Disarm();
			loom::WaitFor(done);
			if (ran != 600)
				_exit(kBroken);
			if (!failed)
				_exit(kNotReached);
		}
		Finish(true);
	});
}

TEST_CASE("faults: a task graph run")
{
	Sweep([](int n) {
		loom::Pool pool(Config());
		loom::TaskGraph g;
		std::atomic<int> ran{ 0 };
		const loom::TaskGraph::Node root = g.Add([&] { ran++; });
		for (int i = 0; i < 300; i++)
			g.Precede(root, g.Add([&] { ran++; }));
		Arm(n);
		bool threw = false;
		try { loom::Run(g); }
		catch (const std::bad_alloc&) { threw = true; }
		const bool failed = Disarm() || threw;
		if (!failed && ran != 301)
			_exit(kBroken);
		if (g.IsRunning())
			_exit(kBroken);
		Finish(failed);
	});
}

TEST_CASE("faults: SetLimit, a dedicated thread, ParkAll and the debug dump")
{
	Sweep([](int n) {
		loom::Pool pool(Config());
		// Indices 0 and 1 held while the limit drops to 1: index 1 is not
		// kept when it comes back, so a raise must make room for it again.
		loom::LimiterHandle l = loom::CreateLimiter(2);
		{
			std::atomic<int> have{ 0 };
			std::atomic<bool> lowered{ false };
			loom::ScopedCounter done(2);
			for (int i = 0; i < 2; i++)
				loom::Submit([&] {
					(void)loom::CurrentLimiterUnit();
					have++;
					while (!lowered)
						std::this_thread::yield();
				}, { .signals = done, .limiter = l });
			while (have.load() < 2)
				std::this_thread::yield();
			loom::SetLimit(l, 1);
			lowered = true;
			loom::WaitFor(done);
		}
		Arm(n);
		bool threw = false;
		// Past every limit used before, the raise needs room for the new
		// units' indices: if it cannot make it, the limit stays as it was.
		bool raised = true;
		try
		{
			loom::SetLimit(l, 64);
		}
		catch (const std::bad_alloc&) { threw = true; raised = false; }
		if (loom::Limit(l) != (raised ? 64u : 1u))
			_exit(kBroken);
		if (raised)
		{
			try
			{
				loom::SetLimit(l, 4);
				const loom::DedicatedId d = loom::RegisterDedicatedThread({ .name = "io" });
				loom::UnregisterDedicatedThread(d);
				if (loom::ParkAll(2000))
					loom::UnparkAll();
				(void)loom::DebugDump();
			}
			catch (const std::bad_alloc&) { threw = true; }
		}
		const bool failed = Disarm() || threw;
		// The limit is whatever the call left, and it holds: 4 at once if
		// the raise went through, 1 if it threw.
		std::atomic<int> running{ 0 }, peak{ 0 };
		{
			loom::ScopedCounter done(32);
			for (int i = 0; i < 32; i++)
				loom::Submit([&] {
					const int now = ++running;
					int p = peak.load();
					while (now > p && !peak.compare_exchange_weak(p, now)) {}
					std::this_thread::sleep_for(std::chrono::microseconds(100));
					running--;
				}, { .signals = done, .limiter = l });
			loom::WaitFor(done);
		}
		if (peak.load() > (int)loom::Limit(l) || loom::IsParkRequested())
			_exit(kBroken);
		loom::DestroyLimiter(l);
		Finish(failed);
	});
}
