// Built with -fno-exceptions, as an engine may build loom: checks that the
// header and library compile that way and that the parts with exception
// handling inside them (futures, promises, continuations, coroutines, sort,
// pipelines) still run. Returns non-zero, naming the check, if one fails. CI builds it
// in its embedded job (.github/workflows/linux.yml), which shows the commands.
#include "loom/loom.h"

#include <algorithm>
#include <cstdio>
#include <vector>

static int s_Failed = 0;

#if defined(LOOM_COROUTINES)
static loom::Future<int> Doubled(int v)
{
	co_await loom::ResumeOn();
	const int x = co_await loom::Async([v] { return v; });
	co_return x * 2;
}
#endif

static void Check(bool ok, const char* what)
{
	if (!ok)
	{
		std::fprintf(stderr, "failed: %s\n", what);
		s_Failed++;
	}
}

int main()
{
	loom::Pool pool(loom::PoolConfig{});

	std::vector<int> v(300000);
	for (size_t i = 0; i < v.size(); i++)
		v[i] = (int)((i * 2654435761u) % 100000);
	loom::ParallelSort(v.begin(), v.end());
	Check(std::is_sorted(v.begin(), v.end()), "ParallelSort");

	std::vector<long> sums(v.size());
	loom::ParallelInclusiveScan(v.begin(), v.end(), sums.begin(), 0L);
	loom::PerThread<long> partial;
	loom::ParallelFor(0, v.size(), [&](size_t b, size_t e) {
		for (size_t i = b; i < e; i++)
			partial.Local() += v[i];
	});
	long total = 0;
	partial.ForEach([&](long& x) { total += x; });
	Check(total == sums.back(), "ParallelInclusiveScan and PerThread");

	loom::Future<int> doubled = loom::Async([] { return 21; }).Then([](int x) { return x * 2; });
	Check(doubled.Get() == 42, "Async and Then");

#if defined(LOOM_COROUTINES)
	Check(Doubled(21).Get() == 42, "coroutines");
#endif

	loom::Promise<int> promise;
	loom::Future<int>  promised = promise.GetFuture();
	promise.SetValue(7);
	Check(promised.Get() == 7, "Promise");

	struct Item
	{
		int i = 0;
	};
	int next = 0, seen = 0;
	bool inOrder = true;
	loom::Pipeline<Item> pipeline(4);
	pipeline.Source([&](Item& it) { it.i = next; return next++ < 100; })
		.Parallel([](Item& it) { it.i *= 2; })
		.Serial([&](Item& it) { inOrder &= it.i == seen * 2; seen++; });
	pipeline.Run();
	Check(seen == 100 && inOrder, "Pipeline");

	loom::ScopedLimiter limiter(2);
	loom::SetLimit(limiter, 3);
	Check(loom::Limit(limiter) == 3, "SetLimit");

	if (s_Failed == 0)
		std::printf("no-exceptions build: all checks passed\n");
	return s_Failed == 0 ? 0 : 1;
}
