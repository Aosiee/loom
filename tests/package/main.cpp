// Built against an installed loom or a source tree (see CMakeLists.txt):
// checks that the target brings the headers and the library, and that the
// version macros agree with each other.
#include <loom/loom.h>

#include <cstdio>
#include <string>

static_assert(LOOM_VERSION == LOOM_MAKE_VERSION(LOOM_VERSION_MAJOR, LOOM_VERSION_MINOR, LOOM_VERSION_PATCH));

int main()
{
	const std::string expected = std::to_string(LOOM_VERSION_MAJOR) + "." + std::to_string(LOOM_VERSION_MINOR) + "." +
		std::to_string(LOOM_VERSION_PATCH);
	if (expected != LOOM_VERSION_STRING)
	{
		std::fprintf(stderr, "LOOM_VERSION_STRING is %s, the numbers say %s\n", LOOM_VERSION_STRING, expected.c_str());
		return 1;
	}
	loom::Pool pool;
	loom::Future<int> answer = loom::Async([] { return 6; }).Then([](int x) { return x * 7; });
	if (answer.Get() != 42)
		return 1;
	std::printf("loom %s: package check passed\n", LOOM_VERSION_STRING);
	return 0;
}
