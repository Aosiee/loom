// Standalone loom suite executable (loom_tests / loom_tests_tsan targets):
// links ONLY loom — no engine, no window, no third-party threads — so the
// TSAN run reports loom's races and nothing else. The doctest main registers
// the suite's TEST_CASEs from Tests/LoomTests.h; tinytest
// runs the same cases in the full-suite binary.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "Tests/LoomTests.h"
