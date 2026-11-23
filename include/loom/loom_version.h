#pragma once

// loom's version, for code that must build against more than one release:
//
//   #if LOOM_VERSION >= LOOM_MAKE_VERSION(0, 20, 0)
//
// loom.h includes this. CMake reads the version from here too, so it is
// set in one place. Before 1.0 a minor release may change the API (see
// CHANGELOG.md), and find_package accepts only the same minor version.
#define LOOM_VERSION_MAJOR 0
#define LOOM_VERSION_MINOR 20
#define LOOM_VERSION_PATCH 0
#define LOOM_VERSION_STRING "0.20.0"

#define LOOM_MAKE_VERSION(major, minor, patch) ((major) * 1000000 + (minor) * 1000 + (patch))
#define LOOM_VERSION LOOM_MAKE_VERSION(LOOM_VERSION_MAJOR, LOOM_VERSION_MINOR, LOOM_VERSION_PATCH)
