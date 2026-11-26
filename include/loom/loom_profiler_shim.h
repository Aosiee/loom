#pragma once

// The loom -> profiler shim.
//
// loom depends on nothing, so it cannot include the engine's profiler.
// Instead it reports through these hooks: it marks its waits and helping and
// each task's execution, and a layer that depends on both (the engine's
// LoomPool) installs hooks that forward to the profiler. With no hook
// installed, the marks cost one null check.
//
// The hooks belong to the active pool: install them after the pool is
// created; they go away with it. Installing a non-null hook with no active
// pool asserts. They are atomic because the owner may unhook while the
// workers are still running.

#include <cstdint>

namespace loom {

	// A thread's state through a wait, as transitions: 0 = waiting (it
	// entered a wait, or a task it ran while waiting has finished), 1 =
	// helping (it is running a task while it waits, from that task's start),
	// 2 = the wait ended. A wait inside a helped task ends with 1: the thread
	// is back in that task.
	using ProfilerStateFn = void (*)(int threadIndex, uint8_t state);

	void SetProfilerStateHook(ProfilerStateFn fn);

	// Per-task zones: loom marks a "Task" zone around each task's execution.
	// The hook maps these onto the profiler's ZoneBegin/ZoneEnd: the begin
	// call returns a handle, and loom passes it back with the matching end.
	// The task's name id (0 = unnamed; the hook falls back to `name`) and its
	// TaskDesc::profilerTag come along, so the host can tell apart named tasks
	// with different tags (the engine's windows) in the timeline. A zone's
	// end goes to the hook its begin went to: one installed or removed while
	// a task runs sees no half pairs.
	using ProfilerZoneFn = uint64_t (*)(int threadIndex, uint32_t nameId, uint8_t profilerTag, const char* name, bool begin, uint64_t handle);

	void SetProfilerZoneHook(ProfilerZoneFn fn);

} // namespace loom
