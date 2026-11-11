#pragma once

// The loom -> profiler state shim.
//
// The dependency direction forbids loom including the engine's profiler
// (loom deps nobody), so the state marks flow through these hooks: loom
// marks its waits/helping and each task's execution, and an app layer that
// depends on BOTH (the engine's LoomPool) registers the hooks that forward
// into the profiler. Null hook = the marks are free no-ops.
//
// The hooks are members of the ACTIVE
// pool's state (the old function-local statics here), so a hook is
// installed after the pool is created and dies with it; installing a
// non-null hook with no active pool asserts. They stay atomic: the owner
// unhooks while the workers are still alive.

#include <cstdint>

namespace loom {

	// 0 = the thread entered a wait, 1 = the thread is helping (running
	// stolen work while waiting), 2 = the wait/helping ended.
	using ProfilerStateFn = void (*)(int threadIndex, uint8_t state);

	void SetProfilerStateHook(ProfilerStateFn fn);

	// Per-task profiler zones: loom
	// marks a "Task" zone around each task's execution. The app hook
	// pairs the profiler's ZoneBegin/ZoneEnd: the BEGIN call returns the
	// handle the matching END must pass back. Null hook = free no-op.
	// The task's interned name id
	// (0 = unnamed - the hook falls back to the generic name) and the
	// window tag ride alongside, so named, multi-window tasks stay
	// distinguishable in the timeline.
	using ProfilerZoneFn = uint64_t (*)(int threadIndex, uint32_t nameId, uint8_t window, const char* name, bool begin, uint64_t handle);

	void SetProfilerZoneHook(ProfilerZoneFn fn);

} // namespace loom
