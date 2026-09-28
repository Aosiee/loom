#pragma once

// The loom -> profiler state shim.
//
// The dependency direction forbids loom including the engine's profiler
// (loom deps nobody), so the state marks flow through this self-contained
// header: loom calls MarkProfilerState around its waits/helping, and an
// app layer that depends on BOTH (the editor) registers the hook that
// forwards into the profiler. Null hook = the marks are free no-ops.

#include <cstdint>

namespace loom {

	// 0 = the thread entered a wait, 1 = the thread is helping (running
	// stolen work while waiting), 2 = the wait/helping ended.
	using ProfilerStateFn = void (*)(int threadIndex, uint8_t state);

	inline ProfilerStateFn& GetProfilerStateHook()
	{
		static ProfilerStateFn hook = nullptr;
		return hook;
	}

	inline void SetProfilerStateHook(ProfilerStateFn fn) { GetProfilerStateHook() = fn; }

	inline void MarkProfilerState(int threadIndex, uint8_t state)
	{
		if (ProfilerStateFn fn = GetProfilerStateHook())
			fn(threadIndex, state);
	}

} // namespace loom
