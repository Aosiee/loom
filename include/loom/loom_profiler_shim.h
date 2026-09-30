#pragma once

// The loom -> profiler state shim.
//
// The dependency direction forbids loom including the engine's profiler
// (loom deps nobody), so the state marks flow through this self-contained
// header: loom calls MarkProfilerState around its waits/helping, and an
// app layer that depends on BOTH (the engine's LoomPool) registers the
// hook that forwards into the profiler. Null hook = the marks are free
// no-ops. The hook pointers are atomic: Shutdown unhooks while workers
// are still alive, and a mid-mark teardown must not race.

#include <atomic>
#include <cstdint>

namespace loom {

	// 0 = the thread entered a wait, 1 = the thread is helping (running
	// stolen work while waiting), 2 = the wait/helping ended.
	using ProfilerStateFn = void (*)(int threadIndex, uint8_t state);

	inline std::atomic<ProfilerStateFn>& GetProfilerStateHook()
	{
		static std::atomic<ProfilerStateFn> hook{ nullptr };
		return hook;
	}

	inline void SetProfilerStateHook(ProfilerStateFn fn)
	{
		GetProfilerStateHook().store(fn, std::memory_order_release);
	}

	inline void MarkProfilerState(int threadIndex, uint8_t state)
	{
		if (ProfilerStateFn fn = GetProfilerStateHook().load(std::memory_order_acquire))
			fn(threadIndex, state);
	}

	// Per-task profiler zones: loom
	// marks a "Task" zone around each task's execution. The app hook
	// pairs the profiler's ZoneBegin/ZoneEnd: the BEGIN call returns the
	// handle the matching END must pass back. Null hook = free no-op.
	using ProfilerZoneFn = uint64_t (*)(int threadIndex, const char* name, bool begin, uint64_t handle);

	inline std::atomic<ProfilerZoneFn>& GetProfilerZoneHook()
	{
		static std::atomic<ProfilerZoneFn> hook{ nullptr };
		return hook;
	}

	inline void SetProfilerZoneHook(ProfilerZoneFn fn)
	{
		GetProfilerZoneHook().store(fn, std::memory_order_release);
	}

	inline uint64_t MarkProfilerZone(int threadIndex, const char* name, bool begin, uint64_t handle)
	{
		if (ProfilerZoneFn fn = GetProfilerZoneHook().load(std::memory_order_acquire))
			return fn(threadIndex, name, begin, handle);
		return 0;
	}

} // namespace loom
