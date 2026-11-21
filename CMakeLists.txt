# loom - a task and threading library: one pool of workers, dedicated
# threads and lanes, atomic counters as the only synchronisation, and
# help-while-waiting. This file is the standalone build and what a host's
# CMake build adds with add_subdirectory.
cmake_minimum_required(VERSION 3.16)
project(loom CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(Threads REQUIRED)

# thread | address | undefined: the sanitizer every target here builds with.
set(LOOM_SANITIZE "" CACHE STRING "Build with a sanitizer (thread, address or undefined)")
set(LOOM_SANITIZE_FLAGS "")
if(LOOM_SANITIZE)
    set(LOOM_SANITIZE_FLAGS -fsanitize=${LOOM_SANITIZE} -fno-omit-frame-pointer)
endif()

add_library(loom STATIC src/loom.cpp)
target_include_directories(loom PUBLIC include)
target_link_libraries(loom PUBLIC Threads::Threads)
if(LOOM_SANITIZE STREQUAL "thread")
    # The suite's TSAN-aware paths (the same define the host's tsan build sets).
    target_compile_definitions(loom PUBLIC LOOM_TSAN)
endif()
if(LOOM_SANITIZE_FLAGS)
    target_compile_options(loom PUBLIC ${LOOM_SANITIZE_FLAGS})
    target_link_options(loom PUBLIC ${LOOM_SANITIZE_FLAGS})
endif()
if(PROJECT_IS_TOP_LEVEL AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(loom PRIVATE -Wall -Wextra -Werror)
endif()

option(LOOM_BUILD_TESTS "Build the suite and the benchmark" ${PROJECT_IS_TOP_LEVEL})
if(LOOM_BUILD_TESTS)
    enable_testing()
    add_executable(loom_tests tests/main.cpp)
    target_include_directories(loom_tests PRIVATE tests tests/third_party)
    target_link_libraries(loom_tests PRIVATE loom)
    add_test(NAME loom_tests COMMAND loom_tests)
    set_tests_properties(loom_tests PROPERTIES ENVIRONMENT "TSAN_OPTIONS=suppressions=${CMAKE_CURRENT_SOURCE_DIR}/tests/tsan.supp")

    add_executable(loom_bench bench/loom_bench.cpp)
    target_link_libraries(loom_bench PRIVATE loom)
endif()
