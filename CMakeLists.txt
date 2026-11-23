# loom - a task and threading library: one pool of workers, dedicated
# threads and lanes, atomic counters as the only synchronisation, and
# help-while-waiting. This file is the standalone build, what a host's
# CMake build adds with add_subdirectory, and what `cmake --install`
# installs for find_package(loom).
cmake_minimum_required(VERSION 3.16)

# The version is set once, in the header.
file(STRINGS include/loom/loom_version.h LOOM_VERSION_LINE REGEX "^#define LOOM_VERSION_STRING ")
string(REGEX MATCH "[0-9]+\\.[0-9]+\\.[0-9]+" LOOM_VERSION_FROM_HEADER "${LOOM_VERSION_LINE}")
project(loom VERSION ${LOOM_VERSION_FROM_HEADER} LANGUAGES CXX)

# PROJECT_IS_TOP_LEVEL arrived in CMake 3.21.
if(CMAKE_VERSION VERSION_LESS 3.21)
    string(COMPARE EQUAL "${CMAKE_SOURCE_DIR}" "${PROJECT_SOURCE_DIR}" PROJECT_IS_TOP_LEVEL)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(Threads REQUIRED)
include(GNUInstallDirs)

# thread | address | undefined: the sanitizer every target here builds with.
set(LOOM_SANITIZE "" CACHE STRING "Build with a sanitizer (thread, address or undefined)")
set(LOOM_SANITIZE_FLAGS "")
if(LOOM_SANITIZE)
    set(LOOM_SANITIZE_FLAGS -fsanitize=${LOOM_SANITIZE} -fno-omit-frame-pointer)
endif()
# Sleep on a mutex and condition variable instead of a futex, as non-Linux
# builds do: lets that path be tested on Linux.
option(LOOM_NO_FUTEX "Use the portable sleep instead of the Linux futex" OFF)
option(LOOM_BUILD_TESTS "Build the suite, the stress test and the benchmark" ${PROJECT_IS_TOP_LEVEL})
option(LOOM_INSTALL "Install the library, headers and CMake package" ${PROJECT_IS_TOP_LEVEL})

# The library, as `name`. The suite links a second copy built with test
# hooks, so the library that is installed and benchmarked never has them.
function(loom_add_library name)
    add_library(${name} STATIC src/loom.cpp)
    target_include_directories(${name} PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
    target_compile_features(${name} PUBLIC cxx_std_20)
    target_link_libraries(${name} PUBLIC Threads::Threads)
    if(LOOM_NO_FUTEX)
        target_compile_definitions(${name} PRIVATE LOOM_NO_FUTEX)
    endif()
    if(LOOM_SANITIZE STREQUAL "thread")
        # The suite's TSAN-aware paths (the same define the host's tsan build sets).
        target_compile_definitions(${name} PUBLIC LOOM_TSAN)
    endif()
    if(LOOM_SANITIZE_FLAGS)
        target_compile_options(${name} PUBLIC ${LOOM_SANITIZE_FLAGS})
        target_link_options(${name} PUBLIC ${LOOM_SANITIZE_FLAGS})
    endif()
    if(PROJECT_IS_TOP_LEVEL AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(${name} PRIVATE -Wall -Wextra -Werror)
    endif()
endfunction()

loom_add_library(loom)
add_library(loom::loom ALIAS loom)

if(LOOM_BUILD_TESTS)
    enable_testing()
    # Test hooks: callbacks at a few points inside loom where a test needs to
    # widen a race to see it. Only this copy, which only the suite links, has them.
    loom_add_library(loom_test_hooks)
    target_compile_definitions(loom_test_hooks PUBLIC LOOM_TEST_HOOKS)

    add_executable(loom_tests tests/main.cpp)
    target_include_directories(loom_tests PRIVATE tests tests/third_party)
    target_link_libraries(loom_tests PRIVATE loom_test_hooks)
    add_test(NAME loom_tests COMMAND loom_tests)

    add_executable(loom_stress tests/stress.cpp)
    target_link_libraries(loom_stress PRIVATE loom)
    add_test(NAME loom_stress COMMAND loom_stress --seconds 10 --quiet)

    set_tests_properties(loom_tests loom_stress PROPERTIES ENVIRONMENT "TSAN_OPTIONS=suppressions=${CMAKE_CURRENT_SOURCE_DIR}/tests/tsan.supp")

    add_executable(loom_bench bench/loom_bench.cpp)
    target_link_libraries(loom_bench PRIVATE loom)
endif()

if(LOOM_INSTALL)
    include(CMakePackageConfigHelpers)
    install(TARGETS loom EXPORT loomTargets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
    install(DIRECTORY include/loom DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
    set(LOOM_CMAKE_DIR ${CMAKE_INSTALL_LIBDIR}/cmake/loom)
    install(EXPORT loomTargets NAMESPACE loom:: DESTINATION ${LOOM_CMAKE_DIR})
    configure_package_config_file(cmake/loomConfig.cmake.in
        ${CMAKE_CURRENT_BINARY_DIR}/loomConfig.cmake
        INSTALL_DESTINATION ${LOOM_CMAKE_DIR})
    # Before 1.0 a minor release may change the API, so only the same
    # minor version is compatible.
    write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/loomConfigVersion.cmake
        COMPATIBILITY SameMinorVersion)
    install(FILES
        ${CMAKE_CURRENT_BINARY_DIR}/loomConfig.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/loomConfigVersion.cmake
        DESTINATION ${LOOM_CMAKE_DIR})
    install(FILES LICENSE DESTINATION ${CMAKE_INSTALL_DATADIR}/licenses/loom)
endif()
