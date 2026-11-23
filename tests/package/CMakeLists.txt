# A program that uses loom as a host project would: from an installed
# package (find_package, given CMAKE_PREFIX_PATH), or from the source tree
# with add_subdirectory (-DLOOM_SOURCE_DIR=...). CI builds it both ways,
# and asks the package for the version in the header (-DLOOM_EXPECT_VERSION).
cmake_minimum_required(VERSION 3.16)
project(loom_package_check CXX)

if(LOOM_SOURCE_DIR)
    add_subdirectory(${LOOM_SOURCE_DIR} loom)
else()
    find_package(loom ${LOOM_EXPECT_VERSION} REQUIRED)
endif()

add_executable(loom_package_check main.cpp)
target_link_libraries(loom_package_check PRIVATE loom::loom)
