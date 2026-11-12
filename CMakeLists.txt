cmake_minimum_required(VERSION 3.20)
project(echo VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(ECHO_SANITIZE "Build with AddressSanitizer + UndefinedBehaviorSanitizer" OFF)
option(ECHO_BUILD_TESTS "Build the GoogleTest suite" ON)

# ---- core library ----
add_library(echo
  src/distance.cpp
  src/dataset.cpp
  src/brute_force.cpp
)
target_include_directories(echo PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_options(echo PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow)

if(ECHO_SANITIZE)
  add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer -fno-sanitize-recover=all)
  add_link_options(-fsanitize=address,undefined)
  target_compile_options(echo PUBLIC -fsanitize=address,undefined -fno-omit-frame-pointer -fno-sanitize-recover=all)
  target_link_options(echo PUBLIC -fsanitize=address,undefined)
endif()

# ---- tools ----
add_executable(echo_query tools/echo_query.cpp)
target_link_libraries(echo_query PRIVATE echo)

add_executable(bench_bruteforce tools/bench_bruteforce.cpp)
target_link_libraries(bench_bruteforce PRIVATE echo)

# ---- tests ----
if(ECHO_BUILD_TESTS)
  include(FetchContent)
  FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.tar.gz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  )
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)

  enable_testing()
  add_executable(echo_tests
    tests/test_distance.cpp
    tests/test_dataset.cpp
    tests/test_brute_force.cpp
  )
  target_link_libraries(echo_tests PRIVATE echo GTest::gtest_main)
  target_compile_definitions(echo_tests PRIVATE ECHO_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/data")
  include(GoogleTest)
  gtest_discover_tests(echo_tests DISCOVERY_MODE PRE_TEST)
endif()
