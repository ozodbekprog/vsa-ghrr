cmake_minimum_required(VERSION 3.20)
project(vsa_ghrr LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(OpenMP)

add_library(vsacore INTERFACE)
target_include_directories(vsacore INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_options(vsacore INTERFACE -O3 -march=native -Wall -Wextra)
if(OpenMP_CXX_FOUND)
  target_link_libraries(vsacore INTERFACE OpenMP::OpenMP_CXX)
endif()

add_executable(vsa-tests tests/test_vsa.cpp)
target_link_libraries(vsa-tests PRIVATE vsacore)

add_executable(test-char-encoder tests/test_char_encoder.cpp)
target_link_libraries(test-char-encoder PRIVATE vsacore)

add_executable(vsa-bench src/bench.cpp)
target_link_libraries(vsa-bench PRIVATE vsacore)

# Optional Python bindings
option(VSA_PYTHON "Build pybind11 module" OFF)
if(VSA_PYTHON)
  find_package(pybind11 CONFIG REQUIRED)
  pybind11_add_module(vsacore_py python/bindings.cpp)
  target_link_libraries(vsacore_py PRIVATE vsacore)
  set_target_properties(vsacore_py PROPERTIES LIBRARY_OUTPUT_DIRECTORY
    ${CMAKE_CURRENT_SOURCE_DIR}/python/gpuvsa)
endif()
