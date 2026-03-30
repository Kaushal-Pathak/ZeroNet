cmake_minimum_required(VERSION 3.15)
project(ZeroNet CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── Optimization flags ─────────────────────────────────────────────────────────
if(MSVC)
    add_compile_options(/O2 /arch:AVX2)
else()
    add_compile_options(-O3 -march=native)
endif()

# ── Find Eigen ─────────────────────────────────────────────────────────────────
# Option A: installed system-wide (most common on Linux/Mac)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# Option B: header-only local copy — uncomment if Eigen3 not found above
# set(EIGEN3_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/third_party/eigen")
# include_directories(${EIGEN3_INCLUDE_DIR})

# ── Executable ────────────────────────────────────────────────────────────────
add_executable(neural_network
    src/main.cpp
)

target_include_directories(neural_network PRIVATE
    include
)

target_link_libraries(neural_network PRIVATE
    Eigen3::Eigen
)
