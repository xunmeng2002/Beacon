cmake_minimum_required(VERSION 3.20)
include(submodules/CMakeCommon/CMakeCommon.cmake)
restore_vcpkg_root()
# 注意：vcpkg 工具链由 CMakePresets.json 的 windows-base 通过 cacheVariables 注入
# （普通 set() 不会让 project() 加载工具链，此处不得再用 set 覆盖）
message(STATUS "ENV{VCPKG_ROOT} = $ENV{VCPKG_ROOT}")
message(STATUS "ENV{VCPKG_DOWNLOADS} = $ENV{VCPKG_DOWNLOADS}")
message(STATUS "ENV{VCPKG_DEFAULT_BINARY_CACHE} = $ENV{VCPKG_DEFAULT_BINARY_CACHE}")
message(STATUS "CMAKE_TOOLCHAIN_FILE = ${CMAKE_TOOLCHAIN_FILE}")
SET(PROJECT_NAME "MdbVector")
project(${PROJECT_NAME} VERSION 1.0.0 LANGUAGES CXX)
include(GNUInstallDirs)
include(GenerateExportHeader)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# x86_64 上开启 AVX2 加速点积；MSVC 用 /arch:AVX2，GCC/Clang 用 -mavx2
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64")
  if(MSVC)
    add_compile_options(/arch:AVX2)
  else()
    add_compile_options(-mavx2)
  endif()
endif()

add_library(mdbvec
  src/HnswIndex.cpp
  src/Metrics.cpp
  src/VectorDb.cpp
  src/VectorTable.cpp
)
target_include_directories(mdbvec PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(mdbvec_demo src/main.cpp)
target_link_libraries(mdbvec_demo PRIVATE mdbvec)

# 可选基准：hnswlib 对比（需 vcpkg toolchain，见 README；库本身保持零第三方依赖）
option(MDBVEC_ENABLE_BENCH "Build hnswlib comparison benchmark (requires vcpkg toolchain)" ON)
if(MDBVEC_ENABLE_BENCH)
  if(NOT DEFINED VCPKG_INSTALLED_DIR)
    message(FATAL_ERROR "MDBVEC_ENABLE_BENCH=ON 需要 vcpkg toolchain：请用 CMakePresets 的 x64-Debug/x64-Release（已内置），"
                        "或配置时追加 -DCMAKE_TOOLCHAIN_FILE=$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake")
  endif()
  find_package(hnswlib CONFIG REQUIRED)
  add_executable(mdbvec_bench bench/bench_compare.cpp)
  target_link_libraries(mdbvec_bench PRIVATE mdbvec hnswlib::hnswlib)
endif()
