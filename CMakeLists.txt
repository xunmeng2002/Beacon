cmake_minimum_required(VERSION 3.20)
include(submodules/CMakeCommon/CMakeCommon.cmake)
restore_vcpkg_root()
set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake")
message(STATUS "ENV{VCPKG_ROOT} = $ENV{VCPKG_ROOT}")
message(STATUS "ENV{VCPKG_DOWNLOADS} = $ENV{VCPKG_DOWNLOADS}")
message(STATUS "ENV{VCPKG_DEFAULT_BINARY_CACHE} = $ENV{VCPKG_DEFAULT_BINARY_CACHE}")
message(STATUS "CMAKE_TOOLCHAIN_FILE = ${CMAKE_TOOLCHAIN_FILE}")
SET(PROJECT_NAME "Beacon")
project(${PROJECT_NAME} VERSION 1.0.0 LANGUAGES CXX)
include(GNUInstallDirs)
include(GenerateExportHeader)

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_DEBUG_POSTFIX "d")
set(CMAKE_VERBOSE_MAKEFILE ON)


IF(CMAKE_SYSTEM_NAME MATCHES "Windows")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc /bigobj /utf-8")
    add_definitions(-DWINDOWS -D_CRT_SECURE_NO_WARNINGS)
    add_compile_options(
        $<$<CONFIG:Debug>:/Od>
        $<$<CONFIG:Debug>:/Zi>
        $<$<CONFIG:Release>:/O2>
        $<$<CONFIG:RelWithDebInfo>:/O2>
        $<$<CONFIG:RelWithDebInfo>:/Zi>
        $<$<CONFIG:MinSizeRel>:/O1>)
ELSEIF(CMAKE_SYSTEM_NAME MATCHES "Linux")
    add_definitions("-DLINUX")
    add_compile_options(
        $<$<CONFIG:Debug>:-O0>
        $<$<CONFIG:Debug>:-g>
        $<$<CONFIG:Debug>:-ggdb>
        $<$<CONFIG:Release>:-O3>
        $<$<CONFIG:RelWithDebInfo>:-O2>
        $<$<CONFIG:RelWithDebInfo>:-g>
        $<$<CONFIG:MinSizeRel>:-Os>)
    add_compile_definitions(
        $<$<CONFIG:Debug>:_DEBUG>
        $<$<CONFIG:Release>:NDEBUG>
        $<$<CONFIG:RelWithDebInfo>:NDEBUG>
        $<$<CONFIG:MinSizeRel>:NDEBUG>)
ENDIF()

# x86_64 上开启 AVX2 加速点积；MSVC 用 /arch:AVX2，GCC/Clang 用 -mavx2
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64")
  if(MSVC)
    add_compile_options(/arch:AVX2)
  else()
    add_compile_options(-mavx2)
  endif()
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin/$<CONFIG>)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib/$<CONFIG>)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib/$<CONFIG>)

# 库本体（零第三方依赖）：对外头 include/（PUBLIC，消费方只看到这里）、对内实现 src/Beacon/
add_static_md_module(Beacon ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/src/Beacon)

# Demo 可执行文件（对齐 Mdb 的 add_application(TestDB ...) 命名约定）
add_application(TestBeacon ${CMAKE_CURRENT_SOURCE_DIR}/test/TestBeacon ${CMAKE_CURRENT_SOURCE_DIR}/test/TestBeacon)
target_link_libraries(TestBeacon PRIVATE BeaconStatic)

# 可选基准：hnswlib 对比（需 vcpkg toolchain，见 README；库本身保持零第三方依赖）
option(BEACON_ENABLE_BENCH "Build hnswlib comparison benchmark (requires vcpkg toolchain)" ON)
if(BEACON_ENABLE_BENCH)
  if(NOT DEFINED VCPKG_INSTALLED_DIR)
    message(FATAL_ERROR "BEACON_ENABLE_BENCH=ON 需要 vcpkg toolchain：请用 CMakePresets 的 x64-Debug/x64-Release（已内置），"
                        "或配置时追加 -DCMAKE_TOOLCHAIN_FILE=$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake")
  endif()
  find_package(hnswlib CONFIG REQUIRED)
  add_executable(beacon_bench bench/bench_compare.cpp)
  target_link_libraries(beacon_bench PRIVATE BeaconStatic hnswlib::hnswlib)
endif()

# 单元测试（GoogleTest，需 vcpkg toolchain；无 vcpkg 构建时与 BEACON_ENABLE_BENCH 同理需显式关闭）
option(BUILD_UNIT_TESTS "Build unit tests (GoogleTest, requires vcpkg)" ON)
if(BUILD_UNIT_TESTS)
    add_subdirectory(test/unittest)
endif()
