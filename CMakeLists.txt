cmake_minimum_required(VERSION 3.16)
project(MdbVector VERSION 0.1.0 LANGUAGES CXX)

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
