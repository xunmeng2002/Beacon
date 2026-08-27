cmake_minimum_required(VERSION 3.20)

# 对齐 Spark 的 test/unittest/CMakeLists.txt：GoogleTest via vcpkg + 单一 UnitTests 可执行目标
find_package(GTest CONFIG REQUIRED)

add_application(UnitTests ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(UnitTests PRIVATE
    GTest::gtest
    GTest::gtest_main
    MdbVectorStatic)
