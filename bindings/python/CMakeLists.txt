cmake_minimum_required(VERSION 3.20)

# Python 模块：MODULE 库走 LIBRARY_OUTPUT_DIRECTORY，强制改到 bin/<CONFIG> 与其他产物对齐
# Windows 产出 beacon.cp311-win_amd64.pyd（SOABI 后缀随 Python 版本，导入名仍为 beacon）
pybind11_add_module(beacon PythonBindings.cpp)
target_link_libraries(beacon PRIVATE BeaconStatic)
set_target_properties(beacon PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin/$<CONFIG>
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin/$<CONFIG>
    # 去掉全局 CMAKE_DEBUG_POSTFIX 的 "d" 后缀：模块名必须始终是 beacon（否则 Debug 产出 beacond.pyd 无法 import beacon）
    DEBUG_POSTFIX "")
