# 最低版本要求
cmake_minimum_required(VERSION 3.20)
# 项目信息
project(global_memory LANGUAGES CXX CUDA)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CUDA_STANDARD 14) # 用于指定CUDA编译器应该使用的CUDA C++标准的版本
set(CMAKE_CUDA_STANDARD_REQUIRED ON)    # 表明如果找不到指定版本的CUDA编译器，将发出错误
set(CMAKE_CXX_STANDARD 14)  # 用于指定 C++ 编译器应该使用的 C++ 标准版本
set(CMAKE_CXX_STANDARD_REQUIRED ON) # 表明如果找不到指定版本的 C++ 编译器，将发出错误

# 这里-arch=sm_86表示计算能力 --ptxas-options=-v表示是否开启L1 Cache
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -arch=sm_86 --ptxas-options=-v")
# set(CMAKE_CUDA_FLAGS_DEBUG="-G -g -O0")
find_package(CUDA REQUIRED)
if (CUDA_FOUND)
    message(STATUS "CUDA_INCLUDE_DIRS: ${CUDA_INCLUDE_DIRS}")
    message(STATUS "CUDA_LIBRARIES: ${CUDA_LIBRARIES}")
    message(STATUS "CUDA_LIBRARY_DIRS: ${CUDA_LIBRARY_DIRS}")
else()
message(FATAL_ERROR "Cannot find CUDA")
endif()

# 添加可执行文件
add_executable(static_global_memory static_global_memory.cu common.cuh)
target_include_directories(static_global_memory PRIVATE ${CUDA_INCLUDE_DIRS})
# 链接库
target_link_libraries(static_global_memory PRIVATE ${CUDA_LIBRARIES})