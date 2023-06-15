cmake_minimum_required(VERSION 3.19)
project(cpp)

set(CMAKE_CXX_STANDARD 14)

# opencv
set(OpenCV_DIR D:/ml/opencv/build)
find_package(OpenCV REQUIRED)
if(OpenCV_FOUND)
    message("OpenCV_INCLUDE_DIRS: " ${OpenCV_INCLUDE_DIRS})
    message("OpenCV_LIBS: " ${OpenCV_LIBS})
endif()
include_directories(${OpenCV_INCLUDE_DIRS})

# 链接所有库，不指定cpp文件
link_libraries(${OpenCV_LIBS})

add_executable(main main.cpp)
