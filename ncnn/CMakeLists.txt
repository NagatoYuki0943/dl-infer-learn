cmake_minimum_required(VERSION 3.19)
project(cpp)

set(CMAKE_CXX_STANDARD 14)

# rapidjson 为相对目录,可以更改为绝对目录
include_directories(include)

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

# ncnn
set(NCNN_DIR D:/ml/ncnn/lib/cmake/ncnn)
find_package(NCNN REQUIRED)
if(NCNN_FOUND)
    message("NCNN_FOUND!")
endif()
# ncnn include
include_directories(D:/ml/ncnn/include)
# ncnn lib
link_directories(D:/ml/ncnn/lib)
link_libraries(ncnn.lib)

#vulkan
include_directories(C:/VulkanSDK/Include)

add_executable(main main.cpp)
