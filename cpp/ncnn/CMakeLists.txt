cmake_minimum_required(VERSION 3.19)
project(cpp)

set(CMAKE_CXX_STANDARD 14)

set(OpenCV_DIR D:/ml/opencv/build)
set(NCNN_DIR D:/ml/ncnn/lib/cmake/ncnn)
set(Vulkan_DIR C:/VulkanSDK)

# opencv
find_package(OpenCV REQUIRED)
if(OpenCV_FOUND)
    message("OpenCV_INCLUDE_DIRS: " ${OpenCV_INCLUDE_DIRS})
    message("OpenCV_LIBS: " ${OpenCV_LIBS})
endif()
include_directories(${OpenCV_INCLUDE_DIRS})

# ncnn
find_package(NCNN REQUIRED)
if(NCNN_FOUND)
    message("NCNN_FOUND!")
endif()
# ncnn include
include_directories(D:/ml/ncnn/include)
# ncnn lib
link_directories(D:/ml/ncnn/lib)

# 链接所有库，不指定cpp文件
link_libraries(${OpenCV_LIBS}
                ncnn.lib)

#vulkan
include_directories(${Vulkan_DIR}/Include)

add_executable(main main.cpp)
