cmake_minimum_required(VERSION 3.19)
project(cpp)

set(CMAKE_CXX_STANDARD 14)

set(OpenCV_DIR D:/ml/opencv/build)
set(ONNXRuntime_DIR D:/ml/onnxruntime)

# opencv
find_package(OpenCV REQUIRED)
if(OpenCV_FOUND)
    message("OpenCV_INCLUDE_DIRS: " ${OpenCV_INCLUDE_DIRS})
    message("OpenCV_LIBS: " ${OpenCV_LIBS})
endif()
include_directories(${OpenCV_INCLUDE_DIRS})

# onnxruntime include
include_directories(${ONNXRuntime_DIR}/include)
# onnxruntime lib
link_directories(${ONNXRuntime_DIR}/lib)

# 链接所有库，不指定cpp文件
link_libraries(${OpenCV_LIBS}
                onnxruntime.lib onnxruntime_providers_cuda.lib onnxruntime_providers_shared.lib onnxruntime_providers_tensorrt.lib)

add_executable(main main.cpp)
