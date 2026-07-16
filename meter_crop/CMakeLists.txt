cmake_minimum_required(VERSION 3.5.1)
SET(CMAKE_SYSTEM_NAME Linux)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wl,--allow-shlib-undefined")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wl,--allow-shlib-undefined")

PROJECT(AML_NN_SDK)
include_directories(./ ./include)

# opencv
set(OPENCV_API_PATRH /usr/include/opencv4/)
set(OpenCV_LIBS /usr/lib/aarch64-linux-gnu/)
include_directories(${OPENCV_API_PATRH})
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} /usr/lib/aarch64-linux-gnu/cmake/opencv4)
find_package(OpenCV REQUIRED)

SET(CMAKE_C_COMPILER aarch64-linux-gnu-gcc)
SET(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)
SET(LINK_DIR /usr/lib/aarch64-linux-gnu/  ./)

link_directories(${LINK_DIR})
aux_source_directory(. SRC_LIST)
link_libraries(nnsdk libjpeg.a )

ADD_EXECUTABLE(yolov8n ${SRC_LIST})
target_link_libraries(yolov8n -lpthread -lm -ldl ${OpenCV_LIBS})

