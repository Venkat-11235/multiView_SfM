cmake_minimum_required(VERSION 3.10)
project(multiViewSfM)
set(OpenCV_DIR "${CMAKE_CURRENT_LIST_DIR}/support/opencv/lib/cmake/opencv4")

find_package(OpenCV REQUIRED)

add_library(multiViewSfM)
target_sources(multiViewSfM
  PRIVATE
    src/cameraIntrinsics.cpp
    src/featureExtractionMatching.cpp
  PUBLIC
    FILE_SET HEADERS
    BASE_DIRS include
    FILES
      include/multiViewSfM/cameraIntrinsics.h
      include/multiViewSfM/featureExtractionMatching.h
)
target_include_directories(multiViewSfM PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:include>)
target_link_libraries(multiViewSfM ${OpenCV_LIBS})


add_executable(multiview_sfm main.cpp)
target_link_libraries(multiview_sfm PUBLIC multiViewSfM)


add_executable(test_intrinsics tests/test_intrinsics.cpp)
target_link_libraries(test_intrinsics PUBLIC multiViewSfM)


add_executable(test_features tests/test_featureExtractionMatching.cpp)
target_link_libraries(test_features PUBLIC multiViewSfM)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_definitions(multiview_sfm PRIVATE DEBUG_MODE=1)
else()
    target_compile_definitions(multiview_sfm PRIVATE DEBUG_MODE=0)
endif()

enable_testing()
add_test(
  NAME MAIN_TESTING COMMAND multiview_sfm -i ${CMAKE_SOURCE_DIR}/tests/door/parameters/cameras.txt -bp ${CMAKE_SOURCE_DIR}/build/featuresv1.bin
)
add_test(
    NAME TESTING_INTRINSICS COMMAND test_intrinsics
            ${CMAKE_SOURCE_DIR}/tests/door/parameters/cameras.txt   
)
add_test(
    NAME TESTING_FEATURE_EXTRACTION COMMAND test_features
              ${CMAKE_SOURCE_DIR}/build/test_featuresv1.bin
)
