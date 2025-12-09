cmake_minimum_required(VERSION 3.10)
project(multiViewSfM)
set(OpenCV_DIR "${CMAKE_CURRENT_LIST_DIR}/support/opencv/lib/cmake/opencv4")
set(Open3D_DIR "${CMAKE_CURRENT_LIST_DIR}/support/open3d/lib/cmake/Open3D")

find_package(OpenCV REQUIRED)
find_package(Open3D REQUIRED)

add_library(multiViewSfM)
target_sources(multiViewSfM
  PRIVATE
    src/cameraIntrinsics.cpp
    src/featureExtractionMatching.cpp
    src/triangulateAndReproject.cpp
    src/localizationAndCorrespondance.cpp
  PUBLIC
    FILE_SET HEADERS
    BASE_DIRS include
    FILES
      include/multiViewSfM/cameraIntrinsics.h
      include/multiViewSfM/featureExtractionMatching.h
      include/multiViewSfM/triangulateAndReproject.h
      include/multiViewSfM/localizationAndCorrespondance.h
)
target_include_directories(multiViewSfM PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:include>)
target_link_libraries(multiViewSfM ${OpenCV_LIBS} Open3D::Open3D)


add_executable(multiview_sfm main.cpp)
target_link_libraries(multiview_sfm PUBLIC multiViewSfM)

add_executable(test_vis tests/test_visualization.cpp)
target_link_libraries(test_vis PUBLIC multiViewSfM)

# add_executable(test_intrinsics tests/test_intrinsics.cpp)
# target_link_libraries(test_intrinsics PUBLIC multiViewSfM)


# add_executable(test_features tests/test_featureExtractionMatching.cpp)
# target_link_libraries(test_features PUBLIC multiViewSfM)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_definitions(multiview_sfm PRIVATE DEBUG_MODE=1)
else()
    target_compile_definitions(multiview_sfm PRIVATE DEBUG_MODE=0)
endif()

enable_testing()
add_test(
  # NAME MAIN_TESTING COMMAND multiview_sfm -i ${CMAKE_SOURCE_DIR}/tests/door/parameters/cameras.txt -bp ${CMAKE_SOURCE_DIR}/build/featuresv1.bin
  NAME MAIN_TESTING COMMAND multiview_sfm -i ${CMAKE_SOURCE_DIR}/tests/door/calibration/cameras.txt -fp ${CMAKE_SOURCE_DIR}/tests/door/images/
 )
# add_test(
#     NAME TESTING_INTRINSICS COMMAND test_intrinsics
#             ${CMAKE_SOURCE_DIR}/tests/door/parameters/cameras.txt   
# )
# add_test(
#     NAME TESTING_FEATURE_EXTRACTION COMMAND test_features
#               ${CMAKE_SOURCE_DIR}/build/test_featuresv1.bin
# )
#  add_test(
#    NAME TESTING_VISUALIZATION COMMAND test_vis -i ${CMAKE_SOURCE_DIR}/tests/statue/calibration/points3D.txt
# )
