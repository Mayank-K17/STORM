cmake_minimum_required(VERSION 3.10)
project(MyCGNSApp LANGUAGES CXX)

# Include the directory where FindCGNS.cmake resides
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

find_package(CGNS REQUIRED)

find_package(CUDA REQUIRED)

set(TARGET_NAME cgns_reader)

add_executable(${TARGET_NAME} IO/cgns_read.cu)

target_include_directories(${TARGET_NAME} PUBLIC ${CGNS_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS})
target_link_libraries(${TARGET_NAME} ${CGNS_LIBRARIES} ${CUDA_LIBRARIES})
