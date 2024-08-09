cmake_minimum_required(VERSION 3.10)

# Assuming read_cgns_file.c is the source file you want to compile
add_library(IO STATIC read_cgns_file.c)

# Specify include directories for the IO library
target_include_directories(IO PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

