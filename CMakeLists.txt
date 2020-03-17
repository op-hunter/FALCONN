cmake_minimum_required(VERSION 3.4.1)
project(falconn CXX)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_VERBOSE_MAKEFILE 1)
add_custom_target(makeFALCONN
        COMMAND make -j8
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})