cmake_minimum_required (VERSION 2.6)
project(devkit_object)

# Set the path to Boost
set(BOOST_ROOT /work/u5832291/yixian/M3D_RPN_edit/boost/boost_1_66_0)

# Find Boost
find_package(Boost REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})

add_executable(evaluate_object evaluate_object.cpp)
