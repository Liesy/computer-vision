cmake_minimum_required(VERSION 3.20)
project(computer_vision)

set(CMAKE_CXX_STANDARD 11)

add_executable(computer_vision MatchTemplate.cpp)

find_package(OpenCV REQUIRED)
target_link_libraries(computer_vision ${OpenCV_LIBS})
