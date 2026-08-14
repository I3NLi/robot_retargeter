cmake_minimum_required(VERSION 3.16)
project(kengo_fullbody_composed_target LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(hdas2 REQUIRED)

add_library(kengo_fullbody_composed_target_core STATIC
  ../src/fullbody_composed_target.cpp)
target_include_directories(kengo_fullbody_composed_target_core
  PUBLIC ../include)
target_compile_options(kengo_fullbody_composed_target_core PRIVATE
  -Wall -Wextra -Wpedantic -Werror)

add_executable(kengo_fullbody_composed_target_node
  ../src/fullbody_composed_target_node.cpp)
target_link_libraries(kengo_fullbody_composed_target_node
  kengo_fullbody_composed_target_core
  hdas2::hdas2__rosidl_generator_cpp
  hdas2::hdas2__rosidl_typesupport_cpp)
ament_target_dependencies(kengo_fullbody_composed_target_node
  rclcpp sensor_msgs)
target_compile_options(kengo_fullbody_composed_target_node PRIVATE
  -Wall -Wextra -Wpedantic -Werror)

include(CTest)
if(BUILD_TESTING)
  add_executable(kengo_fullbody_composed_target_test
    ../src/fullbody_composed_target_test.cpp)
  target_link_libraries(kengo_fullbody_composed_target_test
    kengo_fullbody_composed_target_core)
  target_compile_options(kengo_fullbody_composed_target_test PRIVATE
    -Wall -Wextra -Wpedantic -Werror)
  add_test(NAME kengo_fullbody_composed_target_test
    COMMAND kengo_fullbody_composed_target_test)
endif()

install(TARGETS kengo_fullbody_composed_target_node
  RUNTIME DESTINATION bin)

ament_package()
