cmake_minimum_required(VERSION 3.16)
project(kengo_fullbody_retarget_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(hdas2 REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(yaml-cpp REQUIRED)

set(MUJOCO_ROOT "" CACHE PATH "MuJoCo Python package directory")
find_path(MUJOCO_INCLUDE_DIR
  NAMES mujoco/mujoco.h
  HINTS "${MUJOCO_ROOT}/include"
  REQUIRED)
find_library(MUJOCO_LIBRARY
  NAMES mujoco libmujoco.so.3.3.4 mujoco.3.3.4
  HINTS "${MUJOCO_ROOT}"
  NO_DEFAULT_PATH
  REQUIRED)

add_library(kengo_fullbody_core STATIC
  src/fullbody_retarget_core.cpp)
target_include_directories(kengo_fullbody_core
  PUBLIC include "${MUJOCO_INCLUDE_DIR}")
target_link_libraries(kengo_fullbody_core
  PUBLIC Eigen3::Eigen yaml-cpp "${MUJOCO_LIBRARY}")
target_compile_options(kengo_fullbody_core PRIVATE -Wall -Wextra -Wpedantic -Werror)

add_executable(kengo_fullbody_retarget_node
  src/fullbody_retarget_node.cpp)
target_link_libraries(kengo_fullbody_retarget_node kengo_fullbody_core)
ament_target_dependencies(kengo_fullbody_retarget_node
  rclcpp geometry_msgs sensor_msgs)
target_compile_options(kengo_fullbody_retarget_node PRIVATE -Wall -Wextra -Wpedantic -Werror)

option(BUILD_LEGACY_COMMAND_BRIDGE
  "Build the archived direct /hybrid_body_controller/commands publisher" OFF)

if(BUILD_LEGACY_COMMAND_BRIDGE)
  add_library(kengo_fullbody_command_core STATIC
    src/fullbody_command_composer.cpp)
  target_include_directories(kengo_fullbody_command_core PUBLIC include)
  target_compile_options(kengo_fullbody_command_core PRIVATE
    -Wall -Wextra -Wpedantic -Werror)

  add_executable(kengo_fullbody_command_bridge_node
    src/fullbody_command_bridge_node.cpp)
  target_link_libraries(kengo_fullbody_command_bridge_node
    kengo_fullbody_command_core)
  ament_target_dependencies(kengo_fullbody_command_bridge_node
    rclcpp sensor_msgs)
  target_link_libraries(kengo_fullbody_command_bridge_node
    hdas2::hdas2__rosidl_generator_cpp
    hdas2::hdas2__rosidl_typesupport_cpp)
  target_compile_options(kengo_fullbody_command_bridge_node PRIVATE
    -Wall -Wextra -Wpedantic -Werror)
endif()

add_library(kengo_fullbody_composed_target_core STATIC
  src/fullbody_composed_target.cpp)
target_include_directories(kengo_fullbody_composed_target_core PUBLIC include)
target_compile_options(kengo_fullbody_composed_target_core PRIVATE
  -Wall -Wextra -Wpedantic -Werror)

add_executable(kengo_fullbody_composed_target_node
  src/fullbody_composed_target_node.cpp)
target_link_libraries(kengo_fullbody_composed_target_node
  kengo_fullbody_composed_target_core)
ament_target_dependencies(kengo_fullbody_composed_target_node
  rclcpp sensor_msgs)
target_link_libraries(kengo_fullbody_composed_target_node
  hdas2::hdas2__rosidl_generator_cpp
  hdas2::hdas2__rosidl_typesupport_cpp)
target_compile_options(kengo_fullbody_composed_target_node PRIVATE
  -Wall -Wextra -Wpedantic -Werror)

set_target_properties(kengo_fullbody_retarget_node PROPERTIES
  BUILD_RPATH "${MUJOCO_ROOT}"
  INSTALL_RPATH "$ORIGIN/../lib")

include(CTest)
if(BUILD_TESTING)
  add_executable(kengo_fullbody_graft_test
    src/fullbody_graft_test.cpp)
  target_link_libraries(kengo_fullbody_graft_test kengo_fullbody_core)
  target_compile_options(kengo_fullbody_graft_test PRIVATE
    -Wall -Wextra -Wpedantic -Werror)
  set_target_properties(kengo_fullbody_graft_test PROPERTIES
    BUILD_RPATH "${MUJOCO_ROOT}")
  add_test(NAME kengo_fullbody_graft_test COMMAND kengo_fullbody_graft_test)
  if(BUILD_LEGACY_COMMAND_BRIDGE)
    add_executable(kengo_fullbody_command_composer_test
      src/fullbody_command_composer_test.cpp)
    target_link_libraries(kengo_fullbody_command_composer_test
      kengo_fullbody_command_core)
    target_compile_options(kengo_fullbody_command_composer_test PRIVATE
      -Wall -Wextra -Wpedantic -Werror)
    add_test(NAME kengo_fullbody_command_composer_test
      COMMAND kengo_fullbody_command_composer_test)
  endif()
  add_executable(kengo_fullbody_composed_target_test
    src/fullbody_composed_target_test.cpp)
  target_link_libraries(kengo_fullbody_composed_target_test
    kengo_fullbody_composed_target_core)
  target_compile_options(kengo_fullbody_composed_target_test PRIVATE
    -Wall -Wextra -Wpedantic -Werror)
  add_test(NAME kengo_fullbody_composed_target_test
    COMMAND kengo_fullbody_composed_target_test)
endif()

install(TARGETS kengo_fullbody_retarget_node
  kengo_fullbody_composed_target_node
  RUNTIME DESTINATION bin)
if(BUILD_LEGACY_COMMAND_BRIDGE)
  install(TARGETS kengo_fullbody_command_bridge_node RUNTIME DESTINATION bin)
endif()

ament_package()
