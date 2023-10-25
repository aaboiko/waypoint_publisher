cmake_minimum_required(VERSION 3.2)
project(waypoint_publisher)

find_package(catkin REQUIRED COMPONENTS
  actionlib
  actionlib_msgs
  geometry_msgs
  gazebo_msgs
  move_base
  move_base_msgs
  roscpp
  rospy
  smach
  std_msgs
  tf
  ecl_threads
)

#find_package(PCL 1.13 REQUIRED)
#find_package (Eigen3 3.4 REQUIRED NO_MODULE)


## Generate messages in the 'msg' folder
add_message_files(
   DIRECTORY msg
   FILES
   Point2i.msg
   Point2f.msg
)

## Generate services in the 'srv' folder
# add_service_files(
#   FILES
#   Service1.srv
#   Service2.srv
# )

## Generate actions in the 'action' folder
add_action_files(
   DIRECTORY action
   FILES
   GetPath.action
)

## Generate added messages and services with any dependencies listed here
 generate_messages(
   DEPENDENCIES
   geometry_msgs   
   std_msgs
   move_base_msgs
   actionlib_msgs
   gazebo_msgs
 )

catkin_package(
  #INCLUDE_DIRS include
  LIBRARIES waypoint_publisher

  CATKIN_DEPENDS 
  actionlib 
  geometry_msgs 
  gazebo_msgs
  move_base 
  move_base_msgs 
  roscpp 
  rospy 
  smach 
  std_msgs 
  tf 
  ecl_threads
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  #${PCL_INCLUDE_DIRS}
)

#add_executable(${PROJECT_NAME} 
#   include/pointcloud_processor.h
#   src/pointcloud_processor.cpp
#)

#add_dependencies(${PROJECT_NAME} ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
#link_directories(${PCL_LIBRARY_DIRS})
#add_definitions(${PCL_DEFINITIONS})

 catkin_install_python(PROGRAMS
   scripts/waypoint_publisher_node.py
   scripts/pointcloud_processor.py
   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
 )

 #install(TARGETS ${PROJECT_NAME}
 #  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
 #)

 install(DIRECTORY launch
        DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

#target_link_libraries(waypoint_publisher ${PCL_LIBRARIES})
