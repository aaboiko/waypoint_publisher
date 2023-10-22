#include "ros/ros.h"
#include "../include/pointcloud_processor.h"

int main(int argc, char** argv){
    ros::init(argc, argv, "waypoint_publisher_node");
    ros::NodeHandle nh;
    return 0;
}