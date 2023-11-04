#Waypoint Publisher ROS package

pointcloud_processor.py - node for processing 3D pointcloud. Takes start and destination poses,
segmentates the pointcloud, generates costmap, builds a apth from start to the destination

waypoint_publisher_node.py - uses pointcloud processor to build path between 2 points, generates velocity command to make mobile robot reach the goal

fit_model.py - iteratively searches optimal values of koefs to evaluate map traversability so that the diffenence between standard costmap and generated costmap in minimal