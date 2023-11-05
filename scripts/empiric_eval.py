import rospy, actionlib
import numpy as np
import open3d as pcl
from geometry_msgs.msg import Twist, Pose
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import GetModelState, SetModelState
from tf.transformations import euler_from_quaternion
from numpy.linalg import inv
from csv import writer
import timeit

def float_to_grid(point, offset, cell_size):
    x = int((point[0] - offset[0]) / cell_size)
    y = int((point[1] - offset[1]) / cell_size)
    return [x, y]


def grid_to_float(point, offset, cell_size):
    x = cell_size * point[0] + cell_size / 2 + offset[0]
    y = cell_size * point[1] + cell_size / 2 + offset[1]
    return [x, y]


def read_cloud(filepath):
    cloud = pcl.io.read_point_cloud(filepath)
    rospy.loginfo('Pointcloud obtained: %s', filepath)
    points_array = np.asarray(cloud.points)
    points_number = points_array.shape[0]
    rospy.loginfo('Number of points: %s', points_number)
    return points_array, points_number


def evaluate_cloud(cloud, cell_size):
    xmax, xmin, ymax, ymin, zmax, zmin = -np.inf, np.inf, -np.inf, np.inf, -np.inf, np.inf
    for point in cloud:
        x = point[0]
        y = point[1]
        z = point[2]
        xmax = max(xmax, x)
        xmin = min(xmin, x)
        ymax = max(ymax, y)
        ymin = min(ymin, y)
        zmax = max(zmax, z)
        zmin = min(zmin, z)

    rospy.loginfo('xmin: %f, xmax: %f, ymin: %f, ymax: %f, zmin: %f, zmax: %f',xmin,xmax,ymin,ymax,zmin,zmax)
    
    cloud_size_x = xmax - xmin
    cloud_size_y = ymax - ymin

    grid_size_x = int(cloud_size_x / cell_size) + 1
    grid_size_y = int(cloud_size_y / cell_size) + 1

    rospy.loginfo('Grid size: x = %i, y = %i', grid_size_x, grid_size_y)
    return grid_size_x, grid_size_y, xmin, xmax, ymin, ymax, zmin, zmax


def segmentate_cloud(cloud, size_x, size_y, xmin, xmax, ymin, ymax, cell_size):
    point_grid = dict()
    
    for i in range(size_x):
        for j in range(size_y):
            point_grid.setdefault((i,j), [])

    for point in cloud:
        x = int((point[0] - xmin) / cell_size)
        y = int((point[1] - ymin) / cell_size)
        point_grid[(x,y)].append(point)

    rospy.loginfo('Cloud segmentated')
    return point_grid


def get_plane_inclination(segment):
    n = len(segment)
    big = 10e6
    zmax = -np.inf
    if n < 3:
        return big, big, big, 0.0

    count_param = 1.0 / n
    sign = -1.0
    A = np.zeros((n, 3))
    B = sign * np.ones((n, 1))

    for i in range(n):
        x = segment[i][0]
        y = segment[i][1]
        z = segment[i][2]
        A[i,:] = np.array([x, y, z])
        zmax = max(zmax, z)

    X = inv(np.dot(A.T, A)) @ A.T @ B

    sigma = 0.0
    if n > 3:
        for i in range(n):
            a = X[0]
            b = X[1]
            c = X[2]
            x = segment[i][0]
            y = segment[i][1]
            z = segment[i][2]
            sigma += ((a*x + b*y + c*z - sign)**2) / n

    x_norm = np.linalg.norm(X)
    X = X / x_norm
    
    if abs(X[2]) <= 10e-3:
        return big, sigma, count_param, zmax

    inc_cos = X[2]
    inc_param = np.sqrt(1.0 - inc_cos**2) / inc_cos

    return inc_param, sigma, count_param, zmax


def get_pose(get_topic, robot_name, frame_id):
    rospy.wait_for_service(get_topic)
    state = GetModelState()

    try:
        get_model_state = rospy.ServiceProxy(get_topic, GetModelState)
        state = get_model_state(robot_name, frame_id)
    except:
        rospy.logerr('Error: get state service unavailable')

    return state


def set_pose(set_topic, robot_name, pose):
    x, y, z, w = pose
    
    state_msg = ModelState()
    state_msg.model_name = robot_name
    state_msg.pose.position.x = x
    state_msg.pose.position.y = y
    state_msg.pose.position.z = z
    state_msg.pose.orientation.x = 0
    state_msg.pose.orientation.y = 0
    state_msg.pose.orientation.z = 0
    state_msg.pose.orientation.w = w

    rospy.wait_for_service(set_topic)
    try:
        set_state = rospy.ServiceProxy(set_topic, SetModelState)
        result = set_state(state_msg)
        rospy.loginfo('Pose is set successfully: ' + str(state_msg))
    except:
        rospy.logerr('Error: gazebo server unavailable')


def move_to(get_topic, vel_topic, robot_name, pose, frame_id, cell_size, v):
    rate = rospy.Rate(10)
    vel_pub = rospy.Publisher(vel_topic, Twist, queue_size=10)
    speed = Twist()

    def goal_reached(pose):
        state = get_pose(get_topic, robot_name, frame_id)
        cur_x = state.pose.position.x
        cur_y = state.pose.position.y
        curr_p = np.array([cur_x, cur_y])
        curr_goal = np.array([pose[0], pose[1]])
        delta = np.linalg.norm(curr_p - curr_goal)
        reached = delta < 0.1

        if reached:
            rospy.loginfo('Goal reached: ' + str(curr_goal))

        return reached
    
    timestamp = timeit.default_timer()
    while not goal_reached(pose):
        cur_time = timeit.default_timer()
        dt = cur_time - timestamp
        if dt > 20.0 * cell_size / v:
            rospy.loginfo('Goal ' + str(pose) + ' is not traversable')
            rospy.loginfo('Time to try: ' + str(dt) + ' secs')
            return False
        
        state = get_pose(get_topic, robot_name, frame_id)
        x = state.pose.position.x
        y = state.pose.position.y
        rot = state.pose.orientation
        (roll, pitch, theta) = euler_from_quaternion([rot.x, rot.y, rot.z, rot.w])

        inc_x = pose[0] - x
        inc_y = pose[1] - y

        angle_to_goal = np.arctan2(inc_y, inc_x)
        inc_angle = angle_to_goal - theta
        k = 2.0
        speed.angular.z = k * cell_size * inc_angle
        
        if abs(inc_angle) > 0.1:
            speed.linear.x = 0.0
        else:
            speed.linear.x = v

        vel_pub.publish(speed)
        rate.sleep() 

    return True


def investigate_cell(cell, cell_size, offset, v, get_topic, set_topic, vel_topic, robot_name, frame_id, z):
    x_cell, y_cell = cell
    rospy.loginfo('Starting investigating cell (' + str(x_cell) + ' ' + str(y_cell) + ')')
    center = grid_to_float(cell, offset, cell_size)

    start = [center[0] - cell_size / 2, center[1], z, 0.0]
    dest = [center[0] + cell_size / 2, center[1], z, 0.0]
    set_pose(set_topic, robot_name, start)
    res = move_to(get_topic, vel_topic, robot_name, dest, frame_id, cell_size, v)
    if not res:
        return 0
    
    return 1


def eval_traversability(point_grid, size_x, size_y, cell_size, offset, v, get_topic, set_topic, vel_topic, robot_name, frame_id, save_data, datapath):
    rospy.loginfo('Starting traversability analysis to generate dataset...')
    iter = 0.0
    progress = 0
    prev = 0
    count = size_x * size_y
    n_traversable = 0

    if save_data:
        line = ['inc_param', 'sigma', 'count_param', 'traversable']
        with open(datapath, 'a') as f_obj:
            obj = writer(f_obj)
            obj.writerow(line)
            f_obj.close()
        rospy.loginfo('New dataset file created: ' + datapath)

    for i in range(size_x):
        for j in range(size_y):
            inc_param, sigma, count_param, zmax = get_plane_inclination(point_grid[(i,j)])
            cell = [i, j]
            iter += 1.0
            progress = int((iter / count) * 100)

            if count_param > 1.0:
                continue

            res = investigate_cell(cell, cell_size, offset, v, get_topic, set_topic, vel_topic, robot_name, frame_id, zmax + 0.2)
            if res > 0:
                n_traversable += 1

            if save_data:
                line = [inc_param, sigma, count_param, res]
                with open(datapath, 'a') as f_obj:
                    obj = writer(f_obj)
                    obj.writerow(line)
                    f_obj.close()
                rospy.loginfo('Line written to the dataset: ' + str(line))

            if progress != prev:
                rospy.loginfo('Analysis in progress: [' + str(progress) + '%]')
                prev = progress

    trav_rate = (n_traversable / count) * 100
    rospy.loginfo('Traversability analysis finished successfully')
    rospy.loginfo('Traversability rate: ' + str(trav_rate) + '% (' + str(n_traversable) + '/' + str(count) + ')')


def empiric_eval():
    rospy.init_node('empiric_eval')

    filepath = rospy.get_param('filepath', '/home/anatoliy/cloud.ply')
    datapath = rospy.get_param('datapath', 'src/waypoint_publisher/dataset/data_marsyard2020.csv')
    cell_size = rospy.get_param('cell_size', 1.0)
    get_topic = rospy.get_param('gazebo_get_topic','/gazebo/get_model_state')
    set_topic = rospy.get_param('gazebo_set_topic','/gazebo/set_model_state')
    vel_topic = rospy.get_param('velocity_topic','/cmd_vel')
    robot_name = rospy.get_param('robot_name','leo')
    frame_id = rospy.get_param('frame_id','ground_plane')
    v = rospy.get_param('velocity', 2.0)
    save_data = rospy.get_param('save_data', True)

    rospy.loginfo('Empiric eval node is running')
    cloud, points_number = read_cloud(filepath)
    grid_size_x, grid_size_y, xmin, xmax, ymin, ymax, zmin, zmax = evaluate_cloud(cloud, cell_size)
    offset = [xmin, ymin]
    point_grid = segmentate_cloud(cloud, grid_size_x, grid_size_y, xmin, xmax, ymin, ymax, cell_size)
    eval_traversability(point_grid, grid_size_x, grid_size_y, cell_size, offset, v, get_topic, set_topic, vel_topic, robot_name, frame_id, save_data, datapath)


try:
    empiric_eval()
except rospy.ROSInterruptException:
    rospy.loginfo('Unexpected ROSInterruptException')