import rospy, actionlib
import waypoint_publisher.msg as msg
import numpy as np
from geometry_msgs.msg import Twist, Pose
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import GetModelState, SetModelState
from tf.transformations import euler_from_quaternion
import timeit

def add_orientation_to_path(path):
    for i in range(1, len(path)):
        dx = path[i][0] - path[i - 1][0]
        dy = path[i][1] - path[i - 1][1]

        if dx == -1:
            if dy == -1:
                path[i - 1].append(5)
            elif dy == 0:
                path[i - 1].append(4)
            else:
                path[i - 1].append(3)
        elif dx == 0:
            if dy == -1:
                path[i - 1].append(6)
            else:
                path[i - 1].append(2)
        else:
            if dy == -1:
                path[i - 1].append(7)
            elif dy == 0:
                path[i - 1].append(0)
            else:
                path[i - 1].append(1)

    path[i].append(path[i - 1][2])
    return path


def get_pose_from_gazebo(topic, robot_name, frame_id):
    rospy.wait_for_service(topic)
    state = GetModelState()

    try:
        get_model_state = rospy.ServiceProxy(topic, GetModelState)
        state = get_model_state(robot_name, frame_id)
    except:
        rospy.logerr('Error: get state service unavailable')

    return state


def reset_pose_gazebo(gazebo_set_topic, robot_name, pose):
    x, y, z = pose
    state_msg = ModelState()
    state_msg.model_name = robot_name
    state_msg.pose.position.x = x
    state_msg.pose.position.y = y
    state_msg.pose.position.z = z
    state_msg.pose.orientation.x = 0
    state_msg.pose.orientation.y = 0
    state_msg.pose.orientation.z = 0
    state_msg.pose.orientation.w = 0

    rospy.wait_for_service(gazebo_set_topic)
    try:
        set_state = rospy.ServiceProxy(gazebo_set_topic, SetModelState)
        result = set_state(state_msg)
        rospy.loginfo('Reset state server responded: ', result)
    except:
        rospy.logerr('Error: set state server unavailable')


def move_to_goal(velocity_topic, gazebo_get_topic, cell_size, robot_name, frame_id, traj, vel):
    rate = rospy.Rate(10)
    vel_pub = rospy.Publisher(velocity_topic, Twist, queue_size=10)

    speed = Twist()

    def goal_reached(pose, cell_size):
        state = get_pose_from_gazebo(gazebo_get_topic, robot_name, frame_id)
        cur_x = state.pose.position.x
        cur_y = state.pose.position.y
        curr_p = np.array([cur_x, cur_y])
        curr_goal = np.array([pose[0], pose[1]])
        delta = np.linalg.norm(curr_p - curr_goal)

        if delta < cell_size / 2:
            rospy.loginfo('Goal reached: ' + str(curr_goal))

        return delta < cell_size / 2

    for pose in traj:
        timestamp = timeit.default_timer()
        while not goal_reached(pose, cell_size):
            cur_time = timeit.default_timer()
            dt = cur_time - timestamp
            if dt > 10.0 * np.sqrt(2) * cell_size / vel:
                rospy.loginfo('Goal ' + str(pose) + ' is not traversable. The path is invalid')
                rospy.loginfo('Time to try: ' + str(dt) + ' secs')
                return False
            
            state = get_pose_from_gazebo(gazebo_get_topic, robot_name, frame_id)
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
                speed.linear.x = vel

            vel_pub.publish(speed)
            rate.sleep()  

    return True

def get_path_client(goal):
    client = actionlib.SimpleActionClient('pointcloud_processor', msg.GetPathAction)
    client.wait_for_server()
    client.send_goal(goal, active_cb=callback_get_path_active,
                    feedback_cb=callback_get_path_feedback,
                    done_cb=callback_get_path_done)
    
    client.wait_for_result()
    return client.get_result()


def callback_get_path_active():
    rospy.loginfo("GetPath server is processing the goal")

def callback_get_path_done(state, result):
    rospy.loginfo("GetPath server is done. State: %s, result: %s" % (str(state), str(result)))
    success = result.success
    xmin = float(result.xmin)
    xmax = float(result.xmax)
    ymin = float(result.ymin)
    ymax = float(result.ymax)

    path = []
    for i in result.path:
        x = float(i.x)
        y = float(i.y)
        point = np.array([x, y])
        path.append(point)

    pass


def callback_get_path_feedback(feedback):
    rospy.loginfo("GetPath server feedback:%s" % str(feedback))


def main():
    rospy.init_node('waypoint_publisher')

    velocity_topic = rospy.get_param('velocity_topic','/cmd_vel')
    gazebo_get_topic = rospy.get_param('gazebo_get_topic','/gazebo/get_model_state')
    gazebo_set_topic = rospy.get_param('gazebo_set_topic','/gazebo/set_model_state')
    robot_name = rospy.get_param('robot_name','leo')
    frame_id = rospy.get_param('frame_id','ground_plane')

    state = get_pose_from_gazebo(gazebo_get_topic, robot_name, frame_id)
    x = state.pose.position.x
    y = state.pose.position.y
    start = msg.Point2f(x=x, y=y)

    dest = msg.Point2f(x=10.0, y=10.0)
    offset = msg.Point2f(x=0.0, y=0.0)
    cell_size = 1.0
    koefs = [0.0, 0.0, 0.0]

    goal = msg.GetPathGoal(start=start, destination=dest, offset=offset, cell_size=cell_size, koefs=koefs)
    result = get_path_client(goal)

    vel = 1.0

    grid_path = [[0, 0], [2, 0], [4, 0], [6, 2], [6, 4], [4, 4]]
    #traversable = move_to_goal(velocity_topic, gazebo_get_topic, cell_size, robot_name, frame_id, grid_path, vel)
    
    
    #pose = [0.5, 0.0, 1.0]
    #reset_pose_gazebo(gazebo_set_topic, robot_name, pose)

    #rospy.spin()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        rospy.loginfo('Unexpected ROSInterruptException')
