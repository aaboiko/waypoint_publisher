import rospy, actionlib
import waypoint_publisher.msg as msg
import numpy as np
from geometry_msgs.msg import Twist, Pose
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import GetModelState, SetModelState
from tf.transformations import euler_from_quaternion

def grid_to_float(point, offset, cell_size):
    x = cell_size * point[0] + cell_size / 2 + offset[0]
    y = cell_size * point[1] + cell_size / 2 + offset[1]
    theta = (np.pi / 4) * point[2]
    return [x, y, theta]


def float_to_grid(point, offset, cell_size):
    x = int((point[0] - offset[0]) / cell_size)
    y = int((point[1] - offset[1]) / cell_size)
    return [x, y]


def get_pose_from_gazebo(topic, robot_name, frame_id):
    rospy.wait_for_service(topic)
    state = GetModelState()

    try:
        get_model_state = rospy.ServiceProxy(topic, GetModelState)
        state = get_model_state(robot_name, frame_id)
    except:
        rospy.logerr('Error: get state service unavailable')

    return state


def reset_pose_gazebo(gazebo_set_topic, robot_name):
    state_msg = ModelState()
    state_msg.model_name = robot_name
    state_msg.pose.position.x = 0
    state_msg.pose.position.y = 0
    state_msg.pose.position.z = 0
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


def get_path_client(goal):
    client = actionlib.SimpleActionClient('pointcloud_processor', msg.GetPathAction)
    client.wait_for_server()
    client.send_goal(goal, active_cb=callback_get_path_active,
                    feedback_cb=callback_get_path_feedback,
                    done_cb=callback_get_path_done)
    
    client.wait_for_result()


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
        x = int(i.x)
        y = int(i.y)
        point = np.array([x, y])
        path.append(point)


def callback_get_path_feedback(feedback):
    rospy.loginfo("GetPath server feedback:%s" % str(feedback))


def main():
    rospy.init_node('waypoint_publisher')

    velocity_topic = rospy.get_param('velocity_topic','/mobile_base/commands/velocity')
    gazebo_get_topic = rospy.get_param('gazebo_get_topic','/gazebo/get_model_state')
    gazebo_set_topic = rospy.get_param('gazebo_set_topic','/gazebo/set_model_state')
    robot_name = rospy.get_param('robot_name','mobile_base')
    frame_id = rospy.get_param('frame_id','ground_plane')

    start = msg.Point2i(x=0, y=0)
    dest = msg.Point2i(x=10, y=10)
    cell_size = 1.0
    koefs = [0.0, 0.0, 0.0]

    goal = msg.GetPathGoal(start=start, destination=dest, cell_size=cell_size, koefs=koefs)
    #get_path_client(goal)

    rate = rospy.Rate(10)
    vel_pub = rospy.Publisher(velocity_topic, Twist, queue_size=10)
    path = [[1, 0], [2, 0], [3, 1], [3, 2], [2, 2]]
    init_pose = [0, 0]
    
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

    traj = []
    offset = [-17.0, -22.0]
    for pose in path:
        p = grid_to_float(pose, offset, cell_size)
        traj.append(p)

    speed = Twist()


    while not rospy.is_shutdown():
        state = get_pose_from_gazebo(gazebo_get_topic, robot_name, frame_id)
        x = state.pose.position.y
        y = state.pose.position.x
        rot = state.pose.orientation
        (roll, pitch, theta) = euler_from_quaternion([rot.x, rot.y, rot.z, rot.w])

        if abs(x - dest.x) < 0.1 and abs(y - dest.y < 0.1):
            print('Goal reached')
            break
        else:
            print('To the goal: ' + str(x - dest.x) + ' ' + str(y - dest.y))

        inc_x = dest.x - x
        inc_y = dest.y - y

        angle_to_goal = np.arctan2(inc_y, inc_x)

        if abs(angle_to_goal - theta) > 0.1:
            speed.linear.x = 0.0
            speed.angular.z = 0.3
        else:
            speed.linear.x = 0.5
            speed.angular.z = 0.0

        vel_pub.publish(speed)
        rate.sleep()  

    reset_pose_gazebo(gazebo_set_topic, robot_name)

    #rospy.spin()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        rospy.loginfo('Unexpected ROSInterruptException')
