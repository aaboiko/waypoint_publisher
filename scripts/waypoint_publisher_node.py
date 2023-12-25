import rospy, actionlib
import waypoint_publisher.msg as msg
import numpy as np
from geometry_msgs.msg import Twist, Pose
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import GetModelState, SetModelState
from tf.transformations import euler_from_quaternion, quaternion_from_euler
import timeit
import pickle

class WaypointHandler:
    def __init__(self, params):
        self.velocity_topic   = params[0]
        self.gazebo_get_topic = params[1]
        self.gazebo_set_topic = params[2]
        self.robot_name       = params[3]
        self.frame_id         = params[4]
        self.v                = params[5]
        self.cell_size        = params[6]
        self.model_path       = params[7]
        self.x_init           = params[8]
        self.y_init           = params[9]
        self.heading_init     = params[10]
        self.x_dest           = params[11]
        self.y_dest           = params[12]
        self.heading_dest     = params[13]
        
        x, y, z, w = quaternion_from_euler(0.0, 0.0, self.heading_init)
        pose_init = [self.x_init, self.y_init, 0.1, x, y, z, w]
        #self.reset_pose_gazebo(pose_init)

        start = msg.Point2f(x = self.x_init, y = self.y_init)
        dest = msg.Point2f(x = self.x_dest, y = self.y_dest)
        offset = msg.Point2f(x = 0.0, y = 0.0)
        model = pickle.load(open(self.model_path, 'rb'))
        koefs = [model.coef_[0,0], model.coef_[0,1], model.coef_[0,2]]

        goal = msg.GetPathGoal(start=start, destination=dest, offset=offset, cell_size=self.cell_size, koefs=koefs)
        
        result = self.get_path_client(goal)
        print(result)

        #traversable = self.move_to_goal(self.velocity_topic, self.gazebo_get_topic, self.cell_size, robot_name, frame_id, grid_path, v)

        rospy.spin()


    def add_orientation_to_path(self, path):
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


    def get_pose_from_gazebo(self):
        rospy.wait_for_service(self.gazebo_get_topic)
        state = GetModelState()

        try:
            get_model_state = rospy.ServiceProxy(self.gazebo_get_topic, GetModelState)
            state = get_model_state(self.robot_name, self.frame_id)
        except:
            rospy.logerr('Error: get state service unavailable')

        return state


    def reset_pose_gazebo(self, pose):
        x, y, z, qx, qy, qz, qw = pose
            
        state_msg = ModelState()
        state_msg.model_name = self.robot_name
        state_msg.pose.position.x = x
        state_msg.pose.position.y = y
        state_msg.pose.position.z = z
        state_msg.pose.orientation.x = qx
        state_msg.pose.orientation.y = qy
        state_msg.pose.orientation.z = qz
        state_msg.pose.orientation.w = qw

        rospy.wait_for_service(self.gazebo_set_topic)
        try:
            set_state = rospy.ServiceProxy(self.gazebo_set_topic, SetModelState)
            result = set_state(state_msg)
            rospy.loginfo('New pose is set successfully: ' + str(pose))
        except:
            rospy.logerr('Error: gazebo server unavailable')


    def move_to_goal(self, traj):
        rate = rospy.Rate(10)
        vel_pub = rospy.Publisher(self.velocity_topic, Twist, queue_size=10)

        speed = Twist()

        def goal_reached(pose):
            state = self.get_pose_from_gazebo()
            cur_x = state.pose.position.x
            cur_y = state.pose.position.y
            curr_p = np.array([cur_x, cur_y])
            curr_goal = np.array([pose[0], pose[1]])
            delta = np.linalg.norm(curr_p - curr_goal)

            if delta < self.cell_size / 2:
                rospy.loginfo('Goal reached: ' + str(curr_goal))

            return delta < self.cell_size / 2

        for pose in traj:
            timestamp = timeit.default_timer()
            while not goal_reached(pose):
                cur_time = timeit.default_timer()
                dt = cur_time - timestamp
                if dt > 10.0 * np.sqrt(2) * self.cell_size / self.v:
                    rospy.loginfo('Goal ' + str(pose) + ' is not traversable. The path is invalid')
                    rospy.loginfo('Time to try: ' + str(dt) + ' secs')
                    return False
                
                state = self.get_pose_from_gazebo()
                x = state.pose.position.x
                y = state.pose.position.y
                rot = state.pose.orientation
                (roll, pitch, theta) = euler_from_quaternion([rot.x, rot.y, rot.z, rot.w])

                inc_x = pose[0] - x
                inc_y = pose[1] - y

                angle_to_goal = np.arctan2(inc_y, inc_x)
                inc_angle = angle_to_goal - theta
                k = 2.0
                speed.angular.z = k * self.cell_size * inc_angle
                
                if abs(inc_angle) > 0.1:
                    speed.linear.x = 0.0
                else:
                    speed.linear.x = self.v

                vel_pub.publish(speed)
                rate.sleep()  

        return True


    def get_path_client(self, goal):
        print('get_path_client() invoked')
        client = actionlib.SimpleActionClient('pointcloud_processor', msg.GetPathAction)
        rospy.loginfo('Waiting for action server response...')
        client.wait_for_server()

        rospy.loginfo('Sending goal...')
        client.send_goal(goal, active_cb = self.callback_get_path_active,
                        feedback_cb = self.callback_get_path_feedback,
                        done_cb = self.callback_get_path_done)
        
        rospy.loginfo('Goal sent. Waiting for result..')
        client.wait_for_result()
        rospy.loginfo('result obtained')
        return client.get_result()


    def callback_get_path_active(self):
        rospy.loginfo("GetPath server is processing the goal")


    def callback_get_path_done(self, state, result):
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

        if success:
            rospy.loginfo('The path is built successfully -> move_to_goal() invoking...')
            self.move_to_goal(path)
        else:
            rospy.loginfo('Cannot invoke move_to_goal. No corrent path found')


    def callback_get_path_feedback(self, feedback):
        rospy.loginfo("GetPath server feedback:%s" % str(feedback))


def main():
    rospy.init_node('waypoint_publisher')

    velocity_topic = rospy.get_param('velocity_topic','/mobile_base/commands/velocity')
    gazebo_get_topic = rospy.get_param('gazebo_get_topic','/gazebo/get_model_state')
    gazebo_set_topic = rospy.get_param('gazebo_set_topic','/gazebo/set_model_state')
    robot_name = rospy.get_param('robot_name','mobile_base')
    frame_id = rospy.get_param('frame_id','ground_plane')
    v = rospy.get_param('velocity', 1.0)
    cell_size = rospy.get_param('cell_size', 0.5)
    model_path = rospy.get_param('model_path', 'src/waypoint_publisher/models/house_1.sav')

    x_init = rospy.get_param('x_init', 0.0)
    y_init = rospy.get_param('y_init', 0.0)
    heading_init = rospy.get_param('heading_init', 0)

    x_dest = rospy.get_param('x_dest', 2.0)
    y_dest = rospy.get_param('y_dest', 5.0)
    heading_dest = rospy.get_param('heading_dest', 0)

    params = [
        velocity_topic,
        gazebo_get_topic,
        gazebo_set_topic,
        robot_name,
        frame_id,
        v,
        cell_size,
        model_path,
        x_init,
        y_init,
        heading_init,
        x_dest,
        y_dest,
        heading_dest
    ]

    handler = WaypointHandler(params)


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        rospy.loginfo('Unexpected ROSInterruptException')
