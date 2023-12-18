import rospy, actionlib
import numpy as np
import open3d as pcl
from geometry_msgs.msg import Twist, Pose
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import GetModelState, SetModelState
from sensor_msgs.msg import PointCloud2
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from numpy.linalg import inv
from csv import writer
from sklearn.linear_model import LogisticRegression
import timeit, time
import ros_numpy
import pickle

class PointcloudHandler:
    def __init__(self, params):
        self.model_path =       params[0]
        self.cell_size =        params[1]
        self.get_topic =        params[2]
        self.set_topic =        params[3]
        self.vel_topic =        params[4]
        self.robot_name =       params[5]
        self.frame_id =         params[6]
        self.v =                params[7]
        self.save_data =        params[8]
        self.visualize_cloud =  params[9]
        self.wait =             params[10]
        self.cloudpath =        params[11]
        self.save_cloud =       params[12]
        self.mappath =          params[13]
        self.save_inc_costmap = params[14]
        self.world_x_min =      params[15]
        self.world_x_max =      params[16]
        self.world_y_min =      params[17]
        self.world_y_max =      params[18]
        self.depth_topic =      params[19]

        self.cloud_index = 0
        self.model = pickle.load(open(self.model_path, 'rb'))
        rospy.loginfo('depth_topic subscriber started')

        self.eval_traversability()
        rospy.spin()


    def evaluate_cloud(self, cloud):
        xmax, xmin, ymax, ymin, zmax, zmin = -np.inf, np.inf, -np.inf, np.inf, -np.inf, np.inf
        
        for point in cloud:
            x, z, y = point
            xmax = max(xmax, x)
            xmin = min(xmin, x)
            ymax = max(ymax, y)
            ymin = min(ymin, y)
            zmax = max(zmax, z)
            zmin = min(zmin, z)

        rospy.loginfo('xmin: %f, xmax: %f, ymin: %f, ymax: %f, zmin: %f, zmax: %f',xmin,xmax,ymin,ymax,zmin,zmax)

        return xmin, xmax, ymin, ymax, zmin, zmax


    def save_cloud_segment(self, cloud, cloudpath):
        rospy.loginfo('Starting saving cloud...')
        path = self.cloudpath + 'segment_' + str(self.cloud_index) + '.csv'
        self.cloud_index += 1

        with open(path, 'a') as f_obj:
                    if len(cloud) > 0:
                        for point in cloud:
                            x, y, z = point
                            line = [float(x), float(y), float(z)]
                            obj = writer(f_obj)
                            obj.writerow(line)

                    f_obj.close()

        rospy.loginfo('Cloud saved successfully')


    def get_segment(self, cloud):
        res = []
        s = cloud.shape[0]
        iter = 0.0
        progress = 0
        prev = 0
        rospy.loginfo('Cloud reduction started...')

        for point in cloud:
            x, z, y = point
            if abs(x) <= self.cell_size / 2 and y <= self.cell_size + 0.5:
                res.append(np.array([x, y, z]))

            iter += 1.0
            progress = int((iter / s) * 100)
            if progress != prev:
                rospy.loginfo('Progress: ' + str(progress) + '%')
                prev = progress
        
        rospy.loginfo('Cloud reduced. Number of points: ' + str(len(res)))
        return res


    def get_plane_inclination(self, segment):
        rospy.loginfo('Get plane inclination started')
        n = len(segment)
        big = 10e6

        xmax = -np.inf
        xmin = np.inf
        ymax = -np.inf
        ymin = np.inf
        zmax = -np.inf
        zmin = np.inf

        if n < 3:
            return big, big, 0, big, big

        sign = -1.0
        A = np.zeros((n, 3))
        B = sign * np.ones((n, 1))

        for i in range(n):
            x = segment[i][0]
            y = segment[i][1]
            z = segment[i][2]
            A[i,:] = np.array([x, y, z])
            xmax = max(xmax, x)
            xmin = min(xmin, x)
            ymax = max(ymax, y)
            ymin = min(ymin, y)
            zmax = max(zmax, z)
            zmin = min(zmin, z)

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
        z_range_param = zmax - zmin
        
        if abs(X[2]) <= 10e-3:
            return big, sigma, n, z_range_param, zmax

        inc_cos = abs(X[2])
        inc_param = np.sqrt(1.0 - inc_cos**2) / inc_cos

        rospy.loginfo('Get plane inclination finished')
        rospy.loginfo('xmin = '+str(xmin)+' xmax = '+str(xmax))
        rospy.loginfo('ymin = '+str(ymin)+' ymax = '+str(ymax))
        rospy.loginfo('zmin = '+str(zmin)+' zmax = '+str(zmax))
        return inc_param, sigma, n, z_range_param, zmax
    

    def get_pose(self):
        rospy.wait_for_service(self.get_topic)
        state = GetModelState()

        try:
            get_model_state = rospy.ServiceProxy(self.get_topic, GetModelState)
            state = get_model_state(self.robot_name, self.frame_id)
        except:
            rospy.logerr('Error: get state service unavailable')

        return state


    def set_pose(self, pose):
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

        rospy.wait_for_service(self.set_topic)
        try:
            set_state = rospy.ServiceProxy(self.set_topic, SetModelState)
            result = set_state(state_msg)
            rospy.loginfo('New pose is set successfully: ' + str(pose))
        except:
            rospy.logerr('Error: gazebo server unavailable')


    def move_to(self, pose, k):
        rate = rospy.Rate(10)
        vel_pub = rospy.Publisher(self.vel_topic, Twist, queue_size=10)
        speed = Twist()

        timestamp = timeit.default_timer()
        while True:
            cur_time = timeit.default_timer()
            dt = cur_time - timestamp
            if dt > self.wait:
                rospy.loginfo('Goal ' + str(pose) + ' is not traversable')
                return False
            
            state = self.get_pose()
            x = state.pose.position.x
            y = state.pose.position.y
            z = state.pose.position.z

            '''if z < -10:
                rospy.loginfo('Fallen!')
                return False'''
            
            x_goal = pose[0]
            y_goal = pose[1]
            curr_goal = np.array([x_goal, y_goal])

            print(' x = ' + str(x) + ' y = ' + str(y) + ' x_goal = '+str(x_goal)+' y_goal = '+str(y_goal))

            '''delta = np.array([x - x_goal, y - y_goal])
            if np.linalg.norm(delta) <= 0.5:
                rospy.loginfo('Goal reached: ' + str(curr_goal))
                break'''
            
            cond_0 = k == 0 and x >= x_goal
            cond_1 = k == 1 and x >= x_goal and y >= y_goal
            cond_2 = k == 2 and y >= y_goal
            cond_3 = k == 3 and x <= x_goal and y >= y_goal
            cond_4 = k == 4 and x <= x_goal
            cond_5 = k == 5 and x <= x_goal and y <= y_goal
            cond_6 = k == 6 and y <= y_goal
            cond_7 = k == 7 and x >= x_goal and y <= y_goal

            if cond_0 or cond_1 or cond_2 or cond_3 or cond_4 or cond_5 or cond_6 or cond_7:
                rospy.loginfo('Goal reached: ' + str(curr_goal))
                speed.linear.x = 0
                speed.angular.z = 0
                vel_pub.publish(speed)
                break

            speed.linear.x = self.v
            speed.angular.z = 0

            vel_pub.publish(speed)
            #rate.sleep() 

        return True


    def eval_traversability(self):
        rospy.loginfo('Starting traversability analysis to validate model...')
        size_x = self.world_x_max - self.world_x_min
        size_y = self.world_y_max - self.world_y_min
        iter = 0.0
        progress = 0
        prev = 0
        count = (size_x + 1) * (size_y + 1) * 8
        rate = 0

        for i in range(self.world_x_min, self.world_x_max + 1, int(self.cell_size)):
            for j in range(self.world_y_min, self.world_y_max + 1, int(self.cell_size)):
                for k in range(8):
                    angle = k * np.pi / 4
                    x, y, z, w = quaternion_from_euler(0.0, 0.0, angle)
                    pose = [i, j, 0.1, x, y, z, w]
                    self.set_pose(pose)

                    rospy.loginfo('Waiting for cloud...')
                    pointcloud2_msg = rospy.wait_for_message(self.depth_topic, PointCloud2, timeout=5)
                    self.pointcloud = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(pointcloud2_msg)
                    rospy.loginfo('Poincloud obtained. Number of points: ' + str(self.pointcloud.shape[0]))

                    segment = self.get_segment(self.pointcloud)

                    if self.save_cloud:
                        self.save_cloud_segment(segment, self.cloudpath)

                    if self.visualize_cloud:
                        frame0 = pcl.geometry.TriangleMesh.create_coordinate_frame(size=0.6, origin=[0, 0, 0])
                        pcl.visualization.draw_geometries([pointcloud2_msg, frame0])

                    inc_param, sigma, n_points, z_range_param, zmax = self.get_plane_inclination(segment)
                    iter += 1.0
                    progress = int((iter / count) * 100)

                    if n_points < 3:
                        rospy.loginfo('Pointcloud contains < 3 points')
                        continue

                    x_dest = i + (0.5 + self.cell_size) * np.cos(angle)
                    y_dest = j + (0.5 + self.cell_size) * np.sin(angle)
                    dest = [x_dest, y_dest]
                    
                    res = self.move_to(dest, k)
                    x_test = np.array([[float(inc_param), float(sigma), float(z_range_param)]])
                    y_test = int(self.model.predict(x_test)[0])

                    if res == y_test:
                        rospy.loginfo('Current cell is predicted correctly!')
                        rate += 1
                    else:
                        rospy.loginfo('Model prediction is incorrect')

                    if progress != prev:
                        rospy.loginfo('Validation in progress: [' + str(progress) + '%]')
                        prev = progress

        score = int((rate / count) * 100)
        rospy.loginfo('Model validation finished successfully')
        rospy.loginfo('Score: ' + str(score) + '% (' + str(rate) + '/' + str(count) + ')')
    

def validate():
    rospy.init_node('validate', anonymous=True)
    rospy.loginfo('validate node init')

    model_path = rospy.get_param('model_path', 'src/waypoint_publisher/models/house_1.sav')
    cell_size = rospy.get_param('cell_size', 1.0)
    get_topic = rospy.get_param('gazebo_get_topic','/gazebo/get_model_state')
    set_topic = rospy.get_param('gazebo_set_topic','/gazebo/set_model_state')
    vel_topic = rospy.get_param('velocity_topic','/mobile_base/commands/velocity')
    depth_topic = rospy.get_param('depth_topic', '/camera/depth/points')

    robot_name = rospy.get_param('robot_name','mobile_base')
    frame_id = rospy.get_param('frame_id','ground_plane')
    v = rospy.get_param('velocity', 5.0)
    save_data = rospy.get_param('save_data', True)
    visualize_cloud = rospy.get_param('visualize_cloud', False)
    wait = rospy.get_param('wait', 10.0)

    cloudpath = rospy.get_param('cloudpath', 'src/waypoint_publisher/clouds_realtime/')
    save_cloud = rospy.get_param('save_cloud', False)
    mappath = rospy.get_param('mappath', 'src/waypoint_publisher/maps_realtime/map_cloud_house.txt')
    save_inc_costmap = rospy.get_param('save_inc_costmap', False)

    world_x_min = rospy.get_param('world_x_min', -10)
    world_x_max = rospy.get_param('world_x_max', 10)
    world_y_min = rospy.get_param('world_y_min', -10)
    world_y_max = rospy.get_param('world_y_max', 10)

    params = [  model_path,
                cell_size,
                get_topic,
                set_topic,
                vel_topic,
                robot_name,
                frame_id,
                v,
                save_data,
                visualize_cloud,
                wait,
                cloudpath,
                save_cloud,
                mappath,
                save_inc_costmap,
                world_x_min,
                world_x_max,
                world_y_min,
                world_y_max,
                depth_topic]
    
    handler = PointcloudHandler(params)


try:
    validate()
except rospy.ROSInterruptException:
    rospy.loginfo('Unexpected ROSInterruptException')