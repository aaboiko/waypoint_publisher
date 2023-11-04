import rospy, actionlib
import numpy as np
import waypoint_publisher.msg as msg

def get_path_with_koefs(goal):
    client = actionlib.SimpleActionClient('pointcloud_processor', msg.GetPathAction)
    client.wait_for_server()
    client.send_goal(goal, active_cb=callback_get_path_with_koefs_active,
                    feedback_cb=callback_get_path_with_koefs_feedback,
                    done_cb=callback_get_path_with_koefs_done)
    
    client.wait_for_result()
    return client.get_result()


def callback_get_path_with_koefs_active():
    rospy.loginfo('Getting path with koefs in process...')


def callback_get_path_with_koefs_done(state, result):
    rospy.loginfo("GetPath server is done")
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

    return success


def callback_get_path_with_koefs_feedback(feedback):
    rospy.loginfo("GetPath server feedback:%s" % str(feedback))


def run_fit_model_node():
    rospy.init_node('fit_model')
    
    filename_koefs = 'src/waypoint_publisher/koefs/koefs.txt'
    koefs = np.loadtxt(filename_koefs)
    offset = msg.Point2f(x=-7.990018, y=-11.153976)
    cell_size = 0.050000
    start = msg.Point2f(x=0.0, y=0.0)
    dest = msg.Point2f(x=-5.0, y=-5.0)

    goal = msg.GetPathGoal(start=start, destination=dest, offset=offset, cell_size=cell_size, koefs=koefs)
    result = get_path_with_koefs(goal)
    print('Result')
    print(result)
    

try:
    run_fit_model_node()
except rospy.ROSInterruptException:
    rospy.loginfo('Unexpected ROSInterruptException')

