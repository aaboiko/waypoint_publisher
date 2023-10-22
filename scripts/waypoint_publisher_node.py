import rospy, actionlib
import waypoint_publisher.msg as msg
import numpy as np

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

def callback_get_path_feedback(feedback):
    rospy.loginfo("GetPath server feedback:%s" % str(feedback))


def main():
    rospy.init_node('waypoint_publisher')
    start = msg.Point2i(x=0, y=0)
    dest = msg.Point2i(x=10, y=10)
    cell_size = 1.0
    koefs = [0.0, 0.0, 0.0]

    goal = msg.GetPathGoal(start=start, destination=dest, cell_size=cell_size, koefs=koefs)
    get_path_client(goal)
    
    #rospy.spin()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        rospy.loginfo('Unexpected ROSInterruptException')
