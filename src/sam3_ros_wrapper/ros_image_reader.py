import multiprocessing as mp
import cv2
import sys
import os

class ROSInputProcess(mp.Process):
    def __init__(self, input_q):
        super().__init__()
        self.input_q = input_q

    def run(self):
        # 1. HARD OVERRIDE: Strip ROS launch arguments
        sys.argv = [sys.argv[0]] # Completely wipe launch-injected names/args
        
        import rospy
        from sensor_msgs.msg import Image as ROSImage
        from cv_bridge import CvBridge

        # Now it's impossible for it to conflict with 'sam3_pipeline_wrapper'
        rospy.init_node('sam3_reader_worker', anonymous=True)
        bridge = CvBridge()

        def cb(msg):
            # ZED is BGR, SAM-3 needs RGB
            cv_img = bridge.imgmsg_to_cv2(msg, "bgr8")
            rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            if not self.input_q.full():
                self.input_q.put(rgb_img)

        rospy.Subscriber("/zedxm_left/zed_node/rgb/image_rect_color", ROSImage, cb, queue_size=1)
        rospy.spin()