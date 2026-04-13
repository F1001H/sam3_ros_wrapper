import rospy
import cv2
from sensor_msgs.msg import Image as ROSImage
from cv_bridge import CvBridge
import multiprocessing as mp

class ROSInputProcess(mp.Process):
    def __init__(self, image_queue):
        super().__init__()
        self.image_queue = image_queue

    def callback(self, msg):
        bridge = CvBridge()
        # Convert to BGR then to RGB for SAM-3
        cv_img = bridge.imgmsg_to_cv2(msg, "bgr8")
        rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        
        # Don't let the queue bloat; only keep the freshest frame
        if self.image_queue.full():
            try: self.image_queue.get_nowait()
            except: pass
        self.image_queue.put(rgb_img)

    def run(self):
        rospy.init_node('sam3_input_node')
        rospy.Subscriber("/zedxm_left/zed_node/rgb/image_rect_color", ROSImage, self.callback, queue_size=1)
        rospy.spin()