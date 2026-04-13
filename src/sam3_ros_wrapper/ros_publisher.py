import rospy
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from sensor_msgs.msg import Image as ROSImage
from cv_bridge import CvBridge
import multiprocessing as mp

class ROSOutputProcess(mp.Process):
    def __init__(self, output_queue):
        super().__init__()
        self.output_queue = output_queue

    def run(self):
        rospy.init_node('sam3_output_node')
        pub = rospy.Publisher("/sam3/visualization", ROSImage, queue_size=1)
        bridge = CvBridge()

        while not rospy.is_shutdown():
            rgb_frame, masks_tensor = self.output_queue.get()
            h, w = rgb_frame.shape[:2]

            if masks_tensor.numel() > 0:
                # 5090 Resizing Logic
                if masks_tensor.shape[-2:] != (h, w):
                    masks_tensor = F.interpolate(masks_tensor.float(), size=(h, w), mode='nearest')
                
                mask_np = masks_tensor.squeeze().cpu().numpy().astype(bool)
                
                # Apply green overlay (RGB format here)
                rgb_frame[mask_np] = [0, 255, 0] 
            
            # Convert RGB back to BGR for ROS/OpenCV standard
            bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
            pub.publish(bridge.cv2_to_imgmsg(bgr_frame, "bgr8"))