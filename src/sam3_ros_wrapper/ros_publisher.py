import rospy
import cv2
import numpy as np
import multiprocessing as mp
from sensor_msgs.msg import Image as ROSImage
from cv_bridge import CvBridge

class ROSOutputProcess(mp.Process):
    def __init__(self, output_queue):
        super().__init__()
        self.output_queue = output_queue

    def run(self):
        # Note: No 'import torch' here. We keep this process lightweight.
        rospy.init_node('sam3_output_node', anonymous=True)
        
        mask_pub = rospy.Publisher("/sam3/mask", ROSImage, queue_size=1)
        viz_pub = rospy.Publisher("/sam3/visualization", ROSImage, queue_size=1)
        
        bridge = CvBridge()

        while not rospy.is_shutdown():
            # Get data from queue (now receiving NumPy array for masks)
            # Expected shape from SAM3: (B, 1, H, W) or (1, H, W)
            rgb_frame, mask_array = self.output_queue.get()
            
            if rgb_frame is None: 
                break
                
            h, w = rgb_frame.shape[:2]

            # 1. Handle the mask if it exists
            if mask_array is not None and mask_array.size > 0:
                # Squeeze to get (H, W)
                mask_2d = np.squeeze(mask_array)

                # 2. Match resolution if necessary (Resize in NumPy)
                if mask_2d.shape != (h, w):
                    mask_2d = cv2.resize(mask_2d.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
                
                # 3. Finalize Mask (Boolean to 0-255 uint8)
                mask_bool = mask_2d > 0
                mask_uint8 = (mask_bool.astype(np.uint8)) * 255

                # 4. Create Visualization (Overlay)
                # Keep original RGB for the viz, just flip channels for BGR output
                viz_frame = rgb_frame.copy()
                viz_frame[mask_bool] = [0, 255, 0] # Apply green mask
                viz_frame_bgr = cv2.cvtColor(viz_frame, cv2.COLOR_RGB2BGR)

                # 5. Publish
                try:
                    # Publish standard mask for FoundationPose
                    mask_pub.publish(bridge.cv2_to_imgmsg(mask_uint8, encoding="mono8"))
                    # Publish visual feedback
                    viz_pub.publish(bridge.cv2_to_imgmsg(viz_frame_bgr, encoding="bgr8"))
                except Exception as e:
                    rospy.logerr(f"Publishing error: {e}")

            else:
                # Fallback for no detection
                empty_mask = np.zeros((h, w), dtype=np.uint8)
                mask_pub.publish(bridge.cv2_to_imgmsg(empty_mask, encoding="mono8"))