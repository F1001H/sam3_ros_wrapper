import multiprocessing as mp
import cv2
import numpy as np
import sys

class ROSOutputProcess(mp.Process):
    def __init__(self, output_q):
        super().__init__()
        self.output_q = output_q

    def run(self):
        # 1. Block ROS launch arguments to prevent the 'New node registered' kill
        sys.argv = [sys.argv[0]]
        
        import rospy
        from sensor_msgs.msg import Image as ROSImage
        from cv_bridge import CvBridge

        rospy.init_node('sam3_publisher_worker', anonymous=True)
        bridge = CvBridge()
        m_pub = rospy.Publisher("/sam3/mask", ROSImage, queue_size=1)
        v_pub = rospy.Publisher("/sam3/viz", ROSImage, queue_size=1)

        print("[Publisher] Waiting for first valid frame from SAM...")

        while not rospy.is_shutdown():
            try:
                # 2. Block here until Inference sends something
                data = self.output_q.get()
                print('frame arrived')
                
                # 3. Validation: If SAM isn't ready, it might send None or empty
                if data is None or not isinstance(data, tuple):
                    continue
                
                rgb, mask = data
                if rgb is None or mask is None:
                    continue

                # --- THE NUMPY FIX ---
                # Ensure mask is 2D (H, W)
                mask_2d = np.squeeze(mask)
                
                # Threshold to boolean
                mask_bool = mask_2d > 0
                
                # Create mask for FoundationPose
                mask_uint8 = (mask_bool.astype(np.uint8)) * 255
                
                # 4. BROADCAST OVERLAY (np.where)
                # Expand mask_bool from (H, W) to (H, W, 1) so it matches (H, W, 3)
                mask_3d = mask_bool[:, :, np.newaxis]
                
                # Create a solid green canvas the same size as the image
                green_overlay = np.zeros_like(rgb)
                green_overlay[:] = [0, 255, 0]
                
                # Synthesize: If mask is true, green; else, original rgb
                viz = np.where(mask_3d, green_overlay, rgb).astype(np.uint8)
                
                # 5. Publish
                m_pub.publish(bridge.cv2_to_imgmsg(mask_uint8, "mono8"))
                v_pub.publish(bridge.cv2_to_imgmsg(cv2.cvtColor(viz, cv2.COLOR_RGB2BGR), "bgr8"))

            except Exception as e:
                # If we get here during startup, just log and keep waiting
                rospy.logwarn(f"Publisher waiting for valid data: {e}")
                continue