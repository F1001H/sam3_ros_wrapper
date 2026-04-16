#!/usr/bin/env python3
import multiprocessing as mp
import signal
import sys
import os

from ros_image_reader import ROSInputProcess
from sam3_engine import SAM3InferenceProcess
from ros_publisher import ROSOutputProcess

def main():
    import rospy
    # The Manager keeps the name from the launch file (sam3_pipeline_wrapper)    
    input_q = mp.Queue(maxsize=1)
    output_q = mp.Queue(maxsize=1)
    
    prompt = rospy.get_param('~prompt', 'cube')

    p1 = ROSInputProcess(input_q)
    p2 = SAM3InferenceProcess(input_q, output_q, prompt=prompt)
    p3 = ROSOutputProcess(output_q)

    p1.start()
    p2.start()
    p3.start()

    rospy.init_node('sam3_pipeline_wrapper', anonymous=True)


    rospy.loginfo("SAM-3 Manager: Parallel Workers Active.")
    
    def cleanup(sig, frame):
        rospy.loginfo("Terminating...")
        for p in [p1, p2, p3]: p.terminate()
        sys.exit(0)

    signal.signal(signal.SIGINT, cleanup)
    rospy.spin()

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()