#!/usr/bin/env python3
import multiprocessing as mp
import signal
import sys
import os
import psutil

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

    p1.daemon = True
    p2.daemon = True
    p3.daemon = True

    p1.start()
    p2.start()
    p3.start()

    rospy.init_node('sam3_pipeline_wrapper', anonymous=True)

    processes = [p1, p2, p3]

    rospy.loginfo("SAM-3 Manager: Parallel Workers Active.")
    
    def kill_child_processes():
        """Forcibly finds and kills all child processes and their threads."""
        current_process = psutil.Process()
        children = current_process.children(recursive=True)
        for child in children:
            try:
                rospy.loginfo(f"Force killing child PID: {child.pid}")
                child.kill() # Sends SIGKILL (cannot be ignored)
            except psutil.NoSuchProcess:
                pass

    def cleanup(sig, frame):
        rospy.loginfo("SAM-3 Manager: Hard Shutdown Initiated...")
        
        # 1. Stop the ROS loop immediately
        rospy.signal_shutdown("User requested shutdown")

        # 2. Break the Queues (Prevents the 100% CPU feeder thread spin)
        # We tell the queues to stop trying to sync data
        input_q.cancel_join_thread()
        output_q.cancel_join_thread()
        input_q.close()
        output_q.close()

        # 3. Forcible OS-level kill
        # This catches orphan ZED or CUDA threads that terminate() misses
        kill_child_processes()

        # 4. Final exit to the kernel
        rospy.loginfo("SAM-3 Manager: Cleaning up memory...")
        os._exit(0)

    signal.signal(signal.SIGINT, cleanup)
    rospy.spin()

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()