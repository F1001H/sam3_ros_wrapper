import multiprocessing as mp
import sys

# Script imports here
from ros_image_reader import ROSInputProcess
from sam3_engine import SAM3InferenceProcess
from ros_publisher import ROSOutputProcess

if __name__ == "__main__":
    # CRITICAL: This must be the first line inside the main block
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass # Already set

    # Create high-speed queues
    input_q = mp.Queue(maxsize=2)
    output_q = mp.Queue(maxsize=2)

    # Initialize scripts
    # Note: Pass your prompt here if needed
    p1 = ROSInputProcess(input_q)
    p2 = SAM3InferenceProcess(input_q, output_q, prompt="cube")
    p3 = ROSOutputProcess(output_q)

    # Start processes
    p1.start()
    p2.start()
    p3.start()

    try:
        p1.join()
        p2.join()
        p3.join()
    except KeyboardInterrupt:
        print("\nShutting down SAM-3 Pipeline...")
        p1.terminate()
        p2.terminate()
        p3.terminate()
        sys.exit(0)