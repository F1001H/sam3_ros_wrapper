import torch
import PIL.Image as PILImage
import multiprocessing as mp
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

class SAM3InferenceProcess(mp.Process):
    def __init__(self, input_queue, output_queue, prompt="cube"):
        super().__init__()
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.prompt = prompt

    def run(self):
        # Heavy imports and model loading must happen inside run() for CUDA
        device = torch.device("cuda")
        model = build_sam3_image_model().to(device)
        processor = Sam3Processor(model)

        while True:
            rgb_frame = self.input_queue.get() # Wait for frame
            pil_img = PILImage.fromarray(rgb_frame)

            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                state = processor.set_image(pil_img)
                output = processor.set_text_prompt(state=state, prompt=self.prompt)
            
            # Send the raw image and the mask tensor to the publisher
            # We keep the mask as a tensor to resize it in the next stage
            self.output_queue.put((rgb_frame, output["masks"]))