import multiprocessing as mp
# ONLY light, non-GPU imports here
import numpy as np 

class SAM3InferenceProcess(mp.Process):
    def __init__(self, input_queue, output_queue, prompt="cube"):
        super().__init__()
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.prompt = prompt

    def run(self):
        # --- ALL HEAVY IMPORTS MOVE INSIDE RUN ---
        import torch
        import PIL.Image as PILImage
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
        # -----------------------------------------

        device = torch.device("cuda")
        
        print("SAM3 Process: Loading model onto 5090...")
        model = build_sam3_image_model().to(device)
        processor = Sam3Processor(model)
        print("SAM3 Process: Model Ready.")

        while True:
            rgb_frame = self.input_queue.get() 
            if rgb_frame is None: break # Graceful shutdown
            
            pil_img = PILImage.fromarray(rgb_frame)

            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                state = processor.set_image(pil_img)
                output = processor.set_text_prompt(state=state, prompt=self.prompt)
            
            self.output_queue.put((rgb_frame, output["masks"].cpu().numpy()))