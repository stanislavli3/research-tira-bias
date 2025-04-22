# synthetic.py

import os
import torch
import PIL.Image
from diffusers import StableDiffusionPipeline

class SyntheticGenerator:
    """
    Uses Stable Diffusion to generate synthetic images from captions and saves them locally.
    Generates only one image per caption, and stops generating new images once the output
    directory has reached a maximum number (default is 1,000).
    """
    def __init__(self, output_dir="research-tira-bias/data/dataset_synthetic", 
                 model_id="CompVis/stable-diffusion-v1-4", max_images=1000):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.max_images = max_images  # Maximum number of synthetic images to produce
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
        self.pipe = self.pipe.to(self.device)

    def current_count(self) -> int:
        """Return the number of image files currently in the output directory."""
        files = os.listdir(self.output_dir)
        return len([f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    def generate_synthetic_image(self, prompt: str, num_inference_steps: int = 50) -> PIL.Image.Image:
        """
        Generate a synthetic image from the given text prompt using Stable Diffusion.
        The prompt is modified to encourage photorealistic output.
        """
        # Prepend a realistic description to the original prompt.
        realistic_prompt = "a highly detailed, photorealistic, 8k resolution photograph of " + prompt
        try:
            with torch.no_grad():
                result = self.pipe(realistic_prompt, num_inference_steps=num_inference_steps)
            if result.images:
                return result.images[0]
            return None
        except Exception as e:
            print(f"⚠️ Error generating image for prompt '{realistic_prompt}': {e}")
            return None

    def generate_and_save(self, prompt: str, filename: str = None, num_inference_steps: int = 50):
        """
        Generates a synthetic image from 'prompt', saves it to disk, and returns (file_path, prompt).
        Generates only one image per caption. Checks if the maximum number of images has been reached.
        If 'filename' is None, a filename is created using the hash of the prompt.
        """
        if self.current_count() >= self.max_images:
            print(f"Maximum synthetic images reached ({self.max_images}). Skipping generation for prompt: '{prompt}'")
            return None, prompt

        image = self.generate_synthetic_image(prompt, num_inference_steps=num_inference_steps)
        if image is None:
            return None, prompt
        
        if filename is None:
            filename = f"synthetic_{abs(hash(prompt))}.png"
        file_path = os.path.join(self.output_dir, filename)
        try:
            image.save(file_path)
        except Exception as e:
            print(f"⚠️ Failed to save synthetic image for prompt '{prompt}': {e}")
            return None, prompt
        return file_path, prompt
