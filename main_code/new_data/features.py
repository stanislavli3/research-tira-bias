import torch
import numpy as np
import PIL.Image
from transformers import CLIPProcessor, CLIPModel

class CLIPEncoder:
    """
    A unified CLIP encoder for extracting image and text features.
    Loads the model and processor only once.
    """
    def __init__(self):
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.model.eval()

    def extract_image_features(self, image: PIL.Image.Image) -> np.ndarray:
        """
        Extract features from an image using CLIP.
        """
        try:
            inputs = self.processor(images=image, return_tensors="pt")
            with torch.no_grad():
                feats = self.model.get_image_features(**inputs)
            if feats is None or feats.numel() == 0:
                print("⚠️ No image features extracted.")
                return None
            return feats.squeeze().cpu().numpy()
        except Exception as e:
            print(f"⚠️ Error extracting image features: {e}")
            return None

    def extract_from_path(self, file_path: str) -> np.ndarray:
        """
        Load an image from a file path and extract its features.
        """
        try:
            image = PIL.Image.open(file_path).convert("RGB")
            print(f"✅ Successfully loaded image: {file_path}")
            return self.extract_image_features(image)
        except Exception as e:
            print(f"⚠️ Failed to load image {file_path}: {e}")
            return None

    def encode_text(self, text_query: str) -> np.ndarray:
        """
        Encode a text query using CLIP.
        """
        try:
            inputs = self.processor(text=[text_query], return_tensors="pt", padding=True)
            with torch.no_grad():
                feats = self.model.get_text_features(**inputs)
            if feats is None or feats.numel() == 0:
                print("⚠️ No text features extracted.")
                return None
            return feats.squeeze().cpu().numpy()
        except Exception as e:
            print(f"⚠️ Error encoding text: {e}")
            return None
