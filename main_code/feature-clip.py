#!/usr/bin/env python3
import os
import io
import faiss
import torch
import numpy as np
import urllib.request
import PIL.Image
import random
from tqdm import tqdm
from datasets import load_dataset
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from transformers import CLIPProcessor, CLIPModel
from datasets.utils.file_utils import get_datasets_user_agent

# ----------------------------
# Helper Function: Download an Image from URL (for real images)
# ----------------------------
USER_AGENT = get_datasets_user_agent()

def fetch_single_image(image_url, timeout=5, retries=1):
    """
    Attempts to download an image from the provided URL.
    Retries the download if it fails.
    """
    for _ in range(retries + 1):
        try:
            request = urllib.request.Request(
                image_url,
                data=None,
                headers={"user-agent": USER_AGENT},
            )
            with urllib.request.urlopen(request, timeout=timeout) as req:
                image = PIL.Image.open(io.BytesIO(req.read())).convert("RGB")
            return image
        except Exception:
            continue
    return None

# ----------------------------
# Global Synthetic Image Generator using Diffusers
# ----------------------------
from diffusers import StableDiffusionPipeline

# Initialize the pipeline once (this may take a few minutes)
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16)
pipe = pipe.to(device)

def generate_synthetic_image(prompt: str, num_inference_steps: int = 50) -> PIL.Image.Image:
    """
    Generate a synthetic image from the given text prompt using Stable Diffusion.
    """
    with torch.no_grad():
        image = pipe(prompt, num_inference_steps=num_inference_steps).images[0]
    return image

# ----------------------------
# CLIP Image Encoder (for consistent image & text embeddings)
# ----------------------------
class CLIPImageEncoder:
    def __init__(self):
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    def extract(self, image: PIL.Image.Image) -> np.ndarray:
        """
        Given a PIL image, returns the CLIP image embedding as a NumPy array.
        """
        inputs = self.processor(images=image, return_tensors="pt")
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
        return image_features.squeeze().cpu().numpy()

# ----------------------------
# CLIP Text Encoder
# ----------------------------
class CLIPTextEncoder:
    def __init__(self):
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    def encode_text(self, text_query: str) -> np.ndarray:
        """
        Encodes a text prompt into a feature vector.
        """
        inputs = self.processor(text=[text_query], return_tensors="pt", padding=True)
        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)
        return text_features.squeeze().cpu().numpy()

# ----------------------------
# Helper Functions for Processing
# ----------------------------
def download_and_extract(sample, extractor, timeout=5, retries=1):
    """
    For real images: downloads the image from the sample and extracts its features.
    Returns a tuple (image_identifier, feature_vector) where the identifier is the image URL.
    """
    image_url = sample.get("image_url") or sample.get("url")
    if image_url is None:
        return None, None

    image = fetch_single_image(image_url, timeout=timeout, retries=retries)
    if image is None:
        print(f"⚠️ Failed to fetch image at: {image_url}")
        return image_url, None

    try:
        feature = extractor.extract(image)
    except Exception as e:
        print(f"⚠️ Feature extraction failed for image at {image_url}: {e}")
        return image_url, None

    return image_url, feature

def generate_and_extract(sample, extractor, num_inference_steps=50):
    """
    For synthetic images: uses the sample caption as prompt to generate a synthetic image,
    then extracts its features.
    Returns a tuple (prompt, feature_vector).
    """
    prompt = sample.get("caption")
    if not prompt:
        return None, None
    try:
        image = generate_synthetic_image(prompt, num_inference_steps=num_inference_steps)
    except Exception as e:
        print(f"⚠️ Synthetic generation failed for prompt '{prompt}': {e}")
        return prompt, None

    try:
        feature = extractor.extract(image)
    except Exception as e:
        print(f"⚠️ Feature extraction failed for synthetic image with prompt '{prompt}': {e}")
        return prompt, None

    return prompt, feature

# ----------------------------
# Conceptual Captions Dataset Handler
# ----------------------------
class ConceptualCaptionsDataset:
    def __init__(self, dataset_name="google-research-datasets/conceptual_captions", real_limit=50, synthetic_limit=50):
        """
        Initialize the dataset handler.
        For this example:
          - 'real_limit' images are used for the real index,
          - 'synthetic_limit' images are used for the synthetic index,
          - The 'mixed' index is a combination of both.
        """
        self.dataset_name = dataset_name
        self.real_limit = real_limit
        self.synthetic_limit = synthetic_limit
        self.dataset = load_dataset(self.dataset_name, split="train")
        self.extractor = CLIPImageEncoder()  # Using CLIP for images
        self.text_encoder = CLIPTextEncoder()
        self.index_real = None
        self.index_synthetic = None
        self.index_mixed = None
        self.image_ids_real = []
        self.image_ids_synthetic = []
        self.image_ids_mixed = []

    def build_faiss_index(self, dataset_type="real"):
        """
        Builds a FAISS index for the given dataset type.
        'real': uses the first real_limit samples (downloaded).
        'synthetic': uses the next synthetic_limit samples (generated).
        'mixed': combines both.
        """
        if dataset_type == "real":
            limit = self.real_limit
            index_file = "faiss_index_real.bin"
            ids_file = "image_ids_real.npy"
            samples = [self.dataset[idx] for idx in range(limit)]
            desc = "Processing real images"
            process_func = partial(download_and_extract, extractor=self.extractor, timeout=10, retries=3)
        elif dataset_type == "synthetic":
            limit = self.synthetic_limit
            index_file = "faiss_index_synthetic.bin"
            ids_file = "image_ids_synthetic.npy"
            # For synthetic images, take samples after the real ones.
            samples = [self.dataset[idx] for idx in range(self.real_limit, self.real_limit + limit)]
            desc = "Processing synthetic images"
            process_func = partial(generate_and_extract, extractor=self.extractor, num_inference_steps=50)
        elif dataset_type == "mixed":
            index_file = "faiss_index_mixed.bin"
            ids_file = "image_ids_mixed.npy"
            # The mixed index will be built by combining the already-built real and synthetic indices.
            if self.index_real is None or self.index_synthetic is None:
                print("Real or synthetic index not available; building them first.")
                if self.index_real is None:
                    self.build_faiss_index("real")
                if self.index_synthetic is None:
                    self.build_faiss_index("synthetic")
            # Reconstruct features from the real index.
            real_features = []
            for i in range(self.index_real.ntotal):
                real_features.append(self.index_real.reconstruct(i))
            # Reconstruct features from the synthetic index.
            synthetic_features = []
            for i in range(self.index_synthetic.ntotal):
                synthetic_features.append(self.index_synthetic.reconstruct(i))
            # Combine the features.
            all_features = np.concatenate([np.array(real_features), np.array(synthetic_features)], axis=0).astype(np.float32)
            dimension = all_features.shape[1]
            mixed_index = faiss.IndexFlatL2(dimension)
            mixed_index.add(all_features)
            faiss.write_index(mixed_index, index_file)
            combined_ids = self.image_ids_real + self.image_ids_synthetic
            np.save(ids_file, np.array(combined_ids))
            print(f"✅ Built and saved 'mixed' index with {mixed_index.ntotal} vectors.")
            self.index_mixed = mixed_index
            self.image_ids_mixed = combined_ids
            return  # Exit after building mixed index.
        else:
            print("⚠️ Unknown dataset_type:", dataset_type)
            return

        # For real and synthetic branches, process in parallel.
        with ThreadPoolExecutor(max_workers=20) as executor:
            results = list(tqdm(executor.map(process_func, samples),
                                total=len(samples), desc=desc))

        features_list = []
        image_ids = []

        for image_id, feature in results:
            if feature is not None:
                features_list.append(feature)
                image_ids.append(image_id)

        if not features_list:
            print(f"⚠️ No features extracted for {dataset_type}. Aborting index creation.")
            return

        features_array = np.array(features_list).astype(np.float32)
        dimension = features_array.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(features_array)

        faiss.write_index(index, index_file)
        np.save(ids_file, np.array(image_ids))
        print(f"✅ Built and saved '{dataset_type}' index with {index.ntotal} vectors.")

        if dataset_type == "real":
            self.index_real = index
            self.image_ids_real = image_ids
        elif dataset_type == "synthetic":
            self.index_synthetic = index
            self.image_ids_synthetic = image_ids

    def load_faiss_index(self):
        """
        Loads pre-built FAISS indices and image identifier lists from disk.
        """
        if os.path.exists("faiss_index_real.bin"):
            self.index_real = faiss.read_index("faiss_index_real.bin")
            self.image_ids_real = np.load("image_ids_real.npy", allow_pickle=True).tolist()
        if os.path.exists("faiss_index_synthetic.bin"):
            self.index_synthetic = faiss.read_index("faiss_index_synthetic.bin")
            self.image_ids_synthetic = np.load("image_ids_synthetic.npy", allow_pickle=True).tolist()
        if os.path.exists("faiss_index_mixed.bin"):
            self.index_mixed = faiss.read_index("faiss_index_mixed.bin")
            self.image_ids_mixed = np.load("image_ids_mixed.npy", allow_pickle=True).tolist()

    def get_random_captions(self, num_captions=10):
        """
        Returns a list of random captions from the dataset.
        """
        return random.sample(self.dataset["caption"], num_captions)

    def search_by_text(self, text_query, dataset="real", k=5):
        """
        Retrieves the top-K similar images based on a text prompt.
        Returns a list of image identifiers (URLs or prompts).
        """
        print(f"\n🔍 Searching for '{text_query}' in the {dataset} dataset...")
        query_vector = self.text_encoder.encode_text(text_query).reshape(1, -1)
        if dataset == "real":
            index = self.index_real
            image_ids = self.image_ids_real
        elif dataset == "synthetic":
            index = self.index_synthetic
            image_ids = self.image_ids_synthetic
        elif dataset == "mixed":
            index = self.index_mixed
            image_ids = self.image_ids_mixed
        else:
            print("⚠️ Unknown dataset type for search:", dataset)
            return []

        distances, indices = index.search(query_vector, k)
        retrieved = [image_ids[i] for i in indices[0]]
        return retrieved

# ----------------------------
# Main Execution Block
# ----------------------------
if __name__ == "__main__":
    # For testing, use small numbers.
    dataset_handler = ConceptualCaptionsDataset(real_limit=50, synthetic_limit=50)
    
    # Check and build/load each index branch individually.
    if not os.path.exists("faiss_index_real.bin"):
        print("⚠️ FAISS index for real not found. Building real index...")
        dataset_handler.build_faiss_index("real")
    else:
        print("✅ FAISS index for real found. Loading real index...")
        # Load only the real branch
        dataset_handler.load_faiss_index()
    
    if not os.path.exists("faiss_index_synthetic.bin"):
        print("⚠️ FAISS index for synthetic not found. Building synthetic index...")
        dataset_handler.build_faiss_index("synthetic")
    else:
        print("✅ FAISS index for synthetic found. Loading synthetic index...")
        dataset_handler.load_faiss_index()
    
    if not os.path.exists("faiss_index_mixed.bin"):
        print("⚠️ FAISS index for mixed not found. Building mixed index...")
        dataset_handler.build_faiss_index("mixed")
    else:
        print("✅ FAISS index for mixed found. Loading mixed index...")
        dataset_handler.load_faiss_index()
    
    print("\n🔍 **Select a caption for retrieval:**")
    random_captions = dataset_handler.get_random_captions(10)
    for i, caption in enumerate(random_captions, 1):
        print(f"{i}. {caption}")
    
    while True:
        try:
            selection = int(input("\nEnter the number of the caption you want to use (1-10): "))
            if 1 <= selection <= 10:
                text_prompt = random_captions[selection - 1]
                break
            else:
                print("⚠️ Invalid selection. Please enter a number between 1 and 10.")
        except ValueError:
            print("⚠️ Invalid input. Please enter a number.")
    
    print(f"\n✅ Using caption: '{text_prompt}'")
    
    top_matches_real = dataset_handler.search_by_text(text_prompt, dataset="real", k=5)
    top_matches_synthetic = dataset_handler.search_by_text(text_prompt, dataset="synthetic", k=5)
    top_matches_mixed = dataset_handler.search_by_text(text_prompt, dataset="mixed", k=5)
    
    print("\n🔍 **Top Retrieved Image IDs (Real Dataset):**")
    for i, img_id in enumerate(top_matches_real, 1):
        print(f"{i}. {img_id}")
    
    print("\n🔍 **Top Retrieved Image IDs (Synthetic Dataset):**")
    for i, img_id in enumerate(top_matches_synthetic, 1):
        print(f"{i}. {img_id}")
    
    print("\n🔍 **Top Retrieved Image IDs (Mixed Dataset):**")
    for i, img_id in enumerate(top_matches_mixed, 1):
        print(f"{i}. {img_id}")
