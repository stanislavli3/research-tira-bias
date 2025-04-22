import os
import io
import csv
import urllib.request
import numpy as np
import torch
import PIL.Image
import faiss
from tqdm import tqdm
from datasets import load_dataset
from diffusers import StableDiffusionPipeline
from datasets.utils.file_utils import get_datasets_user_agent
from concurrent.futures import ThreadPoolExecutor
from features import CLIPEncoder

# === CONFIG ===
REAL_CAPTIONS_COUNT = 140
SYNTHETIC_PER_REAL = 5
USER_AGENT = get_datasets_user_agent()

# === FOLDER STRUCTURE ===
BASE_DIR = "research-tira-bias/main_code/new_data"
DATASETS = {
    "1to5": {
        "real": os.path.join(BASE_DIR, "dataset_1to5/real"),
        "synthetic": os.path.join(BASE_DIR, "dataset_1to5/synthetic"),
        "csv": os.path.join(BASE_DIR, "dataset_1to5/dataset_info.csv")
    },
    "1to1": {
        "real": os.path.join(BASE_DIR, "dataset_1to1/real"),
        "synthetic": os.path.join(BASE_DIR, "dataset_1to1/synthetic"),
        "csv": os.path.join(BASE_DIR, "dataset_1to1/dataset_info.csv")
    }
}

# === HELPERS ===
def fetch_single_image(image_url, timeout=5, retries=1):
    for _ in range(retries + 1):
        try:
            req = urllib.request.Request(image_url, headers={"user-agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as response:
                image = PIL.Image.open(io.BytesIO(response.read())).convert("RGB")
            return image
        except Exception:
            continue
    return None

class DataHandler:
    def __init__(self):
        self.encoder = CLIPEncoder()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipe = StableDiffusionPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16
        ).to(self.device)

    def download_image(self, sample):
        url = sample.get("image_url") or sample.get("url")
        caption = sample.get("caption", "")
        if not url:
            return None, caption
        image = fetch_single_image(url)
        return image, caption

    def generate_synthetic_image(self, prompt, save_path):
        full_prompt = "a highly detailed, photorealistic, 8k resolution photograph of " + prompt
        try:
            with torch.no_grad():
                image = self.pipe(full_prompt).images[0]
            image.save(save_path)
            return save_path, prompt
        except Exception as e:
            print(f"⚠️ Failed to generate/save image for prompt '{prompt}': {e}")
            return None, prompt

# === MAIN ===
if __name__ == "__main__":
    handler = DataHandler()
    print("📦 Loading Conceptual Captions dataset...")
    dataset = load_dataset("google-research-datasets/conceptual_captions", split="train")
    samples = dataset.shuffle(seed=42)

    # === Make directories ===
    for config in DATASETS.values():
        os.makedirs(config["real"], exist_ok=True)
        os.makedirs(config["synthetic"], exist_ok=True)

    # === Prepare CSV writers ===
    files = {}
    writers = {}
    for key, config in DATASETS.items():
        f = open(config["csv"], "w", newline="", encoding="utf-8")
        writer = csv.writer(f)
        writer.writerow(["image_name", "caption", "type"])
        f.flush()  # ✨ Immediately flush header
        files[key] = f
        writers[key] = writer

    count = 0
    for sample in samples:
        if count >= REAL_CAPTIONS_COUNT:
            break

        image, caption = handler.download_image(sample)
        if image is None:
            continue

        real_filename = f"real_{count:03}.jpg"
        real_path_1to5 = os.path.join(DATASETS["1to5"]["real"], real_filename)
        real_path_1to1 = os.path.join(DATASETS["1to1"]["real"], real_filename)

        try:
            image.save(real_path_1to5)
            image.save(real_path_1to1)
        except Exception as e:
            print(f"⚠️ Failed to save real image: {e}")
            continue

        # ✅ Write and flush immediately
        writers["1to5"].writerow([real_filename, caption, "real"])
        files["1to5"].flush()

        writers["1to1"].writerow([real_filename, caption, "real"])
        files["1to1"].flush()

        # === 1to5 synthetic images ===
        for j in range(1, SYNTHETIC_PER_REAL + 1):
            synth_filename = f"synthetic_{count:03}_{j}.png"
            synth_path = os.path.join(DATASETS["1to5"]["synthetic"], synth_filename)
            result = handler.generate_synthetic_image(caption, synth_path)
            if result[0]:
                writers["1to5"].writerow([os.path.basename(result[0]), caption, "synthetic"])
                files["1to5"].flush()

        # === 1to1 synthetic image ===
        synth_filename = f"synthetic_bal_{count:03}.png"
        synth_path = os.path.join(DATASETS["1to1"]["synthetic"], synth_filename)
        result = handler.generate_synthetic_image(caption, synth_path)
        if result[0]:
            writers["1to1"].writerow([os.path.basename(result[0]), caption, "synthetic"])
            files["1to1"].flush()

        count += 1

    # ✅ Always close files!
    for f in files.values():
        f.close()

    print(f"\n✅ Finished: {count} real captions")
    print("✅ Dataset 1to5 and 1to1 successfully created.")
