# data.py

import os
import io
import urllib.request
import PIL.Image
from datasets.utils.file_utils import get_datasets_user_agent
from tqdm import tqdm

USER_AGENT = get_datasets_user_agent()

def fetch_single_image(image_url, timeout=5, retries=1):
    """
    Attempts to download an image from the given URL using a proper user agent.
    Retries the download if it fails.
    Returns a PIL.Image or None if download fails.
    """
    for _ in range(retries + 1):
        try:
            req = urllib.request.Request(
                image_url,
                data=None,
                headers={"user-agent": USER_AGENT},
            )
            with urllib.request.urlopen(req, timeout=timeout) as response:
                image = PIL.Image.open(io.BytesIO(response.read())).convert("RGB")
            return image
        except Exception:
            continue
    return None

class DataDownloader:
    """
    Downloads real images from the dataset and saves them locally.
    Before downloading, it checks if the image file already exists.
    """
    def __init__(self, output_dir="data/dataset_real"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
    
    def download_image(self, sample, timeout=10, retries=3):
        """
        Downloads an image given a sample (expects keys "image_url" or "url" and "caption").
        Returns (file_path, caption) if successful; otherwise, (None, caption).
        """
        image_url = sample.get("image_url") or sample.get("url")
        caption = sample.get("caption", "")
        if image_url is None:
            return None, caption
        
        # Create a filename based on the hash of the URL.
        file_name = f"real_{abs(hash(image_url))}.jpg"
        file_path = os.path.join(self.output_dir, file_name)
        
        # Check if the file already exists. If so, return its path.
        if os.path.exists(file_path):
            return file_path, caption
        
        image = fetch_single_image(image_url, timeout=timeout, retries=retries)
        if image is None:
            # Skip if download fails.
            return None, caption
        
        try:
            image.save(file_path)
        except Exception:
            return None, caption
        
        return file_path, caption

    def download_images(self, samples, max_workers=20):
        """
        Downloads images in parallel from the provided list of samples.
        Returns a list of (file_path, caption) tuples for successful downloads.
        """
        from concurrent.futures import ThreadPoolExecutor
        
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for res in tqdm(executor.map(lambda s: self.download_image(s), samples),
                            total=len(samples), desc="Downloading real images"):
                results.append(res)
        # Only return results where a valid file path was returned.
        return [res for res in results if res[0] is not None]
