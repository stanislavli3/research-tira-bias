# indexer.py

import faiss
import numpy as np
from tqdm import tqdm

class IndexBuilder:
    """
    Builds, saves, and loads a FAISS index from local images.
    """
    def __init__(self, image_encoder):
        self.image_encoder = image_encoder

    def build_index(self, image_info_list):
        """
        image_info_list: list of (file_path, caption) tuples.
        Returns a tuple (index, valid_info_list) where valid_info_list contains only those images that were successfully processed.
        """
        features = []
        valid_info = []
        for file_path, caption in tqdm(image_info_list, desc="Extracting features"):
            feat = self.image_encoder.extract_from_path(file_path)
            if feat is not None:
                features.append(feat)
                valid_info.append((file_path, caption))
        if not features:
            raise ValueError("No features extracted.")
        features_array = np.array(features).astype(np.float32)
        d = features_array.shape[1]
        index = faiss.IndexFlatL2(d)
        index.add(features_array)
        return index, valid_info

    def save_index(self, index, info_list, index_file, info_file):
        faiss.write_index(index, index_file)
        np.save(info_file, np.array(info_list, dtype=object))

    def load_index(self, index_file, info_file):
        index = faiss.read_index(index_file)
        info_list = np.load(info_file, allow_pickle=True).tolist()
        return index, info_list
