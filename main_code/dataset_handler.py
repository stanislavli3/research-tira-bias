import os
import numpy as np
import faiss
from retrieval_models import RetrieverFactory
import csv


def l2_normalize(vec: np.ndarray) -> np.ndarray:
    return vec / np.linalg.norm(vec)


class ConceptualCaptionsHandler:
    def __init__(self,
                 real_dir="new_data/real",
                 synthetic_dir="new_data/synthetic",
                 mixed_dir="new_data/mixed",
                 real_index_path="faiss_index_real.bin",
                 synthetic_index_path="faiss_index_synthetic.bin",
                 mixed_index_path="faiss_index_mixed.bin",
                 real_info_path="image_info_real.npy",
                 synthetic_info_path="image_info_synthetic.npy",
                 mixed_info_path="image_info_mixed.npy",
                 real_limit=50000,
                 synthetic_limit=25000,
                 model_type="clip"):

        self.model_type = model_type
        self.encoder = RetrieverFactory.create_retriever(model_type)
        self.real_dir = real_dir
        self.synthetic_dir = synthetic_dir
        self.mixed_dir = mixed_dir

        self.real_index_path = real_index_path
        self.synthetic_index_path = synthetic_index_path
        self.mixed_index_path = mixed_index_path
        self.real_info_path = real_info_path
        self.synthetic_info_path = synthetic_info_path
        self.mixed_info_path = mixed_info_path

        self.real_limit = real_limit
        self.synthetic_limit = synthetic_limit

        self.real_index = None
        self.synthetic_index = None
        self.mixed_index = None
        self.real_info = None
        self.synthetic_info = None
        self.mixed_info = None

    def _build_index_from_folder(self, folder_path, limit=None, metadata_csv=None):
        caption_map = {}
        if metadata_csv and os.path.exists(metadata_csv):
            with open(metadata_csv, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    caption_map[row["image_name"]] = row["caption"]

        features, info = [], []
        for file_name in sorted(os.listdir(folder_path)):
            if limit and len(features) >= limit:
                break
            file_path = os.path.join(folder_path, file_name)
            try:
                vec = self.encoder.extract_from_path(file_path)
                if vec is not None and np.all(np.isfinite(vec)):
                    vec = l2_normalize(vec)
                    features.append(vec)
                    caption = caption_map.get(file_name, f"Caption for {file_name}")
                    info.append({"path": file_path, "caption": caption})
            except Exception as e:
                print(f"⚠️ Failed: {file_path} → {e}")
        
        if not features:
            raise ValueError(f"❌ No features extracted from {folder_path}")
            
        features_np = np.stack(features).astype("float32")
        index = faiss.IndexFlatL2(features_np.shape[1])
        index.add(features_np)
        
        print(f"✅ Built index with {len(features)} vectors")
        return index, info

    def _save_index_and_info(self, index, info, index_path, info_path):
        faiss.write_index(index, index_path)
        np.save(info_path, info, allow_pickle=True)

    def _load_index_and_info(self, index_path, info_path):
        index = faiss.read_index(index_path)
        info = np.load(info_path, allow_pickle=True).tolist()
        print(f"📥 Loaded FAISS index with {index.ntotal} vectors")
        print(f"📚 Metadata size: {len(info)}")
        return index, info

    def build_real_index(self):
        print(f"📦 Building real index from: {self.real_dir}")
        self.real_index, self.real_info = self._build_index_from_folder(self.real_dir, self.real_limit)
        self._save_index_and_info(self.real_index, self.real_info, self.real_index_path, self.real_info_path)

    def load_real_index(self):
        print(f"📥 Loading real index from: {self.real_index_path}")
        self.real_index, self.real_info = self._load_index_and_info(self.real_index_path, self.real_info_path)

    def build_synthetic_index(self):
        print(f"📦 Building synthetic index from: {self.synthetic_dir}")
        self.synthetic_index, self.synthetic_info = self._build_index_from_folder(self.synthetic_dir, self.synthetic_limit)
        self._save_index_and_info(self.synthetic_index, self.synthetic_info, self.synthetic_index_path, self.synthetic_info_path)

    def load_synthetic_index(self):
        print(f"📥 Loading synthetic index from: {self.synthetic_index_path}")
        self.synthetic_index, self.synthetic_info = self._load_index_and_info(self.synthetic_index_path, self.synthetic_info_path)

    def build_mixed_index(self, dataset_info_csv=None):
        print(f"📦 Building mixed index from mixed directory using {self.model_type}")
        print(f"Mixed directory: {self.mixed_dir}")

        # Load captions from CSV if provided
        caption_map = {}
        if dataset_info_csv and os.path.exists(dataset_info_csv):
            print(f"Loading captions from {dataset_info_csv}")
            with open(dataset_info_csv, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    caption_map[row["image_name"]] = row["caption"]
            print(f"Loaded {len(caption_map)} captions from CSV")

        # Initialize the model
        if not self.encoder.is_initialized:
            self.encoder.initialize()

        # Get all image paths
        image_paths = [os.path.join(self.mixed_dir, f) for f in sorted(os.listdir(self.mixed_dir))]
        
        # Build the index using the retriever
        self.encoder.build_index(image_paths)
        self.mixed_index = self.encoder.index
        
        # Create info list with captions
        self.mixed_info = []
        for path in self.encoder.image_paths:
            file_name = os.path.basename(path)
            caption = caption_map.get(file_name, f"Caption for {file_name}")
            self.mixed_info.append({
                "path": path,
                "caption": caption,
                "type": "real" if "real_" in file_name else "synthetic"
            })

        # Save the index and info
        self._save_index_and_info(self.mixed_index, self.mixed_info, self.mixed_index_path, self.mixed_info_path)
        print("✅ Mixed index built and saved to FAISS.")
        print(f"🔢 FAISS index size: {self.mixed_index.ntotal}")
        print(f"📚 Metadata size: {len(self.mixed_info)}")
        
        # Print some statistics
        real_count = sum(1 for info in self.mixed_info if info["type"] == "real")
        synthetic_count = len(self.mixed_info) - real_count
        print(f"📊 Dataset composition:")
        print(f"  - Real images: {real_count}")
        print(f"  - Synthetic images: {synthetic_count}")
        print(f"  - Ratio: 1:{synthetic_count/real_count:.1f}")

    def load_mixed_index(self):
        print(f"📥 Loading mixed index from: {self.mixed_index_path}")
        self.mixed_index, self.mixed_info = self._load_index_and_info(self.mixed_index_path, self.mixed_info_path)
        # Also load the index into the retriever
        if not self.encoder.is_initialized:
            self.encoder.initialize()
        self.encoder.index = self.mixed_index
        self.encoder.image_paths = [info["path"] for info in self.mixed_info]

    def search_by_text(self, text_query, k=5):
        if self.mixed_index is None:
            raise ValueError("Mixed index not loaded!")
            
        print(f"\n🔍 Searching mixed dataset with {self.model_type}, k={k}")
        
        # Use the retriever's search method
        results = self.encoder.search(text_query, k)
        
        # Map results to our format
        return [(path, next(info["caption"] for info in self.mixed_info if info["path"] == path)) 
                for path, _ in results]
