from abc import ABC, abstractmethod
import numpy as np
import faiss
import torch
import open_clip
from PIL import Image
from typing import List, Tuple, Dict, Any, Optional
from transformers import (
    CLIPProcessor, CLIPModel,
    ViltProcessor, ViltModel,
    FlavaProcessor, FlavaModel,
    VisionTextDualEncoderModel, AutoProcessor
)
import torch.nn.functional as F
import os
import clip
import subprocess
import sys

# Import CLIP properly
try:
    import clip
except ImportError:
    print("Installing OpenAI CLIP...")
    import subprocess
    subprocess.check_call(["pip", "install", "git+https://github.com/openai/CLIP.git"])
    import clip

class BaseRetriever(ABC):
    """Base class for all retrieval models."""
    
    def __init__(self, name: str):
        self.name = name
        self.is_initialized = False
        self.index = None
        self.image_paths = []
    
    @abstractmethod
    def initialize(self) -> None:
        """Initialize the model, load weights, etc."""
        pass
    
    @abstractmethod
    def encode_text(self, text: str) -> np.ndarray:
        """Encode text query into embedding vector."""
        pass
    
    @abstractmethod
    def encode_image(self, image_path: str) -> np.ndarray:
        """Encode image into embedding vector."""
        pass
    
    @abstractmethod
    def build_index(self, image_paths: List[str]):
        """Build FAISS index from a list of image paths."""
        pass
    
    def save_index(self, index_path: str):
        """Save FAISS index to disk."""
        if self.index is not None:
            faiss.write_index(self.index, index_path)
    
    def load_index(self, index_path: str):
        """Load FAISS index from disk."""
        self.index = faiss.read_index(index_path)
    
    @abstractmethod
    def search(self, query: str, k: int = 100) -> List[Tuple[str, float]]:
        """Search for images given a text query."""
        pass

class CLIPRetriever(BaseRetriever):
    """OpenAI CLIP-based image retrieval."""
    
    def __init__(self, model_name: str = "ViT-B/32"):
        super().__init__(name="clip")
        self.model_name = model_name
        self.model = None
        self.preprocess = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def initialize(self):
        try:
            print(f"Loading CLIP model {self.model_name}...")
            try:
                self.model, self.preprocess = clip.load(self.model_name, device=self.device)
            except Exception as e:
                print("Failed to load CLIP model. Installing OpenAI CLIP...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", "git+https://github.com/openai/CLIP.git"])
                self.model, self.preprocess = clip.load(self.model_name, device=self.device)
            
            self.model.eval()
            self.is_initialized = True
            print("✅ CLIP model loaded successfully")
        except Exception as e:
            print(f"❌ Error loading CLIP model: {str(e)}")
            raise
    
    def encode_image(self, image_path: str) -> Optional[np.ndarray]:
        try:
            image = Image.open(image_path).convert('RGB')
            image_input = self.preprocess(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                image_features = self.model.encode_image(image_input)
                image_features = F.normalize(image_features, p=2, dim=1)
            return image_features.cpu().numpy()[0]
        except Exception as e:
            print(f"Error encoding image {image_path}: {str(e)}")
            return None
    
    def encode_text(self, text: str) -> Optional[np.ndarray]:
        try:
            text_input = clip.tokenize([text]).to(self.device)
            with torch.no_grad():
                text_features = self.model.encode_text(text_input)
                text_features = F.normalize(text_features, p=2, dim=1)
            return text_features.cpu().numpy()[0]
        except Exception as e:
            print(f"Error encoding text: {str(e)}")
            return None
    
    def build_index(self, image_paths: List[str]):
        """Build FAISS index from a list of image paths."""
        if not self.is_initialized:
            self.initialize()
            
        print(f"Building index for {len(image_paths)} images...")
        features = []
        valid_paths = []
        
        for path in image_paths:
            feat = self.encode_image(path)
            if feat is not None:
                features.append(feat)
                valid_paths.append(path)
        
        if not features:
            raise ValueError("No valid features extracted from images")
            
        features = np.stack(features)
        
        # Build FAISS index
        self.index = faiss.IndexFlatIP(features.shape[1])
        self.index.add(features.astype('float32'))
        self.image_paths = valid_paths
        
        print(f"✅ Built index with {len(valid_paths)} images")
    
    def search(self, query: str, k: int = 100) -> List[Tuple[str, float]]:
        """Search for images given a text query."""
        if not self.is_initialized:
            self.initialize()
            
        if self.index is None:
            raise ValueError("Index not built. Call build_index first.")
            
        # Encode query
        query_features = self.encode_text(query)
        if query_features is None:
            raise ValueError("Failed to encode query text")
        
        # Search index
        distances, indices = self.index.search(query_features.reshape(1, -1).astype('float32'), k)
        
        # Return results with distances
        results = [(self.image_paths[idx], float(dist)) 
                  for idx, dist in zip(indices[0], distances[0])]
        return results

class OpenCLIPRetriever(BaseRetriever):
    """LAION OpenCLIP-based image retrieval."""
    
    def __init__(self, model_name: str = "ViT-B-32", pretrained: str = "laion2b_s34b_b79k"):
        super().__init__(name="openclip")
        self.model_name = model_name
        self.pretrained = pretrained
        self.model = None
        self.tokenizer = None
        self.transform = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def initialize(self):
        try:
            print(f"Loading OpenCLIP model {self.model_name} ({self.pretrained})...")
            self.model, _, self.transform = open_clip.create_model_and_transforms(
                self.model_name,
                pretrained=self.pretrained,
                device=self.device
            )
            self.tokenizer = open_clip.get_tokenizer(self.model_name)
            self.model.eval()
            self.is_initialized = True
            print("✅ OpenCLIP model loaded successfully")
        except Exception as e:
            print(f"❌ Error loading OpenCLIP model: {str(e)}")
            raise
    
    def encode_image(self, image_path: str) -> Optional[np.ndarray]:
        try:
            image = Image.open(image_path).convert('RGB')
            image_input = self.transform(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                image_features = self.model.encode_image(image_input)
                image_features = F.normalize(image_features, p=2, dim=1)
            return image_features.cpu().numpy()[0]
        except Exception as e:
            print(f"Error encoding image {image_path}: {str(e)}")
            return None
    
    def encode_text(self, text: str) -> Optional[np.ndarray]:
        try:
            text_input = self.tokenizer([text]).to(self.device)
            with torch.no_grad():
                text_features = self.model.encode_text(text_input)
                text_features = F.normalize(text_features, p=2, dim=1)
            return text_features.cpu().numpy()[0]
        except Exception as e:
            print(f"Error encoding text: {str(e)}")
            return None
    
    def build_index(self, image_paths: List[str]):
        """Build FAISS index from a list of image paths."""
        if not self.is_initialized:
            self.initialize()
            
        print(f"Building index for {len(image_paths)} images...")
        features = []
        valid_paths = []
        
        for path in image_paths:
            feat = self.encode_image(path)
            if feat is not None:
                features.append(feat)
                valid_paths.append(path)
        
        if not features:
            raise ValueError("No valid features extracted from images")
            
        features = np.stack(features)
        
        # Build FAISS index
        self.index = faiss.IndexFlatIP(features.shape[1])
        self.index.add(features.astype('float32'))
        self.image_paths = valid_paths
        
        print(f"✅ Built index with {len(valid_paths)} images")
    
    def search(self, query: str, k: int = 100) -> List[Tuple[str, float]]:
        """Search for images given a text query."""
        if not self.is_initialized:
            self.initialize()
            
        if self.index is None:
            raise ValueError("Index not built. Call build_index first.")
            
        # Encode query
        query_features = self.encode_text(query)
        if query_features is None:
            raise ValueError("Failed to encode query text")
        
        # Search index
        distances, indices = self.index.search(query_features.reshape(1, -1).astype('float32'), k)
        
        # Return results with distances
        results = [(self.image_paths[idx], float(dist)) 
                  for idx, dist in zip(indices[0], distances[0])]
        return results

class RetrieverFactory:
    """Factory class for creating retrievers."""
    
    @staticmethod
    def create_retriever(model_type: str, **kwargs) -> BaseRetriever:
        if model_type.lower() == "clip":
            return CLIPRetriever(**kwargs)
        elif model_type.lower() == "openclip":
            return OpenCLIPRetriever(**kwargs)
        else:
            raise ValueError(f"Unknown retriever type: {model_type}") 