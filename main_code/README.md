# Image-Text Retrieval Bias Testing Framework

## Overview
This framework tests for potential biases in image-text retrieval models by comparing their performance on real vs. synthetic images. It supports multiple retrieval models and dataset ratios to analyze how different models handle synthetic content.

## Testing Capabilities

### 1. Dataset Configurations
The framework tests two dataset configurations:
- **1:1 Ratio** (`dataset_1to1`): Equal number of real and synthetic images
- **1:5 Ratio** (`dataset_1to5`): One real image paired with five synthetic variants

### 2. Retrieval Models
Currently supports testing with:
- **CLIP** (OpenAI's CLIP model)
- **OpenCLIP** (LAION's open-source CLIP implementation)

### 3. Evaluation Metrics
For each model and dataset, the framework measures:
- **Precision**: Proportion of real images in top-k results
- **Recall**: Proportion of relevant real images retrieved
- **Real vs. Synthetic Distribution**: Analysis of result composition
- Tests at different retrieval depths (k=10, 50, 100)

### 4. Result Analysis
For each test run, the framework generates:
1. **Query Results** (`query_results.csv`):
   - Individual retrieval results
   - Image paths and captions
   - Real/synthetic classification
   - Ranking information

2. **Performance Metrics** (`query_metrics.csv`):
   - Aggregated precision/recall metrics
   - Model-specific performance data
   - Query-level statistics

### 5. Directory Structure
Results are organized by:
```
dataset_[ratio]/
├── mixed/           # Combined image dataset
├── clip/            # CLIP model results
│   ├── faiss_index_mixed.bin
│   ├── image_info_mixed.npy
│   ├── query_results.csv
│   └── query_metrics.csv
└── openclip/       # OpenCLIP model results
    ├── faiss_index_mixed.bin
    ├── image_info_mixed.npy
    ├── query_results.csv
    └── query_metrics.csv
```

## Research Applications

This framework helps investigate several research questions:
1. How do different retrieval models handle synthetic vs. real images?
2. Does the ratio of synthetic to real images affect retrieval performance?
3. Are certain models more biased towards or against synthetic content?
4. How does retrieval performance change with different query types?

## Usage Example
The framework allows interactive testing:
1. Choose a text query from the dataset
2. Test retrieval across multiple models
3. Compare results between 1:1 and 1:5 datasets
4. Analyze precision/recall metrics for bias detection

Each test run automatically:
- Cleans previous results
- Builds or loads necessary indexes
- Generates comprehensive metrics
- Saves results in model-specific directories

This enables systematic analysis of retrieval model biases and performance characteristics when dealing with mixed real and synthetic image content. 