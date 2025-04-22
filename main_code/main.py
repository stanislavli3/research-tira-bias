#!/usr/bin/env python3
import os
import csv
import random
from dataset_handler import ConceptualCaptionsHandler

def load_csv_captions(csv_path, limit=1000):
    captions = set()
    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "caption" in row:
                captions.add(row["caption"])
                if len(captions) >= limit:
                    break
    return list(captions)

def main():
    # List of datasets to test
    dataset_dirs = [
        "research-tira-bias/main_code/new_data/dataset_1to1",
        "research-tira-bias/main_code/new_data/dataset_1to5"
    ]

    # List of retrieval models to test
    retrieval_models = ["clip", "openclip"]  # Add more models as they become available

    for dataset_dir in dataset_dirs:
        print(f"\n{'='*50}")
        print(f"Testing dataset: {dataset_dir}")
        print(f"{'='*50}\n")
        
        # === Mixed paths
        mixed_dir = os.path.join(dataset_dir, "mixed")
        csv_path = os.path.join(dataset_dir, "dataset_info.csv")
        
        for model_type in retrieval_models:
            print(f"\n{'-'*20}")
            print(f"Using model: {model_type}")
            print(f"{'-'*20}\n")

            # Create model-specific directory
            model_dir = os.path.join(dataset_dir, model_type)
            os.makedirs(model_dir, exist_ok=True)

            # Set up paths for this model
            index_path = os.path.join(model_dir, "faiss_index_mixed.bin")
            info_path = os.path.join(model_dir, "image_info_mixed.npy")
            results_csv = os.path.join(model_dir, "query_results.csv")
            metrics_csv = os.path.join(model_dir, "query_metrics.csv")

            # Delete old metric files if they exist
            for old_file in [results_csv, metrics_csv]:
                if os.path.exists(old_file):
                    try:
                        os.remove(old_file)
                        print(f"Deleted old metric file: {old_file}")
                    except Exception as e:
                        print(f"Warning: Could not delete {old_file}: {e}")

            handler = ConceptualCaptionsHandler(
                mixed_dir=mixed_dir,
                mixed_index_path=index_path,
                mixed_info_path=info_path,
                model_type=model_type
            )

            # === Build or load mixed index
            try:
                print("Attempting to load mixed index...")
                handler.load_mixed_index()
                print("✅ Mixed index loaded successfully")
            except Exception as e:
                print(f"⚠️ Error loading mixed index: {str(e)}")
                print("Attempting to build mixed index...")
                handler.build_mixed_index(dataset_info_csv=csv_path)
                print("✅ Mixed index built successfully")

            # === Load captions
            captions = load_csv_captions(csv_path, limit=1000)
            if not captions:
                print("⚠️ No captions loaded from dataset_info.csv!")
                continue

            print("\n🔍 **Select a caption for retrieval:**")
            random_captions = random.sample(captions, min(10, len(captions)))
            for i, cap in enumerate(random_captions, 1):
                print(f"{i}. {cap}")

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
            print("\n🔍 Retrieval results:")

            all_results = []
            metrics_rows = []

            # Test different k values
            for k in [10, 50, 100]:
                results = handler.search_by_text(text_prompt, k=k)

                real_count = sum(1 for path, _ in results if "/real/" in path or "real_" in os.path.basename(path))
                synthetic_count = k - real_count
                precision = real_count / k
                recall = real_count / k  # Since we're only using mixed dataset

                metrics_rows.append({
                    "Model": model_type,
                    "Rank": f"Top {k}",
                    "Precision": f"{precision:.2f}",
                    "Recall": f"{recall:.2f}",
                    "query": text_prompt
                })

                for rank, (path, caption) in enumerate(results, 1):
                    is_real = "/real/" in path or "real_" in os.path.basename(path)
                    result_type = "real" if is_real else "synthetic"

                    all_results.append({
                        "id": rank,
                        "model": model_type,
                        "image_path": path,
                        "caption": caption,
                        "type": result_type,
                        "rank": rank
                    })

                print(f"Top-{k}: {real_count} real ({precision:.2f} precision), {synthetic_count} synthetic")

            # === Save results
            with open(results_csv, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["id", "model", "image_path", "caption", "type", "rank"])
                writer.writeheader()
                writer.writerows(all_results)


            with open(metrics_csv, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["Model", "Rank", "Precision", "Recall", "query"])
                writer.writeheader()
                writer.writerows(metrics_rows)

            print(f"\n✅ Results saved to:")
            print(f"  - {results_csv}")
            print(f"  - {metrics_csv}")
            
            print("\nPress Enter to continue to next model/dataset...")
            input()

if __name__ == "__main__":
    main()
