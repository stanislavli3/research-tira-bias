import os
import numpy as np
import pandas as pd
from glob import glob
from PIL import Image
from sentence_transformers import SentenceTransformer
import faiss

# === Config ===
IMAGES_DIR = "research-tira-bias/main_code/new_data/mixed"
CSV_PATH = "/home/stl227/research-tira-bias/main_code/new_data/dataset_info.csv"
SAVE_DIR = "research-tira-bias/embeddings"
os.makedirs(SAVE_DIR, exist_ok=True)

# === Load Model ===
model = SentenceTransformer("clip-ViT-B-32")

# === Load Captions ===
def load_caption_dict(csv_path):
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    return dict(zip(df["image_name"], df["caption"]))

# === Generate & Save Embeddings ===
def generate_and_store_embeddings(images_dir, caption_dict, save_dir):
    image_paths = glob(os.path.join(images_dir, "**/*.*"), recursive=True)
    image_paths = [p for p in image_paths if p.lower().endswith(('.jpg', '.jpeg', '.png'))]

    vectors, valid_paths, captions = [], [], []

    for img_path in image_paths:
        fname = os.path.basename(img_path)
        if fname not in caption_dict:
            continue
        caption = caption_dict[fname]
        embedding = model.encode(caption)
        vectors.append(embedding)
        valid_paths.append(img_path)
        captions.append(caption)

    vectors = np.array(vectors).astype(np.float32)
    ids = np.arange(len(vectors))

    df = pd.DataFrame({
        "id": ids,
        "image_path": valid_paths,
        "caption": captions,
        "type": ["synthetic" if "synthetic" in p.lower() else "real" for p in valid_paths]
    })

    df.to_csv(os.path.join(save_dir, "image_info.csv"), index=False)
    np.save(os.path.join(save_dir, "vectors.npy"), vectors)
    np.save(os.path.join(save_dir, "ids.npy"), ids)

    return vectors, ids, df

# === Build FAISS Index ===
def build_faiss_index(vectors, ids, save_dir):
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index = faiss.IndexIDMap(index)
    index.add_with_ids(vectors, ids)
    faiss.write_index(index, os.path.join(save_dir, "faiss.index"))
    return index

# === Load FAISS + Metadata ===
def load_faiss_and_metadata(save_dir):
    index = faiss.read_index(os.path.join(save_dir, "faiss.index"))
    df = pd.read_csv(os.path.join(save_dir, "image_info.csv"))
    vectors = np.load(os.path.join(save_dir, "vectors.npy"))
    return index, df, vectors

# === Query ===
def retrieve_text_query(query, model, index, df, top_k=100):
    q_vec = model.encode(query)
    q_vec = np.array(q_vec).astype(np.float32).reshape(1, -1)
    distances, indices = index.search(q_vec, top_k)
    results = df.iloc[indices[0]].copy()
    results["rank"] = np.arange(1, len(results) + 1)
    return query, results

# === Metrics ===
def compute_metrics(results, target_type="real", ranks=(10, 50, 100)):
    metrics = []
    total_relevant = (results["type"] == target_type).sum()

    for k in ranks:
        top_k = results.head(k)
        relevant = (top_k["type"] == target_type).sum()
        precision = relevant / k
        recall = relevant / total_relevant if total_relevant > 0 else 0
        metrics.append({"Rank": f"Top {k}", "Precision": precision, "Recall": recall})

    return pd.DataFrame(metrics)

# === Full Pipeline ===
def run():
    print("📄 Loading captions...")
    full_df = pd.read_csv(CSV_PATH)
    full_df.columns = full_df.columns.str.strip()
    caption_dict = dict(zip(full_df["image_name"], full_df["caption"]))

    print("🔢 Generating embeddings...")
    vectors, ids, df = generate_and_store_embeddings(IMAGES_DIR, caption_dict, SAVE_DIR)

    print("🧠 Building FAISS index...")
    index = build_faiss_index(vectors, ids, SAVE_DIR)

    print("🎯 Selecting 3 random unique captions for query:")

    random_captions = full_df["caption"].drop_duplicates().sample(3, random_state=np.random.randint(0, 10000))
    for i, caption in enumerate(random_captions, 1):
        print(f"   {i}. \"{caption}\"")


    for i, query_text in enumerate(random_captions, 1):
        print(f"\n🔎 Query {i}: \"{query_text}\"\n")
        _, results = retrieve_text_query(query_text, model, index, df)

        print("🏆 Top 10 Results:")
        print(results[["rank", "image_path", "caption", "type"]].head(10))

        print("\n🏅 Top 50 Results:")
        print(results[["rank", "image_path", "caption", "type"]].head(50))

        print("\n🎖 Top 100 Results:")
        print(results[["rank", "image_path", "caption", "type"]])

        metrics = compute_metrics(results)
        metrics["query"] = query_text
        print("\n📊 Precision & Recall:\n")
        print(metrics)

        # Save results per query
        safe_name = f"query_{i}".replace(" ", "_")
        results.to_csv(os.path.join(SAVE_DIR, f"{safe_name}_results.csv"), index=False)
        metrics.to_csv(os.path.join(SAVE_DIR, f"{safe_name}_metrics.csv"), index=False)


if __name__ == "__main__":
    run()
