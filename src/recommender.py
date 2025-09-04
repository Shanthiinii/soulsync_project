import pandas as pd
import os
from sentence_transformers import SentenceTransformer, util

# Base directory (project root)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "combined_df.csv")


# Load model once
model = SentenceTransformer("all-MiniLM-L6-v2")

def load_dataset():
    df = pd.read_csv(DATA_PATH)

    # Normalize Type column
    if "Type" in df.columns:
        df["Type"] = df["Type"].str.strip().str.lower()  # normalize

    # Create Combined_text if not present
    if "Combined_text" not in df.columns:
        df["Combined_text"] = (
            df["Title"].astype(str)
            + " "
            + df.get("Description", "").astype(str)
            + " "
            + df.get("Type", "").astype(str)
        )
    return df


def semantic_recommend(df, Emotion, category="all", top_k=3):
    Emotion = Emotion.strip()
    category = category.strip().lower()

    category_map = {"books": "book", "songs": "songs", "movies": "movie"}
    if category in category_map:
        category = category_map[category]

    # Filter category if not "all"
    if category != "all":
        df = df[df["Type"] == category]

    if df.empty:
        return [{"message": f"No {category} found for mood '{Emotion}'"}]

    # Encode mood
    mood_embedding = model.encode(Emotion, convert_to_tensor=True)

    # ---- If category = all → pick one best from each type ----
    if category == "all":
        recs_all = []
        for t in ["book", "songs", "movie"]:
            subset = df[df["Type"] == t]
            if not subset.empty:
                subset_embeddings = model.encode(
                    subset["Combined_text"].tolist(), convert_to_tensor=True
                )
                subset_scores = util.pytorch_cos_sim(mood_embedding, subset_embeddings)[0]

                # pick best unique result
                best_idx = subset_scores.argmax().item()
                best_rec = subset.iloc[best_idx].to_dict()

                # avoid duplicates
                if best_rec not in recs_all:
                    recs_all.append(best_rec)

        return recs_all

    # ---- Otherwise, top-k results from chosen category ----
    dataset_embeddings = model.encode(
        df["Combined_text"].tolist(), convert_to_tensor=True
    )
    scores = util.pytorch_cos_sim(mood_embedding, dataset_embeddings)[0]

    top_results = scores.topk(k=min(top_k, len(df)))
    recommendations = []
    for idx in top_results.indices:
        rec = df.iloc[idx.item()].to_dict()
        if rec not in recommendations:  # avoid duplicates
            recommendations.append(rec)

    return recommendations
