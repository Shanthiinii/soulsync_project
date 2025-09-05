import pandas as pd
import os
from sentence_transformers import SentenceTransformer, util
from data_loader import load_combined_dataset
from mood_classifier import classify_mood

# Base directory (project root)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "combined_df.csv")


# Load model once
model = SentenceTransformer("all-MiniLM-L6-v2")

def load_dataset():
    df = load_combined_dataset()

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

def recommend_from_text(df, user_input_text, category="all", top_k=3):
    """
    Takes raw user text (e.g., journal entry), classifies mood,
    then recommends content based on that mood.
    """
    detected_mood = classify_mood(user_input_text)
    print(f"Detected mood: {detected_mood}")
    
    recs=semantic_recommend(df, Emotion=detected_mood, category=category, top_k=top_k)
    return recs,detected_mood


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
        seen_titles = set()
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
                title = best_rec.get("Title", "").strip().lower()

                # avoid duplicates
                if title and title not in seen_titles:
                    recs_all.append(best_rec)
                    seen_titles.add(title)

        return recs_all

    # ---- Otherwise, top-k results from chosen category ----
    dataset_embeddings = model.encode(
        df["Combined_text"].tolist(), convert_to_tensor=True
    )
    scores = util.pytorch_cos_sim(mood_embedding, dataset_embeddings)[0]

    top_results = scores.topk(k=min(top_k, len(df)))
    recommendations = []
    seen_titles = set()
    for idx in top_results.indices:
        rec = df.iloc[idx.item()].to_dict()
        title = rec.get("Title", "").strip().lower()
        if title and title not in seen_titles:
            recommendations.append(rec)
            seen_titles.add(title)

    return recommendations
