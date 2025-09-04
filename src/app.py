import streamlit as st
from recommender import load_dataset, semantic_recommend

# Load dataset once
df = load_dataset()

st.title("🎵 SoulSync - Mood-based Recommendations")

# User inputs
Emotion = st.text_input("Enter your mood (e.g., happy, sad, nostalgic):")
category = st.selectbox(" Choose category:", ["All", "Books", "Songs", "Movies"])
category = category.lower()

if st.button("Get Recommendations"):
    if not Emotion:
        st.warning("Please enter your mood first.")
    else:
        recs = semantic_recommend(df, Emotion, category, top_k=3)

        if recs and "message" in recs[0]:
            st.error(recs[0]["message"])
        else:
            st.subheader("Your Recommendations:")
            for r in recs:
                st.markdown(f"**Title:** {r.get('Title', 'Unknown')}")
                st.markdown(f"**Type:** {r.get('Type', 'Unknown')}")
                if "Description" in r and str(r["Description"]).strip():
                    st.markdown(f"**Description:** {r['Description']}")
                st.markdown("---")
