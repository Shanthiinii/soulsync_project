import streamlit as st
from recommender import load_dataset, recommend_from_text

# Load dataset once
df = load_dataset()

st.title("🎵 SoulSync - Mood-based Recommendations")

# User inputs
user_input_text = st.text_area("How are you feeling today? (Write anything — we'll figure out your mood!)")
category = st.selectbox(" Choose category:", ["All", "Books", "Songs", "Movies"])
category = category.lower()

if st.button("Get Recommendations"):
    if not user_input_text:
        st.warning("Please describe how you're feeling.")
    else:
        recs,detected_mood = recommend_from_text(df, user_input_text, category, top_k=3)
        # 🧠 Display the detected mood to the user
        st.info(f"Detected Mood: {detected_mood.capitalize()}")

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
