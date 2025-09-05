from transformers import pipeline


# Load once
classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=1)

def classify_mood(text):
    """
    Classifies mood from text using a pre-trained emotion model.
    """
    result = classifier(text)[0][0]  # top prediction
    return result['label'].lower()  # e.g., 'joy', 'sadness', 'anger'
