import streamlit as st
from transformers import pipeline

# Set up the emotion classifier
@st.cache_resource
def load_pipeline():
    return pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=False)

classifier = load_pipeline()

st.title("🧠 Emotion Detector")
st.write("Enter a sentence and see what emotion is detected!")

# User input
user_input = st.text_area("Enter your sentence:", "")

if st.button("Analyze"):
    if user_input.strip():
        result = classifier(user_input)[0]
        label = result["label"]
        score = result["score"]
        st.success(f"**Emotion Detected:** {label} ({score:.2f})")
    else:
        st.warning("Please enter some text to analyze.")
