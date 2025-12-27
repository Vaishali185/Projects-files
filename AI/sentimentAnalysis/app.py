import streamlit as st
from transformers import pipeline

# Load model once (important for performance)
@st.cache_resource
def load_model():
    return pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )

sentiment_pipeline = load_model()

st.title("💬 Sentiment Analysis App")
st.write("Enter text below to analyze its sentiment using a Hugging Face model.")

user_input = st.text_area("Enter your text here:")

if st.button("Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        result = sentiment_pipeline(user_input)[0]

        label = result["label"]
        score = result["score"]

        st.success(f"**Sentiment:** {label}")
        st.write(f"**Confidence:** {score:.2f}")
