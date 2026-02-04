import streamlit as st
import joblib
import re

# Load saved model & vectorizer
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"\bnot\s+(\w+)\s+(\w+)", r"not_\1 not_\2", text)
    text = re.sub(r"\bnot\s+(\w+)", r"not_\1", text)
    text = re.sub(r"[^a-zA-Z_ ]", "", text)
    return text

st.set_page_config(page_title="Sentiment Analysis App", layout="centered")
st.title("📝 Sentiment Analysis App")
st.write("Paste a product review and get sentiment prediction")

review = st.text_area("Enter review text")

if st.button("Predict Sentiment"):
    if review.strip() == "":
        st.warning("Please enter some text")
    else:
        cleaned = clean_text(review)
        vec = vectorizer.transform([cleaned])
        pred = model.predict(vec)[0]

        if pred == 1:
            st.success("✅ Positive Review")
        else:
            st.error("❌ Negative Review")

