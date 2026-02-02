import streamlit as st
import requests
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

st.set_page_config(page_title="News Clustering App")

st.title("📰 News Clustering App")

API_KEY = st.secrets["GNEWS_API_KEY"]


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z ]", "", text)
    return text

if st.button("Fetch & Cluster News"):
    url = f"https://gnews.io/api/v4/top-headlines?lang=en&country=in&max=30&apikey={API_KEY}"
    data = requests.get(url).json()

    if "articles" in data:
        df = pd.DataFrame(data["articles"])
        df = df[["title", "description"]]

        df["text"] = df["title"] + " " + df["description"]
        df["clean_text"] = df["text"].apply(clean_text)

        vectorizer = TfidfVectorizer(stop_words="english", max_features=1000)
        X = vectorizer.fit_transform(df["clean_text"])

        kmeans = KMeans(n_clusters=3, random_state=42)
        df["cluster"] = kmeans.fit_predict(X)

        for i in range(3):
            st.subheader(f"Cluster {i}")
            for title in df[df["cluster"] == i]["title"].head(5):
                st.write("•", title)
    else:
        st.error("Failed to fetch news. Check API key or limit.")

