# app/streamlit_app.py
import os
import numpy as np
import pandas as pd
import joblib
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

@st.cache_data
def load_metadata():
    meta_path = os.path.join(MODELS_DIR, "papers_metadata.csv")
    return pd.read_csv(meta_path)

@st.cache_resource
def load_vectorizer_and_embeddings():
    vectorizer = joblib.load(os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib"))
    X = sparse.load_npz(os.path.join(MODELS_DIR, "abstract_embeddings.npz"))
    return vectorizer, X

def search_papers(query: str, top_k: int = 5):
    df_meta = load_metadata()
    vectorizer, X = load_vectorizer_and_embeddings()

    query_clean = query.lower()
    q_vec = vectorizer.transform([query_clean])

    sims = cosine_similarity(q_vec, X).flatten()
    top_idx = np.argsort(sims)[::-1][:top_k]

    results = df_meta.iloc[top_idx].copy()
    results["similarity"] = sims[top_idx]

    return results

def main():
    st.title("📚 Academic Literature Search Engine")
    st.write("Enter a research topic to find relevant papers from your dataset.")

    query = st.text_input(
        "🔍 Enter your query:",
        value="convolutional neural networks for image classification"
    )

    top_k = st.slider("Number of results", min_value=3, max_value=25, value=5)

    if st.button("Search") or query:
        results = search_papers(query, top_k)

        st.subheader("🔎 Top Results")
        for _, row in results.iterrows():
            st.markdown(f"### {row['title']} ({int(row['year'])})")
            st.markdown(f"**Authors:** {row['authors_str']}")
            st.markdown(f"**Venue:** {row['venue']}")
            st.markdown(f"**Citations:** {row['n_citation']}")
            st.markdown(f"**Relevance score:** {row['similarity']:.4f}")

            with st.expander("Abstract"):
                st.write(row["abstract"])

            st.markdown("---")

        # Optional visualization
        st.subheader("📊 Year Distribution (Top Results)")
        year_counts = results.groupby("year")["paper_id"].count()
        st.bar_chart(year_counts)

if __name__ == "__main__":
    main()
