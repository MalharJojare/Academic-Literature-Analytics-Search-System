# src/build_index.py
import os
import joblib
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from src.preprocess import load_and_prepare_data

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

def build_tfidf_embeddings(texts, max_features: int = 5000):
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        stop_words="english"
    )
    X = vectorizer.fit_transform(texts)
    return vectorizer, X

def main():
    # 1. Load & preprocess
    df = load_and_prepare_data()

    # 2. Build TF-IDF
    print("Building TF-IDF embeddings...")
    vectorizer, X = build_tfidf_embeddings(df["text_clean"].tolist())

    # 3. Save vectorizer and embeddings
    print("Saving vectorizer and embeddings...")
    joblib.dump(vectorizer, os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib"))
    sparse.save_npz(os.path.join(MODELS_DIR, "abstract_embeddings.npz"), X)

    # 4. Save metadata for the app
    meta_cols = [
        "paper_id",
        "title",
        "authors_str",
        "venue",
        "year",
        "n_citation",
        "references_str",
        "abstract",
    ]
    # Keep only columns that exist
    meta_cols = [c for c in meta_cols if c in df.columns]
    df[meta_cols].to_csv(
        os.path.join(MODELS_DIR, "papers_metadata.csv"),
        index=False
    )

    print("Index built successfully.")
    print(f"Saved to: {MODELS_DIR}")

if __name__ == "__main__":
    main()
