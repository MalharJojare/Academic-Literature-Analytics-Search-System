# src/build_clusters.py
import os
import numpy as np
import pandas as pd
import joblib
from scipy import sparse
from sklearn.cluster import MiniBatchKMeans

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

META_PATH = os.path.join(MODELS_DIR, "papers_metadata.csv")
EMB_PATH = os.path.join(MODELS_DIR, "abstract_embeddings.npz")
VEC_PATH = os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib")

OUTPUT_PAPERS = os.path.join(MODELS_DIR, "papers_with_clusters.csv")
OUTPUT_CLUSTERS = os.path.join(MODELS_DIR, "cluster_summary.csv")

def main(
    n_clusters: int = 20,
    batch_size: int = 1000,
):
    print("Loading metadata and embeddings...")
    df = pd.read_csv(META_PATH)
    X = sparse.load_npz(EMB_PATH)
    vectorizer = joblib.load(VEC_PATH)

    print(f"Running MiniBatchKMeans with {n_clusters} clusters...")
    mbk = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=42,
        batch_size=batch_size,
        n_init=5
    )
    labels = mbk.fit_predict(X)

    df["cluster"] = labels

    print("Saving papers_with_clusters.csv ...")
    df.to_csv(OUTPUT_PAPERS, index=False)

    # ---- Build cluster-level summary ----
    print("Building cluster summary (top terms + stats)...")
    terms = np.array(vectorizer.get_feature_names_out())

    cluster_rows = []
    for k in range(n_clusters):
        center = mbk.cluster_centers_[k]
        top_idx = np.argsort(center)[::-1][:10]
        top_terms = ", ".join(terms[top_idx])

        cluster_rows.append({
            "cluster": k,
            "top_terms": top_terms,
        })

    df_clusters = pd.DataFrame(cluster_rows)

    # Aggregate stats per cluster
    agg_cols = {
        "paper_id": "count",
    }
    if "n_citation" in df.columns:
        agg_cols["n_citation"] = "mean"
    if "year" in df.columns:
        agg_cols["year"] = ["min", "max"]

    stats = df.groupby("cluster").agg(agg_cols)

    # Flatten MultiIndex columns if needed
    stats.columns = [
        "_".join(col) if isinstance(col, tuple) else col
        for col in stats.columns
    ]
    stats = stats.reset_index()

    if "paper_id_count" in stats.columns:
        stats = stats.rename(columns={"paper_id_count": "n_papers"})
    if "n_citation_mean" in stats.columns:
        stats = stats.rename(columns={"n_citation_mean": "avg_citations"})
    if "year_min" in stats.columns:
        stats = stats.rename(columns={"year_min": "min_year"})
    if "year_max" in stats.columns:
        stats = stats.rename(columns={"year_max": "max_year"})

    summary = df_clusters.merge(stats, on="cluster", how="left")

    print("Saving cluster_summary.csv ...")
    summary.to_csv(OUTPUT_CLUSTERS, index=False)

    print("\nDone.")
    print(f"- Papers with clusters: {OUTPUT_PAPERS}")
    print(f"- Cluster summary:     {OUTPUT_CLUSTERS}")

if __name__ == "__main__":
    main()
