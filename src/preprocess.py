# src/preprocess.py
import os
import re
import ast
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "papers.csv")

def clean_text(text: str) -> str:
    """Basic text cleaning: lowercase, remove non-alphanumeric, collapse spaces."""
    if pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def parse_list_like(value):
    """
    Handles cases like:
    - ["Author1", "Author2"] stored as a string
    - comma-separated strings
    - already clean text
    """
    if pd.isna(value):
        return ""
    val = str(value).strip()

    # If it looks like a Python list (starts with [ and ends with ])
    if val.startswith("[") and val.endswith("]"):
        try:
            parsed = ast.literal_eval(val)
            if isinstance(parsed, list):
                return ", ".join(str(x) for x in parsed)
        except Exception:
            pass  # fall back to original

    # Otherwise return as-is
    return val

def load_and_prepare_data(csv_path: str = DATA_PATH) -> pd.DataFrame:
    print(f"Loading CSV from: {csv_path}")
    df = pd.read_csv(csv_path)

    # Standardize column names if needed
    # rename id -> paper_id for clarity
    if "id" in df.columns and "paper_id" not in df.columns:
        df = df.rename(columns={"id": "paper_id"})

    # Ensure required columns exist
    required_cols = ["paper_id", "title", "abstract"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Clean / normalize authors & references
    if "authors" in df.columns:
        df["authors_str"] = df["authors"].apply(parse_list_like)
    else:
        df["authors_str"] = ""

    if "references" in df.columns:
        df["references_str"] = df["references"].apply(parse_list_like)
    else:
        df["references_str"] = ""

    # Combined text for search
    df["text_raw"] = (
        df["title"].fillna("") + " " + df["abstract"].fillna("")
    )
    df["text_clean"] = df["text_raw"].apply(clean_text)

    # Make sure year is integer where possible
    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")

    # Fill missing numeric citations
    if "n_citation" in df.columns:
        df["n_citation"] = pd.to_numeric(df["n_citation"], errors="coerce").fillna(0).astype(int)

    print("Data loaded and processed. Sample:")
    print(df[["paper_id", "title", "year"]].head())

    return df
