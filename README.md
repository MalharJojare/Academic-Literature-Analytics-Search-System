# 📚 Academic Literature Analytics & Search System  
*A complete end-to-end pipeline for academic literature retrieval, clustering, and interactive analytics.*

---

## 🚀 Overview  
This project builds a complete **Academic Literature Analytics & Search System** capable of:

- Processing large-scale academic datasets (1.6 GB+)  
- Cleaning and preparing metadata  
- Generating TF-IDF embeddings  
- Performing topic clustering  
- Running a semantic-style search interface  
- Visualizing insights through a professional Power BI dashboard  

It transforms raw literature into **actionable research insights**.

---

## 🔍 Key Features

### 🔹 1. Data Processing & Cleaning  
- Handles large-scale datasets  
- Normalizes titles, abstracts, authors, references  
- Produces structured CSV metadata  

### 🔹 2. Search Engine  
- TF-IDF vectorization  
- Cosine similarity ranking  
- Top-K paper retrieval  

### 🔹 3. Topic Clustering  
- MiniBatchKMeans for large-scale clustering  
- Extracts top keywords per cluster  
- Generates cluster summary tables  

### 🔹 4. Streamlit Web Application  
- Search bar interface  
- Ranked paper display  
- Abstract preview  
- Cluster statistics  

### 🔹 5. Power BI Dashboard  
Includes:

- Total Papers  
- Total Clusters  
- Avg Citations  
- Most Popular Cluster  
- Highest Citation Cluster  
- Papers per Cluster  
- Papers per Year  
- Paper Explorer  
- Interactive slicers  


## 🗂️ Project Structure
```
Academic_Search_System/
│
├── app/
│ └── streamlit_app.py # Search UI
│
├── data/
│ └── papers.csv # Raw dataset (not included)
│ Dataset source: https://www.kaggle.com/datasets/nechbamohammed/research-papers-dataset

│
├── models/ # Excluded from repo
│ ├── tfidf_vectorizer.joblib
│ ├── abstract_embeddings.npz
│ ├── papers_metadata.csv
│ ├── papers_with_clusters.csv
│ └── cluster_summary.csv
│
├── src/
│ ├── preprocess.py # Data cleaning + normalization
│ ├── build_index.py # TF-IDF + embeddings builder
│ └── build_clusters.py # Topic clustering pipeline
│
└── README.md
```

## 🧪 Installation & Setup

### **1. Clone the repository**
```
git clone https://github.com/<your-username>/Academic-Literature-Analytics-Search-System.git
cd Academic-Literature-Analytics-Search-System
```
### **2. Create virtual environment**
```
python -m venv venv
source venv/bin/activate # Mac/Linux
venv\Scripts\activate # Windows
```
### **3. Running the Pipeline**
```
python -m src.preprocess
python -m src.build_index
python -m src.build_clusters
```
### **4. Run the Streamlit App**
```
streamlit run app/streamlit_app.py
```

## 👤 Author  
**Malhar Jojare**  
Graduate Student — Data Science  
Michigan Technological University  
🔗 LinkedIn: https://linkedin.com/in/malharjojare  



