# Hybrid Recommender System

A **Hybrid Recommender System** combining **Content-Based Filtering**, **Collaborative Filtering (SVD)**, and **NLP Sentiment Analysis** with a weighted scoring engine and modern web UI.

---

## Architecture

```
User Reviews (rating + text)  ──→  NLP Engine (VADER Sentiment)    ──┐
Item Metadata (title/desc)    ──→  Content Vectorization (TF-IDF)  ──┤──→ Weighted Hybrid Score ──→ Ranked Results
User Behavior (clicks/views)  ──→  Matrix Factorization (SVD)      ──┘
```

**Hybrid Score** = `α × content_score + β × collab_score + γ × sentiment_score`

### How It Works

#### 1. Data Adapter (CSV Parser)
The system uses `data_adapter.py` to seamlessly ingest any CSV:
- **Dynamic Column Detection**: Scans and fuzzy-matches column headings against predefined lists (e.g., mapping `product_name`, `name`, or `title` to a standard unified `title` field).
- **Unified Schema**: Maps wildly different datasets into a single standardized schema containing `title`, `description`, `category`, `user_id`, `rating`, and `review_text`.
- **Fault-Tolerant Loading**: Employs `on_bad_lines='skip'` in Pandas to ignore corrupted rows, and aggressively imputes missing values (e.g. assigning `anonymous` to missing users or `0.0` to missing ratings) to prevent crashes on large, messy datasets.

#### 2. Model Scoring

**Content Score (TF-IDF & Cosine Similarity)**
- Metadata (`title` + `description` + `category`) is combined and processed via a **TF-IDF Vectorizer** (unigrams & bigrams, max 5000 features) to extract keyword importance.
- When an item is selected, we calculate the **Cosine Similarity** of its vector against all other item vectors *on the fly*, yielding a `content_score` from `0.0` to `1.0`.

**Collaborative Score (Truncated SVD)**
- User data (`user_id` & `rating`) is mapped into a heavily optimized **Sparse User-Item Matrix** (`scipy.sparse.csr_matrix`).
- **Matrix Factorization (Truncated SVD)** reduces the matrix to a dense layout of 50 latent factors, discovering hidden associations based purely on user voting patterns.
- Comparing items in this latent space via **Cosine Similarity** yields the `collab_score`.

**Sentiment Score (NLP VADER)**
- Text reviews (`review_text`) are analyzed by **NLTK VADER (Valence Aware Dictionary and sEntiment Reasoner)** to compute a compound polarity score between `[-1.0, 1.0]`.
- Scores are averaged per item and Min-Max normalized to `[0.0, 1.0]`. If a query yields zero matches, the system dynamically falls back to surfacing top-rated, highest-sentiment items.

---

## Features

- **Multi-Dataset Support** — upload & merge multiple CSV files with auto-column detection
- **NLP Sentiment Analysis** — VADER-based review sentiment scoring
- **Content-Based Filtering** — TF-IDF + cosine similarity on item metadata
- **Collaborative Filtering** — SVD matrix factorization on user-item interactions
- **Weighted Hybrid Scoring** — configurable α/β/γ weights with normalization & re-ranking
- **Modern Web UI** — dark-mode glassmorphism design with live search & score visualization
- **REST API** — FastAPI backend with upload, build, search, recommend endpoints
- **Evaluation** — Precision@K, Recall@K, NDCG@K across multiple weight configs

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Backend | Python, FastAPI, Uvicorn |
| ML Models | scikit-learn (TF-IDF, TruncatedSVD, cosine similarity) |
| NLP | NLTK (VADER SentimentIntensityAnalyzer) |
| Data | Pandas, NumPy |
| Frontend | HTML5, CSS3, Vanilla JavaScript |

---

## Project Structure

```
├── backend/
│   └── main.py               # FastAPI server + REST API
├── frontend/
│   ├── index.html             # Single-page web UI
│   ├── styles.css             # Premium dark-mode CSS
│   └── app.js                 # Frontend logic
├── scripts/
│   └── generate_sample_data.py # Synthetic dataset generator
├── datasets/                   # CSV data files
├── data_adapter.py            # Auto column detection & schema unification
├── dataset_manager.py         # Multi-dataset manager
├── nlp_engine.py              # VADER sentiment analysis
├── content_model.py           # TF-IDF content-based recommender
├── collaborative_model.py     # SVD collaborative recommender
├── hybrid_model.py            # Weighted hybrid scoring engine
├── evaluation.py              # Precision/Recall/NDCG evaluation
├── requirements.txt           # Python dependencies
└── README.md
```

---

## Quick Start

### 1. Install Dependencies

```bash
python -m pip install -r requirements.txt
python -m pip install pytest  # (optional for testing)
```

### 2. Prepare Datasets

Ensure your actual dataset files (`books.csv`, `booksdata.csv`, `ratings.csv`) are located in the `datasets/` folder. The application's Data Adapter will automatically detect their columns.

> **Note on Dataset Size**: The hybrid engine is optimized using sparse matrices and computes similarity scores on the fly to minimize memory usage! It is now built to smoothly handle large datasets (like the full `books.csv`) without triggering Out of Memory errors on standard machines.

*(Optional: If you don't have datasets, you can generate synthetic data)*
```bash
python scripts/generate_sample_data.py
```

### 3. Run the Server

```bash
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

### 4. Open the UI

Visit **http://localhost:8000** 
- In the UI, click **"Upload CSV"** and select `books.csv`, `booksdata.csv`, or `ratings.csv` from your datasets folder.
- Click **"Build Models"** to train the hybrid engine on the uploaded data.
- Use the **Search & Explore** functionality to get NLP + Content + Collaborative recommendations!

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/upload` | Upload a CSV dataset |
| `GET` | `/api/datasets` | List loaded datasets |
| `DELETE` | `/api/datasets/{id}` | Remove a dataset |
| `POST` | `/api/build` | Build recommendation models |
| `GET` | `/api/search?q=...` | Search items by keyword |
| `GET` | `/api/recommend/{title}` | Get hybrid recommendations |
| `GET/PUT` | `/api/weights` | Get/update α, β, γ weights |
| `GET` | `/api/items` | List all items (paginated) |
| `GET` | `/api/status` | Check model readiness |

---

## Evaluation

```bash
python evaluation.py
```

Compares Content-Only, Collab-Only, Sentiment-Only, and Hybrid configs using Precision@K, Recall@K, and NDCG@K.
