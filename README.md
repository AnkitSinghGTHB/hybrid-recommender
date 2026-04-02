<p align="center">
  <img src="https://img.shields.io/badge/python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/FastAPI-0.100+-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI">
  <img src="https://img.shields.io/badge/Supabase-PostgreSQL-3FCF8E?style=for-the-badge&logo=supabase&logoColor=white" alt="Supabase">
  <img src="https://img.shields.io/badge/scikit--learn-ML-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white" alt="scikit-learn">
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge" alt="MIT License">
</p>

# HybridRec — Hybrid Recommender System

A production-ready **Hybrid Recommender System** that combines **Content-Based Filtering (TF-IDF)**, **Collaborative Filtering (SVD)**, and **NLP Sentiment Analysis (VADER)** with a weighted scoring engine, PostgreSQL full-text search, and a modern Amazon-like web interface.

> **25,000+ products** · **Sub-50ms search** · **Cold-start resilient** · **Supabase-backed**

---

## ✨ Demo

```
┌─────────────────────────────────────────────────────────────────┐
│  ◆ HybridRec        🔍 Search products...         👤 Sign In   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│            Discover products you'll love                        │
│     AI-powered recommendations using content analysis,          │
│       collaborative filtering, and sentiment intelligence.      │
│                                                                 │
│            ● Ready — 25,450 products                            │
│                                                                 │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐       │
│  │ BOOKS    │  │ SPORTS   │  │ SPORTS   │  │ AUTO     │       │
│  │          │  │          │  │          │  │          │       │
│  │ Wireless │  │ Classic  │  │ Ergono.. │  │ Classic  │       │
│  │ Guide    │  │ Jersey X │  │ Racket   │  │ DashCam  │       │
│  │ ★★★★★  │  │ ★★★★★  │  │ ★★★★★  │  │ ★★★★★  │       │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘       │
└─────────────────────────────────────────────────────────────────┘
```

---

## Architecture

```
User Reviews (text)           ──→  NLP Engine (VADER Sentiment)    ──┐
Item Metadata (title/desc)    ──→  Content Vectorization (TF-IDF)  ──┤──→ Weighted Hybrid ──→ Ranked Results
User Purchases (clicks/buys)  ──→  Matrix Factorization (SVD)      ──┘         Score

                    Hybrid Score = α·content + β·collab + γ·sentiment
```

### How It Works

#### 1. Data Adapter (`data_adapter.py`)
- **Dynamic Column Detection** — fuzzy-matches column names (`product_name` → `title`, `stars` → `rating`)
- **Multi-Format** — CSV and JSON with automatic encoding fallback (UTF-8 → Latin-1 → CP1252)
- **Fault-Tolerant** — skips bad lines, imputes missing values, handles NaN/Infinity gracefully

#### 2. Content Score — TF-IDF & Cosine Similarity
- Item metadata (`title` + `description` + `category`) vectorized with TF-IDF (unigrams + bigrams, max 5000 features)
- On-the-fly cosine similarity yields `content_score` ∈ [0, 1]

#### 3. Collaborative Score — Truncated SVD
- User-item interaction matrix built from purchases + implicit feedback (views, clicks)
- SVD reduces to 50 latent factors; cosine similarity in latent space yields `collab_score`
- **Adaptive rank** — automatically reduces SVD components for sparse matrices

#### 4. Sentiment Score — NLTK VADER
- Review text analyzed for compound polarity ∈ [-1, 1]
- Per-item aggregation → Min-Max normalization → `sentiment_score` ∈ [0, 1]

#### 5. Cold-Start Handling
- **Bayesian average rating** — prevents 1-review 5-star bias
- **Popularity-based fallback** — new items ranked by review count and category similarity
- **Mock user seeding** — synthetic purchase history to bootstrap collaborative filtering

---

## Features

| Feature | Description |
|---------|-------------|
| 🔍 **PostgreSQL FTS** | Full-text search with GIN indexes — sub-50ms on 250k+ rows |
| 🔐 **Supabase Auth** | Guest (anonymous) and email/password authentication |
| 📊 **3 ML Models** | Content (TF-IDF), Collaborative (SVD), Sentiment (VADER) |
| ⚖️ **Tunable Weights** | Live α/β/γ sliders to adjust recommendation blend |
| 📦 **Multi-Format Upload** | CSV and JSON with auto-column detection |
| ⌨️ **Type-to-Search** | Global keyboard capture — start typing anywhere to search |
| 🎨 **Modern UI** | Amazon-inspired dark header, card grid, skeleton loaders |
| 📱 **Responsive** | 4 → 3 → 2 → 1 column grid across breakpoints |
| 🛡️ **Row-Level Security** | Supabase RLS policies on all tables |
| 🧊 **Cold-Start Resilient** | Bayesian avg + popularity fallback for new items |

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| **Backend** | Python 3.10+, FastAPI, Uvicorn |
| **Database** | Supabase (PostgreSQL), Row-Level Security |
| **Search** | PostgreSQL Full-Text Search (GIN indexes, `ts_rank`) |
| **Auth** | Supabase Auth (anonymous + email/password) |
| **ML Models** | scikit-learn (TF-IDF, TruncatedSVD, cosine similarity) |
| **NLP** | NLTK VADER SentimentIntensityAnalyzer |
| **Data** | Pandas, NumPy, SciPy (sparse matrices) |
| **Frontend** | HTML5, CSS3, Vanilla JavaScript, Supabase JS v2 |

---

## Project Structure

```
hybrid-recommender/
├── backend/
│   └── main.py                 # FastAPI server — search, upload, build, recommend
├── frontend/
│   ├── index.html              # Single-page UI (Amazon-like layout)
│   ├── styles.css              # Design system (dark header, cards, animations)
│   └── app.js                  # Frontend logic (auth, search, rendering)
├── scripts/
│   ├── generate_sample_data.py # Synthetic test dataset generator
│   ├── import_to_supabase.py   # Batch import CSV/JSON → PostgreSQL
│   └── seed_mock_data.py       # Create mock users + purchases (cold-start)
├── datasets/                   # CSV/JSON data files
├── data_adapter.py             # Auto column detection + schema normalization
├── db.py                       # Supabase client singleton (anon + admin)
├── content_model.py            # TF-IDF content-based recommender
├── collaborative_model.py      # SVD collaborative recommender + implicit feedback
├── hybrid_model.py             # Weighted hybrid engine (Bayesian avg, popularity)
├── nlp_engine.py               # VADER sentiment analysis
├── evaluation.py               # Precision@K, Recall@K, NDCG@K benchmarks
├── requirements.txt            # Python dependencies
├── .env.example                # Environment variable template
└── SETUP.md                    # Detailed setup guide
```

---

## Quick Start

### Prerequisites

- Python 3.10+
- A [Supabase](https://supabase.com) account (free tier works)

### 1. Clone & Install

```bash
git clone https://github.com/your-username/hybrid-recommender.git
cd hybrid-recommender
python -m pip install -r requirements.txt
```

### 2. Configure Supabase

Create a `.env` file from the template:

```bash
cp .env.example .env
```

Fill in your Supabase credentials (from [Supabase Dashboard → Settings → API](https://supabase.com/dashboard)):

```env
SUPABASE_URL=https://your-project-ref.supabase.co
SUPABASE_ANON_KEY=your-anon-key
SUPABASE_SERVICE_KEY=your-service-role-key   # Optional — needed for bulk import scripts
```

### 3. Set Up Database

Run the SQL migrations in your Supabase SQL Editor — see [`SETUP.md`](SETUP.md) for the full schema.

### 4. Start the Server

```bash
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

### 5. Use It

Open **http://localhost:8000**:

1. Click **Upload Dataset** → select any CSV/JSON file from `datasets/`
2. Click **Build Models** → trains TF-IDF, SVD, and sentiment models
3. **Type anywhere** to search → get instant results via PostgreSQL FTS
4. Click a product → see hybrid recommendations

---

## Data Import (Optional — Bulk)

For large datasets (250k+ rows), use the import script:

```bash
# Import a specific file
python scripts/import_to_supabase.py --file datasets/Books.csv --batch-size 2000

# Import all files in datasets/
python scripts/import_to_supabase.py

# Seed mock users and purchases (solves cold-start)
python scripts/seed_mock_data.py --users 50 --purchases 2000
```

> **Note**: Bulk import scripts require `SUPABASE_SERVICE_KEY` to be set in `.env`.

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/config` | Supabase public config (for frontend init) |
| `GET` | `/api/status` | System status + product count |
| `GET` | `/api/search?q=...&limit=20` | Full-text search (PostgreSQL FTS) |
| `POST` | `/api/upload` | Upload CSV/JSON dataset |
| `POST` | `/api/build` | Build recommendation models |
| `GET` | `/api/recommend/{title}` | Get hybrid recommendations |
| `GET` | `/api/items?page=1&per_page=50` | Paginated product list |
| `GET` | `/api/categories` | List all categories |
| `GET/PUT` | `/api/weights` | Get/update α, β, γ weights |
| `GET` | `/api/purchases/{user_id}` | User purchase history |
| `POST` | `/api/purchases` | Record a purchase |

---

## Evaluation

```bash
python evaluation.py
```

Benchmarks Content-Only, Collab-Only, Sentiment-Only, and Hybrid configurations using:
- **Precision@K** — fraction of relevant items in top-K
- **Recall@K** — fraction of relevant items retrieved
- **NDCG@K** — ranking quality (discounted cumulative gain)

---

## Security

- ✅ No hardcoded credentials — config served via `/api/config`
- ✅ `.env` excluded from git (`.gitignore`)
- ✅ CORS restricted to configured origins
- ✅ Row-Level Security (RLS) on all Supabase tables
- ✅ Input validation via Pydantic models
- ✅ Generic error messages to clients (no stack trace leakage)
- ✅ SQL injection safe (Supabase SDK parameterized queries)

---

## License

[MIT](LICENSE)
