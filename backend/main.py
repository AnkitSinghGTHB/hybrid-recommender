"""
FastAPI Backend for the Hybrid Recommender System.
Serves REST API + static frontend files.
"""
import os
import sys

# Add parent directory to path so imports work when running as a module
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional
import io

from dataset_manager import DatasetManager
from nlp_engine import batch_analyze, aggregate_sentiment_by_item
from content_model import ContentRecommender
from collaborative_model import CollaborativeRecommender
from hybrid_model import HybridRecommender

# ── App ──────────────────────────────────────────────────────────────
app = FastAPI(title="Hybrid Recommender API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── State ────────────────────────────────────────────────────────────
dm = DatasetManager()
models = {
    "content": None,
    "collab": None,
    "hybrid": None,
    "ready": False,
    "item_df": None,
}


# ── Pydantic models ─────────────────────────────────────────────────
class WeightsUpdate(BaseModel):
    alpha: float = 0.4
    beta: float = 0.35
    gamma: float = 0.25


# ── Endpoints ────────────────────────────────────────────────────────

@app.get("/api/status")
def status():
    return {
        "status": "ready" if models["ready"] else "no_data",
        "datasets": dm.get_stats(),
    }


@app.post("/api/upload")
async def upload_dataset(file: UploadFile = File(...)):
    """Upload a CSV dataset."""
    if not file.filename.endswith('.csv'):
        raise HTTPException(400, "Only CSV files are supported.")
    try:
        contents = await file.read()
        buf = io.BytesIO(contents)
        ds_id = dm.load_csv(buf, name=file.filename)
        return {
            "id": ds_id,
            "name": file.filename,
            "datasets": dm.list_datasets(),
            "stats": dm.get_stats(),
        }
    except Exception as e:
        raise HTTPException(400, str(e))


@app.get("/api/datasets")
def list_datasets():
    return {"datasets": dm.list_datasets(), "stats": dm.get_stats()}


@app.delete("/api/datasets/{ds_id}")
def delete_dataset(ds_id: str):
    if dm.remove_dataset(ds_id):
        models["ready"] = False
        return {"message": "Removed", "stats": dm.get_stats()}
    raise HTTPException(404, "Dataset not found.")


@app.post("/api/build")
def build_models():
    """Build/rebuild recommendation models from all loaded datasets."""
    if dm.get_stats()['dataset_count'] == 0:
        raise HTTPException(400, "No datasets loaded. Upload a CSV first.")

    try:
        interaction_df, item_df = dm.merge_all()

        # NLP sentiment
        interaction_df = batch_analyze(interaction_df, 'review_text')
        sentiment_agg = aggregate_sentiment_by_item(interaction_df, 'title')

        # Merge sentiment into item_df
        item_df = item_df.merge(sentiment_agg, on='title', how='left')
        item_df['avg_sentiment'] = item_df['avg_sentiment'].fillna(0.0)

        # Content model (uses item-level data)
        content_model = ContentRecommender(item_df)

        # Collaborative model (uses interaction-level data)
        collab_model = None
        has_collab = (
            'user_id' in interaction_df.columns and
            interaction_df['user_id'].nunique() > 1 and
            'rating' in interaction_df.columns and
            interaction_df['rating'].sum() > 0
        )
        if has_collab:
            collab_model = CollaborativeRecommender(interaction_df)

        # Hybrid model
        hybrid_model = HybridRecommender(
            content_model, collab_model, item_df
        )

        models["content"] = content_model
        models["collab"] = collab_model
        models["hybrid"] = hybrid_model
        models["item_df"] = item_df
        models["ready"] = True

        return {
            "message": "Models built successfully!",
            "items": len(item_df),
            "has_collaborative": collab_model is not None,
            "has_sentiment": 'avg_sentiment' in item_df.columns,
        }
    except Exception as e:
        raise HTTPException(500, f"Model build error: {str(e)}")


@app.get("/api/search")
def search_items(q: str = "", top_n: int = 10):
    """Search items by query text."""
    hybrid = models["hybrid"]
    if hybrid is None or hybrid.item_df is None:
        return {"results": [], "is_fallback": False, "total": 0}

    results = []
    is_fallback = False
    
    if len(q) > 1:
        # Search via content model
        results = hybrid.content_model.search(q, top_n)
        
    # If no results (or query too short), fallback to top sentiment/rating
    if not results:
        is_fallback = True
        item_df = hybrid.item_df # Use hybrid's item_df
        # Fallback to top sentiment items
        if 'avg_sentiment' in item_df.columns and 'rating' in item_df.columns:
            fallback_df = item_df.sort_values(by=['avg_sentiment', 'rating'], ascending=[False, False]).head(top_n)
        elif 'rating' in item_df.columns:
            fallback_df = item_df.sort_values(by='rating', ascending=False).head(top_n)
        else:
            fallback_df = item_df.head(top_n)
            
        for _, row in fallback_df.iterrows():
            tp = row.get('top_reviews', [])
            top_reviews = tp if isinstance(tp, list) else []
            
            results.append({
                'title': row['title'],
                'score': 0.0,
                'item_id': str(row.get('item_id', '')),
                'category': row.get('category', ''),
                'description': str(row.get('description', ''))[:200],
                'top_reviews': top_reviews,
            })

    # Enrich with sentiment & rating if available
    item_df = hybrid.item_df
    for r in results:
        # We might already have it for fallback items, but for content search hits we need it
        if r['title'] in item_df['title'].values:
            row = item_df[item_df['title'] == r['title']].iloc[0]
            r['avg_sentiment'] = float(row.get('avg_sentiment', 0.0))
            r['rating'] = float(row.get('rating', 0.0))
        else:
            r['avg_sentiment'] = 0.0
            r['rating'] = 0.0

    return {"results": results, "is_fallback": is_fallback, "total": len(results)}


@app.get("/api/recommend/{item_title}")
def get_recommendations(item_title: str, top_n: int = 10):
    """Get hybrid recommendations for an item."""
    if not models["ready"]:
        raise HTTPException(400, "Models not built. Upload data and build first.")
    recs = models["hybrid"].recommend(item_title, top_n=top_n)
    if not recs:
        raise HTTPException(404, "Item not found or no recommendations available.")
    return {
        "query_item": item_title,
        "recommendations": recs,
        "weights": models["hybrid"].get_weights(),
    }


@app.get("/api/weights")
def get_weights():
    if not models["ready"]:
        return {"alpha": 0.4, "beta": 0.35, "gamma": 0.25}
    return models["hybrid"].get_weights()


@app.put("/api/weights")
def update_weights(w: WeightsUpdate):
    if not models["ready"]:
        raise HTTPException(400, "Models not built yet.")
    models["hybrid"].set_weights(w.alpha, w.beta, w.gamma)
    return {"message": "Weights updated", "weights": models["hybrid"].get_weights()}


@app.get("/api/items")
def list_items(page: int = 1, per_page: int = 50):
    """List all items with pagination."""
    if not models["ready"]:
        raise HTTPException(400, "Models not built.")
    df = models["item_df"]
    start = (page - 1) * per_page
    end = start + per_page
    items = []
    for _, row in df.iloc[start:end].iterrows():
        items.append({
            'title': row['title'],
            'category': row.get('category', ''),
            'rating': round(float(row.get('rating', 0)), 2),
            'avg_sentiment': round(float(row.get('avg_sentiment', 0)), 4),
            'description': str(row.get('description', ''))[:200],
        })
    return {"items": items, "total": len(df), "page": page, "per_page": per_page}


# ── Serve Frontend ───────────────────────────────────────────────────
frontend_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'frontend')

if os.path.isdir(frontend_dir):
    app.mount("/static", StaticFiles(directory=frontend_dir), name="frontend")

    @app.get("/")
    def serve_frontend():
        return FileResponse(os.path.join(frontend_dir, "index.html"))
