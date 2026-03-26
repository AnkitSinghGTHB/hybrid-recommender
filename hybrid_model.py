"""
Hybrid Recommender
Combines content-based, collaborative, and NLP sentiment scores
using a weighted scoring system with normalization and re-ranking.

final_score = α × content_score + β × collab_score + γ × sentiment_score
"""
import numpy as np


class HybridRecommender:
    def __init__(self, content_model, collab_model=None, item_df=None,
                 alpha=0.4, beta=0.35, gamma=0.25):
        """
        content_model:  ContentRecommender instance
        collab_model:   CollaborativeRecommender instance (optional)
        item_df:        DataFrame with item-level 'avg_sentiment' column (optional)
        alpha:          weight for content-based score
        beta:           weight for collaborative score
        gamma:          weight for sentiment score
        """
        self.content_model = content_model
        self.collab_model = collab_model
        self.item_df = item_df
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        # Build sentiment lookup
        self._sentiment_map = {}
        if item_df is not None and 'avg_sentiment' in item_df.columns:
            for _, row in item_df.iterrows():
                self._sentiment_map[row['title']] = row['avg_sentiment']

    def set_weights(self, alpha, beta, gamma):
        """Update the scoring weights. They are normalized to sum to 1."""
        total = alpha + beta + gamma
        if total == 0:
            total = 1
        self.alpha = alpha / total
        self.beta = beta / total
        self.gamma = gamma / total

    def get_weights(self):
        return {'alpha': self.alpha, 'beta': self.beta, 'gamma': self.gamma}

    def _normalize(self, scores):
        """Min-max normalize a list of scores to [0, 1]."""
        if not scores:
            return scores
        values = [s for s in scores]
        mn, mx = min(values), max(values)
        if mx - mn == 0:
            return [0.5] * len(values)
        return [(v - mn) / (mx - mn) for v in values]

    def recommend(self, title, top_n=10):
        """
        Get hybrid recommendations for a given item title.
        Returns list of dicts sorted by hybrid_score:
        [{ 'title', 'content_score', 'collab_score', 'sentiment_score',
           'hybrid_score', 'rating', 'category', 'description' }, ...]
        """
        # 1. Content-based scores
        content_recs = self.content_model.recommend(title, top_n=top_n * 3)
        all_titles = {r['title'] for r in content_recs}

        # 2. Collaborative scores
        collab_map = {}
        if self.collab_model:
            collab_recs = self.collab_model.recommend(title, top_n=top_n * 3)
            for r in collab_recs:
                collab_map[r['title']] = r['collab_score']
                all_titles.add(r['title'])

        # 3. Build unified score table
        candidates = {}
        for r in content_recs:
            candidates[r['title']] = {
                'title': r['title'],
                'raw_content': r['content_score'],
                'raw_collab': collab_map.get(r['title'], 0.0),
                'raw_sentiment': self._sentiment_map.get(r['title'], 0.0),
            }
        # Add collab-only items
        for t in collab_map:
            if t not in candidates:
                candidates[t] = {
                    'title': t,
                    'raw_content': 0.0,
                    'raw_collab': collab_map[t],
                    'raw_sentiment': self._sentiment_map.get(t, 0.0),
                }

        if not candidates:
            return []

        items = list(candidates.values())

        # 4. Normalize each component
        content_scores = self._normalize([it['raw_content'] for it in items])
        collab_scores = self._normalize([it['raw_collab'] for it in items])
        # Sentiment: map from [-1,1] to [0,1]
        sentiment_scores = [(it['raw_sentiment'] + 1) / 2 for it in items]

        # 5. Determine active weights (redistribute if model is missing)
        a, b, g = self.alpha, self.beta, self.gamma
        if self.collab_model is None:
            # No collaborative model — redistribute its weight
            total = a + g
            if total > 0:
                a, g = a / total, g / total
            b = 0
        if not self._sentiment_map:
            # No sentiment data — redistribute
            total = a + b
            if total > 0:
                a, b = a / total, b / total
            g = 0

        # 6. Compute hybrid score
        results = []
        for i, item in enumerate(items):
            hybrid = (
                a * content_scores[i] +
                b * collab_scores[i] +
                g * sentiment_scores[i]
            )

            # Lookup extra info from content model's df
            row_data = self.content_model.df[
                self.content_model.df['title'] == item['title']
            ]
            avg_rating = 0.0
            category = ''
            description = ''
            top_reviews = []
            if len(row_data) > 0:
                avg_rating = float(row_data['rating'].mean())
                category = str(row_data.iloc[0].get('category', ''))
                description = str(row_data.iloc[0].get('description', ''))[:200]
                tp = row_data.iloc[0].get('top_reviews', [])
                top_reviews = tp if isinstance(tp, list) else []

            results.append({
                'title': item['title'],
                'content_score': round(content_scores[i], 4),
                'collab_score': round(collab_scores[i], 4),
                'sentiment_score': round(sentiment_scores[i], 4),
                'hybrid_score': round(hybrid, 4),
                'rating': round(avg_rating, 2),
                'category': category,
                'description': description,
                'top_reviews': top_reviews,
            })

        # 7. Sort by hybrid score descending
        results.sort(key=lambda x: x['hybrid_score'], reverse=True)
        return results[:top_n]