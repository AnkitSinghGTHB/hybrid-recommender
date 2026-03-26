"""
Collaborative Recommender
Uses Truncated SVD (matrix factorization) on the user-item interaction
matrix to discover latent factors and predict ratings.
"""
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import coo_matrix

class CollaborativeRecommender:
    def __init__(self, interaction_df, n_factors=50):
        """
        interaction_df: DataFrame with columns 'user_id', 'title', 'rating'.
        n_factors: number of latent factors for SVD decomposition.
        """
        self.df = interaction_df.copy()

        # Map users and items to integer indices
        self.users = self.df['user_id'].astype('category')
        self.titles = self.df['title'].astype('category')
        
        self._user_to_idx = {u: i for i, u in enumerate(self.users.cat.categories)}
        self._title_to_idx = {t: i for i, t in enumerate(self.titles.cat.categories)}
        self.title_list = list(self.titles.cat.categories)

        # Build sparse user-item matrix
        row = self.users.cat.codes
        col = self.titles.cat.codes
        data = self.df['rating'].values

        self.user_item_sparse = coo_matrix((data, (row, col)), 
                                         shape=(len(self._user_to_idx), len(self._title_to_idx))).tocsr()

        # SVD decomposition
        n_components = min(n_factors, min(self.user_item_sparse.shape) - 1)
        if n_components < 1:
            n_components = 1

        self.svd = TruncatedSVD(n_components=n_components, random_state=42)
        self.user_factors = self.svd.fit_transform(self.user_item_sparse)
        self.item_factors = self.svd.components_  # (n_components, n_items)

    def recommend(self, title, top_n=10):
        """
        Get collaborative-filtering recommendations for a given item title.
        Uses item-item similarity in the SVD latent space.
        Returns list of dicts: [{ 'title', 'collab_score' }, ...]
        """
        if title not in self._title_to_idx:
            return []

        idx = self._title_to_idx[title]
        query_vec = self.item_factors[:, idx].reshape(1, -1)
        scores = cosine_similarity(query_vec, self.item_factors.T).flatten()
        
        sim_scores = list(enumerate(scores))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        results = []
        seen = set()
        for i, score in sim_scores:
            t = self.title_list[i]
            if t == title or t in seen:
                continue
            seen.add(t)
            results.append({
                'title': t,
                'collab_score': float(score),
            })
            if len(results) >= top_n:
                break

        return results

    def predict_rating(self, user_id, title):
        """Predict the rating a user would give to an item."""
        if user_id not in self._user_to_idx or title not in self._title_to_idx:
            return None
        u_idx = self._user_to_idx[user_id]
        i_idx = self._title_to_idx[title]
        return float(np.dot(self.user_factors[u_idx], self.item_factors[:, i_idx]))