class HybridRecommender:
    def __init__(self, content_model, collab_model=None, alpha=0.5):
        self.content_model = content_model
        self.collab_model = collab_model
        self.alpha = alpha

    def recommend(self, title, top_n=5):
        content_recs = self.content_model.recommend(title, top_n)

        if self.collab_model:
            collab_recs = self.collab_model.recommend(title, top_n)

            # Merge both
            combined = list(dict.fromkeys(content_recs + collab_recs))
            return combined[:top_n]

        return content_recs