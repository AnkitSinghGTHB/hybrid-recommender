from sklearn.metrics.pairwise import cosine_similarity

class CollaborativeRecommender:
    def __init__(self, df):
        self.df = df
        self.user_item = df.pivot_table(
            index='user_id',
            columns='title',
            values='rating'
        ).fillna(0)

        self.similarity = cosine_similarity(self.user_item.T)
        self.titles = self.user_item.columns

    def recommend(self, title, top_n=5):
        if title not in self.titles:
            return []

        idx = list(self.titles).index(title)

        scores = list(enumerate(self.similarity[idx]))
        scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:top_n+1]

        return [self.titles[i[0]] for i in scores]