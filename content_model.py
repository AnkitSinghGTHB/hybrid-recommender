from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class ContentRecommender:
    def __init__(self, df):
        self.df = df
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.matrix = self.vectorizer.fit_transform(df['combined'])
        self.similarity = cosine_similarity(self.matrix)

    def recommend(self, title, top_n=5):
        idx = self.df[self.df['title'] == title].index

        if len(idx) == 0:
            return []

        idx = idx[0]

        scores = list(enumerate(self.similarity[idx]))
        scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:top_n+1]

        return [self.df.iloc[i[0]]['title'] for i in scores]