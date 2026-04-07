import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

data = pd.read_csv("games.csv")
print(data.head())
print(data.info())
print(data.shape)

tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(data["Genres"])

cos_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
indices = pd.Series(data.index, index=data["Title"])
title = data["Title"]

def GetRecommendation(title, cos_sim=cos_sim):
    if title not in indices:
        return "Game not found in dataset"
    idx = indices[title]
    sim_scores = list(enumerate(cos_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x:x[1], reverse=True)
    sim_scores = sim_scores[1:10]
    games_indices = [i[0] for i in sim_scores]
    return data["Title".iloc[games_indices]]


choice = input("what game do you like?")
if choice not in indices:
    print("The game is not in my dataset. Try another!")

print("If you like",choice,"you might also like")
print(GetRecommendation(choice))