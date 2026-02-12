import pandas as pd
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

# -------------------------------
# Load datasets
# -------------------------------
movies = pd.read_csv("movies_metadata.csv", low_memory=False)
credits = pd.read_csv("credits.csv")
keywords = pd.read_csv("keywords.csv")

# Keep useful columns
movies = movies[['id', 'title', 'genres']]
movies = movies.dropna()

# Convert id to int
movies['id'] = movies['id'].astype(int)
credits['id'] = credits['id'].astype(int)
keywords['id'] = keywords['id'].astype(int)

# Merge datasets
movies = movies.merge(credits, on='id')
movies = movies.merge(keywords, on='id')

# 🔥 LIMIT DATA SIZE (VERY IMPORTANT FOR RAM)
movies = movies.head(5000)


# -------------------------------
# Helper functions
# -------------------------------
def convert(text):
    L = []
    for i in ast.literal_eval(text):
        L.append(i['name'])
    return L


def convert_cast(text):
    L = []
    counter = 0
    for i in ast.literal_eval(text):
        if counter < 3:
            L.append(i['name'])
            counter += 1
    return L


def fetch_director(text):
    L = []
    for i in ast.literal_eval(text):
        if i['job'] == 'Director':
            L.append(i['name'])
    return L


# Apply processing
movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)
movies['cast'] = movies['cast'].apply(convert_cast)
movies['crew'] = movies['crew'].apply(fetch_director)

# Create tags column
movies['tags'] = movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']

# Keep only needed columns
new_df = movies[['id', 'title', 'tags']].copy()

# Convert list → string
new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x))


# -------------------------------
# Vectorization + similarity
# -------------------------------
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(new_df['tags']).toarray()

similarity = cosine_similarity(vectors)


# -------------------------------
# Save model files
# -------------------------------
pickle.dump(new_df, open("movies.pkl", "wb"))
pickle.dump(similarity, open("similarity.pkl", "wb"))

print("✅ Model built and saved successfully!")