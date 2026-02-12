import streamlit as st
import pandas as pd
import ast

st.set_page_config(page_title="Movie Recommendation System", layout="wide")

st.title("🎬 Movie Recommendation System")
st.write("Recommendations based on **Genre and Language**")

# -------------------------------
# Load dataset
# -------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("movies_metadata.csv", low_memory=False)

    df = df[['title', 'genres', 'vote_average', 'original_language', 'release_date']]
    df = df.dropna()

    # -------- Language Mapping --------
    language_map = {
        'en': 'English',
        'hi': 'Hindi',
        'mr': 'Marathi'
    }

    # Keep only required languages
    df = df[df['original_language'].isin(language_map.keys())]

    # Create readable language column
    df['language'] = df['original_language'].map(language_map)

    # -------- Extract genres --------
    def extract_genres(text):
        try:
            return [g['name'] for g in ast.literal_eval(text)]
        except:
            return []

    df['genre_list'] = df['genres'].apply(extract_genres)

    return df


movies = load_data()

# -------------------------------
# Sidebar Filters
# -------------------------------
all_genres = sorted({g for lst in movies['genre_list'] for g in lst})
all_languages = ["English", "Hindi", "Marathi"]  # 🔥 Only 3 languages

selected_genre = st.sidebar.selectbox("Select Genre", all_genres)
selected_language = st.sidebar.selectbox("Select Language", all_languages)
top_n = st.sidebar.slider("Number of Movies", 5, 50, 10)

# -------------------------------
# Recommendation Logic
# -------------------------------
filtered = movies[
    movies['genre_list'].apply(lambda x: selected_genre in x)
]

filtered = filtered[filtered['language'] == selected_language]

filtered = filtered.sort_values(by="vote_average", ascending=False).head(top_n)

# -------------------------------
# Display Results
# -------------------------------
st.subheader("🎥 Recommended Movies")

if len(filtered) > 0:
    st.dataframe(
        filtered[['title', 'language', 'vote_average', 'release_date']],
        use_container_width=True
    )
else:
    st.warning("No movies found for selected Genre and Language.")