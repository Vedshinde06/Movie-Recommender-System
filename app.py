import streamlit as st
import pandas as pd
import numpy as np
import requests

from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity

session = requests.Session()

import os
from dotenv import load_dotenv

load_dotenv()

TMDB_API_KEY = os.getenv("TMDB_API_KEY")

if not TMDB_API_KEY:
    st.error("TMDB API key not found. Please set it in .env file.")
    st.stop()


@st.cache_data
def load_movies():
    return pd.read_pickle("movies.pkl")

new_df = load_movies()



st.title("Movie Recommender System")

st.sidebar.markdown("---")
st.sidebar.title("ℹ️ About")

st.sidebar.markdown(
    """
    **Movie Recommendation System 🎬**

    This is a **content-based movie recommender system** built using
    **Word2Vec embeddings** trained on movie metadata such as plot,
    genres, cast, and director.

    🔹 **How it works**
    - Movie descriptions are converted into dense vector embeddings
    - Similarity between movies is computed using **cosine similarity**
    - The system recommends movies with the most similar content

    🔹 **Tech Stack**
    - Python, Pandas, NumPy
    - Gensim (Word2Vec)
    - Scikit-learn
    - Streamlit
    - TMDB API (for posters)

    🔹 **Features**
    - Search-based movie selection
    - Top-N adjustable recommendations
    - Visual poster-based results
    - Fast inference using precomputed embeddings

    ---
    **Built by Vedant Shinde**  
    AI & Data Science | Machine Learning
    """
)


selected_movie = st.selectbox(
    "Select a movie",
    new_df['title'].values
)




w2v_model = Word2Vec.load("word2vec.model")
movie_vectors = np.load("movie_vectors.npy", allow_pickle=True)
similarity = cosine_similarity(movie_vectors)



top_n = st.slider("Number of recommendations", 5, 15, 5)

get_recs = st.button("🎬 Get Recommendations")


def recommend(movie_title, df, similarity_matrix, top_n=5):
    # find index of the selected movie
    matches = df[df['title'] == movie_title]

    if matches.empty:
        return []

    idx = matches.index[0]

    # get similarity scores
    scores = list(enumerate(similarity_matrix[idx]))

    # sort by similarity (descending)
    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    # exclude the movie itself
    scores = scores[1:top_n + 1]

    return scores


@st.cache_data(show_spinner=False)
def fetch_poster(movie_title):
    url = "https://api.themoviedb.org/3/search/movie"
    params = {
        "api_key": TMDB_API_KEY,
        "query": movie_title
    }

    try:
        response = session.get(url, params=params, timeout=5)
        response.raise_for_status()
        data = response.json()

        if data.get("results"):
            poster_path = data["results"][0].get("poster_path")
            if poster_path:
                return f"https://image.tmdb.org/t/p/w500{poster_path}"

    except requests.exceptions.RequestException:
        return None

    return None



if get_recs and selected_movie:
    st.subheader("🎯 Recommended Movies")

    with st.spinner("Finding similar movies..."):
        results = recommend(selected_movie, new_df, similarity, top_n)

    if not results:
        st.warning("No recommendations found.")
    else:
        with st.spinner("Loading posters..."):
            cols = st.columns(5)

            for i, (idx, score) in enumerate(results):
                with cols[i % 5]:
                    title = new_df.iloc[idx].title
                    poster = fetch_poster(title)

                    if poster:
                        st.image(poster, use_container_width=True)
                    else:
                        st.write("❌ No poster")

                    st.markdown(f"**{title}**")
                    st.caption(f"Similarity: {score:.2f}")
