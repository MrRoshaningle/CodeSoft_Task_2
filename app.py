import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Movie Recommendation System", page_icon="🎬", layout="wide")

@st.cache_data
def load_movies():
    return pd.DataFrame([
        {"title": "The Dark Knight", "genres": "action crime drama"},
        {"title": "Batman Begins", "genres": "action crime drama"},
        {"title": "Inception", "genres": "action sci-fi thriller"},
        {"title": "Interstellar", "genres": "sci-fi drama adventure"},
        {"title": "The Matrix", "genres": "action sci-fi"},
        {"title": "Titanic", "genres": "romance drama"},
        {"title": "The Notebook", "genres": "romance drama"},
        {"title": "Avengers: Endgame", "genres": "action sci-fi adventure"},
        {"title": "Finding Nemo", "genres": "animation family adventure"},
        {"title": "Toy Story", "genres": "animation family comedy"},
        {"title": "The Hangover", "genres": "comedy"},
        {"title": "Superbad", "genres": "comedy romance"},
        {"title": "The Conjuring", "genres": "horror thriller"},
        {"title": "A Quiet Place", "genres": "horror sci-fi thriller"},
        {"title": "Jurassic Park", "genres": "adventure sci-fi"},
        {"title": "The Lion King", "genres": "animation drama adventure"},
        {"title": "Gladiator", "genres": "action drama adventure"},
        {"title": "The Pursuit of Happyness", "genres": "drama"},
        {"title": "John Wick", "genres": "action thriller crime"},
        {"title": "Coco", "genres": "animation family drama"},
    ])

movies = load_movies()
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(movies["genres"])
similarity = cosine_similarity(tfidf_matrix)

def recommend_movies(title, selected_genres=None, count=5):
    selected_genres = selected_genres or []
    movie_index = movies.index[movies["title"] == title][0]
    scores = sorted(enumerate(similarity[movie_index]), key=lambda item: item[1], reverse=True)
    results = []
    for index, score in scores:
        if index == movie_index:
            continue
        genres = movies.iloc[index]["genres"].split()
        if selected_genres and not any(genre in genres for genre in selected_genres):
            continue
        results.append({
            "Movie": movies.iloc[index]["title"],
            "Genres": movies.iloc[index]["genres"].title(),
            "Similarity": round(float(score) * 100, 1),
        })
        if len(results) >= count:
            break
    return pd.DataFrame(results)

st.title("🎬 Movie Recommendation System")
st.write("Discover movies based on your interests using content-based machine learning.")
with st.sidebar:
    st.header("Your preferences")
    selected_movie = st.selectbox("Choose a movie you like", movies["title"].tolist())
    all_genres = sorted({genre for value in movies["genres"] for genre in value.split()})
    selected_genres = st.multiselect("Preferred genres (optional)", all_genres)
    count = st.slider("Number of recommendations", min_value=1, max_value=10, value=5)
    show_dataset = st.checkbox("Show movie dataset")

if st.button("✨ Recommend Movies", type="primary", use_container_width=True):
    recommendations = recommend_movies(selected_movie, selected_genres, count)
    st.subheader(f"Because you like {selected_movie}")
    if recommendations.empty:
        st.info("No matching movies found. Remove some genre filters and try again.")
    else:
        cols = st.columns(2)
        for i, (_, movie) in enumerate(recommendations.iterrows()):
            with cols[i % 2]:
                with st.container(border=True):
                    st.markdown(f"### 🎥 {movie['Movie']}")
                    st.caption(movie["Genres"])
                    st.progress(min(int(movie["Similarity"]), 100), text=f"Genre similarity: {movie['Similarity']}%")
else:
    st.info("Select a movie and optional genres in the sidebar, then click **Recommend Movies**.")
    st.markdown("### How it works")
    st.write("TF-IDF converts movie genres into numerical vectors. Cosine similarity ranks movies by how closely their genre profiles match the selected movie.")
    st.markdown("### Included features")
    st.write("- 20 sample movies\n- Movie selection and genre filters\n- Adjustable recommendation count\n- Similarity scores\n- Optional dataset preview")

if show_dataset:
    st.subheader("Available movies")
    st.dataframe(movies, use_container_width=True, hide_index=True)

st.divider()
st.caption("Educational prototype • Python • Streamlit • Pandas • Scikit-learn")
