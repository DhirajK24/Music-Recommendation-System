import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import linear_kernel
import pickle
import os

@st.cache_data
def load_data():
    filepath = 'dataset_with_clusters.csv'
    if os.path.exists(filepath):
        df = pd.read_csv(filepath)
        return df
    return None

@st.cache_resource
def load_model():
    model_path = 'tfidf_model.pkl'
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            tfidf_matrix, tfidf, kmeans = pickle.load(f)
        return tfidf_matrix, tfidf, kmeans
    return None, None, None

def recommend_similar_songs(song_name, df, tfidf_matrix, top_n=5):
    idx = df[df['song_name'] == song_name].index[0]
    song_cluster = df['cluster'].iloc[idx]
    
    # Find songs in the dataset belonging to the same cluster
    cluster_indices = np.where(df['cluster'] == song_cluster)[0]
    cluster_matrix = tfidf_matrix[cluster_indices]
    target_matrix = tfidf_matrix[idx]
    
    # Calculate similarity from selected song vector to all songs in that cluster
    similarities = linear_kernel(target_matrix, cluster_matrix)[0]
    
    # Get top_n most similar (excluding itself)
    similar_idx_in_cluster = similarities.argsort()[-(top_n+1):-1][::-1]
    similar_global_indices = cluster_indices[similar_idx_in_cluster]
    
    recommendations = df.iloc[similar_global_indices][['song_name', 'artist', 'album', 'genre', 'thumbnail']]
    return recommendations, song_cluster


def display_recommendations(recs, top_n):
    st.write(f"### Top {top_n} Tracks:")
    for idx, row in recs.iterrows():
        with st.container():
            c1, c2 = st.columns([1, 4])
            with c1:
                if pd.notna(row['thumbnail']) and row['thumbnail'] != 'Unknown':
                    st.image(row['thumbnail'], width=100)
                else:
                    st.write("🎵 No Image")
            with c2:
                st.markdown(f"**{row['song_name']}**")
                st.markdown(f"*Artist:* {row['artist']}")
                st.markdown(f"*Album:* {row['album']}")
        st.markdown("---")

def main():
    st.set_page_config(page_title="Music Recommender", layout="centered", page_icon="🎵")
    
    hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)
    
    st.title("🎵 Music Recommendation System")
    st.write("Discover new songs! Find tracks similar to your favorites using **TF-IDF & K-Means clustering**.")
    
    df = load_data()
    tfidf_matrix, tfidf, kmeans = load_model()
    
    if df is None or tfidf_matrix is None:
        st.error("Data or Models not found! Please run `python recommendation_model.py` to generate the required files.")
        return
        
    st.markdown("---")
    
    tab1, tab2 = st.tabs(["🎵 Song-Based", "📂 Browse Clusters"])
    
    with tab1:
        col1, col2 = st.columns([2, 1])
        with col1:
            song_list = sorted(df['song_name'].dropna().unique())
            selected_song = st.selectbox("Search for a song:", song_list)
        with col2:
            top_n_song = st.slider("Count", min_value=1, max_value=10, value=5, key="song_slider")
            
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button(f"Recommend similar to '{selected_song}' 🎧", use_container_width=True):
            recs, cluster_id = recommend_similar_songs(selected_song, df, tfidf_matrix, top_n=top_n_song)
            st.success(f"Your song belongs to **Cluster {cluster_id}**! Here are similar tracks:")
            display_recommendations(recs, top_n_song)
            
    with tab2:
        st.subheader("Explore Songs by Cluster")
        clusters_available = sorted(df['cluster'].unique())
        selected_cluster = st.selectbox("Select a Cluster to view its songs:", clusters_available)
        
        st.markdown("<br>", unsafe_allow_html=True)
        cluster_songs = df[df['cluster'] == selected_cluster]
        st.write(f"### Found {len(cluster_songs)} songs in Cluster {selected_cluster}")
        
        st.dataframe(cluster_songs[['song_name', 'artist', 'album', 'genre']].reset_index(drop=True), use_container_width=True)

    st.markdown("---")
if __name__ == "__main__":
    main()
