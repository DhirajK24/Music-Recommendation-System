import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import linear_kernel
import os
import warnings
import pickle
warnings.filterwarnings('ignore')

def load_data(filepath='Bollywood_Songs_With_Album_Genre.csv'):
    df = pd.read_csv(filepath)
    return df

def preprocess_data(df):
    # Handle missing values
    for col in ['artist', 'genre', 'lyrics', 'album']:
        df[col] = df[col].fillna('')
    
    # Combine text features
    df['combined_features'] = df['genre'] + " " + df['artist'] + " " + df['album'] + " " + df['lyrics']
    
    # TF-IDF Vectorization
    tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
    tfidf_matrix = tfidf.fit_transform(df['combined_features'])
    
    return tfidf_matrix, tfidf

def find_optimal_k(tfidf_matrix, max_k=15):
    inertia = []
    silhouette_scores = []
    
    # K-Means expects array-like, tfidf_matrix is sparse which is fine for KMeans
    # but silhouette_score expects dense for some distance metrics or it parses it internally.
    matrix_dense = tfidf_matrix.toarray()
    
    K = range(2, max_k + 1)
    
    
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(tfidf_matrix)
        inertia.append(kmeans.inertia_)
        score = silhouette_score(matrix_dense, kmeans.labels_)
        silhouette_scores.append(score)
        
    return list(K), inertia, silhouette_scores

def plot_evaluation(K, inertia, silhouette_scores):
    plt.figure(figsize=(14, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(K, inertia, 'bo-')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia')
    plt.title('Elbow Method For Optimal k')
    
    plt.subplot(1, 2, 2)
    plt.plot(K, silhouette_scores, 'ro-')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score For Optimal k')
    
    plt.tight_layout()
    plt.savefig('elbow_silhouette_plot.png', dpi=300)
    print("Saved evaluation plot to 'elbow_silhouette_plot.png'")
    plt.clf()

def cluster_data(tfidf_matrix, n_clusters=5):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(tfidf_matrix)
    return kmeans, labels

def plot_clusters(tfidf_matrix, labels):
    svd = TruncatedSVD(n_components=2, random_state=42)
    pca_features = svd.fit_transform(tfidf_matrix)
    
    plt.figure(figsize=(8, 6))
    str_labels = [f"Cluster {l}" for l in labels]
    
    sns.scatterplot(x=pca_features[:, 0], y=pca_features[:, 1], hue=str_labels, palette='viridis', alpha=0.7)
    plt.title('2D PCA of Song Clusters')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('pca_clusters.png', dpi=300)
    print("Saved PCA cluster plot to 'pca_clusters.png'")

def recommend_songs(song_name, df, tfidf_matrix, labels, top_n=5):
    if song_name not in df['song_name'].values:
        return f"Song '{song_name}' not found in the dataset."
    
    idx = df[df['song_name'] == song_name].index[0]
    song_cluster = labels[idx]
    
    cluster_indices = np.where(labels == song_cluster)[0]
    cluster_matrix = tfidf_matrix[cluster_indices]
    target_matrix = tfidf_matrix[idx]
    
    similarities = linear_kernel(target_matrix, cluster_matrix)[0]
    
    # Get top_n most similar (excluding itself)
    similar_idx_in_cluster = similarities.argsort()[-(top_n+1):-1][::-1]
    similar_global_indices = cluster_indices[similar_idx_in_cluster]
    
    recommendations = df.iloc[similar_global_indices][['song_name', 'artist', 'album', 'genre']]
    return recommendations

if __name__ == "__main__":
    filepath = 'Bollywood_Songs_With_Album_Genre.csv'
    if not os.path.exists(filepath):
        print(f"Error: {filepath} not found in the current directory.")
    else:
        print("Loading data...")
        df = load_data(filepath)
        
        # Remove duplicate song names for easier recommendation testing
        df = df.drop_duplicates(subset=['song_name']).reset_index(drop=True)
        
        print("Preprocessing data and TF-IDF Vectorization...")
        tfidf_matrix, tfidf = preprocess_data(df)
        
        print("Evaluating optimal K...")
        K, inertia, silhouette_scores = find_optimal_k(tfidf_matrix, max_k=15)
        plot_evaluation(K, inertia, silhouette_scores)
        
        # Pick best k based on silhouette
        best_k = K[np.argmax(silhouette_scores)]
        print(f"Optimal K selected based on max silhouette score: {best_k}")
        
        print(f"Applying K-Means clustering with k={best_k}...")
        kmeans, labels = cluster_data(tfidf_matrix, n_clusters=best_k)
        df['cluster'] = labels
        
        print("Plotting clusters...")
        plot_clusters(tfidf_matrix, labels)
        
        print("\nTesting Recommendation System:")
        test_song = df['song_name'].iloc[0]
        print(f"Recommendations for '{test_song}':")
        recs = recommend_songs(test_song, df, tfidf_matrix, labels)
        print(recs.to_string(index=False))
        
        output_file = 'dataset_with_clusters.csv'
        df.to_csv(output_file, index=False)
        print(f"\nSaved clustered dataset to '{output_file}'")
        
        # Save TF-IDF matrix and model using pickle
        with open('tfidf_model.pkl', 'wb') as f:
            pickle.dump((tfidf_matrix, tfidf, kmeans), f)
        print("Saved tfidf_model.pkl")
