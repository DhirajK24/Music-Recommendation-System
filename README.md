
# Music Recommendation System 🎵

A machine learning-based music recommendation engine that groups similar songs using Natural Language Processing (NLP) and Clustering techniques. It features a responsive web interface built with Streamlit.

## Features
- **Data Preprocessing & NLP:** Uses TF-IDF (Term Frequency-Inverse Document Frequency) to vectorize song features like genre, artist, album, and lyrics.
- **K-Means Clustering:** Automatically finds the optimal number of clusters using the Elbow method and Silhouette scores, grouping similar songs together.
- **Cosine Similarity:** Provides fast, localized recommendations by calculating similarity only between the target song and other songs within its cluster.
- **Dimensionality Reduction:** Visualizes high-dimensional song clusters in 2D using PCA (TruncatedSVD).
- **Interactive UI:** A Streamlit-based interface to search for songs, tune the number of recommendations, and explore different song clusters.

## Project Structure
- `app.py`: The main Streamlit web application script.
- `recommendation_model.py`: The backend machine learning script. It loads the raw data, trains the TF-IDF and K-Means models, generates evaluation plots, and saves the trained models.
- `requirements.txt`: A list of required Python packages for this project.
- `Bollywood_Songs_With_Album_Genre.csv`: The initial raw dataset of songs.

*Generated files during training:*
- `dataset_with_clusters.csv`: The dataset appended with assigned cluster IDs.
- `tfidf_model.pkl`: A serialized file containing the trained TF-IDF matrix, vectorizer, and K-Means model.
- `elbow_silhouette_plot.png` & `pca_clusters.png`: Visual evaluation of the clustering models.

## How to Run

### 1. Install Dependencies
Make sure you have Python installed. Open your terminal in the project directory and run:
```bash
pip install -r requirements.txt
```

### 2. (Optional) Train the Model
If you need to regenerate the models or you've added new data to `Bollywood_Songs_With_Album_Genre.csv`, run the model training script:
```bash
python recommendation_model.py
```
This process might take a few moments. It will output the updated `dataset_with_clusters.csv` and `tfidf_model.pkl` needed for the web app to function.

### 3. Start the Web App
Start the interactive application with Streamlit:
```bash
streamlit run app.py
```
This will open your default browser where you can start finding new music!

## Technologies Used
- **Python 3**
- **Pandas & NumPy:** Data manipulation and processing.
- **Scikit-Learn:** TF-IDF Vectorization, K-Means Clustering, PCA, and Cosine Similarity.
- **Matplotlib & Seaborn:** Data visualization.
- **Streamlit:** Frontend web framework.
