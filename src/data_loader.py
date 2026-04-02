import pandas as pd
import numpy as np
import os
import urllib.request
import zipfile


class DataLoader:
    """
    Handles the downloading, extraction, and loading of the MovieLens 10M dataset.
    """
    URL = "https://files.grouplens.org/datasets/movielens/ml-10m.zip"

    def __init__(self, base_path='data'):
        # We define paths relative to the project root
        self.base_path = base_path
        self.zip_path = os.path.join(self.base_path, 'ml-10m.zip')
        self.extract_path = os.path.join(self.base_path, 'ml-10M100K')
        self.movies = None
        self.ratings = None

    def _prepare_data(self):
        """Downloads and extracts data if it doesn't exist."""
        if not os.path.exists(self.base_path):
            os.makedirs(self.base_path)

        if not os.path.exists(self.extract_path):
            if not os.path.exists(self.zip_path):
                print("Downloading ML-10M dataset (this may take a minute)...")
                urllib.request.urlretrieve(self.URL, self.zip_path)
                print("Download complete.")

            print("Extracting files...")
            with zipfile.ZipFile(self.zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.base_path)
            print("Extraction complete.")

    def load_movies(self):
        self._prepare_data()
        file_path = os.path.join(self.extract_path, 'movies.dat')
        self.movies = pd.read_csv(
            file_path, sep='::', engine='python',
            names=['movieId', 'title', 'genres'],
            encoding='latin-1'
        )
        return self.movies

    def load_ratings(self):
        self._prepare_data()
        file_path = os.path.join(self.extract_path, 'ratings.dat')
        self.ratings = pd.read_csv(
            file_path, sep='::', engine='python',
            names=['userId', 'movieId', 'rating', 'timestamp'],
            dtype={'userId': np.int32, 'movieId': np.int32, 'rating': np.float32},
            encoding='latin-1'
        )
        return self.ratings

    def get_processed_data(self):
        """Merges movies and ratings and creates the 'year' column."""
        if self.movies is None: self.load_movies()
        if self.ratings is None: self.load_ratings()

        print("Merging data...")
        df = pd.merge(self.ratings, self.movies, on='movieId')

        print("Converting timestamps...")
        df['year'] = pd.to_datetime(df['timestamp'], unit='s').dt.year
        df.drop(columns=['timestamp'], inplace=True)

        return df