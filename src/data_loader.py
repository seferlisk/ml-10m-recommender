import pandas as pd
import numpy as np
import os


class DataLoader:
    """
    Handles the loading and initial preprocessing of the MovieLens 10M dataset.
    """

    def __init__(self, data_path='data/ml-10M100K'):
        self.data_path = data_path
        self.movies = None
        self.ratings = None

    def load_movies(self):
        file_path = os.path.join(self.data_path, 'movies.dat')
        # movies.dat: MovieID::Title::Genres
        self.movies = pd.read_csv(
            file_path, sep='::', engine='python',
            names=['movieId', 'title', 'genres'],
            encoding='latin-1'
        )
        return self.movies

    def load_ratings(self):
        file_path = os.path.join(self.data_path, 'ratings.dat')
        # ratings.dat: UserID::MovieID::Rating::Timestamp
        # Optimization: Use int32 and float32 to save RAM
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

        # Merge DataFrames
        df = pd.merge(self.ratings, self.movies, on='movieId')

        # Create 'year' column from rating timestamp
        df['year'] = pd.to_datetime(df['timestamp'], unit='s').dt.year

        # Drop timestamp to free up memory immediately
        df.drop(columns=['timestamp'], inplace=True)

        return df