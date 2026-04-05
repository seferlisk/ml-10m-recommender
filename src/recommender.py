import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

class MatrixFactorizer:
    def __init__(self, n_factors=10, learning_rate=0.01, reg=0.02, epochs=5):
        self.n_factors = n_factors
        self.lr = learning_rate
        self.reg = reg
        self.epochs = epochs

        self.user_map = {}
        self.movie_map = {}
        self.P = None  # User matrix
        self.Q = None  # Movie matrix

    def split_data(self, df):
        """Temporal split."""
        print("Splitting data into train (<2008) and test (2008-2009)...")
        train = df[df['year'] < 2008].copy()
        test = df[df['year'] >= 2008].copy()
        return train, test

    def _create_mappings(self, train_df):
        """Maps IDs to 0-indexed integers."""
        unique_users = train_df['userId'].unique()
        unique_movies = train_df['movieId'].unique()

        self.user_map = {id: i for i, id in enumerate(unique_users)}
        self.movie_map = {id: i for i, id in enumerate(unique_movies)}

        return len(unique_users), len(unique_movies)

    def fit(self, train_df):
        """Apply recommendation system using Gradient Descent."""
        n_users, n_movies = self._create_mappings(train_df)

        # Initialize latent matrices with small random values
        self.P = np.random.normal(scale=1. / self.n_factors, size=(n_users, self.n_factors))
        self.Q = np.random.normal(scale=1. / self.n_factors, size=(n_movies, self.n_factors))

        # Prepare training data as indices for speed
        u_indices = train_df['userId'].map(self.user_map).values
        i_indices = train_df['movieId'].map(self.movie_map).values
        ratings = train_df['rating'].values

        print(f"Starting Gradient Descent for {self.epochs} epochs...")
        for epoch in range(self.epochs):
            for i in range(len(ratings)):
                u, m, r = u_indices[i], i_indices[i], ratings[i]

                # Predict and calculate error
                prediction = np.dot(self.P[u, :], self.Q[m, :].T)
                error = r - prediction

                # Update Latent Factors (Gradient Descent Step)
                self.P[u, :] += self.lr * (error * self.Q[m, :] - self.reg * self.P[u, :])
                self.Q[m, :] += self.lr * (error * self.P[u, :] - self.reg * self.Q[m, :])

            print(f"Epoch {epoch + 1}/{self.epochs} complete.")

    def predict_rating(self, user_id, movie_id):
        """Estimates rating for a specific user and movie."""
        u = self.user_map.get(user_id)
        m = self.movie_map.get(movie_id)

        # If user/movie was not in training set (Cold Start), return global mean
        if u is None or m is None:
            return 3.5

        return np.dot(self.P[u, :], self.Q[m, :].T)

    def evaluate(self, test_df):
        """Compute MSE for the test set."""
        print("Evaluating model on test set...")
        predictions = test_df.apply(lambda x: self.predict_rating(x['userId'], x['movieId']), axis=1)
        mse = mean_squared_error(test_df['rating'], predictions)
        return mse