import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

class MatrixFactorizer:
    """
    THE MODEL: Handles only the mathematical training and raw prediction.
    """
    def __init__(self, n_factors=10, learning_rate=0.01, reg=0.02, epochs=3):
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

    def predict_test_set(self, test_df):
        """
        Estimates a rating for every user-movie pair in the test set.
        Returns the DataFrame with an additional 'predicted_rating' column.
        """
        print("Generating predictions for the test set...")

        # We use a copy to avoid SettingWithCopyWarning
        results_df = test_df.copy()

        # Map the IDs to their latent indices
        # We use .get() to handle users/movies not seen in training (Cold Start)
        results_df['u_idx'] = results_df['userId'].map(self.user_map)
        results_df['m_idx'] = results_df['movieId'].map(self.movie_map)

        # Vectorized prediction for speed:
        # For rows where we have both user and movie in training: Dot product of P and Q
        # For others (Cold Start): We default to the global mean (3.5)

        def get_prediction(row):
            if pd.isna(row['u_idx']) or pd.isna(row['m_idx']):
                return 3.5  # Simple Cold Start baseline

            u = int(row['u_idx'])
            m = int(row['m_idx'])
            return np.dot(self.P[u, :], self.Q[m, :])

        results_df['predicted_rating'] = results_df.apply(get_prediction, axis=1)

        # Drop the temporary index columns
        results_df.drop(columns=['u_idx', 'm_idx'], inplace=True)

        return results_df

class RecommenderEngine:
    """
    THE SERVICE: Implements specific business logic for different types of users.
    """
    def __init__(self, model, full_df):
        self.model = model
        self.df = full_df
        # Cache movie information for quick lookup
        self.movie_info = full_df[['movieId', 'title', 'genres']].drop_duplicates('movieId').set_index('movieId')

    def recommend_cold_start(self, n=10, min_ratings=10000):
        """
        Step 5: Strategy for users with NO history (Popularity-based).
        """
        stats = self.df.groupby('movieId').agg(
            avg_rating=('rating', 'mean'),
            count=('rating', 'count')
        )
        top_ids = stats[stats['count'] >= min_ratings].sort_values('avg_rating', ascending=False).head(n).index
        return self.movie_info.loc[top_ids]

    def recommend_for_existing_user(self, user_id, n=10):
        """
        Strategy for users in our system (Model-based).
        """
        # Logic to predict ratings for all movies the user hasn't seen
        pass

    def recommend_by_context(self, movie_titles, n=10):
        """
        Step 6: Strategy for users who gave us a few samples (Content/Similarity).
        """
        # Logic to find similar items based on latent factors
        pass