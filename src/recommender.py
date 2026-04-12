import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error

class MatrixFactorizer:
    """
    THE MODEL: Handles only the mathematical training and raw prediction.
    """
    def __init__(self, n_factors=20, learning_rate=0.01, reg=0.02, epochs=5):
        self.n_factors = n_factors
        self.lr = learning_rate
        self.reg = reg
        self.epochs = epochs
        self.user_map = {}
        self.movie_map = {}
        # Latent Matrices
        self.P = None  # User matrix
        self.Q = None  # Movie matrix
        # Bias Terms
        self.user_biases = None
        self.movie_biases = None
        self.global_mean = 0

    def split_data(self, df):
        """Temporal split."""
        print("Splitting data into train (<2008) and test (2008-2009)...")
        train = df[df['year'] < 2008].copy()
        test = df[df['year'] >= 2008].copy()
        return train, test

    def _create_mappings(self, train_df):
        """Maps IDs to 0-indexed integers for matrix indices."""
        unique_users = train_df['userId'].unique()
        unique_movies = train_df['movieId'].unique()

        self.user_map = {id: i for i, id in enumerate(unique_users)}
        self.movie_map = {id: i for i, id in enumerate(unique_movies)}

        return len(unique_users), len(unique_movies)

    def fit(self, train_df):
        """Training using Stochastic Gradient Descent (SGD)."""
        # Calculate actual global mean from training data
        self.global_mean = train_df['rating'].mean()
        n_users, n_movies = self._create_mappings(train_df)

        # Initialize latent matrices with small random values
        self.P = np.random.normal(scale=0.1, size=(n_users, self.n_factors))
        self.Q = np.random.normal(scale=0.1, size=(n_movies, self.n_factors))

        # Initialize Biases at Zero
        self.user_biases = np.zeros(n_users)
        self.movie_biases = np.zeros(n_movies)

        # Prepare training data as indices for speed
        u_indices = train_df['userId'].map(self.user_map).values
        i_indices = train_df['movieId'].map(self.movie_map).values
        ratings = train_df['rating'].values

        print(f"Training on {len(train_df)} rows (Mean: {self.global_mean:.2f})...")
        for epoch in range(self.epochs):
            for i in range(len(ratings)):
                u, m, r = u_indices[i], i_indices[i], ratings[i]

                # Prediction formula: Global Mean + User_Bias + Movie_Bias + (P dot Q)
                prediction = self.global_mean + self.user_biases[u] + self.movie_biases[m] + \
                             np.dot(self.P[u, :], self.Q[m, :].T)

                error = r - prediction

                # Update Biases
                self.user_biases[u] += self.lr * (error - self.reg * self.user_biases[u])
                self.movie_biases[m] += self.lr * (error - self.reg * self.movie_biases[m])

                # Update Latent Factors
                self.P[u, :] += self.lr * (error * self.Q[m, :] - self.reg * self.P[u, :])
                self.Q[m, :] += self.lr * (error * self.P[u, :] - self.reg * self.Q[m, :])

            print(f"Epoch {epoch + 1}/{self.epochs} complete.")

    def predict_rating(self, user_id, movie_id):
        """Predicts a rating for a specific user and movie pair."""
        u = self.user_map.get(user_id)
        m = self.movie_map.get(movie_id)

        if u is None and m is None: return self.global_mean
        if u is None: return self.global_mean + self.movie_biases[m]
        if m is None: return self.global_mean + self.user_biases[u]

        pred = self.global_mean + self.user_biases[u] + self.movie_biases[m] + \
               np.dot(self.P[u, :], self.Q[m, :].T)
        return np.clip(pred, 0.5, 5.0)

    def predict_test_set(self, test_df):
        """
        Estimates a rating for every user-movie pair in the test set.
        Returns the DataFrame with an additional 'predicted_rating' column.
        """
        print("Generating predictions for the test set...")

        # We use a copy to avoid SettingWithCopyWarning
        results_df = test_df.copy()

        # Pre-calculate indices to avoid repeated map lookups
        results_df['u_idx'] = results_df['userId'].map(self.user_map)
        results_df['m_idx'] = results_df['movieId'].map(self.movie_map)

        # Vectorized prediction for speed:
        # For rows where we have both user and movie in training: Dot product of P and Q
        # For others (Cold Start): We default to the global mean (3.5)

        def get_prediction(row):
            if pd.isna(row['u_idx']) or pd.isna(row['m_idx']):
                return self.global_mean  # Simple Cold Start baseline

            u, m = int(row['u_idx']), int(row['m_idx'])
            pred = self.global_mean + np.dot(self.P[u, :], self.Q[m, :].T)

            # Ensures predictions stay within the valid 0.5 - 5.0 range
            return np.clip(pred, 0.5, 5.0)

        # results_df['predicted_rating'] = results_df.apply(get_prediction, axis=1)
        results_df['predicted_rating'] = results_df.apply(lambda x: self.predict_rating(x['userId'], x['movieId']), axis=1)

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
        # Distinct movie reference for metadata
        self.movie_info = full_df[['movieId', 'title', 'genres']].drop_duplicates('movieId')

    def recommend_cold_start(self, n=10, min_ratings=10000):
        """Popularity-based recommendations for brand-new users."""
        stats = self.df.groupby('movieId').agg(
            avg_rating=('rating', 'mean'),
            count=('rating', 'count')
        ).reset_index()

        top_ids = stats[stats['count'] >= min_ratings].sort_values('avg_rating', ascending=False).head(n)['movieId']
        return self.movie_info[self.movie_info['movieId'].isin(top_ids)]

    def recommend_by_context(self, movie_titles, n=10):
        """Content-Similarity based on latent factors (Item-Item)."""
        # 1. Identify movieIds from the provided titles
        liked_ids = []
        for title in movie_titles:
            matches = self.movie_info[self.movie_info['title'].str.contains(title, case=False, na=False)]
            if not matches.empty:
                liked_ids.append(matches.iloc[0]['movieId'])

        # 2. Get indices in the model's Q matrix
        latent_indices = [self.model.movie_map[mid] for mid in liked_ids if mid in self.model.movie_map]

        if not latent_indices:
            return "Could not find any of those movies in the training data."

        # 3. Create a 'pseudo-user' vector by averaging the latent factors of liked movies
        user_vector = self.model.Q[latent_indices].mean(axis=0).reshape(1, -1)

        # 4. Calculate similarity between this vector and all movies in Q
        similarities = cosine_similarity(user_vector, self.model.Q).flatten()

        # 5. Retrieve top N (filtering out the original liked movies)
        top_indices = similarities.argsort()[::-1]

        # Mapping indices back to movieIds
        reverse_map = {v: k for k, v in self.model.movie_map.items()}

        recommendations = []
        for idx in top_indices:
            movie_id = reverse_map[idx]
            if movie_id not in liked_ids:
                recommendations.append(movie_id)
            if len(recommendations) == n:
                break

        return self.movie_info[self.movie_info['movieId'].isin(recommendations)]