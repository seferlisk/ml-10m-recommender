import pandas as pd

class GenreAnalyzer:
    def __init__(self, df):
        """
        :param df: The merged DataFrame from DataLoader
        """
        self.df = df
        self.exploded_df = None

    def _prepare_exploded_data(self):
        """Splits the 'genres' string and expands the DataFrame."""
        if self.exploded_df is not None:
            return self.exploded_df

        print("Exploding genres (this may take a moment)...")
        # Copy to avoid SettingWithCopyWarning
        df_copy = self.df[['year', 'rating', 'genres']].copy()

        # Split 'genres' string into a list
        df_copy['genre'] = df_copy['genres'].str.split('|')

        # 'Explode' converts [Action, Sci-Fi] into two separate rows
        self.exploded_df = df_copy.explode('genre')
        return self.exploded_df

    def get_top_decreases(self, top_n=5):
        """Step 2: Find genres with greatest decrease from first to last year."""
        data = self._prepare_exploded_data()

        # Group by genre and year to get average annual rating
        annual_stats = data.groupby(['genre', 'year'])['rating'].mean().reset_index()

        results = []
        for genre in annual_stats['genre'].unique():
            genre_data = annual_stats[annual_stats['genre'] == genre].sort_values('year')

            first_year_rating = genre_data.iloc[0]['rating']
            last_year_rating = genre_data.iloc[-1]['rating']
            decrease = first_year_rating - last_year_rating

            results.append({
                'genre': genre,
                'first_year': int(genre_data.iloc[0]['year']),
                'last_year': int(genre_data.iloc[-1]['year']),
                'decrease': decrease
            })

        return pd.DataFrame(results).sort_values('decrease', ascending=False).head(top_n)

    def get_weighted_decreases(self, min_ratings=1000, top_n=5):
        """Step 3: Adjusted analysis considering the volume of ratings."""
        data = self._prepare_exploded_data()

        # Group by genre and year, but this time get mean AND count
        stats = data.groupby(['genre', 'year'])['rating'].agg(['mean', 'count']).reset_index()

        # Filter out 'noisy' data where a year has very few ratings
        stats = stats[stats['count'] >= min_ratings]

        results = []
        for genre in stats['genre'].unique():
            genre_data = stats[stats['genre'] == genre].sort_values('year')

            if len(genre_data) < 2: continue  # Need at least two years to show a trend

            decrease = genre_data.iloc[0]['mean'] - genre_data.iloc[-1]['mean']
            results.append({
                'genre': genre,
                'decrease': decrease,
                'total_ratings_tracked': genre_data['count'].sum()
            })

        return pd.DataFrame(results).sort_values('decrease', ascending=False).head(top_n)