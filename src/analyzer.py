import pandas as pd

class GenreAnalyzer:
    def __init__(self, df):
        self.df = df
        self.exploded_df = None
        self.annual_stats = None  # To store the yearly averages
        self.summary_df = None  # To store the final comparison table

    def prepare_data(self):
        """Splits genres and expands the DataFrame."""
        print("Exploding genres...")
        df_copy = self.df[['year', 'rating', 'genres']].copy()
        df_copy['genre'] = df_copy['genres'].str.split('|')
        self.exploded_df = df_copy.explode('genre')
        return self.exploded_df

    def analyze_trends(self):
        """Calculates annual stats and the raw decrease for all genres."""
        if self.exploded_df is None:
            self.prepare_data()

        # 1. Create the Annual Stats DataFrame
        self.annual_stats = self.exploded_df.groupby(['genre', 'year']).agg(
            avg_rating=('rating', 'mean'),
            rating_count=('rating', 'count')
        ).reset_index()

        # 2. Create the Summary DataFrame (First year vs Last year)
        results = []
        for genre in self.annual_stats['genre'].unique():
            genre_data = self.annual_stats[self.annual_stats['genre'] == genre].sort_values('year')

            first_row = genre_data.iloc[0]
            last_row = genre_data.iloc[-1]

            results.append({
                'genre': genre,
                'start_year': int(first_row['year']),
                'end_year': int(last_row['year']),
                'start_rating': first_row['avg_rating'],
                'end_rating': last_row['avg_rating'],
                'start_count': int(first_row['rating_count']),
                'decrease': first_row['avg_rating'] - last_row['avg_rating']
            })

        self.summary_df = pd.DataFrame(results).sort_values('decrease', ascending=False)
        return self.summary_df