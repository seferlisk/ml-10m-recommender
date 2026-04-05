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

    def get_adjusted_trends(self, confidence_quantile=0.25):
        """
        Adjusts for number of ratings using a Bayesian-style weighted average.
        :param confidence_quantile: Used to determine 'm' (minimum ratings required).
        """
        if self.annual_stats is None:
            self.analyze_trends()

        # C = Global mean rating across the entire dataset
        C = self.df['rating'].mean()

        # m = Minimum ratings required to 'trust' the average.
        # We use a quantile of existing counts to make it dynamic.
        m = self.annual_stats['rating_count'].quantile(confidence_quantile)

        # Calculate Weighted Rating: (v/(v+m) * R) + (m/(v+m) * C)
        # v = count, R = mean
        stats = self.annual_stats.copy()
        stats['weighted_rating'] = (
                (stats['rating_count'] / (stats['rating_count'] + m)) * stats['avg_rating'] +
                (m / (stats['rating_count'] + m)) * C
        )

        results = []
        for genre in stats['genre'].unique():
            genre_data = stats[stats['genre'] == genre].sort_values('year')

            if len(genre_data) < 2: continue

            first_val = genre_data.iloc[0]['weighted_rating']
            last_val = genre_data.iloc[-1]['weighted_rating']

            results.append({
                'genre': genre,
                'raw_decrease': genre_data.iloc[0]['avg_rating'] - genre_data.iloc[-1]['avg_rating'],
                'adjusted_decrease': first_val - last_val,
                'start_count': int(genre_data.iloc[0]['rating_count'])
            })

        self.adjusted_summary = pd.DataFrame(results).sort_values('adjusted_decrease', ascending=False)
        return self.adjusted_summary