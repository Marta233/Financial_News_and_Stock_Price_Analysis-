# src/data_analysis.py

import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

class Descriptives:
    def __init__(self, df):
        # Initialize the class with a DataFrame
        self.df = df

    def get_headline_lengths(self):
        # Drop the unnamed column and calculate headline lengths
        self.df = self.df.drop(columns=['Unnamed: 0']) 
        self.df['headline_length'] = self.df['headline'].apply(len)
        return self.df

    def get_plot_headline_length(self):
        # Create a histogram of headline lengths
        fig = px.histogram(self.df, x="headline_length")
        return fig

    def get_top_ten_headline_len(self):
        # Get the top ten headlines by length
        top_ten_headline_len = self.df.nlargest(10, 'headline_length')
        return top_ten_headline_len

    def count_headline_by_publisher(self):
        # Count headlines by publisher and sort them
        publisher_count = self.df.groupby('publisher').size().reset_index(name='publisher_count')
        publisher_count = publisher_count.sort_values(by='publisher_count', ascending=False)
        return publisher_count

    def display_top_publishers(self):
        # Display the top ten publishers by headline count
        top_publishers = self.count_headline_by_publisher().head(10)
        return top_publishers

    def visualize_count_headline_by_publisher(self):
        # Visualize the count of headlines by the top publishers
        fig = px.bar(self.display_top_publishers(), x='publisher', y='publisher_count')
        return fig

    def format_publication_dates(self):
        # Convert the 'publication_date' column to datetime if it's not already
        self.df['date'] = pd.to_datetime(self.df['date'], format='ISO8601')

        # Extract different time components
        self.df['year'] = self.df['date'].dt.year
        self.df['month'] = self.df['date'].dt.month
        self.df['day_of_week'] = self.df['date'].dt.day_name()

        # Group by day of the week
        day_of_week_trends = self.df['day_of_week'].value_counts().reindex(
            ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        )
        # Group by year and month
        year_month_trends = self.df.groupby(['year', 'month']).size().reset_index(name='counts')
        return day_of_week_trends, year_month_trends

    def plot_publication_trends(self, day_of_week_trends, year_month_trends):
        # Plot trends of news frequency by day of the week
        bar_fig = go.Figure(go.Bar(
            x=day_of_week_trends.index,
            y=day_of_week_trends.values,
            marker_color="green"
        ))
        bar_fig.update_layout(
            title='News Frequency by Day of the Week',
            xaxis_title='Day of the Week',
            yaxis_title='Number of Publications',
            xaxis_tickangle=-45
        )
        bar_fig.show()

        # Plot trends over time using Plotly
        line_fig = go.Figure()
        for year in year_month_trends['year'].unique():
            monthly_data = year_month_trends[year_month_trends['year'] == year]
            line_fig.add_trace(go.Scatter(
                x=monthly_data['month'],
                y=monthly_data['counts'],
                mode='lines+markers',
                name=f'Year {year}'
            ))
        
        line_fig.update_layout(
            title='News Frequency Over Time',
            xaxis_title='Month',
            yaxis_title='Number of Publications',
            xaxis=dict(tickmode='linear', tick0=1, dtick=1),
            legend_title_text='Year'
        )
        line_fig.show()