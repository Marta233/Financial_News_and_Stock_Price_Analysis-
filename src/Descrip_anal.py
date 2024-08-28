# src/data_analysis.py
import pandas as pd
import matplotlib.pyplot as plt
class Descriptives:
    def __init__(self, df):
        self.df= df
    def get_headline_lengths(self):
        self.df['headline_length'] = self.df['headline'].apply(len)
        return self.df
    def get_top_ten_headline_len(self):
        top_ten_headline_len = self.df.nlargest(10, 'headline_length')
        return top_ten_headline_len
    def count_healine_by_publisher(self):
        publisher_count = self.df.groupby('publisher').size().reset_index(name ='publisher_count')
        publisher_count = publisher_count.sort_values(by = 'publisher_count', ascending = False)
        return publisher_count
    def format_date_time(self):
        self.df['date'] = pd.to_datetime(self.df['date'], format='ISO8601')
        