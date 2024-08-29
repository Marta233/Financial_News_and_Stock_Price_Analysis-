import pandas as pd
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt
import re

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

class TextAnalyzer:
    def __init__(self, df):
        self.df = df
    def remove_unname(self):
        self.df = self.df.drop(columns=['Unnamed: 0']) 
        return self.df
    # def sentiment_analysis(self):
    #     # Clean the headlines and calculate sentiment polarity
    #     self.df['cleaned_headline'] = self.df['headline'].apply(self._clean_text)
    #     self.df['sentiment'] = self.df['cleaned_headline'].apply(self._get_sentiment)
    #     return self.df
    def sentiment_analysis(self):
        # Clean the headlines and calculate sentiment polarity
        self.df['cleaned_headline'] = self.df['headline'].apply(self._clean_text)
        self.df['sentiment'], self.df['polarity'] = zip(*self.df['cleaned_headline'].apply(self._get_sentiment))
        return self.df
    def keyword_extraction(self):
        # Clean the headlines
        self.df['cleaned_headline'] = self.df['headline'].apply(self._clean_text)
        
        # Use TF-IDF to extract keywords
        tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=10)
        tfidf_matrix = tfidf_vectorizer.fit_transform(self.df['cleaned_headline'])
        feature_names = tfidf_vectorizer.get_feature_names_out()
        tfidf_scores = tfidf_matrix.sum(axis=0).A1
        keywords = {feature_names[i]: tfidf_scores[i] for i in range(len(feature_names))}
        
        # Convert keywords to DataFrame
        keywords_df = pd.DataFrame(list(keywords.items()), columns=['Keyword', 'TF-IDF Score'])
        
        # Extract common bigrams (two-word phrases)
        bigrams = self.df['cleaned_headline'].apply(lambda x: self._extract_bigrams(x))
        bigram_counts = Counter([item for sublist in bigrams for item in sublist])
        
        # Convert bigram counts to DataFrame
        bigram_counts_df = pd.DataFrame(bigram_counts.most_common(10), columns=['Bigram', 'Count'])
        
        return keywords_df, bigram_counts_df
    def plot_keyword_extraction(self, keywords_df, bigram_counts_df):
        import matplotlib.pyplot as plt

        # Plot top 10 keywords
        plt.figure(figsize=(10, 6))
        plt.bar(keywords_df['Keyword'], keywords_df['TF-IDF Score'])
        plt.xlabel('Keywords')
        plt.ylabel('TF-IDF Score')
        plt.title('Top 10 Keywords')
        plt.xticks(rotation=45)
        plt.show()

        # Plot top 10 bigrams
        plt.figure(figsize=(10, 6))
        plt.bar(bigram_counts_df['Bigram'], bigram_counts_df['Count'])
        plt.xlabel('Bigrams')
        plt.ylabel('Count')
        plt.title('Top 10 Bigrams')
        plt.xticks(rotation=45)
        plt.show()
    
    def _clean_text(self, text):
        # Lowercase and remove non-alphanumeric characters
        text = re.sub(r'\W+', ' ', text.lower())
        text = ' '.join([word for word in text.split() if word not in stop_words])
        return text

    def _get_sentiment(self, text):
        # Analyze sentiment using TextBlob
        analysis = TextBlob(text)
        polarity = analysis.sentiment.polarity
        if polarity > 0:
            sentiment = 'Positive'
        elif polarity < 0:
            sentiment = 'Negative'
        else:
            sentiment = 'Neutral'
        return sentiment, polarity

    def _extract_bigrams(self, text):
        # Extract bigrams from the text
        words = text.split()
        return list(nltk.bigrams(words))
    # Assuming `df` has a 'stock symbol' column
    def count_publisher_per_symbole(self):
        publisher_counts = self.df.groupby(['stock', 'publisher']).size().reset_index()
        publisher_counts.columns = ['stock', 'publisher', 'count']
        return publisher_counts
    def plot_publisher_counts(self, publisher_counts):
        plt.figure(figsize=(10, 6))
        sns.barplot(x='publisher', y='count', hue='stock', data=publisher_counts)
        plt.xlabel('Publisher')
        plt.ylabel('Count')
        plt.title('Publisher Counts by Stock Symbol')
        plt.xticks(rotation=45)
        plt.show()
    def contains_email(self, publisher_name):
        email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
        return bool(re.search(email_pattern, publisher_name))
    def filter_publishers_with_email(self):
        # Apply the contains_email function to the 'publisher' column and filter the rows
        self.df['has_email'] = self.df['publisher'].apply(self.contains_email)
        filtered_data = self.df[self.df['has_email'] == True]
        # Drop the temporary 'has_email' column
        filtered_data = filtered_data.drop(columns=['has_email'])
        filtered_data['domain'] = filtered_data['publisher'].apply(lambda x: x.split('@')[-1])
        return filtered_data
    def count_top_domains(self):
        self.df = self.filter_publishers_with_email()
        domain_counts = self.df['domain'].value_counts().reset_index()
        domain_counts.columns = ['domain', 'count']
        return domain_counts
    def plot_email_have_publisher(self):
        self.df = self.count_top_domains()
        plt.figure(figsize=(10, 6))
        sns.barplot(x='domain', y='count', data=self.df)
        plt.xlabel('Domain')
        plt.ylabel('Count')
        plt.title('Email Publishers by Domain')
        plt.xticks(rotation=45)
        plt.show()
    





