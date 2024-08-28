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
    def sentiment_analysis(self):
        # Clean the headlines and calculate sentiment polarity
        self.df['cleaned_headline'] = self.df['headline'].apply(self._clean_text)
        self.df['sentiment'] = self.df['cleaned_headline'].apply(self._get_sentiment)
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
        if analysis.sentiment.polarity > 0:
            return 'Positive'
        elif analysis.sentiment.polarity < 0:
            return 'Negative'
        else:
            return 'Neutral'

    def _extract_bigrams(self, text):
        # Extract bigrams from the text
        words = text.split()
        return list(nltk.bigrams(words))
    # Assuming `df` has a 'stock symbol' column
    def count_publisher_per_symbole(self):
        stock_publisher = pd.crosstab(self.df['stock'], self.df['publisher'])
        plt.figure(figsize=(12,8))
        sns.heatmap(stock_publisher, cmap='coolwarm', annot=True, fmt="d")
        plt.title('Stock Symbols by Publisher')
        plt.show()

