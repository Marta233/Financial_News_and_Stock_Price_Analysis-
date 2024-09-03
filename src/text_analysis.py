import pandas as pd
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import nltk
from nltk.corpus import stopwords
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt
import re

# Download the stopwords from NLTK
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

class TextAnalyzer:
    def __init__(self, df):
        """
        Initializes the TextAnalyzer class with a DataFrame.
        
        :param df: DataFrame containing the text data (e.g., headlines).
        """
        self.df = df

    def remove_unname(self):
        """Removes the unnamed column from the DataFrame."""
        self.df = self.df.drop(columns=['Unnamed: 0']) 
        return self.df

    def sentiment_analysis(self):
        """Cleans the headlines and calculates sentiment polarity."""
        self.df['cleaned_headline'] = self.df['headline'].apply(self._clean_text)
        # Extract sentiment and polarity as separate columns
        self.df['sentiment'], self.df['polarity'] = zip(*self.df['cleaned_headline'].apply(self._get_sentiment))
        return self.df

    def keyword_extraction(self):
        """Extracts keywords and common bigrams from the headlines."""
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
        """Plots the extracted keywords and bigrams."""
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
        """Cleans the input text by lowering case and removing non-alphanumeric characters."""
        text = re.sub(r'\W+', ' ', text.lower())
        text = ' '.join([word for word in text.split() if word not in stop_words])
        return text

    def _get_sentiment(self, text):
        """Analyzes sentiment using TextBlob and returns sentiment label and polarity."""
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
        """Extracts bigrams from the cleaned text."""
        words = text.split()
        return list(nltk.bigrams(words))

    def count_publisher_per_symbole(self):
        """Counts the number of headlines per stock symbol and publisher."""
        publisher_counts = self.df.groupby(['stock', 'publisher']).size().reset_index()
        publisher_counts.columns = ['stock', 'publisher', 'count']
        return publisher_counts

    def plot_publisher_counts(self, publisher_counts):
        """Plots the counts of publishers by stock symbol."""
        plt.figure(figsize=(10, 6))
        sns.barplot(x='publisher', y='count', hue='stock', data=publisher_counts)
        plt.xlabel('Publisher')
        plt.ylabel('Count')
        plt.title('Publisher Counts by Stock Symbol')
        plt.xticks(rotation=45)
        plt.show()

    def contains_email(self, publisher_name):
        """Checks if the publisher name contains an email address."""
        email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
        return bool(re.search(email_pattern, publisher_name))

    def filter_publishers_with_email(self):
        """Filters publishers that have an email address and extracts the domain."""
        self.df['has_email'] = self.df['publisher'].apply(self.contains_email)
        filtered_data = self.df[self.df['has_email'] == True]
        # Drop the temporary 'has_email' column
        filtered_data = filtered_data.drop(columns=['has_email'])
        filtered_data['domain'] = filtered_data['publisher'].apply(lambda x: x.split('@')[-1])
        return filtered_data

    def count_top_domains(self):
        """Counts the occurrence of each domain from publishers with email addresses."""
        self.df = self.filter_publishers_with_email()
        domain_counts = self.df['domain'].value_counts().reset_index()
        domain_counts.columns = ['domain', 'count']
        return domain_counts

    def plot_email_have_publisher(self):
        """Plots the counts of publishers by their email domains."""
        self.df = self.count_top_domains()
        plt.figure(figsize=(10, 6))
        sns.barplot(x='domain', y='count', data=self.df)
        plt.xlabel('Domain')
        plt.ylabel('Count')
        plt.title('Email Publishers by Domain')
        plt.xticks(rotation=45)
        plt.show()
    def topic_modeling(self, n_topics=5):
        """Performs topic modeling using LDA and returns the topics."""
        # Ensure 'cleaned_headline' exists
        self.df['cleaned_headline'] = self.df['headline'].apply(self._clean_text)
        
        # Create TF-IDF matrix
        tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf_vectorizer.fit_transform(self.df['cleaned_headline'])
        
        # Initialize and fit LDA
        lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
        lda.fit(tfidf_matrix)
        
        # Get the words associated with each topic
        topic_words = {}
        feature_names = tfidf_vectorizer.get_feature_names_out()
        for topic_idx, topic in enumerate(lda.components_):
            topic_words[topic_idx] = [feature_names[i] for i in topic.argsort()[:-6:-1]]
        
        return topic_words