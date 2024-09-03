import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

class CorrelationAnalysis:
    def __init__(self, file_paths):
        self.file_paths = file_paths
        self.merged_data = self.merge_stock_data()

    def merge_stock_data(self):   
        # Load data from CSV files
        df_headlines = pd.read_csv(self.file_paths[0])
        df_prices = pd.read_csv(self.file_paths[1])

        # Rename 'stock' to 'Symbol' if it exists
        if 'stock' in df_headlines.columns:
            df_headlines.rename(columns={'stock': 'Symbol'}, inplace=True)

        # Convert 'date' columns to datetime
        df_headlines['date'] = pd.to_datetime(df_headlines['date'], format='ISO8601').dt.tz_localize(None)
        df_prices['Date'] = pd.to_datetime(df_prices['Date'], format='ISO8601')

        # Merge DataFrames on 'date' and 'Symbol'
        merged_df = pd.merge(df_headlines, df_prices, left_on=['date', 'Symbol'], right_on=['Date', 'Symbol'])
        merged_df.drop(columns=['Unnamed: 0_x', 'Unnamed: 0_y'], inplace=True)

        # Group by date and symbol, calculate average polarity
        grouped_df = merged_df.groupby(['date', 'Symbol'])['polarity'].mean().reset_index()
        merged_df.drop(columns=['polarity'], inplace=True)
        # Get the first row of each group to retain other attributes
        first_rows = merged_df.groupby(['date', 'Symbol']).first().reset_index()
        
        # Merge the average polarity back with the first rows
        merged_df = first_rows.merge(grouped_df, on=['date', 'Symbol'], how='left')
        merged_df.drop(columns=['Sharpe_Ratio_x', 'Cumulative_Max', 'Drawdown', 'Max_Drawdown','Sharpe_Ratio_y'], inplace=True)
        # merged_df.rename(columns={'polarity_x': 'polarity'}, inplace=True)
        return merged_df
    def correlation_for_cols(self, col1, col2):
        # Calculate the correlation between two columns
        correlation = self.merged_data[col1].corr(self.merged_data[col2])

        # Create a scatter plot
        fig = px.scatter(self.merged_data, x=col1, y=col2, color='Symbol', title=f'{col1} vs {col2} (Correlation: {correlation:.2f})')

        # Show the plot
        fig.show()

    def plot_correlation_matrix(self):
        # Select numerical columns
        numerical_cols = self.merged_data.select_dtypes(include=['float64', 'int64']).columns

        # Calculate the correlation matrix
        corr_matrix = self.merged_data[numerical_cols].corr()

        # Create a heatmap
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='Viridis'
        ))

        # Add labels and title
        fig.update_layout(
            title='Correlation Matrix',
            xaxis_title='Attributes',
            yaxis_title='Attributes'
        )

        # Show the plot
        fig.show()
    def correl_selected_attributes(self):
        # Select numerical columns
        numerical_cols = self.merged_data.select_dtypes(include=['float64', 'int64']).columns

        # Calculate the correlation matrix
        corr_matrix = self.merged_data[numerical_cols].corr()

        # Select the desired attributes
        selected_attributes = ['polarity', 'Daily_Return', 'Cumulative_Return', 'Volatility', 'Close', 'SMA', 'RSI']

        # Create a heatmap for the selected attributes
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.loc[selected_attributes, selected_attributes].values,
            x=selected_attributes,
            y=selected_attributes,
            colorscale='Viridis',
            text=corr_matrix.loc[selected_attributes, selected_attributes].values,  # Display correlation values
            texttemplate="%{text:.2f}",  # Format the text to 2 decimal places
            hoverinfo='text'  # Show text on hover
        ))

        # Add labels and title
        fig.update_layout(
            title='Correlation Matrix for Selected Attributes',
            xaxis_title='Attributes',
            yaxis_title='Attributes'
        )

        # Show the plot
        fig.show()
    def corr_spec_att_for_each_symbol_attrib():
        pass
    def corr_spec_att_for_each_symbol(self):
        # Select numerical columns
        numerical_cols = self.merged_data.select_dtypes(include=['float64', 'int64']).columns

        # Calculate the correlation matrix
        corr_matrix = self.merged_data[numerical_cols].corr()

        # Create a list of symbols
        symbols = self.merged_data['Symbol'].unique()

        # Iterate over each symbol
        for symbol in symbols:
            # Select the rows for the current symbol
            symbol_data = self.merged_data[self.merged_data['Symbol'] == symbol]

            # Select the desired attributes
            selected_attributes = ['polarity', 'Daily_Return', 'Cumulative_Return', 'Close', 'SMA', 'RSI']
            symbol_corr_matrix = symbol_data[selected_attributes].corr()

            # Create a heatmap for the current symbol
            fig = go.Figure(data=go.Heatmap(
                z=symbol_corr_matrix.values,
                x=selected_attributes,
                y=selected_attributes,
                colorscale='Viridis',
                text=symbol_corr_matrix.values,  # Display correlation values
                texttemplate="%{text:.2f}",  # Format the text to 2 decimal places
                hoverinfo='text'  # Show text on hover
            ))

            # Add labels and title
            fig.update_layout(
                title=f'Correlation Matrix for {symbol}',
                xaxis_title='Attributes',
                yaxis_title='Attributes'
            )

            # Show the plot
            fig.show()