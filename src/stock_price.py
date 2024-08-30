import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import talib as ta
import plotly.express as px
import matplotlib.pyplot as plt



class StockPrice:
    def __init__(self, file_paths, symbols):
        self.file_paths = file_paths
        self.symbols = symbols
        self.merged_data = self.merge_stock_data()

    def merge_stock_data(self):
        """
        Merges stock data from multiple CSV files and adds a Symbol column.
        """
        data_frames = []

        for file_path, symbol in zip(self.file_paths, self.symbols):
            df = pd.read_csv(file_path)
            df['Symbol'] = symbol
            df['Date'] = pd.to_datetime(df['Date'])  # Convert Date to datetime
            data_frames.append(df)

        merged_data = pd.concat(data_frames, ignore_index=True)
        return merged_data
    def calcu_info(self):
        infos = self.merged_data.info()
        return infos

    def plot_stock_subplots(self):
        for symbol in self.symbols:
            # Filter the DataFrame for the specified symbol
            symbol_df = self.merged_data[self.merged_data['Symbol'] == symbol]
            
            # Create a figure with 4 subplots in one row
            fig = make_subplots(rows=1, cols=4, subplot_titles=[f'{symbol} Close', f'{symbol} Open', f'{symbol} High', f'{symbol} Low'])
            
            # Add traces for Close, Open, High, and Low prices
            fig.add_trace(go.Scatter(x=symbol_df['Date'], y=symbol_df['Close'], name=f'{symbol} Close'), row=1, col=1)
            fig.add_trace(go.Scatter(x=symbol_df['Date'], y=symbol_df['Open'], name=f'{symbol} Open'), row=1, col=2)
            fig.add_trace(go.Scatter(x=symbol_df['Date'], y=symbol_df['High'], name=f'{symbol} High'), row=1, col=3)
            fig.add_trace(go.Scatter(x=symbol_df['Date'], y=symbol_df['Low'], name=f'{symbol} Low'), row=1, col=4)
            
            # Update layout for better visualization
            fig.update_layout(title_text=f'{symbol} Stock Prices Over Time', showlegend=False)
            
            # Show the figure
            fig.show()
    def calculate_moving_average(self,df,window_size):
        return ta.SMA( df,timeperiod=window_size)

    def calculate_indicators(self):
        # Initialize empty columns for indicators
        self.merged_data['SMA'] = pd.NA
        self.merged_data['RSI'] = pd.NA
        self.merged_data['EMA'] = pd.NA
        self.merged_data['MACD'] = pd.NA
        self.merged_data['MACD_Signal'] = pd.NA
        # Calculate RSI, EMA, and MACD for each symbol
        for symbol in self.symbols:
            symbol_df = self.merged_data[self.merged_data['Symbol'] == symbol].copy()
            
            # Calculate indicators
            symbol_df['SMA'] = self.calculate_moving_average(symbol_df['Close'], 20)
            symbol_df['RSI'] = ta.RSI(symbol_df['Close'], timeperiod=14)
            symbol_df['EMA'] = ta.EMA(symbol_df['Close'], timeperiod=14)
            macd, macd_signal, _ = ta.MACD(symbol_df['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
            symbol_df['MACD'] = macd
            symbol_df['MACD_Signal'] = macd_signal
            # Update the main DataFrame with calculated indicators
            self.merged_data.loc[self.merged_data['Symbol'] == symbol, 'SMA'] = symbol_df['SMA']
            self.merged_data.loc[self.merged_data['Symbol'] == symbol, 'RSI'] = symbol_df['RSI']
            self.merged_data.loc[self.merged_data['Symbol'] == symbol, 'EMA'] = symbol_df['EMA']
            self.merged_data.loc[self.merged_data['Symbol'] == symbol, 'MACD'] = symbol_df['MACD']
            self.merged_data.loc[self.merged_data['Symbol'] == symbol, 'MACD_Signal'] = symbol_df['MACD_Signal']

        # Sort by Date in ascending order
        self.merged_data.sort_index(inplace=True)  # Index is 'Date' after setting it
        return self.merged_data

    # def plot_data(self):
    #     # Ensure metrics are added
    #     self.add_financial_metrics()
        
    #     # Ensure 'Date' is a datetime object
    #     self.data['Date'] = pd.to_datetime(self.data['Date'])
        
    #     # Determine the number of symbols
    #     symbols = self.data['Symbol'].unique()
    #     num_symbols = len(symbols)
        
    #     # Create a figure with subplots
    #     fig, axes = plt.subplots(nrows=num_symbols, ncols=1, figsize=(14, 5 * num_symbols))
        
    #     # Adjust for cases with a single symbol
    #     if num_symbols == 1:
    #         axes = [axes]
        
    #     # Loop through each symbol and plot
    #     for i, symbol in enumerate(symbols):
    #         symbol_data = self.data[self.data['Symbol'] == symbol]
            
    #         # Plot 'Close' price
    #         ax1 = axes[i]
    #         ax1.plot(symbol_data['Date'], symbol_data['Close'], label='Close Price', color='blue')
    #         ax1.set_title(f'Stock Price and P/E Ratio for {symbol}')
    #         ax1.set_xlabel('Date')
    #         ax1.set_ylabel('Price', color='blue')
    #         ax1.tick_params(axis='y', labelcolor='blue')
            
    #         # Plot P/E Ratio on a secondary y-axis
    #         ax2 = ax1.twinx()
    #         ax2.plot(symbol_data['Date'], symbol_data['PE_Ratio'], label='P/E Ratio', color='green')
    #         ax2.set_ylabel('P/E Ratio', color='green')
    #         ax2.tick_params(axis='y', labelcolor='green')
            
    #         # Add legends
    #         ax1.legend(loc='upper left')
    #         ax2.legend(loc='upper right')
        
    #     # Adjust layout
    #     plt.tight_layout()
        
    #     # Show the plot
    #     plt.show()
    def plot_sema(self):
        # Ensure indicators are calculated before plotting
        self.calculate_indicators()
        
        # Convert all relevant columns to float to avoid data type issues
        self.merged_data['Close'] = self.merged_data['Close'].astype(float)
        self.merged_data['SMA'] = self.merged_data['SMA'].astype(float)
        
        # Handle NaN values and infer the correct data types
        self.merged_data.fillna(method='ffill', inplace=True)
        self.merged_data = self.merged_data.infer_objects(copy=False)
        
        # Ensure 'Date' is a datetime object
        self.merged_data['Date'] = pd.to_datetime(self.merged_data['Date'])
        
        # Determine the number of symbols
        symbols = self.merged_data['Symbol'].unique()
        num_symbols = len(symbols)
        
        # Create a figure with subplots
        fig, axes = plt.subplots(nrows=num_symbols, ncols=1, figsize=(14, 5 * num_symbols))
        
        # Adjust for cases with a single symbol
        if num_symbols == 1:
            axes = [axes]
        
        # Loop through each symbol and plot
        for i, symbol in enumerate(symbols):
            symbol_data = self.merged_data[self.merged_data['Symbol'] == symbol]
            
            # Plot 'Close' price on the primary y-axis
            ax1 = axes[i]
            ax1.plot(symbol_data['Date'], symbol_data['Close'], label='Close Price', color='blue')
            ax1.set_title(f'Stock Price and SMA for {symbol}')
            ax1.set_xlabel('Date')
            ax1.set_ylabel('Price', color='blue')
            ax1.tick_params(axis='y', labelcolor='blue')
            
            # Create a secondary y-axis for RSI
            ax2 = ax1.twinx()
            ax2.plot(symbol_data['Date'], symbol_data['SMA'], label='SMA', color='orange')
            ax2.set_ylabel('SMA', color='orange')
            ax2.tick_params(axis='y', labelcolor='orange')
            
            # Add legends
            ax1.legend(loc='upper left')
            ax2.legend(loc='upper right')
        
        # Adjust layout
        plt.tight_layout()
        
        # Show the plot
        plt.show()
    def plot_stock_data(self):
        # Ensure indicators are calculated before plotting
        self.calculate_indicators()
        
        # Convert all relevant columns to float to avoid data type issues
        self.merged_data['Close'] = self.merged_data['Close'].astype(float)
        self.merged_data['RSI'] = self.merged_data['RSI'].astype(float)
        
        # Handle NaN values and infer the correct data types
        self.merged_data.fillna(method='ffill', inplace=True)
        self.merged_data = self.merged_data.infer_objects(copy=False)
        
        # Ensure 'Date' is a datetime object
        self.merged_data['Date'] = pd.to_datetime(self.merged_data['Date'])
        
        # Determine the number of symbols
        symbols = self.merged_data['Symbol'].unique()
        num_symbols = len(symbols)
        
        # Create a figure with subplots
        fig, axes = plt.subplots(nrows=num_symbols, ncols=1, figsize=(14, 5 * num_symbols))
        
        # Adjust for cases with a single symbol
        if num_symbols == 1:
            axes = [axes]
        
        # Loop through each symbol and plot
        for i, symbol in enumerate(symbols):
            symbol_data = self.merged_data[self.merged_data['Symbol'] == symbol]
            
            # Plot 'Close' price on the primary y-axis
            ax1 = axes[i]
            ax1.plot(symbol_data['Date'], symbol_data['Close'], label='Close Price', color='blue')
            ax1.set_title(f'Stock Price and RSI for {symbol}')
            ax1.set_xlabel('Date')
            ax1.set_ylabel('Price', color='blue')
            ax1.tick_params(axis='y', labelcolor='blue')
            
            # Create a secondary y-axis for RSI
            ax2 = ax1.twinx()
            ax2.plot(symbol_data['Date'], symbol_data['RSI'], label='RSI', color='orange')
            ax2.set_ylabel('RSI', color='orange')
            ax2.tick_params(axis='y', labelcolor='orange')
            
            # Add legends
            ax1.legend(loc='upper left')
            ax2.legend(loc='upper right')
        
        # Adjust layout
        plt.tight_layout()
        
        # Show the plot
        plt.show()



    def plot_ema(self):
        # Ensure indicators are calculated before plotting
        self.calculate_indicators()
        
        # Convert all relevant columns to float to avoid data type issues
        self.merged_data['Close'] = self.merged_data['Close'].astype(float)
        self.merged_data['EMA'] = self.merged_data['EMA'].astype(float)
        
        # Handle NaN values and infer the correct data types
        self.merged_data.fillna(method='ffill', inplace=True)
        self.merged_data = self.merged_data.infer_objects(copy=False)
        
        # Ensure 'Date' is a datetime object
        self.merged_data['Date'] = pd.to_datetime(self.merged_data['Date'])
        
        # Determine the number of symbols
        symbols = self.merged_data['Symbol'].unique()
        num_symbols = len(symbols)
        
        # Create a figure with subplots
        fig, axes = plt.subplots(nrows=num_symbols, ncols=1, figsize=(14, 5 * num_symbols))
        
        # Adjust for cases with a single symbol
        if num_symbols == 1:
            axes = [axes]
        
        # Loop through each symbol and plot
        for i, symbol in enumerate(symbols):
            symbol_data = self.merged_data[self.merged_data['Symbol'] == symbol]
            
            # Plot 'Close' price on the primary y-axis
            ax1 = axes[i]
            ax1.plot(symbol_data['Date'], symbol_data['Close'], label='Close Price', color='blue')
            ax1.set_title(f'Stock Price and EMA for {symbol}')
            ax1.set_xlabel('Date')
            ax1.set_ylabel('Price', color='blue')
            ax1.tick_params(axis='y', labelcolor='blue')
            
            # Create a secondary y-axis for RSI
            ax2 = ax1.twinx()
            ax2.plot(symbol_data['Date'], symbol_data['EMA'], label='EMA', color='orange')
            ax2.set_ylabel('EMA', color='orange')
            ax2.tick_params(axis='y', labelcolor='orange')
            
            # Add legends
            ax1.legend(loc='upper left')
            ax2.legend(loc='upper right')
        
        # Adjust layout
        plt.tight_layout()
        
        # Show the plot
        plt.show()

    def plot_macd(self):
        # Ensure indicators are calculated before plotting
        self.calculate_indicators()
        
        # Convert all relevant columns to float to avoid data type issues
        self.merged_data['Close'] = self.merged_data['Close'].astype(float)
        self.merged_data['MACD'] = self.merged_data['MACD'].astype(float)
        
        # Handle NaN values and infer the correct data types
        self.merged_data.fillna(method='ffill', inplace=True)
        self.merged_data = self.merged_data.infer_objects(copy=False)
        
        # Ensure 'Date' is a datetime object
        self.merged_data['Date'] = pd.to_datetime(self.merged_data['Date'])
        
        # Determine the number of symbols
        symbols = self.merged_data['Symbol'].unique()
        num_symbols = len(symbols)
        
        # Create a figure with subplots
        fig, axes = plt.subplots(nrows=num_symbols, ncols=1, figsize=(14, 5 * num_symbols))
        
        # Adjust for cases with a single symbol
        if num_symbols == 1:
            axes = [axes]
        
        # Loop through each symbol and plot
        for i, symbol in enumerate(symbols):
            symbol_data = self.merged_data[self.merged_data['Symbol'] == symbol]
            
            # Plot 'Close' price on the primary y-axis
            ax1 = axes[i]
            ax1.plot(symbol_data['Date'], symbol_data['Close'], label='Close Price', color='blue')
            ax1.set_title(f'Stock Price and MACD for {symbol}')
            ax1.set_xlabel('Date')
            ax1.set_ylabel('Price', color='blue')
            ax1.tick_params(axis='y', labelcolor='blue')
            
            # Create a secondary y-axis for RSI
            ax2 = ax1.twinx()
            ax2.plot(symbol_data['Date'], symbol_data['MACD'], label='MACD', color='orange')
            ax2.set_ylabel('MACD', color='orange')
            ax2.tick_params(axis='y', labelcolor='orange')
            
            # Add legends
            ax1.legend(loc='upper left')
            ax2.legend(loc='upper right')
        
        # Adjust layout
        plt.tight_layout()
        
        # Show the plot
        plt.show()

if __name__ == "__main__":
    file_paths = [
        '../data/yfinance_data/AAPL_historical_data.csv',
        '../data/yfinance_data/AMZN_historical_data.csv',
        '../data/yfinance_data/GOOG_historical_data.csv',
        '../data/yfinance_data/META_historical_data.csv',
        '../data/yfinance_data/MSFT_historical_data.csv',
        '../data/yfinance_data/NVDA_historical_data.csv',
        '../data/yfinance_data/TSLA_historical_data.csv'
    ]
    symbols = ['AAPL', 'AMZN', 'GOOG', 'META', 'MSFT', 'NVDA', 'TSLA'] 

    stock_price_analyzer = StockPrice(file_paths, symbols)
    stock_price_analyzer.plot_stock_subplots()
