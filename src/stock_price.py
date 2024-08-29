import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import talib as ta
import plotly.express as px

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
    def calculate_moving_average(self,data, window_size):
        return ta.SMA(data, timeperiod=window_size)

    def calculate_indicators(self):
        # Initialize empty columns for indicators
        self.merged_data['RSI'] = pd.NA
        self.merged_data['EMA'] = pd.NA
        self.merged_data['MACD'] = pd.NA
        self.merged_data['MACD_Signal'] = pd.NA
        
        # Calculate RSI, EMA, and MACD for each symbol
        for symbol in self.symbols:
            symbol_df = self.merged_data[self.merged_data['Symbol'] == symbol].copy()
            
            # Calculate indicators
            symbol_df['RSI'] = ta.RSI(symbol_df['Close'], timeperiod=14)
            symbol_df['EMA'] = ta.EMA(symbol_df['Close'], timeperiod=14)
            macd, macd_signal, _ = ta.MACD(symbol_df['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
            symbol_df['MACD'] = macd
            symbol_df['MACD_Signal'] = macd_signal
            
            # Update the main DataFrame with calculated indicators
            self.merged_data.loc[self.merged_data['Symbol'] == symbol, 'RSI'] = symbol_df['RSI']
            self.merged_data.loc[self.merged_data['Symbol'] == symbol, 'EMA'] = symbol_df['EMA']
            self.merged_data.loc[self.merged_data['Symbol'] == symbol, 'MACD'] = symbol_df['MACD']
            self.merged_data.loc[self.merged_data['Symbol'] == symbol, 'MACD_Signal'] = symbol_df['MACD_Signal']

        # Sort by Date in ascending order
        self.merged_data.sort_index(inplace=True)  # Index is 'Date' after setting it
        return self.merged_data
    def plot_stock_data(self):
        # Ensure indicators are calculated before plotting
        self.calculate_indicators()
        
        # Plot using Plotly Express
        fig = px.line(self.merged_data, x=self.merged_data.index, y=['Close', 'RSI'],
                      title='Stock Price with Moving Average',
                      labels={'value': 'Price', 'variable': 'Legend'})
        fig.show()

    def plot_rsi(self, data):
        data = self.calculate_indicators()
        fig = px.line(data, x=data.index, y='RSI', title='Relative Strength Index (RSI)')
        fig.show()

    def plot_ema(self, data):
        data = self.calculate_indicators()
        fig = px.line(data, x=data.index, y=['Close', 'EMA'], title='Stock Price with Exponential Moving Average')
        fig.show()

    def plot_macd(self, data):
        data = self.calculate_indicators()
        fig = px.line(data, x=data.index, y=['MACD', 'MACD_Signal'], title='Moving Average Convergence Divergence (MACD)')
        fig.show()




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
