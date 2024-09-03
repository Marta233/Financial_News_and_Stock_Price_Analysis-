import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import talib as ta
import matplotlib.pyplot as plt

class StockPrice:
    def __init__(self, file_paths, symbols):
        """
        Initializes the StockPrice class with file paths and stock symbols.
        
        :param file_paths: List of paths to CSV files containing stock data.
        :param symbols: List of stock symbols corresponding to the data files.
        """
        self.file_paths = file_paths
        self.symbols = symbols
        self.merged_data = self.merge_stock_data()

    def merge_stock_data(self):
        """Merges stock data from multiple CSV files and adds a Symbol column."""
        data_frames = []

        for file_path, symbol in zip(self.file_paths, self.symbols):
            df = pd.read_csv(file_path)
            df['Symbol'] = symbol  # Add a column for the stock symbol
            df['Date'] = pd.to_datetime(df['Date'])  # Convert Date to datetime
            data_frames.append(df)

        return pd.concat(data_frames, ignore_index=True)  # Concatenate all data frames

    def calcu_info(self):
        """Returns information about the merged DataFrame."""
        return self.merged_data.info()

    def plot_stock_subplots(self):
        """Plots subplots for Close, Open, High, and Low prices for each symbol."""
        for symbol in self.symbols:
            symbol_df = self.merged_data[self.merged_data['Symbol'] == symbol]
            fig = make_subplots(rows=1, cols=4, subplot_titles=[
                f'{symbol} Close', f'{symbol} Open', f'{symbol} High', f'{symbol} Low'
            ])

            # Create traces for each price type
            fig.add_trace(go.Scatter(x=symbol_df['Date'], y=symbol_df['Close'], name=f'{symbol} Close'), row=1, col=1)
            fig.add_trace(go.Scatter(x=symbol_df['Date'], y=symbol_df['Open'], name=f'{symbol} Open'), row=1, col=2)
            fig.add_trace(go.Scatter(x=symbol_df['Date'], y=symbol_df['High'], name=f'{symbol} High'), row=1, col=3)
            fig.add_trace(go.Scatter(x=symbol_df['Date'], y=symbol_df['Low'], name=f'{symbol} Low'), row=1, col=4)

            # Update layout
            fig.update_layout(title_text=f'{symbol} Stock Prices Over Time', showlegend=False)
            fig.show()

    def calculate_moving_average(self, df, window_size):
        """Calculates Simple Moving Average (SMA) for a given DataFrame and window size."""
        return ta.SMA(df, timeperiod=window_size)

    def calculate_indicators(self):
        """Calculates various stock indicators and updates the merged DataFrame."""
        # Initialize columns for indicators
        self.merged_data['SMA'] = pd.NA
        self.merged_data['RSI'] = pd.NA
        self.merged_data['EMA'] = pd.NA
        self.merged_data['MACD'] = pd.NA
        self.merged_data['MACD_Signal'] = pd.NA

        for symbol in self.symbols:
            symbol_df = self.merged_data[self.merged_data['Symbol'] == symbol].copy()
            symbol_df['SMA'] = self.calculate_moving_average(symbol_df['Close'], 20)
            symbol_df['RSI'] = ta.RSI(symbol_df['Close'], timeperiod=14)
            symbol_df['EMA'] = ta.EMA(symbol_df['Close'], timeperiod=14)
            macd, macd_signal, _ = ta.MACD(symbol_df['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
        
            symbol_df['MACD'] = macd
            symbol_df['MACD_Signal'] = macd_signal
            
            # Update main DataFrame with calculated indicators
            for indicator in ['SMA', 'RSI', 'EMA', 'MACD', 'MACD_Signal']:
                self.merged_data.loc[self.merged_data['Symbol'] == symbol, indicator] = symbol_df[indicator]

        # Sort by Date in ascending order
        self.merged_data.sort_values(by='Date', inplace=True)
        return self.merged_data

    def plot_indicator(self, indicator, title):
        """Plots stock prices along with a specified indicator."""
        self.calculate_indicators()
        symbols = self.merged_data['Symbol'].unique()
        num_symbols = len(symbols)

        fig, axes = plt.subplots(nrows=num_symbols, ncols=1, figsize=(14, 5 * num_symbols), sharex=True)

        for i, symbol in enumerate(symbols):
            symbol_data = self.merged_data[self.merged_data['Symbol'] == symbol]

            # Drop NA values for the Close and the specified indicator
            symbol_data = symbol_data.dropna(subset=['Close', indicator])

            ax1 = axes[i]
            ax1.plot(symbol_data['Date'], symbol_data['Close'], label='Close Price', color='blue')
            ax1.set_title(f'Stock Price and {title} for {symbol}')
            ax1.set_ylabel('Price', color='blue')
            ax1.tick_params(axis='y', labelcolor='blue')

            ax2 = ax1.twinx()  # Create a second y-axis for the indicator
            ax2.plot(symbol_data['Date'], symbol_data[indicator], label=indicator, color='orange')
            ax2.set_ylabel(indicator, color='orange')
            ax2.tick_params(axis='y', labelcolor='orange')

            ax1.legend(loc='upper left')
            ax2.legend(loc='upper right')

        plt.tight_layout()  # Adjust layout to prevent overlap
        plt.show()

    def calculate_returns(self):
        """Calculates daily and cumulative returns."""
        self.merged_data['Daily_Return'] = self.merged_data.groupby('Symbol')['Close'].pct_change()
        self.merged_data['Cumulative_Return'] = (1 + self.merged_data['Daily_Return']).groupby(self.merged_data['Symbol']).cumprod() - 1

    def calculate_volatility(self, period=30):
        """Calculates rolling volatility."""
        volatility = self.merged_data.groupby('Symbol')['Daily_Return'].rolling(window=period).std().reset_index(level=0, drop=True)
        self.merged_data['Volatility'] = volatility

    def calculate_sharpe_ratio(self, risk_free_rate=0.0):
        """Calculates the Sharpe Ratio for each symbol."""
        def sharpe_ratio(group):
            returns = group['Daily_Return'].dropna()
            return (returns.mean() - risk_free_rate) / returns.std() if returns.std() != 0 else pd.NA

        sharpe_ratios = self.merged_data.groupby('Symbol').apply(sharpe_ratio).reset_index(name='Sharpe_Ratio')
        self.merged_data = self.merged_data.merge(sharpe_ratios, on='Symbol', how='left')

    def calculate_max_drawdown(self):
        """Calculates the maximum drawdown for each symbol."""
        self.merged_data['Cumulative_Max'] = self.merged_data.groupby('Symbol')['Close'].cummax()
        self.merged_data['Drawdown'] = self.merged_data['Close'] / self.merged_data['Cumulative_Max'] - 1
        self.merged_data['Max_Drawdown'] = self.merged_data.groupby('Symbol')['Drawdown'].cummin()

    def calculate_all_metrics(self):
        """Calculates all financial metrics."""
        self.calculate_indicators()
        self.calculate_returns()
        self.calculate_volatility()
        self.calculate_sharpe_ratio()
        self.calculate_max_drawdown()
        return self.merged_data

    def plot_metric(self, metric, title):
        """Plots a specified metric for all symbols."""
        fig = make_subplots(rows=len(self.symbols), cols=1, shared_xaxes=True, vertical_spacing=0.02,
                            subplot_titles=[f'{symbol} {title}' for symbol in self.symbols])

        for i, symbol in enumerate(self.symbols):
            symbol_df = self.merged_data[self.merged_data['Symbol'] == symbol]
            fig.add_trace(go.Scatter(x=symbol_df['Date'], y=symbol_df[metric], mode='lines', name=f'{symbol} {title}'), row=i + 1, col=1)

        fig.update_layout(title_text=title, showlegend=False)
        fig.show()


if __name__ == "__main__":
    # Define file paths for historical stock data
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

    # Create an instance of StockPrice and plot subplots for stock data
    stock_price_analyzer = StockPrice(file_paths, symbols)
    stock_price_analyzer.plot_stock_subplots()