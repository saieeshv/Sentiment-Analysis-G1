import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import logging
from typing import List
from config.config import Config


class YFinanceCollector:
    def __init__(self, tickers: List[str]):
        """
        Initialize collector with tickers
        
        Args:
            tickers: List of ticker symbols
        """
        self.tickers = tickers
        self.logger = logging.getLogger(__name__)
        
    def get_stock_data(self, start_date=None, end_date=None, interval: str = '1d') -> pd.DataFrame:
        """
        Collect stock price data for the configured date range
        
        Args:
            start_date: Start date (optional, uses default if None)
            end_date: End date (optional, uses datetime.now() if None)
            interval: Data interval ('1d' for daily, '1h' for hourly, etc.)
        """
        all_data = []
        
        # Set default dates if not provided
        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            start_date = end_date - timedelta(days=Config.DEFAULT_NEWS_DAYS_BACK)
        
        for ticker in self.tickers:
            try:
                self.logger.info(f"📈 Fetching data for {ticker}")
                
                stock = yf.Ticker(ticker)
                
                # Use the configured date range
                hist = stock.history(
                    start=start_date.strftime('%Y-%m-%d') if hasattr(start_date, 'strftime') else start_date,
                    end=end_date.strftime('%Y-%m-%d') if hasattr(end_date, 'strftime') else end_date,
                    interval=interval
                )
                
                if hist.empty:
                    self.logger.warning(f"⚠️ No data for {ticker}")
                    continue
                
                # Reset index to get date as column
                hist = hist.reset_index()
                hist['ticker'] = ticker
                hist['category'] = Config.get_sector(ticker)
                
                all_data.append(hist)
                self.logger.info(f"✅ Collected {len(hist)} data points for {ticker}")
                
            except Exception as e:
                self.logger.error(f"❌ Error fetching {ticker}: {e}")
                continue
        
        if all_data:
            df = pd.concat(all_data, ignore_index=True)
            df.columns = df.columns.str.lower()
            
            self.logger.info(f"📊 Total stock data: {len(df)} rows across {len(self.tickers)} tickers")
            return df
        else:
            self.logger.warning("⚠️ No stock data collected")
            return pd.DataFrame()
