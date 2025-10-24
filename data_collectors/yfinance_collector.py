import yfinance as yf
import pandas as pd
from typing import List, Optional
from datetime import datetime, timedelta
import logging
from config.config import Config 

class YFinanceCollector:
    def __init__(self, tickers: List[str]):
        self.tickers = tickers
        self.logger = logging.getLogger(__name__)
        
    def get_stock_data(self, start: Optional[str] = None, end: Optional[str] = None, interval: str = "1d") -> pd.DataFrame:
        all_stock_data = []
        
        for ticker in self.tickers:
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(start=start, end=end, interval=interval)
                
                if not hist.empty:
                    hist['Ticker'] = ticker
                    hist['category'] = Config.get_sector(ticker)  # ✅ Now Config is available
                    hist.reset_index(inplace=True)
                    all_stock_data.append(hist)
                    self.logger.info(f"✅ Collected data for {ticker}")
                else:
                    self.logger.warning(f"⚠️ No data for {ticker}")
                    
            except Exception as e:
                self.logger.error(f"❌ Error collecting data for {ticker}: {str(e)}")
        
        if all_stock_data:
            combined_df = pd.concat(all_stock_data, ignore_index=True)
            return combined_df
        else:
            return pd.DataFrame()
    
    
        
