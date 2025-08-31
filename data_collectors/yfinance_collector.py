import yfinance as yf
import pandas as pd
from typing import List, Dict
from datetime import datetime
import logging

class YFinanceCollector:
    def __init__(self, tickers: List[str]):
        self.tickers = tickers
        self.logger = logging.getLogger(__name__)
        
    def get_stock_data(self, period: str = "1mo") -> Dict[str, pd.DataFrame]:
        """Collect basic stock data"""
        stock_data = {}
        
        for ticker in self.tickers:
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(period=period)
                
                if not hist.empty:
                    hist['Ticker'] = ticker
                    stock_data[ticker] = hist
                    self.logger.info(f"✅ Collected data for {ticker}")
                else:
                    self.logger.warning(f"⚠️  No data for {ticker}")
                    
            except Exception as e:
                self.logger.error(f"❌ Error with {ticker}: {str(e)}")
                
        return stock_data
    
    def get_company_news(self) -> List[Dict]:
        """Get company news from yfinance"""
        all_news = []
        
        for ticker in self.tickers:
            try:
                stock = yf.Ticker(ticker)
                news = stock.news
                
                for article in news:
                    article['ticker'] = ticker
                    article['collected_at'] = datetime.now()
                    all_news.append(article)
                
                self.logger.info(f"📰 Got {len(news)} news for {ticker}")
                
            except Exception as e:
                self.logger.error(f"❌ News error for {ticker}: {str(e)}")
        
        return all_news
