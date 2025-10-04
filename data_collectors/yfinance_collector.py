import yfinance as yf
import pandas as pd
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import logging

class YFinanceCollector:
    def __init__(self, tickers: List[str]):
        self.tickers = tickers
        self.logger = logging.getLogger(__name__)
        
    def get_stock_data(self, start: Optional[str] = None, end: Optional[str] = None, interval: str = "1d") -> Dict[str, pd.DataFrame]:
        """
        Collect historical stock data with flexible date ranges and intervals.
        
        Args:
            start (str): start date in 'YYYY-MM-DD' format. Defaults to None (fetch max allowed).
            end (str): end date in 'YYYY-MM-DD' format. Defaults to None (up to current date).
            interval (str): data interval. Supported intervals: '1d', '1h', '5m', etc.
        
        Returns:
            Dict[str, pd.DataFrame]: Dictionary of ticker to DataFrame with historical data.
        """
        stock_data = {}
        
        for ticker in self.tickers:
            try:
                stock = yf.Ticker(ticker)
                
                hist = stock.history(start=start, end=end, interval=interval)
                
                if not hist.empty:
                    hist['Ticker'] = ticker
                    stock_data[ticker] = hist
                    self.logger.info(f"✅ Collected data for {ticker} from {start or 'max'} to {end or 'now'} at {interval} interval")
                else:
                    self.logger.warning(f"⚠️ No data for {ticker} - start: {start} end: {end} interval: {interval}")
                    
            except Exception as e:
                self.logger.error(f"❌ Error collecting data for {ticker}: {str(e)}")
                
        return stock_data
    
    def get_company_news(self, days_back: int = 30) -> List[Dict]:
        all_news = []
        cutoff_date = datetime.now() - timedelta(days=days_back)
        
        for ticker in self.tickers:
            try:
                stock = yf.Ticker(ticker)
                news = stock.news
                
                filtered_news = []
                for article in news:
                    # Filter out older articles based on 'providerPublishTime' if available
                    publish_time = article.get('providerPublishTime')
                    if publish_time:
                        article_date = datetime.utcfromtimestamp(publish_time)
                        if article_date < cutoff_date:
                            continue
                    
                    article['ticker'] = ticker
                    article['collected_at'] = datetime.now()
                    filtered_news.append(article)
                
                all_news.extend(filtered_news)
                self.logger.info(f"📰 Got {len(filtered_news)} recent news for {ticker} within last {days_back} days")
                
            except Exception as e:
                self.logger.error(f"❌ News error for {ticker}: {str(e)}")
        
        return all_news
