import yfinance as yf
import pandas as pd
from typing import List, Optional
from datetime import datetime, timedelta
import logging


class YFinanceCollector:
    def __init__(self, tickers: List[str]):
        self.tickers = tickers
        self.logger = logging.getLogger(__name__)
        
    def get_stock_data(self, start: Optional[str] = None, end: Optional[str] = None, interval: str = "1d") -> pd.DataFrame:
        """
        Collect historical stock data with flexible date ranges and intervals.
        
        Args:
            start (str): start date in 'YYYY-MM-DD' format. Defaults to None (fetch max allowed).
            end (str): end date in 'YYYY-MM-DD' format. Defaults to None (up to current date).
            interval (str): data interval. Supported intervals: '1d', '1h', '5m', etc.
        
        Returns:
            pd.DataFrame: Combined DataFrame with historical data for all tickers.
        """
        all_stock_data = []
        
        for ticker in self.tickers:
            try:
                stock = yf.Ticker(ticker)
                
                hist = stock.history(start=start, end=end, interval=interval)
                
                if not hist.empty:
                    hist['Ticker'] = ticker
                    hist.reset_index(inplace=True)  # Make Date a column
                    all_stock_data.append(hist)
                    self.logger.info(f"✅ Collected data for {ticker} from {start or 'max'} to {end or 'now'} at {interval} interval")
                else:
                    self.logger.warning(f"⚠️ No data for {ticker} - start: {start} end: {end} interval: {interval}")
                    
            except Exception as e:
                self.logger.error(f"❌ Error collecting data for {ticker}: {str(e)}")
        
        # Combine all ticker data into a single DataFrame
        if all_stock_data:
            combined_df = pd.concat(all_stock_data, ignore_index=True)
            
            # ADD CATEGORY HERE ⬇️
            from config.config import Config
            combined_df['category'] = combined_df['Ticker'].apply(lambda x: Config.get_sector(x))
            
            self.logger.info(f"📊 Combined stock data: {len(combined_df)} rows across {len(self.tickers)} tickers")
            return combined_df
        else:
            return pd.DataFrame()
    
    def get_company_news(self, days_back: int = 30) -> pd.DataFrame:

        all_news = []
        cutoff_date = datetime.now() - timedelta(days=days_back)
        
        for ticker in self.tickers:
            try:
                stock = yf.Ticker(ticker)
                news = stock.news
                
                for article in news:
                    # Filter out older articles based on 'providerPublishTime' if available
                    publish_time = article.get('providerPublishTime')
                    if publish_time:
                        article_date = datetime.utcfromtimestamp(publish_time)
                        if article_date < cutoff_date:
                            continue
                    
                    # Flatten the article dictionary
                    article_data = {
                        'ticker': ticker,
                        'title': article.get('title', ''),
                        'publisher': article.get('publisher', ''),
                        'link': article.get('link', ''),
                        'providerPublishTime': datetime.utcfromtimestamp(publish_time) if publish_time else None,
                        'type': article.get('type', ''),
                        'collected_at': datetime.now()
                    }
                    
                    all_news.append(article_data)
                
                self.logger.info(f"📰 Got {len([a for a in all_news if a['ticker'] == ticker])} recent news for {ticker} within last {days_back} days")
                
            except Exception as e:
                self.logger.error(f"❌ News error for {ticker}: {str(e)}")
        
        # Convert to DataFrame
        if all_news:
            news_df = pd.DataFrame(all_news)
            
            # ADD CATEGORY HERE ⬇️
            from config.config import Config
            news_df['category'] = news_df['ticker'].apply(lambda x: Config.get_sector(x))
            
            self.logger.info(f"📰 Combined news data: {len(news_df)} articles")
            return news_df
        else:
            return pd.DataFrame()