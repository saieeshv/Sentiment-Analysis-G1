import requests
import pandas as pd
from typing import List, Dict
from datetime import datetime, timedelta
import logging
from config.config import Config

class NewsCollector:
    def __init__(self):
        self.api_key = Config.NEWSAPI_KEY
        self.base_url = "https://newsapi.org/v2"
        self.logger = logging.getLogger(__name__)
        
    def test_connection(self):
        """Test NewsAPI connection"""
        url = f"{self.base_url}/top-headlines"
        params = {
            'apiKey': self.api_key,
            'country': 'us',
            'pageSize': 1
        }
        
        try:
            response = requests.get(url, params=params)
            if response.status_code == 200:
                self.logger.info("✅ NewsAPI connection successful")
                return True
            else:
                self.logger.error(f"❌ NewsAPI returned status: {response.status_code}")
                return False
        except Exception as e:
            self.logger.error(f"❌ NewsAPI connection failed: {str(e)}")
            return False
    
    def collect_financial_news(self, days_back: int = 1) -> pd.DataFrame:
        """Collect general financial news"""
        from_date = datetime.now() - timedelta(days=days_back)
        
        url = f"{self.base_url}/everything"
        params = {
            'apiKey': self.api_key,
            'q': 'stock market OR financial news OR earnings',
            'from': from_date.strftime('%Y-%m-%d'),
            'language': 'en',
            'sortBy': 'publishedAt',
            'pageSize': 50
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            articles = data.get('articles', [])
            
            news_data = []
            for article in articles:
                news_item = {
                    'title': article.get('title'),
                    'description': article.get('description'),
                    'content': article.get('content'),
                    'source': article.get('source', {}).get('name'),
                    'published_at': article.get('publishedAt'),
                    'url': article.get('url'),
                    'collected_at': datetime.now()
                }
                news_data.append(news_item)
            
            self.logger.info(f"📰 Collected {len(news_data)} financial news articles")
            return pd.DataFrame(news_data)
            
        except Exception as e:
            self.logger.error(f"❌ Error collecting financial news: {str(e)}")
            return pd.DataFrame()
    
    def collect_ticker_news(self, tickers: List[str]) -> pd.DataFrame:
        """Collect news for specific tickers"""
        all_news = []
        
        for ticker in tickers:
            url = f"{self.base_url}/everything"
            params = {
                'apiKey': self.api_key,
                'q': f'"{ticker}" OR "${ticker}"',
                'language': 'en',
                'sortBy': 'publishedAt',
                'pageSize': 20
            }
            
            try:
                response = requests.get(url, params=params)
                response.raise_for_status()
                
                data = response.json()
                articles = data.get('articles', [])
                
                for article in articles:
                    news_item = {
                        'ticker': ticker,
                        'title': article.get('title'),
                        'description': article.get('description'),
                        'source': article.get('source', {}).get('name'),
                        'published_at': article.get('publishedAt'),
                        'url': article.get('url'),
                        'collected_at': datetime.now()
                    }
                    all_news.append(news_item)
                
                self.logger.info(f"📰 Collected {len(articles)} articles for {ticker}")
                
            except Exception as e:
                self.logger.error(f"❌ Error collecting news for {ticker}: {str(e)}")
        
        return pd.DataFrame(all_news)
