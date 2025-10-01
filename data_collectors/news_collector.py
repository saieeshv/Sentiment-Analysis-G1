import requests
import pandas as pd
from typing import List, Optional
from datetime import datetime, timedelta
import logging
from config.config import Config
from newspaper import Article
from bs4 import BeautifulSoup
import re
import nltk


# Ensure required NLTK corpora are downloaded
nltk.download('punkt')
nltk.download('punkt_tab')


class NewsCollector:
    def __init__(self, source: str = "newsapi"):
        """
        Initialize NewsCollector with a specific source.
        source: "newsapi" or "eventregistry"
        """
        self.source = source.lower()
        if self.source == "newsapi":
            self.api_key = Config.NEWSAPI_KEY
            self.base_url = "https://newsapi.org/v2"
        elif self.source == "eventregistry":
            self.api_key = Config.EVENTREGISTRY_KEY
            self.base_url = "https://eventregistry.org/api/v1/article/getArticles"
        else:
            raise ValueError("Unsupported source. Choose 'newsapi' or 'eventregistry'.")

        self.logger = logging.getLogger(__name__)

    def test_connection(self):
        """Test API connection"""
        if self.source == "newsapi":
            url = f"{self.base_url}/top-headlines"
            params = {'apiKey': self.api_key, 'country': 'us', 'pageSize': 1}
        else:
            url = self.base_url
            params = {'apiKey': self.api_key, 'action': 'getArticles', 'keyword': 'test', 'articlesCount': 1}

        try:
            response = requests.get(url, params=params)
            if response.status_code == 200:
                self.logger.info(f"✅ {self.source} connection successful")
                return True
            else:
                self.logger.error(f"❌ {self.source} returned status: {response.status_code}")
                return False
        except Exception as e:
            self.logger.error(f"❌ {self.source} connection failed: {str(e)}")
            return False

    def _fetch_summary(self, url: str, existing_content: str = None) -> Optional[str]:
        """Fast content extraction - use existing content if available"""
        # For Event Registry, use the 'body' field which often contains good content
        if existing_content and len(existing_content) > 100:
            return existing_content[:1000]  # Use first 1000 chars
        
        # Only try newspaper for high-value sources
        trusted_sources = ['reuters.com', 'bloomberg.com', 'wsj.com', 'ft.com', 'cnbc.com']
        if any(source in url for source in trusted_sources):
            return self._fetch_summary(url)
        
        return existing_content  # Return whatever we have


    def _clean_fallback_content(self, raw_text: str) -> str:
        """Clean fallback content"""
        if not raw_text:
            return ""

        text = BeautifulSoup(raw_text, "html.parser").get_text()
        text = re.sub(r'\{.*?\}', '', text)
        text = re.sub(r'&[a-z]+;', ' ', text)

        boilerplate_patterns = [
            r'Read more', r'Click here', r'Subscribe',
            r'Advertisement', r'Related:', r'Continue reading'
        ]
        for pattern in boilerplate_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)

        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def collect_financial_news(self, days_back: int = Config.DEFAULT_NEWS_DAYS_BACK, max_results: int = 1000) -> pd.DataFrame:
        """Collect financial news with proper pagination and fallback strategies"""
        days_back = min(days_back, Config.MAX_DAYS_BACK)
        from_date = datetime.now() - timedelta(days=days_back)
        all_articles = []

        if self.source == "newsapi":
            # NewsAPI with pagination (limited to 100 articles on free tier)
            url = f"{self.base_url}/everything"
            page = 1
            collected = 0
            
            while collected < min(max_results, 100):  # Free tier limit
                params = {
                    'apiKey': self.api_key,
                    'q': 'stock market OR financial news OR earnings',
                    'from': from_date.strftime('%Y-%m-%d'),
                    'language': 'en',
                    'sortBy': 'publishedAt',
                    'pageSize': min(100, max_results - collected),
                    'page': page
                }
                
                try:
                    response = requests.get(url, params=params)
                    if response.status_code == 426:
                        self.logger.warning("⚠️ NewsAPI free tier limited to page 1 (100 articles)")
                        break
                    
                    response.raise_for_status()
                    data = response.json()
                    articles = data.get('articles', [])
                    total_results = data.get('totalResults', 0)
                    
                    self.logger.info(f"📊 NewsAPI Page {page}: {len(articles)} articles, Total available: {total_results}")
                    
                    if not articles:
                        self.logger.info("📄 No more articles available")
                        break
                    
                    for article in articles:
                        content = self._fetch_summary(article.get('url')) or self._clean_fallback_content(article.get('content'))
                        all_articles.append({
                            'title': article.get('title'),
                            'content': content,
                            'source': article.get('source', {}).get('name'),
                            'published_at': article.get('publishedAt'),
                            'url': article.get('url'),
                            'collected_at': datetime.now()
                        })
                    
                    collected += len(articles)
                    page += 1
                    
                    # Break if we've reached all available results
                    if collected >= total_results:
                        break
                        
                except Exception as e:
                    self.logger.error(f"❌ Error collecting NewsAPI page {page}: {str(e)}")
                    break

        elif self.source == "eventregistry":
            # Event Registry with category-based search
            page = 1
            collected = 0
            
            while collected < max_results:
                params = {
                    "apiKey": self.api_key,
                    "action": "getArticles",
                    # Use business category for financial news
                    "categoryUri": "news/Business",
                    "lang": "eng",
                    "articlesPage": page,
                    "articlesCount": min(100, max_results - collected),
                    "articlesSortBy": "date",
                    "includeArticleSentiment": True,
                    # Use broader date range for better results
                    "dateStart": (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d'),
                    "dateEnd": datetime.now().strftime('%Y-%m-%d')
                }
                
                try:
                    response = requests.get(self.base_url, params=params)
                    response.raise_for_status()
                    data = response.json()
                    
                    self.logger.info(f"🔍 Event Registry response info: {data.get('info', 'No info')}")
                    
                    articles = data.get("articles", {}).get("results", [])
                    total_results = data.get("articles", {}).get("totalResults", 0)
                    
                    self.logger.info(f"📊 Event Registry Page {page}: {len(articles)} articles, Total available: {total_results}")
                    
                    if not articles:
                        self.logger.info("📄 No more Event Registry articles available")
                        break

                    for article in articles:
                        content = self._fetch_summary(article.get('url')) or self._clean_fallback_content(article.get('body'))
                        all_articles.append({
                            'title': article.get('title'),
                            'content': content,
                            'source': article.get('source', {}).get('title'),
                            'published_at': article.get('dateTimePub'),
                            'url': article.get('url'),
                            'sentiment': article.get('sentiment'),
                            'collected_at': datetime.now()
                        })

                    collected += len(articles)
                    page += 1
                    
                    # Break if we've reached all available results
                    if collected >= total_results:
                        break
                        
                except Exception as e:
                    self.logger.error(f"❌ Error collecting Event Registry page {page}: {str(e)}")
                    break

        self.logger.info(f"🎯 Total articles collected: {len(all_articles)}")
        return pd.DataFrame(all_articles)

    def collect_financial_news_combined(self, days_back: int = Config.DEFAULT_NEWS_DAYS_BACK, max_results: int = 1000) -> pd.DataFrame:
        """Collect news from both sources with fallback strategies"""
        all_articles = []
        
        # Try NewsAPI first (limited to 100)
        try:
            newsapi_collector = NewsCollector("newsapi")
            newsapi_df = newsapi_collector.collect_financial_news(days_back, 100)
            if not newsapi_df.empty:
                all_articles.extend(newsapi_df.to_dict('records'))
                self.logger.info(f"✅ Collected {len(newsapi_df)} NewsAPI articles")
        except Exception as e:
            self.logger.warning(f"⚠️ NewsAPI failed: {str(e)}")
        
        # Try Event Registry for remaining articles
        remaining = max_results - len(all_articles)
        if remaining > 0:
            try:
                er_collector = NewsCollector("eventregistry")
                er_df = er_collector.collect_financial_news(days_back, remaining)
                if not er_df.empty:
                    all_articles.extend(er_df.to_dict('records'))
                    self.logger.info(f"✅ Collected {len(er_df)} Event Registry articles")
            except Exception as e:
                self.logger.warning(f"⚠️ Event Registry failed: {str(e)}")
        
        self.logger.info(f"🎯 Total combined articles collected: {len(all_articles)}")
        return pd.DataFrame(all_articles)

    def collect_ticker_news(self, tickers: List[str], max_results: int = 100) -> pd.DataFrame:
        """Collect ticker-specific news with proper pagination"""
        all_news = []

        for ticker in tickers:
            if self.source == "newsapi":
                url = f"{self.base_url}/everything"
                page = 1
                collected = 0
                
                while collected < min(max_results, 100):  # Free tier limit
                    params = {
                        'apiKey': self.api_key,
                        'q': f'"{ticker}" OR "${ticker}"',
                        'language': 'en',
                        'sortBy': 'publishedAt',
                        'pageSize': min(100, max_results - collected),
                        'page': page
                    }
                    
                    try:
                        response = requests.get(url, params=params)
                        if response.status_code == 426:
                            self.logger.warning(f"⚠️ NewsAPI free tier limited to page 1 for {ticker}")
                            break
                        
                        response.raise_for_status()
                        articles = response.json().get('articles', [])
                        
                        if not articles:
                            break
                        
                        for article in articles:
                            content = self._fetch_summary(article.get('url')) or self._clean_fallback_content(article.get('content'))
                            all_news.append({
                                'ticker': ticker,
                                'title': article.get('title'),
                                'content': content,
                                'source': article.get('source', {}).get('name'),
                                'published_at': article.get('publishedAt'),
                                'url': article.get('url'),
                                'collected_at': datetime.now()
                            })
                        
                        collected += len(articles)
                        page += 1
                        
                    except Exception as e:
                        self.logger.error(f"❌ Error collecting news for {ticker} (NewsAPI): {str(e)}")
                        break
                
                self.logger.info(f"📰 Collected {collected} articles for {ticker} (NewsAPI)")

            elif self.source == "eventregistry":
                page = 1
                collected = 0
                
                while collected < max_results:
                    params = {
                        "apiKey": self.api_key,
                        "action": "getArticles",
                        "keyword": ticker,
                        "lang": "eng",
                        "articlesPage": page,
                        "articlesCount": min(100, max_results - collected),
                        "articlesSortBy": "date",
                        "includeArticleSentiment": True
                    }
                    
                    try:
                        response = requests.get(self.base_url, params=params)
                        response.raise_for_status()
                        articles = response.json().get("articles", {}).get("results", [])
                        
                        if not articles:
                            break

                        for article in articles:
                            content = self._fetch_summary(article.get('url')) or self._clean_fallback_content(article.get('body'))
                            all_news.append({
                                'ticker': ticker,
                                'title': article.get('title'),
                                'content': content,
                                'source': article.get('source', {}).get('title'),
                                'published_at': article.get('dateTimePub'),
                                'url': article.get('url'),
                                'sentiment': article.get('sentiment'),
                                'collected_at': datetime.now()
                            })

                        collected += len(articles)
                        page += 1

                    except Exception as e:
                        self.logger.error(f"❌ Error collecting news for {ticker} (Event Registry): {str(e)}")
                        break
                
                self.logger.info(f"📰 Collected {collected} Event Registry articles for {ticker}")

        return pd.DataFrame(all_news)
