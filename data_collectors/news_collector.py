import requests
import pandas as pd
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import logging
import time
import random
import json
import hashlib
import shutil
import sys
from pathlib import Path

# Content extraction libraries
try:
    from newspaper import Article, Config as NewspaperConfig
    NEWSPAPER4K_AVAILABLE = True
except ImportError:
    NEWSPAPER4K_AVAILABLE = False
    logging.warning("newspaper4k not available, falling back to trafilatura only")

try:
    import trafilatura
    TRAFILATURA_AVAILABLE = True
except ImportError:
    TRAFILATURA_AVAILABLE = False
    logging.warning("trafilatura not available")

import nltk
from config.config import Config

# Ensure required NLTK corpora are downloaded
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)


class TokenBucket:
    """Token bucket rate limiter for API requests"""
    def __init__(self, capacity: int = 10, refill_rate: float = 1.0):
        self.capacity = capacity
        self.tokens = capacity
        self.refill_rate = refill_rate
        self.last_refill = time.time()
    
    def consume(self, tokens: int = 1) -> bool:
        """Attempt to consume tokens. Returns True if successful."""
        self._refill()
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False
    
    def _refill(self):
        """Refill tokens based on time elapsed"""
        now = time.time()
        elapsed = now - self.last_refill
        tokens_to_add = elapsed * self.refill_rate
        self.tokens = min(self.capacity, self.tokens + tokens_to_add)
        self.last_refill = now
    
    def wait_time(self) -> float:
        """Calculate time to wait for next token"""
        if self.tokens >= 1:
            return 0
        return (1 - self.tokens) / self.refill_rate


class NewsCollector:
    def __init__(self, source: str = "stocknewsapi"):
        self.source = source.lower()
        
        if self.source == "newsapi":
            self.api_key = Config.NEWSAPI_KEY
            self.base_url = "https://newsapi.org/v2"
        elif self.source == "stocknewsapi":
            self.api_key = Config.STOCKNEWS_API_KEY
            self.base_url = "https://stocknewsapi.com/api/v1"
        else:
            raise ValueError("Unsupported source. Choose 'newsapi' or 'stocknewsapi'.")
        
        self.logger = logging.getLogger(__name__)
        self.broadmarketkeywords = Config.BROAD_MARKET_KEYWORDS
        self.ratelimiter = TokenBucket(capacity=10, refill_rate=1.0)
        self.cache_dir = Path("cache")
        self.cache_dir.mkdir(exist_ok=True)

    def _make_api_request(self, url: str, params: Dict[str, Any], max_retries: int = 3) -> Optional[Dict]:
        """Make API request with rate limiting and retries"""
        for attempt in range(max_retries):
            # Rate limiting
            if not self.ratelimiter.consume():
                wait_time = self.ratelimiter.wait_time()
                self.logger.info(f"Rate limit reached, waiting {wait_time:.2f}s")
                time.sleep(wait_time)
            
            try:
                response = requests.get(url, params=params, timeout=30)
                response.raise_for_status()
                return response.json()
            
            except requests.exceptions.HTTPError as e:
                if response.status_code == 429:  # Rate limit
                    wait_time = min(2 ** attempt, 60)
                    self.logger.warning(f"Rate limited (429), waiting {wait_time}s")
                    time.sleep(wait_time)
                    continue
                elif response.status_code == 401:
                    self.logger.error(f"Authentication failed (401): Check API key")
                    return None
                else:
                    self.logger.error(f"HTTP {response.status_code}: {e}")
                    return None
            
            except requests.exceptions.RequestException as e:
                self.logger.error(f"Request failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                return None
        
        return None

    def _get_content_with_fallback(self, article: Dict, url_key: str, content_key: str) -> str:
        """Extract full article content with multiple fallback methods"""
        url = article.get(url_key)
        content = article.get(content_key, '')
        
        # If content is already substantial, use it
        if content and len(content.strip()) > 200:
            return content
        
        # Try to fetch full content from URL
        if url:
            cache_key = hashlib.md5(url.encode()).hexdigest()
            cache_file = self.cache_dir / f"{cache_key}.txt"
            
            # Check cache first
            if cache_file.exists():
                try:
                    return cache_file.read_text(encoding='utf-8')
                except Exception:
                    pass
            
            # Try newspaper4k first
            if NEWSPAPER4K_AVAILABLE:
                try:
                    config = NewspaperConfig()
                    config.browser_user_agent = 'Mozilla/5.0'
                    config.request_timeout = 10
                    
                    news_article = Article(url, config=config)
                    news_article.download()
                    news_article.parse()
                    
                    if news_article.text and len(news_article.text) > 100:
                        # Cache the content
                        try:
                            cache_file.write_text(news_article.text, encoding='utf-8')
                        except Exception:
                            pass
                        return news_article.text
                except Exception as e:
                    self.logger.debug(f"newspaper4k failed for {url}: {e}")
            
            # Fallback to trafilatura
            if TRAFILATURA_AVAILABLE:
                try:
                    downloaded = trafilatura.fetch_url(url)
                    if downloaded:
                        extracted = trafilatura.extract(downloaded)
                        if extracted and len(extracted) > 100:
                            try:
                                cache_file.write_text(extracted, encoding='utf-8')
                            except Exception:
                                pass
                            return extracted
                except Exception as e:
                    self.logger.debug(f"trafilatura failed for {url}: {e}")
        
        # Return whatever content we have
        return content if content else ""

    def _detect_etf_category(self, title: str, content: str) -> str:
        """Detect which ETF this article is about based on title and content"""
        text = f"{title} {content}".upper()
        
        # Check each ETF ticker in the text
        for ticker, sector in Config.TICKER_SECTORS.items():
            if ticker == 'MACRO':
                continue
            if ticker in text:
                return sector
        
        return 'Market-Wide'

    def collect_etf_news(self, etf_tickers: List[str], days_back: int = Config.DEFAULT_NEWS_DAYS_BACK, 
                         max_results_per_etf: int = 50) -> pd.DataFrame:
        """Collect news for specific ETF tickers individually"""
        all_articles = []
        
        for ticker in etf_tickers:
            self.logger.info(f"🔍 Collecting ETF news for {ticker}")
            
            if self.source == "stocknewsapi":
                # Use base URL with tickers parameter
                params = {
                    "tickers": ticker,
                    "items": min(max_results_per_etf, 50),
                    "page": 1,
                    "token": self.api_key
                }
                
                data = self._make_api_request(self.base_url, params)
                
                if not data or "data" not in data:
                    self.logger.warning(f"❌ StockNewsAPI: No data for {ticker}")
                    continue
                
                articles = data.get("data", [])
                self.logger.info(f"📰 StockNewsAPI: Found {len(articles)} articles for {ticker}")
                
                for article in articles:
                    article_data = {
                        'ticker': ticker,
                        'title': article.get('title'),
                        'content': article.get('text', ''),
                        'source': article.get('source_name'),
                        'published_at': article.get('date'),
                        'url': article.get('news_url'),
                        'category': Config.get_sector(ticker),
                        'sentiment': article.get('sentiment'),
                        'tickers': article.get('tickers', []),
                        'collected_at': datetime.now()
                    }
                    all_articles.append(article_data)
            
            elif self.source == "newsapi":
                # Fallback to NewsAPI
                url = f"{self.base_url}/everything"
                params = {
                    'apiKey': self.api_key,
                    'q': ticker,
                    'from': (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d'),
                    'language': 'en',
                    'sortBy': 'publishedAt',
                    'pageSize': min(100, max_results_per_etf)
                }
                
                data = self._make_api_request(url, params)
                if not data:
                    continue
                
                articles = data.get('articles', [])
                self.logger.info(f"📰 NewsAPI: Found {len(articles)} articles for {ticker}")
                
                for article in articles:
                    content = self._get_content_with_fallback(article, 'url', 'content')
                    if len(content.strip()) < 50:
                        continue
                    
                    article_data = {
                        'ticker': ticker,
                        'title': article.get('title'),
                        'content': content,
                        'source': article.get('source', {}).get('name'),
                        'published_at': article.get('publishedAt'),
                        'url': article.get('url'),
                        'category': Config.get_sector(ticker),
                        'collected_at': datetime.now()
                    }
                    all_articles.append(article_data)
        
        df = pd.DataFrame(all_articles)
        self.logger.info(f"✅ Total ETF news collected: {len(df)} articles across {len(etf_tickers)} ETFs")
        return df

    def collect_financial_news(self, days_back: int = Config.DEFAULT_NEWS_DAYS_BACK, max_results: int = 500) -> pd.DataFrame:
        """Collect general broad market news"""
        if self.source == "stocknewsapi":
            all_articles = []
            
            # Use /trending-headlines endpoint for general market news
            url = f"{self.base_url}/trending-headlines"
            params = {
                "items": min(max_results, 50),
                "page": 1,
                "token": self.api_key
            }
            
            self.logger.info(f"📡 StockNewsAPI: Requesting trending market headlines")
            
            data = self._make_api_request(url, params)
            
            if not data or "data" not in data:
                self.logger.warning(f"❌ StockNewsAPI: No trending headlines data")
                return pd.DataFrame()
            
            articles = data.get("data", [])
            self.logger.info(f"📰 StockNewsAPI: Found {len(articles)} trending headlines")
            
            for article in articles:
                article_data = {
                    'title': article.get('title'),
                    'content': article.get('text', ''),
                    'source': article.get('source_name'),
                    'published_at': article.get('date'),
                    'url': article.get('news_url'),
                    'category': 'Market-Wide',
                    'sentiment': article.get('sentiment'),
                    'tickers': article.get('tickers', []),
                    'collected_at': datetime.now()
                }
                all_articles.append(article_data)
            
            df = pd.DataFrame(all_articles)
            return df
        elif self.source == "newsapi":
            url = f"{self.base_url}/everything"
            page = 1
            collected = 0
            
            query = " OR ".join(self.broadmarketkeywords)
            from_date = datetime.now() - timedelta(days=days_back)
            
            all_articles = []
            while collected < max_results:
                params = {
                    'apiKey': self.api_key,
                    'q': query,
                    'from': from_date.strftime('%Y-%m-%d'),
                    'language': 'en',
                    'sortBy': 'publishedAt',
                    'pageSize': min(100, max_results - collected),
                    'page': page
                }
                
                data = self._make_api_request(url, params)
                if not data or data.get("status") == "error":
                    self.logger.warning(f"NewsAPI error or no data on page {page}")
                    break
                
                articles = data.get('articles', [])
                if not articles:
                    self.logger.info(f"NewsAPI no more articles available at page {page}")
                    break
                
                for article in articles:
                    content = self._get_content_with_fallback(article, 'url', 'content')
                    if len(content.strip()) < 50:
                        continue
                    
                    article_data = {
                        'title': article.get('title'),
                        'content': content,
                        'source': article.get('source', {}).get('name'),
                        'published_at': article.get('publishedAt'),
                        'url': article.get('url'),
                        'category': self._detect_etf_category(article.get('title', ''), content),
                        'collected_at': datetime.now()
                    }
                    all_articles.append(article_data)
                
                collected += len(articles)
                page += 1
                
            df = pd.DataFrame(all_articles)
            return df
        
        else:
            self.logger.error(f"Unsupported source: {self.source}")
            return pd.DataFrame()

    def collect_ticker_news(self, tickers: List[str], days_back: int = Config.DEFAULT_NEWS_DAYS_BACK, 
                            max_results: int = 50) -> pd.DataFrame:
        """Collect news for specific stock tickers"""
        all_articles = []
        
        for ticker in tickers:
            self.logger.info(f"🔍 Collecting news for {ticker}")
            
            if self.source == "stocknewsapi":
                # Use base URL with tickers parameter
                params = {
                    "tickers": ticker,
                    "items": min(max_results, 50),
                    "page": 1,
                    "token": self.api_key
                }
                
                data = self._make_api_request(self.base_url, params)
                
                if not data or "data" not in data:
                    self.logger.warning(f"❌ StockNewsAPI: No data for {ticker}")
                    continue
                
                articles = data.get("data", [])
                self.logger.info(f"✅ Collected {len(articles)} articles for {ticker}")
                
                for article in articles:
                    article_data = {
                        'ticker': ticker,
                        'title': article.get('title'),
                        'content': article.get('text', ''),
                        'source': article.get('source_name'),
                        'published_at': article.get('date'),
                        'url': article.get('news_url'),
                        'category': Config.get_sector(ticker),
                        'sentiment': article.get('sentiment'),
                        'tickers': article.get('tickers', []),
                        'collected_at': datetime.now()
                    }
                    all_articles.append(article_data)
            
            elif self.source == "newsapi":
                url = f"{self.base_url}/everything"
                params = {
                    'apiKey': self.api_key,
                    'q': ticker,
                    'from': (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d'),
                    'language': 'en',
                    'sortBy': 'publishedAt',
                    'pageSize': min(100, max_results)
                }
                
                data = self._make_api_request(url, params)
                if not data:
                    continue
                
                articles = data.get('articles', [])
                self.logger.info(f"✅ Collected {len(articles)} articles for {ticker}")
                
                for article in articles:
                    content = self._get_content_with_fallback(article, 'url', 'content')
                    if len(content.strip()) < 50:
                        continue
                    
                    article_data = {
                        'ticker': ticker,
                        'title': article.get('title'),
                        'content': content,
                        'source': article.get('source', {}).get('name'),
                        'published_at': article.get('publishedAt'),
                        'url': article.get('url'),
                        'category': Config.get_sector(ticker),
                        'collected_at': datetime.now()
                    }
                    all_articles.append(article_data)
        
        return pd.DataFrame(all_articles)

    def test_connection(self):
        """Test API connection"""
        if self.source == "stocknewsapi":
            params = {
                "token": self.api_key,
                "items": 1,
                "page": 1,
                "tickers": "AAPL"
            }
            data = self._make_api_request(self.base_url, params)
            return data is not None and "data" in data
        
        elif self.source == "newsapi":
            url = f"{self.base_url}/top-headlines"
            params = {'apiKey': self.api_key, 'country': 'us', 'pageSize': 1}
            data = self._make_api_request(url, params)
            return data is not None and data.get('status') == 'ok'
        
        return False

    @staticmethod
    def clear_cache_dir(cache_dir: str):
        """Clear the cache directory"""
        cache_path = Path(cache_dir)
        if cache_path.exists():
            shutil.rmtree(cache_path)
            cache_path.mkdir(exist_ok=True)
