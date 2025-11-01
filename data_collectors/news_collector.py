import requests
import pandas as pd
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import logging
import time
from pathlib import Path

from config.config import Config


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
        
        if self.source == "stocknewsapi":
            self.api_key = Config.STOCKNEWS_API_KEY
            self.base_url = "https://stocknewsapi.com/api/v1"
        else:
            raise ValueError("Only 'stocknewsapi' is supported.")
        
        self.logger = logging.getLogger(__name__)
        self.ratelimiter = TokenBucket(capacity=10, refill_rate=1.0)


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


    def collect_etf_news_for_correlation(self, etf_tickers: List[str],
                                     min_articles: int = 50,
                                     min_days_back: int = 365) -> pd.DataFrame:
        """
        Collect ETF news for correlation analysis: MINIMUM 50 articles AND MINIMUM 1 year of data.
        Continues collecting beyond 1 year if needed to reach 50 articles.
        
        Args:
            etf_tickers: List of ETF ticker symbols
            min_articles: Minimum articles per ticker (default 50)
            min_days_back: Minimum days to look back (default 365 = 1 year)
        
        Returns:
            DataFrame with ETF news articles
        """
        all_articles = []
        
        for ticker in etf_tickers:
            self.logger.info(f"🔍 Collecting ETF news for {ticker} (min: {min_articles} articles AND {min_days_back} days)")
            
            page = 1
            ticker_articles = []
            reached_one_year = False
            
            # Calculate cutoff date (1 year back) - TIMEZONE-AWARE
            from datetime import timezone
            one_year_cutoff = datetime.now(timezone.utc) - timedelta(days=min_days_back)
            
            while True:
                params = {
                    "tickers": ticker,
                    "items": 50,
                    "page": page,
                    "token": self.api_key
                }
                
                data = self._make_api_request(self.base_url, params)
                
                if not data or "data" not in data:
                    self.logger.warning(f"❌ No more data for {ticker}")
                    break
                
                articles = data.get("data", [])
                
                if not articles:
                    self.logger.info(f"📭 {ticker}: No more articles at page {page}")
                    break
                
                # Process articles and check conditions
                for article in articles:
                    article_date = pd.to_datetime(article.get('date'), format='mixed', utc=True)
                    
                    # Mark when we've passed 1 year
                    if not reached_one_year and article_date < one_year_cutoff:
                        reached_one_year = True
                        self.logger.info(f"📅 {ticker}: Passed 1-year mark at {article_date.date()} ({len(ticker_articles)} articles so far)")
                    
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
                        'collected_at': datetime.now(timezone.utc)
                    }
                    
                    ticker_articles.append(article_data)
                
                # Check if we've met BOTH conditions
                if reached_one_year and len(ticker_articles) >= min_articles:
                    self.logger.info(f"✅ {ticker}: Met both conditions - {len(ticker_articles)} articles AND passed 1 year")
                    break
                
                page += 1
                
                # Safety limit on pages (increased for longer collection)
                if page > 50:
                    self.logger.warning(f"⚠️ {ticker}: Reached page limit (50), stopping")
                    self.logger.warning(f"  • Articles collected: {len(ticker_articles)}/{min_articles}")
                    self.logger.warning(f"  • Passed 1 year: {'Yes' if reached_one_year else 'No'}")
                    break
            
            all_articles.extend(ticker_articles)
            
            # Log what we collected
            if ticker_articles:
                df_temp = pd.DataFrame(ticker_articles)
                df_temp['published_at'] = pd.to_datetime(df_temp['published_at'], format='mixed', utc=True)
                date_range_days = (df_temp['published_at'].max() - df_temp['published_at'].min()).days
                
                self.logger.info(
                    f"✅ {ticker}: Collected {len(ticker_articles)} articles "
                    f"spanning {date_range_days} days "
                    f"({df_temp['published_at'].min().date()} to {df_temp['published_at'].max().date()})"
                )
            else:
                self.logger.warning(f"⚠️ {ticker}: No articles collected")
        
        df = pd.DataFrame(all_articles)
        self.logger.info(f"✅ Total ETF news: {len(df)} articles across {len(etf_tickers)} ETFs")
        
        return df




    def collect_ticker_news_for_correlation(self, tickers: List[str], 
                                       min_articles: int = 50,
                                       min_days_back: int = 365) -> pd.DataFrame:
        """
        Collect news for correlation analysis: MINIMUM 50 articles AND MINIMUM 1 year of data.
        Continues collecting beyond 1 year if needed to reach 50 articles.
        
        Args:
            tickers: List of ticker symbols
            min_articles: Minimum articles per ticker (default 50)
            min_days_back: Minimum days to look back (default 365 = 1 year)
        
        Returns:
            DataFrame with news articles and date ranges per ticker
        """
        all_articles = []
        
        for ticker in tickers:
            self.logger.info(f"🔍 Collecting news for {ticker} (min: {min_articles} articles AND {min_days_back} days)")
            
            page = 1
            ticker_articles = []
            reached_one_year = False
            
            # Calculate cutoff date (1 year back) - TIMEZONE-AWARE
            from datetime import timezone
            one_year_cutoff = datetime.now(timezone.utc) - timedelta(days=min_days_back)
            
            while True:
                params = {
                    "tickers": ticker,
                    "items": 50,
                    "page": page,
                    "token": self.api_key
                }
                
                data = self._make_api_request(self.base_url, params)
                
                if not data or "data" not in data:
                    self.logger.warning(f"❌ No more data for {ticker}")
                    break
                
                articles = data.get("data", [])
                
                if not articles:
                    self.logger.info(f"📭 {ticker}: No more articles at page {page}")
                    break
                
                # Process articles and check conditions
                for article in articles:
                    article_date = pd.to_datetime(article.get('date'), format='mixed', utc=True)
                    
                    # Mark when we've passed 1 year
                    if not reached_one_year and article_date < one_year_cutoff:
                        reached_one_year = True
                        self.logger.info(f"📅 {ticker}: Passed 1-year mark at {article_date.date()} ({len(ticker_articles)} articles so far)")
                    
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
                        'collected_at': datetime.now(timezone.utc)
                    }
                    
                    ticker_articles.append(article_data)
                
                # Check if we've met BOTH conditions
                if reached_one_year and len(ticker_articles) >= min_articles:
                    self.logger.info(f"✅ {ticker}: Met both conditions - {len(ticker_articles)} articles AND passed 1 year")
                    break
                
                page += 1
                
                # Safety limit on pages (increased for longer collection)
                if page > 50:
                    self.logger.warning(f"⚠️ {ticker}: Reached page limit (50), stopping")
                    self.logger.warning(f"  • Articles collected: {len(ticker_articles)}/{min_articles}")
                    self.logger.warning(f"  • Passed 1 year: {'Yes' if reached_one_year else 'No'}")
                    break
            
            all_articles.extend(ticker_articles)
            
            # Log what we collected
            if ticker_articles:
                df_temp = pd.DataFrame(ticker_articles)
                df_temp['published_at'] = pd.to_datetime(df_temp['published_at'], format='mixed', utc=True)
                date_range_days = (df_temp['published_at'].max() - df_temp['published_at'].min()).days
                
                self.logger.info(
                    f"✅ {ticker}: Collected {len(ticker_articles)} articles "
                    f"spanning {date_range_days} days "
                    f"({df_temp['published_at'].min().date()} to {df_temp['published_at'].max().date()})"
                )
            else:
                self.logger.warning(f"⚠️ {ticker}: No articles collected")
        
        df = pd.DataFrame(all_articles)
        self.logger.info(f"✅ Total news: {len(df)} articles across {len(tickers)} tickers")
        
        return df



    def collect_financial_news(self, days_back: int = Config.DEFAULT_NEWS_DAYS_BACK, max_results: int = 500) -> pd.DataFrame:
        """Collect general broad market news"""
        all_articles = []
        
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


    def collect_trending_stocks(self, date_range: str = "last7days", 
                                 min_mentions: int = 0,
                                 max_results: int = 20) -> pd.DataFrame:
        """
        Collect trending stocks based on mention frequency and sentiment.
        Returns raw API data without additional sentiment classification.
        
        Args:
            date_range: Time period for data ('last7days', 'last30days')
            min_mentions: Minimum number of mentions to include
            max_results: Maximum number of stocks to return
            
        Returns:
            DataFrame with trending stock data from the API
        """
        self.logger.info(f"📈 Collecting trending stocks for {date_range}...")
        
        url = f"{self.base_url}/top-mention"
        params = {
            "date": date_range,
            "token": self.api_key
        }
        
        data = self._make_api_request(url, params)
        
        if not data or "data" not in data:
            self.logger.warning("❌ No trending stocks data returned")
            return pd.DataFrame()
        
        stocks = data.get("data", {}).get("all", [])
        
        if not stocks:
            self.logger.warning("📭 No stocks found in trending data")
            return pd.DataFrame()
        
        self.logger.info(f"✅ Retrieved {len(stocks)} trending stocks")
        
        # Process the data - keep API structure as-is
        processed_stocks = []
        for stock in stocks:
            # Filter by minimum mentions
            if stock.get('total_mentions', 0) < min_mentions:
                continue
            
            stock_data = {
                'ticker': stock.get('ticker'),
                'name': stock.get('name'),
                'total_mentions': stock.get('total_mentions', 0),
                'positive_mentions': stock.get('positive_mentions', 0),
                'negative_mentions': stock.get('negative_mentions', 0),
                'neutral_mentions': stock.get('neutral_mentions', 0),
                'sentiment_score': stock.get('sentiment_score', 0),
                'category': Config.get_sector(stock.get('ticker', '')),
                'collected_at': datetime.now(),
                'date_range': date_range
            }
            processed_stocks.append(stock_data)
        
        # Sort by total mentions and limit results
        processed_stocks = sorted(processed_stocks, 
                                  key=lambda x: x['total_mentions'], 
                                  reverse=True)[:max_results]
        
        df = pd.DataFrame(processed_stocks)
        
        self.logger.info(f"📊 Processed {len(df)} trending stocks")
        if not df.empty:
            self.logger.info(f"   Top stock: {df.iloc[0]['ticker']} with {df.iloc[0]['total_mentions']} mentions")
            self.logger.info(f"   Sentiment range: {df['sentiment_score'].min():.3f} to {df['sentiment_score'].max():.3f}")
        
        return df


    def collect_all_news(self, items_per_page: int = 100, 
                         max_pages: int = 20,
                         section: str = "alltickers") -> pd.DataFrame:
        """
        Collect all news articles from the category endpoint with pagination.
        Fetches news across multiple pages to gather comprehensive market coverage.
        
        Args:
            items_per_page: Number of items per page (max 100)
            max_pages: Maximum number of pages to fetch (default 20)
            section: Category section to query (default 'alltickers')
            
        Returns:
            DataFrame with all collected news articles
        """
        self.logger.info(f"📰 Collecting news from category endpoint ({max_pages} pages)...")
        
        all_articles = []
        
        for page in range(1, max_pages + 1):
            self.logger.info(f"   Fetching page {page}/{max_pages}...")
            
            url = f"{self.base_url}/category"
            params = {
                "section": section,
                "items": items_per_page,
                "page": page,
                "token": self.api_key
            }
            
            data = self._make_api_request(url, params)
            
            if not data or "data" not in data:
                self.logger.warning(f"❌ No data returned for page {page}")
                continue
            
            articles = data.get("data", [])
            
            if not articles:
                self.logger.info(f"📭 No more articles found at page {page}, stopping pagination")
                break
            
            self.logger.info(f"   ✅ Retrieved {len(articles)} articles from page {page}")
            all_articles.extend(articles)
            
            # Small delay to respect rate limits
            time.sleep(0.5)
        
        if not all_articles:
            self.logger.warning("📭 No articles collected")
            return pd.DataFrame()
        
        self.logger.info(f"✅ Total articles collected: {len(all_articles)}")
        
        # Process articles into DataFrame
        processed_articles = []
        for article in all_articles:
            article_data = {
                'news_url': article.get('news_url'),
                'image_url': article.get('image_url'),
                'title': article.get('title'),
                'text': article.get('text'),
                'source_name': article.get('source_name'),
                'date': article.get('date'),
                'sentiment': article.get('sentiment'),
                'type': article.get('type'),
                'tickers': ','.join(article.get('tickers', [])) if article.get('tickers') else '',
                'topics': ','.join(article.get('topics', [])) if article.get('topics') else '',
                'collected_at': datetime.now()
            }
            processed_articles.append(article_data)
        
        df = pd.DataFrame(processed_articles)
        
        # Convert date string to datetime
        if not df.empty and 'date' in df.columns:
            try:
                df['date'] = pd.to_datetime(df['date'])
            except Exception as e:
                self.logger.warning(f"Could not parse dates: {e}")
        
        self.logger.info(f"📊 Processed {len(df)} articles")
        if not df.empty:
            self.logger.info(f"   Date range: {df['date'].min()} to {df['date'].max()}")
            self.logger.info(f"   Unique tickers: {df['tickers'].str.split(',').explode().nunique()}")
            sentiment_dist = df['sentiment'].value_counts()
            self.logger.info(f"   Sentiment: {sentiment_dist.to_dict()}")
        
        return df


    def test_connection(self):
        """Test API connection"""
        params = {
            "token": self.api_key,
            "items": 1,
            "page": 1,
            "tickers": "AAPL"
        }
        data = self._make_api_request(self.base_url, params)
        return data is not None and "data" in data
