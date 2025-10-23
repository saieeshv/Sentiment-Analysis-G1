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
        """Attempt to consume tokens, return True if successful"""
        self._refill()
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False
    
    def _refill(self):
        """Refill tokens based on time elapsed"""
        now = time.time()
        elapsed = now - self.last_refill
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
        self.last_refill = now

class ContentCache:
    """Simple file-based cache for article content"""
    def __init__(self, cache_dir: str = "cache", max_age_hours: int = 24):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.max_age = timedelta(hours=max_age_hours)
    
    def _get_cache_key(self, url: str) -> str:
        """Generate cache key from URL"""
        return hashlib.md5(url.encode()).hexdigest()
    
    def get(self, url: str) -> Optional[Dict]:
        """Get cached content if available and fresh"""
        cache_file = self.cache_dir / f"{self._get_cache_key(url)}.json"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cached_data = json.load(f)
                
                cached_time = datetime.fromisoformat(cached_data['cached_at'])
                if datetime.now() - cached_time < self.max_age:
                    return cached_data['content']
            except Exception:
                pass
        return None
    
    def set(self, url: str, content: Dict):
        """Cache content with timestamp"""
        cache_file = self.cache_dir / f"{self._get_cache_key(url)}.json"
        
        cache_data = {
            'cached_at': datetime.now().isoformat(),
            'content': content
        }
        
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logging.warning(f"Failed to cache content: {e}")

class HybridContentExtractor:
    """Hybrid content extractor using newspaper4k + trafilatura fallback"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.cache = ContentCache()
        
        # Configure newspaper4k if available
        if NEWSPAPER4K_AVAILABLE:
            self.newspaper_config = NewspaperConfig()
            self.newspaper_config.browser_user_agent = (
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                '(KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36'
            )
            self.newspaper_config.request_timeout = 8
            self.newspaper_config.fetch_images = False
            self.newspaper_config.memoize_articles = False
            self.newspaper_config.verbose = False
        
        # Problematic domains to skip
        self.problematic_domains = [
            'thecambodianews.net', 'bna.bh', 'forbes.com', 'mybroadband.co.za',
            'webhostingtalk.com', 'morningstar.com', 'citizen.co.za',
            'globenewswire.com', 'biztoc.com', 'slickdeals.net', 'reddit.com'
        ]
    
    def extract_content(self, url: str, html: Optional[str] = None) -> Dict[str, Any]:
        """
        Extract content using hybrid approach:
        1. Check cache first
        2. Try newspaper4k for rich extraction
        3. Fallback to trafilatura for robust extraction
        4. Last resort: basic HTML parsing
        """
        # Check cache first
        cached_content = self.cache.get(url)
        if cached_content:
            cached_content['from_cache'] = True
            return cached_content
        
        # Skip known problematic domains
        if any(domain in url for domain in self.problematic_domains):
            return self._create_empty_result('skipped_domain')
        
        result = {
            'title': None,
            'content': None,
            'authors': [],
            'publish_date': None,
            'top_image': None,
            'keywords': [],
            'summary': None,
            'extraction_method': None,
            'quality_score': 0.0,
            'from_cache': False
        }
        
        # Add random delay to avoid rate limiting
        time.sleep(random.uniform(0.3, 0.8))
        
        # Method 1: Try newspaper4k first
        if NEWSPAPER4K_AVAILABLE:
            try:
                if html:
                    article = Article('', config=self.newspaper_config)
                    article.set_html(html)
                else:
                    article = Article(url, config=self.newspaper_config)
                    article.download()
                
                article.parse()
                
                # Check if extraction was successful
                if article.text and len(article.text.strip()) > 150:
                    try:
                        article.nlp()  # Extract keywords and summary
                    except Exception:
                        pass  # NLP is optional
                    
                    result.update({
                        'title': article.title,
                        'content': article.text,
                        'authors': article.authors,
                        'publish_date': article.publish_date.isoformat() if article.publish_date else None,
                        'top_image': article.top_image,
                        'keywords': article.keywords,
                        'summary': article.summary,
                        'extraction_method': 'newspaper4k'
                    })
                    
                    result['quality_score'] = self._calculate_quality_score(result)
                    
                    # Cache successful extraction
                    self.cache.set(url, result)
                    
                    self.logger.debug(f"✅ newspaper4k successful for {url}")
                    return result
                    
            except Exception as e:
                self.logger.debug(f"⚠️ newspaper4k failed for {url}: {str(e)}")
        
        # Method 2: Fallback to trafilatura
        if TRAFILATURA_AVAILABLE:
            try:
                if not html:
                    # Fetch HTML with proper headers
                    headers = {
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                        'Accept-Language': 'en-US,en;q=0.9',
                        'Connection': 'keep-alive',
                        'DNT': '1'
                    }
                    response = requests.get(url, timeout=8, headers=headers)
                    response.raise_for_status()
                    html = response.text
                
                # Extract with trafilatura
                extracted_text = trafilatura.extract(
                    html,
                    include_comments=False,
                    include_tables=True,
                    favor_precision=True,
                    include_formatting=False
                )
                
                if extracted_text and len(extracted_text.strip()) > 100:
                    # Extract metadata separately
                    metadata = trafilatura.extract_metadata(html)
                    
                    result.update({
                        'title': metadata.title if metadata else None,
                        'content': extracted_text,
                        'authors': [metadata.author] if metadata and metadata.author else [],
                        'publish_date': metadata.date if metadata else None,
                        'extraction_method': 'trafilatura'
                    })
                    
                    result['quality_score'] = self._calculate_quality_score(result)
                    
                    # Cache successful extraction
                    self.cache.set(url, result)
                    
                    self.logger.debug(f"✅ trafilatura successful for {url}")
                    return result
                    
            except Exception as e:
                self.logger.debug(f"⚠️ trafilatura failed for {url}: {str(e)}")
        
        # Method 3: Last resort - basic text extraction
        if html and TRAFILATURA_AVAILABLE:
            try:
                basic_content = trafilatura.html2txt(html)
                if basic_content and len(basic_content.strip()) > 50:
                    result.update({
                        'content': basic_content,
                        'extraction_method': 'basic_html2txt'
                    })
                    result['quality_score'] = self._calculate_quality_score(result)
                    
                    self.logger.debug(f"⚠️ Basic extraction used for {url}")
                    return result
            except Exception:
                pass
        
        # All methods failed
        result['extraction_method'] = 'failed'
        self.logger.debug(f"❌ All extraction methods failed for {url}")
        return result
    
    def _calculate_quality_score(self, result: Dict) -> float:
        """Calculate quality score for extracted content"""
        content_length = len(result['content']) if result['content'] else 0
        has_title = bool(result['title'])
        has_date = bool(result['publish_date'])
        has_author = bool(result['authors'])
        
        return (
            min(content_length / 1000, 1.0) * 0.4 +  # Content length (max 1.0)
            has_title * 0.3 +                        # Has title
            has_date * 0.2 +                         # Has publish date  
            has_author * 0.1                         # Has author
        )
    
    def _create_empty_result(self, method: str) -> Dict[str, Any]:
        """Create empty result with specified extraction method"""
        return {
            'title': None, 'content': None, 'authors': [], 'publish_date': None,
            'top_image': None, 'keywords': [], 'summary': None,
            'extraction_method': method, 'quality_score': 0.0, 'from_cache': False
        }

class NewsCollector:
    def __init__(self, source: str = "newsapi"):
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

        # Broad market 
        self.broad_market_tickers = Config.BROAD_MARKET_ETFS
        self.broad_market_keywords = Config.BROAD_MARKET_KEYWORDS
        
        # Enhanced components
        self.content_extractor = HybridContentExtractor()
        self.rate_limiter = TokenBucket(capacity=10, refill_rate=2.0)  # 2 requests per second average
        
        # Session for connection reuse
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json,text/html,application/xhtml+xml,*/*;q=0.8',
            'Connection': 'keep-alive',
            'DNT': '1'
        })
        
        # Exponential backoff settings
        self.max_retries = 3
        self.base_delay = 1.0

    def test_connection(self):
        """Test API connection with enhanced error handling"""
        if self.source == "newsapi":
            url = f"{self.base_url}/top-headlines"
            params = {'apiKey': self.api_key, 'country': 'us', 'pageSize': 1}
        else:
            url = self.base_url
            params = {'apiKey': self.api_key, 'action': 'getArticles', 'keyword': 'test', 'articlesCount': 1}

        for attempt in range(self.max_retries):
            try:
                if not self.rate_limiter.consume():
                    time.sleep(0.5)
                
                response = self.session.get(url, params=params, timeout=10)
                if response.status_code == 200:
                    self.logger.info(f"✅ {self.source} connection successful")
                    return True
                else:
                    self.logger.warning(f"⚠️ {self.source} returned status: {response.status_code}")
                    if attempt < self.max_retries - 1:
                        delay = self.base_delay * (2 ** attempt) + random.uniform(0, 1)
                        time.sleep(delay)
                    
            except Exception as e:
                self.logger.warning(f"⚠️ {self.source} connection attempt {attempt + 1} failed: {str(e)}")
                if attempt < self.max_retries - 1:
                    delay = self.base_delay * (2 ** attempt) + random.uniform(0, 1)
                    time.sleep(delay)
        
        self.logger.error(f"❌ {self.source} connection failed after {self.max_retries} attempts")
        return False

    def _fetch_summary(self, url: str) -> Optional[str]:
        """Enhanced content extraction using hybrid approach"""
        try:
            result = self.content_extractor.extract_content(url)
            return result.get('content')
        except Exception as e:
            self.logger.debug(f"Content extraction failed for {url}: {e}")
            return None

    def _get_content_with_fallback(self, article_data: dict, url_key: str = 'url', content_key: str = 'content') -> str:
        """Enhanced content extraction with quality filtering"""
        url = article_data.get(url_key)
        if url:
            result = self.content_extractor.extract_content(url)
            
            # Use extracted content if quality is good enough
            if result.get('quality_score', 0) > 0.3 and result.get('content'):
                # Store additional metadata in the article_data if extraction was successful
                if result.get('extraction_method') in ['newspaper4k', 'trafilatura']:
                    article_data['_extracted_title'] = result.get('title')
                    article_data['_extracted_authors'] = result.get('authors')
                    article_data['_extracted_keywords'] = result.get('keywords')
                    article_data['_quality_score'] = result.get('quality_score')
                
                return result['content']
        
        # Fallback to existing content fields
        existing_content = (
            article_data.get(content_key) or 
            article_data.get('body') or 
            article_data.get('description')
        )
        
        if existing_content:
            return existing_content[:800]  # Limit length
        
        # Final fallback
        return article_data.get('title', 'Content unavailable')[:200]

    def _make_api_request(self, url: str, params: dict) -> Optional[dict]:
        """Make API request with rate limiting and exponential backoff"""
        for attempt in range(self.max_retries):
            try:
                # Rate limiting
                while not self.rate_limiter.consume():
                    time.sleep(0.1)
                
                response = self.session.get(url, params=params, timeout=15)
                
                # Handle rate limiting
                if response.status_code == 429:
                    retry_after = int(response.headers.get('Retry-After', 60))
                    self.logger.warning(f"Rate limited, waiting {retry_after} seconds")
                    time.sleep(retry_after)
                    continue
                
                response.raise_for_status()
                return response.json()
                
            except requests.exceptions.Timeout:
                self.logger.warning(f"Request timeout on attempt {attempt + 1}")
            except requests.exceptions.RequestException as e:
                self.logger.warning(f"Request failed on attempt {attempt + 1}: {e}")
            
            if attempt < self.max_retries - 1:
                delay = self.base_delay * (2 ** attempt) + random.uniform(0, 1)
                time.sleep(delay)
        
        return None

    def collect_financial_news(self, days_back: int = Config.DEFAULT_NEWS_DAYS_BACK, max_results: int = 1000) -> pd.DataFrame:
        """Collect news related to broad US equity market ETFs such as VTI, SCHB, IWV."""
        if self.source == "newsapi":
            url = f"{self.base_url}/everything"
            page = 1
            collected = 0
            
            # Keywords to capture news related to broad market ETFs
            query = " OR ".join([
                "VTI", "SCHB", "IWV",
                "Vanguard Total Stock Market",
                "Schwab U.S. Broad Market",
                "iShares Russell 3000"
            ])
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
                        'collected_at': datetime.now()
                    }
                    all_articles.append(article_data)
                
                collected += len(articles)
                page += 1
                
            df = pd.DataFrame(all_articles)
            return df

        elif self.source == "eventregistry":
            page = 1
            collected = 0
            all_articles = []
            
            # Event Registry API uses a keyword parameter similarly
            query = " OR ".join([
                "VTI", "SCHB", "IWV",
                "Vanguard Total Stock Market",
                "Schwab U.S. Broad Market",
                "iShares Russell 3000"
            ])
            
            while collected < max_results:
                params = {
                    "apiKey": self.api_key,
                    "action": "getArticles",
                    "keyword": query,
                    "lang": "eng",
                    "articlesPage": page,
                    "articlesCount": min(100, max_results - collected),
                    "articlesSortBy": "date",
                    "includeArticleSentiment": True,
                    "dateStart": (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d'),
                    "dateEnd": datetime.now().strftime('%Y-%m-%d')
                }
                
                data = self._make_api_request(self.base_url, params)
                if not data:
                    self.logger.warning(f"Event Registry API returned no data at page {page}")
                    break
                
                articles = data.get("articles", {}).get("results", [])
                if not articles:
                    self.logger.info(f"Event Registry no more articles available at page {page}")
                    break
                
                for article in articles:
                    content = self._get_content_with_fallback(article, 'url', 'body')
                    if len(content.strip()) < 50:
                        continue
                    
                    article_data = {
                        'title': article.get('title'),
                        'content': content,
                        'source': article.get('source', {}).get('title'),
                        'published_at': article.get('dateTimePub'),
                        'url': article.get('url'),
                        'sentiment': article.get('sentiment'),
                        'collected_at': datetime.now()
                    }
                    all_articles.append(article_data)
                
                collected += len(articles)
                page += 1
            
            df = pd.DataFrame(all_articles)
            return df

        """Enhanced financial news collection with batching and quality filtering"""
        days_back = min(days_back, Config.MAX_DAYS_BACK)
        from_date = datetime.now() - timedelta(days=days_back)
        all_articles = []
        
        if self.source == "newsapi":
            url = f"{self.base_url}/everything"
            page = 1
            collected = 0
            consecutive_failures = 0
            
            while collected < min(max_results, 100) and consecutive_failures < 3:  # Free tier limit with failure protection
                params = {
                    'apiKey': self.api_key,
                    'q': 'stock market OR financial news OR earnings OR "financial markets"',
                    'from': from_date.strftime('%Y-%m-%d'),
                    'language': 'en',
                    'sortBy': 'publishedAt',
                    'pageSize': min(100, max_results - collected),
                    'page': page
                }
                
                data = self._make_api_request(url, params)
                if not data:
                    consecutive_failures += 1
                    self.logger.warning(f"API request failed for page {page}")
                    break
                
                if data.get('status') == 'error':
                    self.logger.error(f"NewsAPI error: {data.get('message', 'Unknown error')}")
                    break
                
                articles = data.get('articles', [])
                total_results = data.get('totalResults', 0)
                
                self.logger.info(f"📊 NewsAPI Page {page}: {len(articles)} articles, Total available: {total_results}")
                
                if not articles:
                    self.logger.info("📄 No more articles available")
                    break
                
                # Process articles in batches for better memory management
                batch_articles = []
                for article in articles:
                    content = self._get_content_with_fallback(article, 'url', 'content')
                    
                    # Skip articles with very poor content quality
                    if len(content.strip()) < 50:
                        continue
                    
                    article_data = {
                        'title': article.get('title'),
                        'content': content,
                        'source': article.get('source', {}).get('name'),
                        'published_at': article.get('publishedAt'),
                        'url': article.get('url'),
                        'collected_at': datetime.now()
                    }
                    
                    # Add enhanced metadata if available
                    if article.get('_quality_score'):
                        article_data['quality_score'] = article.get('_quality_score')
                    if article.get('_extracted_keywords'):
                        article_data['keywords'] = article.get('_extracted_keywords')
                    
                    batch_articles.append(article_data)
                
                all_articles.extend(batch_articles)
                collected += len(batch_articles)
                page += 1
                consecutive_failures = 0  # Reset on success
                
                # Break if we've reached all available results
                if collected >= total_results:
                    break
                    
        elif self.source == "eventregistry":
            page = 1
            collected = 0
            consecutive_failures = 0
            
            while collected < max_results and consecutive_failures < 3:
                params = {
                    "apiKey": self.api_key,
                    "action": "getArticles",
                    "categoryUri": "news/Business",
                    "lang": "eng",
                    "articlesPage": page,
                    "articlesCount": min(100, max_results - collected),
                    "articlesSortBy": "date",
                    "includeArticleSentiment": True,
                    "dateStart": (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d'),
                    "dateEnd": datetime.now().strftime('%Y-%m-%d')
                }
                
                data = self._make_api_request(self.base_url, params)
                if not data:
                    consecutive_failures += 1
                    self.logger.warning(f"API request failed for page {page}")
                    continue
                
                self.logger.info(f"🔍 Event Registry response info: {data.get('info', 'No info')}")
                
                articles = data.get("articles", {}).get("results", [])
                total_results = data.get("articles", {}).get("totalResults", 0)
                
                self.logger.info(f"📊 Event Registry Page {page}: {len(articles)} articles, Total available: {total_results}")
                
                if not articles:
                    self.logger.info("📄 No more Event Registry articles available")
                    break
                
                # Process articles with enhanced content extraction
                batch_articles = []
                for article in articles:
                    content = self._get_content_with_fallback(article, 'url', 'body')
                    
                    # Skip low-quality content
                    if len(content.strip()) < 50:
                        continue
                    
                    article_data = {
                        'title': article.get('title'),
                        'content': content,
                        'source': article.get('source', {}).get('title'),
                        'published_at': article.get('dateTimePub'),
                        'url': article.get('url'),
                        'sentiment': article.get('sentiment'),
                        'collected_at': datetime.now()
                    }
                    
                    # Add enhanced metadata
                    if article.get('_quality_score'):
                        article_data['quality_score'] = article.get('_quality_score')
                    if article.get('_extracted_keywords'):
                        article_data['keywords'] = article.get('_extracted_keywords')
                    
                    batch_articles.append(article_data)
                
                all_articles.extend(batch_articles)
                collected += len(batch_articles)
                page += 1
                consecutive_failures = 0
                
                # Break if we've reached all available results
                if collected >= total_results:
                    break
        
        # Filter out duplicates based on URL and title similarity
        df = pd.DataFrame(all_articles)
        if not df.empty:
            # Remove exact URL duplicates
            df = df.drop_duplicates(subset=['url'], keep='first')
            
            # Log final statistics
            self.logger.info(f"🎯 Total unique articles collected: {len(df)}")
            if 'quality_score' in df.columns:
                avg_quality = df['quality_score'].mean()
                self.logger.info(f"📈 Average content quality score: {avg_quality:.2f}")
        
        return df

    def collect_financial_news_combined(self, days_back: int = Config.DEFAULT_NEWS_DAYS_BACK, max_results: int = 1000) -> pd.DataFrame:
        """Enhanced combined collection with intelligent source balancing"""
        all_articles = []
        
        # Try NewsAPI first (limited but high quality)
        newsapi_target = min(100, max_results // 2)  # Reserve half for NewsAPI
        try:
            newsapi_collector = NewsCollector("newsapi")
            newsapi_collector.rate_limiter = self.rate_limiter  # Share rate limiter
            newsapi_df = newsapi_collector.collect_financial_news(days_back, newsapi_target)
            
            if not newsapi_df.empty:
                newsapi_articles = newsapi_df.to_dict('records')
                # Mark source origin
                for article in newsapi_articles:
                    article['source_api'] = 'newsapi'
                
                all_articles.extend(newsapi_articles)
                self.logger.info(f"✅ Collected {len(newsapi_df)} NewsAPI articles")
        except Exception as e:
            self.logger.warning(f"⚠️ NewsAPI collection failed: {str(e)}")
        
        # Fill remaining quota with Event Registry
        remaining = max_results - len(all_articles)
        if remaining > 0:
            try:
                er_collector = NewsCollector("eventregistry")
                er_collector.rate_limiter = self.rate_limiter  # Share rate limiter
                er_df = er_collector.collect_financial_news(days_back, remaining)
                
                if not er_df.empty:
                    er_articles = er_df.to_dict('records')
                    # Mark source origin
                    for article in er_articles:
                        article['source_api'] = 'eventregistry'
                    
                    all_articles.extend(er_articles)
                    self.logger.info(f"✅ Collected {len(er_df)} Event Registry articles")
            except Exception as e:
                self.logger.warning(f"⚠️ Event Registry collection failed: {str(e)}")
        
        # Create combined DataFrame with deduplication
        df = pd.DataFrame(all_articles)
        if not df.empty:
            # Advanced deduplication: remove articles with similar titles (>80% similarity)
            df = self._deduplicate_articles(df)
            
            # Sort by quality score if available, then by publish date
            if 'quality_score' in df.columns:
                df = df.sort_values(['quality_score', 'published_at'], ascending=[False, False])
            elif 'published_at' in df.columns:
                df = df.sort_values('published_at', ascending=False)
            
            self.logger.info(f"🎯 Total combined articles after deduplication: {len(df)}")
        
        return df

    def collect_ticker_news(self, tickers: List[str], max_results: int = 100) -> pd.DataFrame:
        """Enhanced ticker-specific collection with relevance scoring"""
        all_news = []
        
        for ticker in tickers:
            self.logger.info(f"🔍 Collecting news for {ticker}")
            
            if self.source == "newsapi":
                url = f"{self.base_url}/everything"
                page = 1
                collected = 0
                ticker_articles = []
                
                while collected < min(max_results, 100):  # Free tier limit
                    # Enhanced query with multiple ticker variations
                    query_variations = [
                        f'"{ticker}"',  # Exact ticker
                        f'"${ticker}"',  # With dollar sign
                        f'{ticker} stock',  # With "stock"
                        f'{ticker} earnings'  # With "earnings"
                    ]
                    
                    params = {
                        'apiKey': self.api_key,
                        'q': ' OR '.join(query_variations),
                        'language': 'en',
                        'sortBy': 'publishedAt',
                        'pageSize': min(50, max_results - collected),
                        'page': page
                    }
                    
                    data = self._make_api_request(url, params)
                    if not data:
                        break
                    
                    articles = data.get('articles', [])
                    if not articles:
                        break
                    
                    for article in articles:
                        content = self._get_content_with_fallback(article, 'url', 'content')
                        
                        # Calculate relevance score for this ticker
                        relevance_score = self._calculate_ticker_relevance(content, article.get('title', ''), ticker)
                        
                        # Skip articles with very low relevance
                        if relevance_score < 0.3:
                            continue
                        
                        article_data = {
                            'ticker': ticker,
                            'title': article.get('title'),
                            'content': content,
                            'source': article.get('source', {}).get('name'),
                            'published_at': article.get('publishedAt'),
                            'url': article.get('url'),
                            'relevance_score': relevance_score,
                            'collected_at': datetime.now()
                        }
                        
                        # Add quality metadata if available
                        if article.get('_quality_score'):
                            article_data['quality_score'] = article.get('_quality_score')
                        
                        ticker_articles.append(article_data)
                    
                    collected += len(articles)
                    page += 1
                
                # Sort ticker articles by relevance and quality
                ticker_articles.sort(key=lambda x: (x['relevance_score'], x.get('quality_score', 0)), reverse=True)
                
                # Keep top articles for this ticker
                all_news.extend(ticker_articles[:max_results])
                
                self.logger.info(f"📰 Collected {len(ticker_articles)} relevant articles for {ticker} (NewsAPI)")
                
            elif self.source == "eventregistry":
                page = 1
                collected = 0
                ticker_articles = []
                
                while collected < max_results:
                    params = {
                        "apiKey": self.api_key,
                        "action": "getArticles",
                        "keyword": ticker,
                        "lang": "eng",
                        "articlesPage": page,
                        "articlesCount": min(50, max_results - collected),
                        "articlesSortBy": "date",
                        "includeArticleSentiment": True
                    }
                    
                    data = self._make_api_request(self.base_url, params)
                    if not data:
                        break
                    
                    articles = data.get("articles", {}).get("results", [])
                    if not articles:
                        break
                    
                    for article in articles:
                        content = self._get_content_with_fallback(article, 'url', 'body')
                        
                        # Calculate relevance score
                        relevance_score = self._calculate_ticker_relevance(content, article.get('title', ''), ticker)
                        
                        if relevance_score < 0.3:
                            continue
                        
                        article_data = {
                            'ticker': ticker,
                            'title': article.get('title'),
                            'content': content,
                            'source': article.get('source', {}).get('title'),
                            'published_at': article.get('dateTimePub'),
                            'url': article.get('url'),
                            'sentiment': article.get('sentiment'),
                            'relevance_score': relevance_score,
                            'collected_at': datetime.now()
                        }
                        
                        ticker_articles.append(article_data)
                    
                    collected += len(articles)
                    page += 1
                
                # Sort and filter ticker articles
                ticker_articles.sort(key=lambda x: (x['relevance_score'], x.get('quality_score', 0)), reverse=True)
                all_news.extend(ticker_articles[:max_results])
                
                self.logger.info(f"📰 Collected {len(ticker_articles)} relevant Event Registry articles for {ticker}")
        
        return pd.DataFrame(all_news)

    def _calculate_ticker_relevance(self, content: str, title: str, ticker: str) -> float:
        """Calculate how relevant an article is to a specific ticker"""
        if not content and not title:
            return 0.0
        
        text = f"{title} {content}".lower()
        ticker_lower = ticker.lower()
        
        score = 0.0
        
        # Exact ticker mentions
        exact_matches = text.count(f" {ticker_lower} ") + text.count(f"${ticker_lower}") + text.count(f"({ticker_lower})")
        score += exact_matches * 0.4
        
        # Ticker in title gets higher weight
        if ticker_lower in title.lower():
            score += 0.3
        
        # Financial context keywords
        financial_keywords = ['stock', 'earnings', 'revenue', 'profit', 'shares', 'market', 'trading', 'investment']
        for keyword in financial_keywords:
            if keyword in text:
                score += 0.1
        
        # Normalize score to 0-1 range
        return min(score, 1.0)

    def _deduplicate_articles(self, df: pd.DataFrame) -> pd.DataFrame:
        """Advanced deduplication using title similarity"""
        if df.empty:
            return df
        
        try:
            from difflib import SequenceMatcher
            
            def similar(a, b):
                return SequenceMatcher(None, a, b).ratio()
            
            # Group by similar titles
            to_remove = set()
            titles = df['title'].fillna('').tolist()
            
            for i, title1 in enumerate(titles):
                if i in to_remove:
                    continue
                for j, title2 in enumerate(titles[i+1:], i+1):
                    if j in to_remove:
                        continue
                    if similar(title1, title2) > 0.8:  # 80% similarity threshold
                        # Keep the one with higher quality score, or the first one
                        if 'quality_score' in df.columns:
                            if df.iloc[i]['quality_score'] >= df.iloc[j]['quality_score']:
                                to_remove.add(j)
                            else:
                                to_remove.add(i)
                        else:
                            to_remove.add(j)
            
            # Remove duplicates
            if to_remove:
                df = df.drop(df.index[list(to_remove)])
                self.logger.info(f"Removed {len(to_remove)} duplicate articles based on title similarity")
        
        except Exception as e:
            self.logger.warning(f"Deduplication failed: {e}, proceeding without advanced deduplication")
        
        return df


    def clear_cache_dir(cache_dir: str = "cache"):
        """Delete all files and directories in the cache directory"""
        path = Path(cache_dir)
        if path.exists() and path.is_dir():
            try:
                shutil.rmtree(path)
                print(f"✅ Cleared cache directory: {cache_dir}")
            except Exception as e:
                print(f"⚠️ Failed to clear cache directory {cache_dir}: {e}")
        else:
            print(f"Cache directory {cache_dir} does not exist or is not a directory.")
