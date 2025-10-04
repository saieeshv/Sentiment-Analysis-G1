import praw
import pandas as pd
import re
import time
import random
import pickle
import hashlib
from typing import List, Dict, Optional, Set, Tuple
from datetime import datetime, timedelta
from collections import deque
from pathlib import Path
import logging
import pytz
from config.config import Config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

class RedditRateLimiter:
    """Smart rate limiter with request history tracking"""
    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
        self.request_times = deque()
    
    def wait_if_needed(self):
        """Smart rate limiting with request history tracking"""
        now = time.time()
        
        # Remove requests older than 1 minute
        while self.request_times and now - self.request_times[0] > 60:
            self.request_times.popleft()
        
        if len(self.request_times) >= self.requests_per_minute:
            sleep_time = 60 - (now - self.request_times[0])
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        self.request_times.append(now)

class RedditCollector:
    """Enhanced Reddit collector for financial sentiment analysis"""
    
    # Financial subreddits with weights and focus areas
    FINANCIAL_SUBREDDITS = {
        'wallstreetbets': {'weight': 1.0, 'focus': 'retail_sentiment'},
        'investing': {'weight': 0.8, 'focus': 'fundamental_analysis'},
        'stocks': {'weight': 0.9, 'focus': 'general_discussion'},
        'SecurityAnalysis': {'weight': 0.7, 'focus': 'professional_analysis'},
        'ValueInvesting': {'weight': 0.6, 'focus': 'value_strategy'},
        'financialindependence': {'weight': 0.5, 'focus': 'long_term_strategy'},
        'StockMarket': {'weight': 0.8, 'focus': 'market_discussion'},
        'pennystocks': {'weight': 0.6, 'focus': 'speculative_trading'},
        'options': {'weight': 0.7, 'focus': 'derivatives_trading'}
    }
    
    # Financial keywords for relevance scoring
    FINANCIAL_KEYWORDS = [
        'bull', 'bear', 'calls', 'puts', 'earnings', 'dividend', 'shorts', 'squeeze', 
        'moon', 'diamond hands', 'paper hands', 'hodl', 'yolo', 'stonks', 'tendies',
        'rocket', 'baghold', 'dip', 'rally', 'breakout', 'support', 'resistance',
        'volume', 'market cap', 'revenue', 'profit', 'loss', 'eps', 'pe ratio',
        'insider trading', 'sec filing', 'merger', 'acquisition', 'ipo', 'split'
    ]
    
    def __init__(self, cache_dir: str = "./reddit_cache"):
        """Initialize enhanced Reddit collector"""
        self.reddit = praw.Reddit(
            client_id=Config.REDDIT_CLIENT_ID,
            client_secret=Config.REDDIT_CLIENT_SECRET,
            user_agent=Config.REDDIT_USER_AGENT
        )
        self.logger = logging.getLogger(__name__)
        self.cutoff_date = datetime.utcnow() - timedelta(days=30)
        
        # Enhanced components
        self.rate_limiter = RedditRateLimiter(requests_per_minute=50)
        self._setup_cache(cache_dir)
        
        # Retry settings
        self.max_retries = 3
        self.base_delay = 1.0
    
    def _setup_cache(self, cache_dir: str):
        """Setup persistent caching for collected data"""
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
    def _get_cache_key(self, operation: str, params: Dict) -> str:
        """Generate cache key from operation and parameters"""
        param_str = str(sorted(params.items()))
        return hashlib.md5(f"{operation}_{param_str}".encode()).hexdigest()
    
    def _get_cached_data(self, cache_key: str) -> Optional[pd.DataFrame]:
        """Retrieve cached data if available and fresh"""
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                    if datetime.utcnow() - cached_data['timestamp'] < timedelta(hours=1):
                        self.logger.info("📋 Using cached data")
                        return cached_data['data']
            except Exception as e:
                self.logger.warning(f"Cache read error: {e}")
        return None
    
    def _cache_data(self, cache_key: str, data: pd.DataFrame):
        """Cache data with timestamp"""
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump({
                    'timestamp': datetime.utcnow(),
                    'data': data
                }, f)
        except Exception as e:
            self.logger.warning(f"Cache write error: {e}")
    
    def test_connection(self) -> bool:
        """Test Reddit API connection with enhanced error handling"""
        try:
            _ = self.reddit.user.me()
            self.logger.info("✅ Reddit API connection successful")
            return True
        except Exception as e:
            self.logger.error(f"❌ Reddit API connection failed: {str(e)}")
            return False
    
    def _clean_text(self, text) -> str:
        """Enhanced text cleaning for financial analysis"""
        if not isinstance(text, str):
            return ""
        if text.lower() in ['[removed]', '[deleted]', '[deleted by user]']:
            return ''
        
        # Remove excessive whitespace and newlines
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n+', ' ', text)
        
        return text.strip()
    
    def _extract_financial_entities(self, text: str) -> Dict[str, any]:
        """Extract financial entities and calculate relevance scores"""
        if not text:
            return {'tickers': [], 'relevance_score': 0.0, 'financial_keywords_found': []}
        
        text_upper = text.upper()
        text_lower = text.lower()
        
        # Multiple ticker patterns
        ticker_patterns = [
            r'\$([A-Z]{1,5})\b',  # $AAPL format
            r'\b([A-Z]{2,5})\b(?=\s+(?:stock|share|call|put|options?))',  # Context-based
            r'\b([A-Z]{2,5})(?:\s+(?:to|@)\s+(?:the\s+)?(?:moon|mars))',  # Meme patterns
            r'\b([A-Z]{1,5})(?:\s+\d+[cp])\b',  # Options format (AAPL 150c)
        ]
        
        tickers = set()
        for pattern in ticker_patterns:
            matches = re.findall(pattern, text_upper)
            # Filter out common false positives
            filtered_matches = [t for t in matches if t not in ['THE', 'AND', 'FOR', 'YOU', 'ARE', 'CAN', 'BUT', 'NOT', 'ALL']]
            tickers.update(filtered_matches)
        
        # Financial keywords for relevance scoring
        keywords_found = [kw for kw in self.FINANCIAL_KEYWORDS if kw in text_lower]
        relevance_score = min(len(keywords_found) / len(self.FINANCIAL_KEYWORDS), 1.0)
        
        return {
            'tickers': list(tickers),
            'relevance_score': relevance_score,
            'financial_keywords_found': keywords_found
        }
    
    def _calculate_post_quality(self, post_data: Dict) -> float:
        """Calculate post quality score for financial sentiment analysis"""
        quality_score = 0.0
        
        # Text length scoring
        text_length = len(post_data.get('text', '')) + len(post_data.get('title', ''))
        if text_length > 100: 
            quality_score += 0.3
        if text_length > 300: 
            quality_score += 0.2
        if text_length > 500:
            quality_score += 0.1
        
        # Engagement scoring
        score = post_data.get('score', 0)
        num_comments = post_data.get('num_comments', 0)
        
        if score > 10: 
            quality_score += 0.1
        if score > 50: 
            quality_score += 0.1
        if num_comments > 5: 
            quality_score += 0.1
        
        # Score to comment ratio (engagement quality)
        if num_comments > 0:
            score_ratio = score / num_comments
            if score_ratio > 2: 
                quality_score += 0.1
        
        # Financial relevance
        financial_entities = self._extract_financial_entities(
            f"{post_data.get('title', '')} {post_data.get('text', '')}"
        )
        quality_score += financial_entities['relevance_score'] * 0.2
        
        return min(quality_score, 1.0)
    
    def _calculate_comment_quality(self, comment) -> float:
        """Calculate comment quality score"""
        quality_score = 0.0
        
        # Length scoring
        if hasattr(comment, 'body') and len(comment.body) > 50:
            quality_score += 0.4
        if hasattr(comment, 'body') and len(comment.body) > 200:
            quality_score += 0.2
        
        # Score scoring
        if hasattr(comment, 'score'):
            if comment.score > 5:
                quality_score += 0.2
            if comment.score > 20:
                quality_score += 0.2
        
        return min(quality_score, 1.0)
    
    def _collect_with_retry(self, collection_func, *args, max_retries: int = 3, **kwargs):
        """Execute collection function with exponential backoff retry"""
        for attempt in range(max_retries):
            try:
                self.rate_limiter.wait_if_needed()
                return collection_func(*args, **kwargs)
            except Exception as e:
                if attempt == max_retries - 1:
                    self.logger.error(f"Collection failed after {max_retries} attempts: {str(e)}")
                    raise e
                
                # Exponential backoff with jitter
                sleep_time = (2 ** attempt) + random.uniform(0, 1)
                self.logger.warning(f"Attempt {attempt + 1} failed, retrying in {sleep_time:.2f}s: {str(e)}")
                time.sleep(sleep_time)
    
    def _prioritize_subreddits_by_ticker(self, ticker: str) -> List[str]:
        """Dynamically prioritize subreddits based on ticker type and historical popularity"""
        # Default prioritization based on ticker characteristics
        if ticker in ['SPY', 'QQQ', 'VTI', 'VOO']:  # ETFs
            return ['investing', 'Bogleheads', 'financialindependence', 'stocks']
        elif len(ticker) <= 3:  # Major stocks
            return ['wallstreetbets', 'stocks', 'investing', 'SecurityAnalysis']
        else:  # Potentially penny stocks
            return ['pennystocks', 'wallstreetbets', 'stocks']
    
    def collect_posts_last_month(self, subreddits: List[str] = None, limit: int = 1000) -> pd.DataFrame:
        """Enhanced collection with caching and quality scoring"""
        if subreddits is None:
            subreddits = list(self.FINANCIAL_SUBREDDITS.keys())
        
        # Check cache first
        cache_key = self._get_cache_key("posts", {"subreddits": subreddits, "limit": limit})
        cached_data = self._get_cached_data(cache_key)
        if cached_data is not None:
            return cached_data
        
        all_posts = []
        
        for sub_name in subreddits:
            try:
                subreddit = self.reddit.subreddit(sub_name)
                post_count = 0
                
                def collect_subreddit_posts():
                    nonlocal post_count
                    posts_for_sub = []
                    
                    for submission in subreddit.new(limit=limit):
                        created = datetime.utcfromtimestamp(submission.created_utc)
                        if created < self.cutoff_date:
                            break
                        
                        post_data = {
                            'id': submission.id,
                            'subreddit': sub_name,
                            'title': submission.title,
                            'text': self._clean_text(submission.selftext),
                            'score': submission.score,
                            'num_comments': submission.num_comments,
                            'created_utc': created,
                            'url': submission.url,
                            'collected_at': datetime.utcnow(),
                            'subreddit_weight': self.FINANCIAL_SUBREDDITS.get(sub_name, {}).get('weight', 0.5),
                            'subreddit_focus': self.FINANCIAL_SUBREDDITS.get(sub_name, {}).get('focus', 'general')
                        }
                        
                        # Extract financial entities
                        financial_entities = self._extract_financial_entities(
                            f"{post_data['title']} {post_data['text']}"
                        )
                        post_data.update({
                            'tickers': financial_entities['tickers'],
                            'relevance_score': financial_entities['relevance_score'],
                            'financial_keywords': financial_entities['financial_keywords_found'],
                            'quality_score': self._calculate_post_quality(post_data)
                        })
                        
                        # Only keep posts with minimum quality
                        if post_data['quality_score'] >= 0.2:
                            posts_for_sub.append(post_data)
                            post_count += 1
                    
                    return posts_for_sub
                
                subreddit_posts = self._collect_with_retry(collect_subreddit_posts)
                all_posts.extend(subreddit_posts)
                
                self.logger.info(f"📱 Collected {post_count} quality posts from r/{sub_name}")
                
            except Exception as e:
                self.logger.error(f"❌ Error fetching posts for r/{sub_name}: {str(e)}")
        
        df = pd.DataFrame(all_posts)
        
        # Cache the results
        self._cache_data(cache_key, df)
        
        return df
    
    def search_tickers_last_month(self, tickers: List[str], limit: int = 1000) -> pd.DataFrame:
        """Enhanced ticker search with relevance scoring and prioritized subreddits"""
        cache_key = self._get_cache_key("tickers", {"tickers": tickers, "limit": limit})
        cached_data = self._get_cached_data(cache_key)
        if cached_data is not None:
            return cached_data
        
        ticker_posts = []
        
        for ticker in tickers:
            # Get prioritized subreddits for this ticker
            prioritized_subs = self._prioritize_subreddits_by_ticker(ticker)
            all_subs = prioritized_subs + [s for s in self.FINANCIAL_SUBREDDITS.keys() if s not in prioritized_subs]
            
            # Enhanced query variations
            query_variations = [
                f"${ticker}",  # Exact cashtag
                f"{ticker}",   # Exact ticker
                f"{ticker} stock",  # With "stock"
                f"{ticker} earnings",  # With "earnings"
            ]
            
            for sub_name in all_subs[:6]:  # Limit to top 6 subreddits per ticker
                try:
                    subreddit = self.reddit.subreddit(sub_name)
                    
                    def search_ticker_in_sub():
                        posts_for_ticker = []
                        
                        for query in query_variations:
                            try:
                                for submission in subreddit.search(query, sort="new", limit=limit//4):
                                    created = datetime.utcfromtimestamp(submission.created_utc)
                                    if created < self.cutoff_date:
                                        continue
                                    
                                    # Calculate relevance to the specific ticker
                                    full_text = f"{submission.title} {self._clean_text(submission.selftext)}"
                                    relevance_score = self._calculate_ticker_relevance(
                                        full_text, submission.title, ticker
                                    )
                                    
                                    # Only include posts with decent relevance
                                    if relevance_score < 0.3:
                                        continue
                                    
                                    post_data = {
                                        'ticker': ticker,
                                        'id': submission.id,
                                        'subreddit': sub_name,
                                        'title': submission.title,
                                        'text': self._clean_text(submission.selftext),
                                        'score': submission.score,
                                        'num_comments': submission.num_comments,
                                        'created_utc': created,
                                        'url': submission.url,
                                        'collected_at': datetime.utcnow(),
                                        'relevance_score': relevance_score,
                                        'query_used': query,
                                        'subreddit_weight': self.FINANCIAL_SUBREDDITS.get(sub_name, {}).get('weight', 0.5)
                                    }
                                    
                                    # Extract additional financial entities
                                    financial_entities = self._extract_financial_entities(full_text)
                                    post_data.update({
                                        'all_tickers': financial_entities['tickers'],
                                        'financial_keywords': financial_entities['financial_keywords_found'],
                                        'quality_score': self._calculate_post_quality(post_data)
                                    })
                                    
                                    posts_for_ticker.append(post_data)
                                    
                            except Exception as query_e:
                                self.logger.debug(f"Query '{query}' failed in r/{sub_name}: {query_e}")
                                continue
                        
                        return posts_for_ticker
                    
                    subreddit_posts = self._collect_with_retry(search_ticker_in_sub)
                    ticker_posts.extend(subreddit_posts)
                    
                    self.logger.info(f"🔍 Found {len(subreddit_posts)} posts for {ticker} in r/{sub_name}")
                    
                except Exception as e:
                    self.logger.error(f"❌ Error searching {ticker} in r/{sub_name}: {str(e)}")
        
        df = pd.DataFrame(ticker_posts)
        
        # Remove duplicates and sort by relevance
        if not df.empty:
            df = df.drop_duplicates(subset=['id'], keep='first')
            df = df.sort_values(['relevance_score', 'quality_score'], ascending=[False, False])
        
        # Cache the results
        self._cache_data(cache_key, df)
        
        return df
    
    def _calculate_ticker_relevance(self, content: str, title: str, ticker: str) -> float:
        """Calculate how relevant an article is to a specific ticker"""
        if not content and not title:
            return 0.0
        
        text = f"{title} {content}".lower()
        ticker_lower = ticker.lower()
        
        score = 0.0
        
        # Exact ticker mentions (with different formats)
        exact_matches = (
            text.count(f"${ticker_lower}") + 
            text.count(f" {ticker_lower} ") + 
            text.count(f"{ticker_lower} stock") +
            text.count(f"{ticker_lower} earnings")
        )
        score += min(exact_matches * 0.4, 0.8)
        
        # Ticker in title gets higher weight
        if ticker_lower in title.lower():
            score += 0.3
        
        # Context-based scoring
        financial_context_words = ['buy', 'sell', 'hold', 'price', 'target', 'analyst', 'upgrade', 'downgrade']
        context_score = sum(1 for word in financial_context_words if word in text)
        score += min(context_score * 0.05, 0.2)
        
        return min(score, 1.0)
    
    def collect_post_comments(self, submission_id: str, max_comments: int = 100) -> List[Dict]:
        """Collect and analyze comments for sentiment context"""
        try:
            def get_comments():
                submission = self.reddit.submission(id=submission_id)
                submission.comments.replace_more(limit=0)
                
                comments_data = []
                for comment in submission.comments[:max_comments]:
                    if hasattr(comment, 'body') and len(comment.body) > 10:
                        financial_entities = self._extract_financial_entities(comment.body)
                        
                        comment_data = {
                            'comment_id': comment.id,
                            'parent_id': submission_id,
                            'body': self._clean_text(comment.body),
                            'score': getattr(comment, 'score', 0),
                            'created_utc': datetime.utcfromtimestamp(comment.created_utc),
                            'tickers': financial_entities['tickers'],
                            'relevance_score': financial_entities['relevance_score'],
                            'quality_score': self._calculate_comment_quality(comment),
                            'financial_keywords': financial_entities['financial_keywords_found']
                        }
                        
                        if comment_data['quality_score'] >= 0.3:
                            comments_data.append(comment_data)
                
                return comments_data
            
            return self._collect_with_retry(get_comments)
            
        except Exception as e:
            self.logger.error(f"Error collecting comments for {submission_id}: {str(e)}")
            return []
    
    def collect_market_hours_posts(self, target_date: datetime, market_timezone='US/Eastern') -> pd.DataFrame:
        """Collect posts during specific market hours for better correlation"""
        try:
            market_tz = pytz.timezone(market_timezone)
            market_open = market_tz.localize(target_date.replace(hour=9, minute=30, second=0))
            market_close = market_tz.localize(target_date.replace(hour=16, minute=0, second=0))
            
            # Convert to UTC for Reddit API
            start_utc = market_open.astimezone(pytz.UTC)
            end_utc = market_close.astimezone(pytz.UTC)
            
            # Collect posts within market hours
            all_posts = self.collect_posts_last_month()
            
            if not all_posts.empty:
                # Filter posts within market hours
                market_posts = all_posts[
                    (all_posts['created_utc'] >= start_utc) &
                    (all_posts['created_utc'] <= end_utc)
                ]
                
                self.logger.info(f"📈 Filtered {len(market_posts)} posts during market hours for {target_date.date()}")
                return market_posts
            
            return pd.DataFrame()
            
        except Exception as e:
            self.logger.error(f"Error collecting market hours posts: {str(e)}")
            return pd.DataFrame()
    
    def get_collection_stats(self, df: pd.DataFrame) -> Dict:
        """Get comprehensive statistics about collected data"""
        if df.empty:
            return {}
        
        stats = {
            'total_posts': len(df),
            'unique_subreddits': df['subreddit'].nunique() if 'subreddit' in df.columns else 0,
            'date_range': {
                'start': df['created_utc'].min() if 'created_utc' in df.columns else None,
                'end': df['created_utc'].max() if 'created_utc' in df.columns else None
            },
            'avg_quality_score': df['quality_score'].mean() if 'quality_score' in df.columns else 0,
            'avg_relevance_score': df['relevance_score'].mean() if 'relevance_score' in df.columns else 0,
        }
        
        if 'tickers' in df.columns:
            all_tickers = []
            for ticker_list in df['tickers'].dropna():
                if isinstance(ticker_list, list):
                    all_tickers.extend(ticker_list)
            stats['unique_tickers'] = len(set(all_tickers))
            stats['most_mentioned_tickers'] = pd.Series(all_tickers).value_counts().head(10).to_dict()
        
        return stats
