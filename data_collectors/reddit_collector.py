import praw
import pandas as pd
from typing import List
from datetime import datetime, timedelta
import logging
import requests
from config.config import Config

class RedditCollector:
    def __init__(self):
        self.reddit = praw.Reddit(
            client_id=Config.REDDIT_CLIENT_ID,
            client_secret=Config.REDDIT_CLIENT_SECRET,
            user_agent=Config.REDDIT_USER_AGENT
        )
        self.logger = logging.getLogger(__name__)
    
    def test_connection(self):
        """Test Reddit API connection"""
        try:
            self.reddit.user.me()
            self.logger.info("✅ Reddit API connection successful")
            return True
        except Exception as e:
            self.logger.error(f"❌ Reddit API connection failed: {str(e)}")
            return False
    
    def _get_timestamp_range(self, days_back: int = 30):
        now = int(datetime.utcnow().timestamp())
        past = int((datetime.utcnow() - timedelta(days=days_back)).timestamp())
        return past, now

    def _clean_text(self, text: str) -> str:
        if not text or text.lower() in ['[removed]', '[deleted]']:
            return ''
        return text.strip()
    
    def collect_posts_last_month(self, subreddits: List[str] = None, limit_per_request: int = 500) -> pd.DataFrame:
        """Collect posts from the past 30 days using Pushshift API"""
        if subreddits is None:
            subreddits = Config.REDDIT_SUBREDDITS
        
        all_posts = []
        after, before = self._get_timestamp_range(30)
        
        for sub_name in subreddits:
            url = "https://api.pushshift.io/reddit/submission/search/"
            params = {
                "subreddit": sub_name,
                "after": after,
                "before": before,
                "size": limit_per_request,
                "sort": "desc",
                "sort_type": "created_utc"
            }
            try:
                response = requests.get(url, params=params)
                response.raise_for_status()
                data = response.json().get("data", [])
                
                for post in data:
                    post_data = {
                        'id': post.get('id'),
                        'subreddit': sub_name,
                        'title': post.get('title'),
                        'text': self._clean_text(post.get('selftext', '')),
                        'score': post.get('score', 0),
                        'num_comments': post.get('num_comments', 0),
                        'created_utc': datetime.fromtimestamp(post.get('created_utc')),
                        'url': post.get('url'),
                        'collected_at': datetime.now()
                    }
                    all_posts.append(post_data)
                
                self.logger.info(f"📱 Collected {len(data)} posts from r/{sub_name} (last 30 days)")
            except Exception as e:
                self.logger.error(f"❌ Error fetching posts for r/{sub_name}: {str(e)}")
        
        return pd.DataFrame(all_posts)
    
    def search_tickers_last_month(self, tickers: List[str], limit_per_request: int = 500) -> pd.DataFrame:
        """Search for specific tickers in the last 30 days using Pushshift"""
        ticker_posts = []
        after, before = self._get_timestamp_range(30)
        
        for ticker in tickers:
            for sub_name in Config.REDDIT_SUBREDDITS:
                url = "https://api.pushshift.io/reddit/search/submission/"
                params = {
                    "subreddit": sub_name,
                    "after": after,
                    "before": before,
                    "q": f"${ticker} OR {ticker}",
                    "size": limit_per_request,
                    "sort": "desc",
                    "sort_type": "created_utc"
                }
                try:
                    response = requests.get(url, params=params)
                    response.raise_for_status()
                    data = response.json().get("data", [])
                    
                    for post in data:
                        post_data = {
                            'ticker': ticker,
                            'id': post.get('id'),
                            'subreddit': sub_name,
                            'title': post.get('title'),
                            'text': self._clean_text(post.get('selftext', '')),
                            'score': post.get('score', 0),
                            'created_utc': datetime.fromtimestamp(post.get('created_utc')),
                            'collected_at': datetime.now()
                        }
                        ticker_posts.append(post_data)
                
                    self.logger.info(f"🔍 Collected {len(data)} posts for {ticker} in r/{sub_name}")
                except Exception as e:
                    self.logger.error(f"❌ Error searching {ticker} in r/{sub_name}: {str(e)}")
        
        return pd.DataFrame(ticker_posts)
