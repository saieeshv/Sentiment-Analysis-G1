import praw
import pandas as pd
from typing import List, Dict
from datetime import datetime
import logging
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
        
    def collect_posts(self, subreddits: List[str] = None, limit: int = 50) -> pd.DataFrame:
        """Collect recent posts from financial subreddits"""
        if subreddits is None:
            subreddits = Config.REDDIT_SUBREDDITS
            
        all_posts = []
        
        for sub_name in subreddits:
            try:
                subreddit = self.reddit.subreddit(sub_name)
                posts = subreddit.hot(limit=limit)
                
                for post in posts:
                    post_data = {
                        'id': post.id,
                        'subreddit': sub_name,
                        'title': post.title,
                        'text': post.selftext,
                        'score': post.score,
                        'num_comments': post.num_comments,
                        'created_utc': datetime.fromtimestamp(post.created_utc),
                        'url': post.url,
                        'collected_at': datetime.now()
                    }
                    all_posts.append(post_data)
                
                self.logger.info(f"📱 Collected {len(all_posts)} posts from r/{sub_name}")
                
            except Exception as e:
                self.logger.error(f"❌ Error with r/{sub_name}: {str(e)}")
        
        return pd.DataFrame(all_posts)
    
    def search_tickers(self, tickers: List[str]) -> pd.DataFrame:
        """Search for specific ticker mentions"""
        ticker_posts = []
        
        for ticker in tickers:
            for sub_name in Config.REDDIT_SUBREDDITS:
                try:
                    subreddit = self.reddit.subreddit(sub_name)
                    # Search for ticker mentions
                    results = subreddit.search(f"${ticker} OR {ticker}", time_filter='day', limit=25)
                    
                    for post in results:
                        post_data = {
                            'ticker': ticker,
                            'id': post.id,
                            'subreddit': sub_name,
                            'title': post.title,
                            'text': post.selftext,
                            'score': post.score,
                            'created_utc': datetime.fromtimestamp(post.created_utc),
                            'collected_at': datetime.now()
                        }
                        ticker_posts.append(post_data)
                
                except Exception as e:
                    self.logger.error(f"❌ Search error for {ticker} in r/{sub_name}: {str(e)}")
        
        return pd.DataFrame(ticker_posts)
