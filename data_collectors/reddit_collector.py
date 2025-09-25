import praw
import pandas as pd
from typing import List
from datetime import datetime, timedelta
import logging
from config.config import Config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

class RedditCollector:
    def __init__(self):
        self.reddit = praw.Reddit(
            client_id=Config.REDDIT_CLIENT_ID,
            client_secret=Config.REDDIT_CLIENT_SECRET,
            user_agent=Config.REDDIT_USER_AGENT
        )
        self.logger = logging.getLogger(__name__)
        self.cutoff_date = datetime.utcnow() - timedelta(days=30)
    
    def test_connection(self):
        """Test Reddit API connection"""
        try:
            _ = self.reddit.user.me()
            self.logger.info("✅ Reddit API connection successful")
            return True
        except Exception as e:
            self.logger.error(f"❌ Reddit API connection failed: {str(e)}")
            return False
    
    def _clean_text(self, text) -> str:
        if not isinstance(text, str):
            return ""
        if text.lower() in ['[removed]', '[deleted]']:
            return ''
        return text.strip()
    
    def collect_posts_last_month(self, subreddits: List[str] = None, limit: int = 1000) -> pd.DataFrame:
        """Collect up to ~1 month of posts using Reddit API (stop when cutoff is reached)"""
        if subreddits is None:
            subreddits = Config.REDDIT_SUBREDDITS

        all_posts = []
        for sub_name in subreddits:
            subreddit = self.reddit.subreddit(sub_name)
            try:
                for submission in subreddit.new(limit=limit):
                    created = datetime.utcfromtimestamp(submission.created_utc)
                    if created < self.cutoff_date:
                        break  # stop if post is older than 30 days
                    
                    post_data = {
                        'id': submission.id,
                        'subreddit': sub_name,
                        'title': submission.title,
                        'text': self._clean_text(submission.selftext),
                        'score': submission.score,
                        'num_comments': submission.num_comments,
                        'created_utc': created,
                        'url': submission.url,
                        'collected_at': datetime.utcnow()
                    }
                    all_posts.append(post_data)
                self.logger.info(f"📱 Collected {len(all_posts)} posts from r/{sub_name} (last 30 days approx.)")
            except Exception as e:
                self.logger.error(f"❌ Error fetching posts for r/{sub_name}: {str(e)}")

        return pd.DataFrame(all_posts)
    
    def search_tickers_last_month(self, tickers: List[str], limit: int = 1000) -> pd.DataFrame:
        """Search for specific tickers in ~last 30 days using Reddit API"""
        ticker_posts = []
        
        for ticker in tickers:
            query = f"${ticker} OR {ticker}"
            for sub_name in Config.REDDIT_SUBREDDITS:
                subreddit = self.reddit.subreddit(sub_name)
                try:
                    for submission in subreddit.search(query, sort="new", limit=limit):
                        created = datetime.utcfromtimestamp(submission.created_utc)
                        if created < self.cutoff_date:
                            break
                        
                        post_data = {
                            'ticker': ticker,
                            'id': submission.id,
                            'subreddit': sub_name,
                            'title': submission.title,
                            'text': self._clean_text(submission.selftext),
                            'score': submission.score,
                            'num_comments': submission.num_comments,
                            'created_utc': created,
                            'collected_at': datetime.utcnow()
                        }
                        ticker_posts.append(post_data)
                    self.logger.info(f"🔍 Collected posts for {ticker} in r/{sub_name}")
                except Exception as e:
                    self.logger.error(f"❌ Error searching {ticker} in r/{sub_name}: {str(e)}")

        return pd.DataFrame(ticker_posts)
