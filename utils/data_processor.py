import pandas as pd
import re
from datetime import datetime
import logging

class DataProcessor:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Clean text for sentiment analysis"""
        if not isinstance(text, str):
            return ""
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+', '', text)
        # Remove mentions and hashtags for now
        text = re.sub(r'@\w+|#\w+', '', text)
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text.strip()
    
    @staticmethod
    def save_data(df: pd.DataFrame, filename: str) -> str:
        """Save DataFrame with timestamp"""
        if df.empty:
            return ""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = f"data/{filename}_{timestamp}.csv"
        df.to_csv(filepath, index=False)
        return filepath
    
    def combine_text_data(self, reddit_df: pd.DataFrame, news_df: pd.DataFrame) -> pd.DataFrame:
        """Combine Reddit and news data for analysis"""
        combined_data = []
        
        # Process Reddit data
        if not reddit_df.empty:
            for _, row in reddit_df.iterrows():
                text = self.clean_text(f"{row.get('title', '')} {row.get('text', '')}")
                if text:
                    combined_data.append({
                        'text': text,
                        'source': 'reddit',
                        'subreddit': row.get('subreddit'),
                        'score': row.get('score', 0),
                        'created_at': row.get('created_utc'),
                        'ticker': row.get('ticker', 'general')
                    })
        
        # Process news data
        if not news_df.empty:
            for _, row in news_df.iterrows():
                text = self.clean_text(f"{row.get('title', '')} {row.get('description', '')}")
                if text:
                    combined_data.append({
                        'text': text,
                        'source': 'news',
                        'news_source': row.get('source'),
                        'published_at': row.get('published_at'),
                        'ticker': row.get('ticker', 'general')
                    })
        
        return pd.DataFrame(combined_data)
