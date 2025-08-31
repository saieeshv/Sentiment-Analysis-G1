import os
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    # API Keys
    NEWSAPI_KEY = os.getenv('NEWSAPI_KEY')
    REDDIT_CLIENT_ID = os.getenv('REDDIT_CLIENT_ID')
    REDDIT_CLIENT_SECRET = os.getenv('REDDIT_CLIENT_SECRET')
    REDDIT_USER_AGENT = os.getenv('REDDIT_USER_AGENT', 'FinancialSentimentBot/1.0')
    
    # Stock tickers to analyze
    DEFAULT_TICKERS = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN', 'NVDA', 'META', 'NFLX']
    
    # Reddit settings
    REDDIT_SUBREDDITS = ['wallstreetbets', 'investing', 'stocks', 'SecurityAnalysis']
    REDDIT_POST_LIMIT = 50
    
    # News settings
    NEWS_SOURCES = ['reuters', 'bloomberg', 'cnbc', 'marketwatch']
    NEWS_KEYWORDS = ['stock market', 'earnings', 'financial news']
    
    # Data collection settings
    DATA_DAYS_BACK = 7
    
    @staticmethod
    def validate_config():
        """Check if all required API keys are set"""
        missing = []
        if not Config.NEWSAPI_KEY:
            missing.append('NEWSAPI_KEY')
        if not Config.REDDIT_CLIENT_ID:
            missing.append('REDDIT_CLIENT_ID')
        if not Config.REDDIT_CLIENT_SECRET:
            missing.append('REDDIT_CLIENT_SECRET')
        
        if missing:
            raise ValueError(f"Missing required environment variables: {', '.join(missing)}")
        
        return True
