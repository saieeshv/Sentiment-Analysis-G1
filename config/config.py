import os
from datetime import datetime, timedelta
from dotenv import load_dotenv
import yfinance as yf
import logging

# Load environment variables
load_dotenv()

class Config:
    # API Keys
    NEWSAPI_KEY = os.getenv('NEWSAPI_KEY')
    REDDIT_CLIENT_ID = os.getenv('REDDIT_CLIENT_ID')
    REDDIT_CLIENT_SECRET = os.getenv('REDDIT_CLIENT_SECRET')
    REDDIT_USER_AGENT = os.getenv('REDDIT_USER_AGENT', 'FinancialSentimentBot/1.0')
    STOCKNEWS_API_KEY = os.getenv('STOCKNEWS_API_KEY')  
    
    # Data Collection Settings
    DEFAULT_NEWS_DAYS_BACK = 30
    DEFAULT_REDDIT_DAYS_BACK = 30
    MAX_DAYS_BACK = 365
    MAX_NEWS_RESULTS = 1000
    MAX_REDDIT_POSTS = 1000
    
    # Stock tickers to analyze
    DEFAULT_TICKERS = [
        'AAPL',  # Information Technology
        'JPM',   # Financials
        'UNH',   # Healthcare
        'AMZN',  # Consumer Discretionary
        'WMT',   # Consumer Staples
        'T',     # Communication Services
        'BA'     # Industrials
    ]

    # Broad Market ETFs (for individual news collection)
    BROAD_MARKET_ETFS = [
        "VOO",   # Vanguard S&P 500
        "SPY",   # SPDR S&P 500
        "QQQ",   # Invesco QQQ
        "SOXX",  # iShares Semiconductor
        "IWM",   # iShares Russell 2000
        "ARKX"   # ARK Space Exploration
    ]
    
    # Broad market keywords (for NewsAPI general market news)
    BROAD_MARKET_KEYWORDS = [
        "stock market",
        "financial markets",
        "Wall Street",
        "market outlook",
        "bull market",
        "bear market",
        "market volatility"
    ]

    # Sector mapping (manual for ETFs and special cases)
    TICKER_SECTORS = {
        # Broad Market ETFs
        'VOO': 'Broad Market ETF',
        'SOXX': 'Semiconductors ETF',
        'IWM': 'Small Cap ETF',
        'SPY': 'Broad Market ETF',
        'QQQ': 'Technology ETF',
        'ARKX': 'Innovation ETF',
        'VTI': 'Broad Market ETF',
        'IWV': 'Broad Market ETF',
        'IJH': 'Mid Cap ETF',
        'MACRO': 'Market-Wide'
    }

    # Date range settings
    START_DATE = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    END_DATE = datetime.now().strftime('%Y-%m-%d')
    
    # Data directories
    DATA_DIR = 'data'
    CACHE_DIR = 'cache'
    
    @staticmethod
    def get_sector(ticker: str) -> str:
        """Get sector for any ticker using yfinance with fallback to manual mapping."""
        if ticker in Config.TICKER_SECTORS:
            return Config.TICKER_SECTORS[ticker]
        
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            sector = info.get('sector', None)
            if sector:
                return sector
            industry = info.get('industry', None)
            if industry:
                return industry
        except Exception as e:
            logging.warning(f"Could not fetch sector for {ticker}: {e}")
        
        return 'Other'
    
    @staticmethod
    def validate_config():
        """Validate that all required API keys are present"""
        required_keys = {
            'NEWSAPI_KEY': Config.NEWSAPI_KEY,
            'REDDIT_CLIENT_ID': Config.REDDIT_CLIENT_ID,
            'REDDIT_CLIENT_SECRET': Config.REDDIT_CLIENT_SECRET,
            'STOCKNEWS_API_KEY': Config.STOCKNEWS_API_KEY
        }
        
        missing_keys = [key for key, value in required_keys.items() if not value]
        
        if missing_keys:
            raise ValueError(f"Missing required API keys: {', '.join(missing_keys)}")
        
        return True
