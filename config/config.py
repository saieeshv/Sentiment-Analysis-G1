import os
from datetime import datetime, timedelta
from dotenv import load_dotenv
import yfinance as yf
import logging

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

    # Broad Market ETFs
    BROAD_MARKET_ETFS = [
        "VOO",   # Vanguard S&P 500
        "SPY",   # SPDR S&P 500
        "QQQ",   # Invesco QQQ
        "SOXX",  # iShares Semiconductor
        "IWM",   # iShares Russell 2000
        "ARKX"   # ARK Space Exploration
    ]
    
    # Broad market keywords
    BROAD_MARKET_KEYWORDS = [
        "stock market",
        "financial markets",
        "Wall Street",
        "market outlook",
        "bull market",
        "bear market",
        "market volatility"
    ]

    # ✅ ADD THIS LINE - Cache for sector lookups
    _SECTOR_CACHE = {}

    # ✅ ADD THIS - Sector mapping to prevent API calls
    TICKER_SECTORS = {
        # Broad Market ETFs
        'VOO': 'Broad Market ETF',
        'SPY': 'Broad Market ETF',
        'QQQ': 'Technology ETF',
        'SOXX': 'Semiconductors ETF',
        'IWM': 'Small Cap ETF',
        'ARKX': 'Innovation ETF',
        'VTI': 'Broad Market ETF',
        'IWV': 'Broad Market ETF',
        'IJH': 'Mid Cap ETF',
        'MACRO': 'Market-Wide',
        
        # Individual stocks - prevents API calls
        'AAPL': 'Technology',
        'JPM': 'Financial Services',
        'UNH': 'Healthcare',
        'AMZN': 'Consumer Cyclical',
        'WMT': 'Consumer Defensive',
        'T': 'Communication Services',
        'BA': 'Industrials'
    }

    # Date range settings
    START_DATE = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    END_DATE = datetime.now().strftime('%Y-%m-%d')
    
    # Data directories
    DATA_DIR = 'data'
    CACHE_DIR = 'cache'
    
    @staticmethod
    def get_sector(ticker: str) -> str:
        """Get sector for any ticker with caching to prevent rate limits."""
        # Check manual mapping first
        if ticker in Config.TICKER_SECTORS:
            return Config.TICKER_SECTORS[ticker]
        
        # Check cache
        if ticker in Config._SECTOR_CACHE:
            return Config._SECTOR_CACHE[ticker]
        
        # Only fetch from yfinance if not in manual mapping or cache
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            sector = info.get('sector', None)
            if sector:
                Config._SECTOR_CACHE[ticker] = sector
                return sector
            
            industry = info.get('industry', None)
            if industry:
                Config._SECTOR_CACHE[ticker] = industry
                return industry
                
        except Exception as e:
            logging.debug(f"Could not fetch sector for {ticker}: {e}")
        
        # Cache 'Other' to avoid repeated failed lookups
        Config._SECTOR_CACHE[ticker] = 'Other'
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
