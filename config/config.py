import os
from datetime import datetime, timedelta
from dotenv import load_dotenv
import yfinance as yf
import logging


load_dotenv()


class Config:
    # API Keys
    NEWSAPI_KEY = userdata.get('NEWSAPI_KEY')
    REDDIT_CLIENT_ID = userdata.get('REDDIT_CLIENT_ID')
    REDDIT_CLIENT_SECRET = userdata.get('REDDIT_CLIENT_SECRET')
    STOCKNEWS_API_KEY = userdata.get('STOCKNEWS_API_KEY')
    REDDIT_USER_AGENT = 'Sentiment-Analysis-G1/1.0'
        

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

    # GICS 11 Sector ETFs (SPDR Select Sector ETFs)
    BROAD_MARKET_ETFS = [
        # Core Market ETFs
        "VOO",   # Vanguard S&P 500
        "SPY",   # SPDR S&P 500
        "QQQ",   # Invesco QQQ (Nasdaq 100)
        
        # GICS 11 Sectors (SPDR Select Sector ETFs)
        "XLK",   # Technology
        "XLF",   # Financials
        "XLV",   # Healthcare
        "XLC",   # Communication Services
        "XLE",   # Energy
        "XLY",   # Consumer Discretionary
        "XLI",   # Industrials
        "XLU",   # Utilities
        "XLP",   # Consumer Staples
        "XLRE",  # Real Estate
        "XLB"    # Materials
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

    # ✅ Cache for sector lookups
    _SECTOR_CACHE = {}

    # ✅ Sector mapping - ETFs only (prevents unnecessary API calls)
    TICKER_SECTORS = {
        # Core Market ETFs
        'VOO': 'Broad Market ETF',
        'SPY': 'Broad Market ETF',
        'QQQ': 'Technology ETF',
        'VTI': 'Broad Market ETF',
        'IWV': 'Broad Market ETF',
        'IWM': 'Small Cap ETF',
        'IJH': 'Mid Cap ETF',
        
        # GICS 11 Sector ETFs (SPDR)
        'XLK': 'Technology',
        'XLF': 'Financial Services',
        'XLV': 'Healthcare',
        'XLC': 'Communication Services',
        'XLE': 'Energy',
        'XLY': 'Consumer Discretionary',
        'XLI': 'Industrials',
        'XLU': 'Utilities',
        'XLP': 'Consumer Staples',
        'XLRE': 'Real Estate',
        'XLB': 'Materials',
        
        # Market-wide
        'MACRO': 'Market-Wide',
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
