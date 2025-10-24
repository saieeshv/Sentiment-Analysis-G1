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
    EVENTREGISTRY_KEY = os.getenv('EVENT_REGISTRY_API_KEY')
    
    # Data Collection Settings
    DEFAULT_NEWS_DAYS_BACK = 30
    DEFAULT_REDDIT_DAYS_BACK = 30
    MAX_DAYS_BACK = 365  # Maximum lookback period for news
    MAX_NEWS_RESULTS = 1000
    MAX_REDDIT_POSTS = 1000
    
    # Stock tickers to analyze
    DEFAULT_TICKERS = [
        'AAPL',  # Information Technology
        'JPM',
        'XOM',  # Energy
        'JNJ',  # Healthcare   


    ]

    # Broad Market ETFs
    BROAD_MARKET_ETFS = [
        "VOO", "SOXX", "IWM", "SPY", "QQQ", "VOO", "ARKX"                                
    ]
    
    BROAD_MARKET_KEYWORDS = [
    "market", "stocks", "economy", "inflation", "interest rates",
    "NASDAQ", "S&P 500", "Dow Jones", "ETF", "bull market", "bear market"
    ]

    # Sector mapping (manual for ETFs and special cases)
    TICKER_SECTORS = {
    # Broad Market ETFs
    'VOO': 'Broad Market ETF',
    'SOXX': 'Semiconductors ETF',
    'IWM': 'Small Cap ETF',
    'SPY': 'Broad Market ETF',
    'QQQ': 'Technology ETF',
    'ARKX': 'Innovation ETF',  # ARKX focuses on space innovation
    
    'VTI': 'Broad Market ETF',
    'IWV': 'Broad Market ETF',
    
    # Specialized ETFs
    'IJH': 'Mid Cap ETF',
    
    # Special
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
        """
        Automatically get sector for any ticker using yfinance.
        Falls back to manual mapping for ETFs.
        """
        # Check manual mapping first (for ETFs and special cases)
        if ticker in Config.TICKER_SECTORS:
            return Config.TICKER_SECTORS[ticker]
        
        # Fetch sector from yfinance for regular stocks
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Try to get sector
            sector = info.get('sector', None)
            if sector:
                return sector
            
            # Fallback to industry if sector not available
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
            'EVENTREGISTRY_KEY': Config.EVENTREGISTRY_KEY
        }
        
        missing_keys = [key for key, value in required_keys.items() if not value]
        
        if missing_keys:
            raise ValueError(f"Missing required API keys: {', '.join(missing_keys)}")
        
        return True
