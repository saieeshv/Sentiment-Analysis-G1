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
    EVENTREGISTRY_KEY = os.getenv('EVENT_REGISTRY_API_KEY')
    
    DEFAULT_TICKERS = [
        'AAPL',  
        'JPM',   
        'UNH',   
        'AMZN',  
        'WMT',   
        'T',     
        'BA',    
        'XOM',   
        'LIN',   
        'NEE',  
        'PLD'   
    ]

    BROAD_MARKET_ETFS = [
        "VTI", "SCHB", "IWV", "SPY", "VOO", "IVV",  # Broad market
        "QQQ",                                        # Tech-heavy
        "IWM", "IJH"                                 # Small/mid cap
    ]
    BROAD_MARKET_KEYWORDS = [
        "VTI", "SCHB", "IWV", "SPY", "QQQ", "VOO",
        "NASDAQ Composite", "S&P 500", "Dow Jones",
        "Vanguard Total Stock Market",
        "Schwab U.S. Broad Market",
        "iShares Russell 3000",
        "market sentiment", "market outlook",
        "bull market", "bear market", "economic indicators"
    ]

    TICKER_SECTORS = {
    # Individual Stocks (11 GICS Sectors)
    'AAPL': 'Information Technology',
    'MSFT': 'Information Technology',
    'GOOGL': 'Information Technology',
    'JPM': 'Financials',
    'UNH': 'Healthcare',
    'AMZN': 'Consumer Discretionary',
    'WMT': 'Consumer Staples',
    'T': 'Communication Services',
    'BA': 'Industrials',
    'XOM': 'Energy',
    'LIN': 'Materials',
    'NEE': 'Utilities',
    'PLD': 'Real Estate',
    
    # Broad Market ETFs
    'VTI': 'Broad Market ETF',
    'SCHB': 'Broad Market ETF',
    'IWV': 'Broad Market ETF',
    'SPY': 'Broad Market ETF',
    'VOO': 'Broad Market ETF',
    'IVV': 'Broad Market ETF',
    'QQQ': 'Technology ETF',
    'IWM': 'Small Cap ETF',
    'IJH': 'Mid Cap ETF',
     
    # Sector ETFs (SPDR Sector Select)
    'XLF': 'Financial Sector ETF',
    'XLK': 'Technology Sector ETF',
    'XLE': 'Energy Sector ETF',
    'XLV': 'Healthcare Sector ETF',
    'XLY': 'Consumer Discretionary Sector ETF',
    'XLP': 'Consumer Staples Sector ETF',
    'XLI': 'Industrial Sector ETF',
    'XLU': 'Utilities Sector ETF',
    'XLRE': 'Real Estate Sector ETF',
    
    # Special
    'MACRO': 'Market-Wide'
    }

    
    @staticmethod
    def get_sector(ticker: str) -> str:
        """Get sector for a ticker, default to 'Other' if not found"""
        return Config.TICKER_SECTORS.get(ticker, 'Other')
    
    # Reddit settings
    REDDIT_SUBREDDITS = ['GrowthStocks', 'investing', 'stocks', 'SecurityAnalysis', 'wallstreetbets']
    REDDIT_POST_LIMIT = 50
    
    # News settings
    NEWS_SOURCES = ['reuters', 'bloomberg', 'cnbc', 'marketwatch']
    NEWS_KEYWORDS = ['stock market', 'earnings', 'financial news']
    
    # Timeline settings
    DEFAULT_NEWS_DAYS_BACK = 1
    DEFAULT_REDDIT_TIME_FILTER = 'day'  # 'hour', 'day', 'week', 'month', 'year', 'all'
    
    # Breaking news settings (very recent)
    BREAKING_NEWS_HOURS = 6
    RECENT_REDDIT_HOURS = 12
    
    # Historical analysis
    MAX_DAYS_BACK = 30  # NewsAPI free plan limit
    
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
