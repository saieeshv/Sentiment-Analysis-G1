import logging
from config.config import Config
from data_collectors.reddit_collector import RedditCollector
from data_collectors.news_collector import NewsCollector

logger = logging.getLogger(__name__)

def test_apis():
    """Test all API connections"""
    logger.info("🔍 Testing API connections...")

    # Test configuration
    try:
        Config.validate_config()
        logger.info("✅ Configuration is valid")
    except Exception as e:
        logger.error(f"❌ Configuration error: {e}")
        return False

    # Test Reddit API
    reddit_collector = RedditCollector()
    reddit_ok = reddit_collector.test_connection()

    # Test News API
    news_collector = NewsCollector()
    news_ok = news_collector.test_connection()

    return reddit_ok and news_ok
