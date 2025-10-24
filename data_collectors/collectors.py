import logging
import pandas as pd
from config.config import Config
from data_collectors.yfinance_collector import YFinanceCollector
from data_collectors.reddit_collector import RedditCollector
from data_collectors.news_collector import NewsCollector
from utils.data_processor import DataProcessor

logger = logging.getLogger(__name__)

def collect_all_data():
    """Main data collection function"""
    logger.info("🚀 Starting data collection...")

    tickers = Config.DEFAULT_TICKERS
    broad_market_tickers = Config.BROAD_MARKET_ETFS 
    processor = DataProcessor()

    # Initialize collectors
    yf_collector = YFinanceCollector(tickers + broad_market_tickers)
    reddit_collector = RedditCollector()
    
    # Two separate NewsCollector instances
    news_collector_newsapi = NewsCollector(source="newsapi")
    news_collector_er = NewsCollector(source="eventregistry")

    # Collect data
    logger.info("📊 Collecting stock data...")
    stock_data = yf_collector.get_stock_data()

    logger.info("📱 Collecting Reddit data...")
    reddit_posts = reddit_collector.collect_posts_last_month()
    ticker_mentions = reddit_collector.search_tickers_last_month(tickers)
    broad_market_reddit_posts = reddit_collector.collect_broad_market_posts_last_month()

    logger.info("📰 Collecting broad market news...")
    financial_news_newsapi = news_collector_newsapi.collect_financial_news()
    financial_news_er = news_collector_er.collect_financial_news(max_results=100)
    financial_news = pd.concat([financial_news_newsapi, financial_news_er], ignore_index=True)

    logger.info("📰 Collecting ticker-specific news...")
    ticker_news_newsapi = news_collector_newsapi.collect_ticker_news(tickers)
    ticker_news_er = news_collector_er.collect_ticker_news(tickers, max_results=10)
    ticker_news = pd.concat([ticker_news_newsapi, ticker_news_er], ignore_index=True)

    # Save data
    logger.info("💾 Saving collected data...")

    # Save stock data
    if not stock_data.empty:
        stock_file = processor.save_data(stock_data, "stock_data")
        logger.info(f"📁 Saved stock data: {stock_file}")
    else:
        logger.warning("⚠️ No stock data to save")

    # Save Reddit data
    if not reddit_posts.empty:
        reddit_file = processor.save_data(reddit_posts, "reddit_posts")
        logger.info(f"📁 Saved Reddit posts: {reddit_file}")

    if not ticker_mentions.empty:
        mentions_file = processor.save_data(ticker_mentions, "ticker_mentions")
        logger.info(f"📁 Saved ticker mentions: {mentions_file}")
    
    if not broad_market_reddit_posts.empty:
        broad_reddit_file = processor.save_data(broad_market_reddit_posts, "broad_market_reddit_posts")
        logger.info(f"📁 Saved broad market Reddit posts: {broad_reddit_file}")

    # Save news data
    if not financial_news.empty:
        news_file = processor.save_data(financial_news, "financial_news")
        logger.info(f"📁 Saved financial news (NewsAPI + Event Registry): {news_file}")
    else:
        logger.warning("⚠️ No broad market financial news collected - check API keys/quota")

    if not ticker_news.empty:
        ticker_news_file = processor.save_data(ticker_news, "ticker_news")
        logger.info(f"📁 Saved ticker news (NewsAPI + Event Registry): {ticker_news_file}")
    else:
        logger.warning("⚠️ No ticker-specific news collected")

    # Combine for sentiment analysis
    logger.info("🔗 Combining data sources...")
    
    # Combine all Reddit data
    reddit_dataframes = [df for df in [reddit_posts, ticker_mentions, broad_market_reddit_posts] if not df.empty]
    combined_reddit = pd.concat(reddit_dataframes, ignore_index=True) if reddit_dataframes else pd.DataFrame()

    # Combine all news data
    news_dataframes = [df for df in [financial_news, ticker_news] if not df.empty]
    combined_news = pd.concat(news_dataframes, ignore_index=True) if news_dataframes else pd.DataFrame()

    # Combine text data and add categories
    if not combined_reddit.empty or not combined_news.empty:
        text_data = processor.combine_text_data(combined_reddit, combined_news)
        
        if not text_data.empty:
            # Add category column
            if 'ticker' in text_data.columns:
                text_data['category'] = text_data['ticker'].apply(
                    lambda x: Config.get_sector(x) if pd.notna(x) else 'General'
                )
            elif 'tickers' in text_data.columns:
                # Handle posts with ticker lists
                text_data['category'] = text_data['tickers'].apply(
                    lambda x: Config.get_sector(x[0]) if isinstance(x, list) and len(x) > 0 else 'General'
                )
            else:
                text_data['category'] = 'General'
            
            text_file = processor.save_data(text_data, "combined_text_data")
            logger.info(f"📁 Saved combined text data: {text_file}")
        else:
            logger.warning("⚠️ Combined text data is empty")
    else:
        logger.warning("⚠️ No text data to combine")

    # Clean up cache
    try:
        NewsCollector.clear_cache_dir("cache")
        logger.info("🧹 Cleaned cache directory")
    except Exception as e:
        logger.warning(f"⚠️ Could not clean cache: {e}")
    
    logger.info("✅ Data collection completed!")
