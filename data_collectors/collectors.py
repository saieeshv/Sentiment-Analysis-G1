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
    
    # Three separate NewsCollector instances
    news_collector_newsapi = NewsCollector(source="newsapi")
    news_collector_er = NewsCollector(source="eventregistry")
    bz_collector = NewsCollector(source="benzinga")

    # Collect data
    logger.info("📊 Collecting stock data...")
    stock_data = yf_collector.get_stock_data()
    logger.info(f"✅ Stock data collected: {len(stock_data)} rows")

    logger.info("📱 Collecting Reddit data...")
    reddit_posts = reddit_collector.collect_posts_last_month()
    logger.info(f"✅ Reddit posts collected: {len(reddit_posts)} posts")
    
    ticker_mentions = reddit_collector.search_tickers_last_month(tickers)
    logger.info(f"✅ Ticker mentions collected: {len(ticker_mentions)} mentions")
    
    broad_market_reddit_posts = reddit_collector.collect_broad_market_posts_last_month()
    logger.info(f"✅ Broad market Reddit posts collected: {len(broad_market_reddit_posts)} posts")

    # Collect broad market news with detailed logging
    logger.info("📰 Collecting broad market news...")
    logger.info("  🔍 Fetching from NewsAPI...")
    financial_news_newsapi = news_collector_newsapi.collect_financial_news()
    logger.info(f"  ✅ NewsAPI returned: {len(financial_news_newsapi)} articles")
    
    logger.info("  🔍 Fetching from Event Registry...")
    financial_news_er = news_collector_er.collect_financial_news(max_results=500)
    logger.info(f"  ✅ Event Registry returned: {len(financial_news_er)} articles")
    
    logger.info("  🔍 Fetching from Benzinga...")
    bz_df = bz_collector.collect_financial_news(days_back=7, max_results=500)
    logger.info(f"  ✅ Benzinga returned: {len(bz_df)} articles")
    
    # Combine financial news
    financial_news = pd.concat([financial_news_newsapi, financial_news_er, bz_df], ignore_index=True)
    logger.info(f"📊 Total broad market news (before dedup): {len(financial_news)} articles")
    
    # Log category breakdown if articles exist
    if not financial_news.empty and 'category' in financial_news.columns:
        logger.info("📋 Category breakdown:")
        for category, count in financial_news['category'].value_counts().items():
            logger.info(f"  • {category}: {count} articles")

    # Collect ticker-specific news with detailed logging
    logger.info("📰 Collecting ticker-specific news...")
    logger.info("  🔍 Fetching from NewsAPI...")
    ticker_news_newsapi = news_collector_newsapi.collect_ticker_news(tickers)
    logger.info(f"  ✅ NewsAPI returned: {len(ticker_news_newsapi)} ticker articles")
    
    logger.info("  🔍 Fetching from Event Registry...")
    ticker_news_er = news_collector_er.collect_ticker_news(tickers, max_results=50)
    logger.info(f"  ✅ Event Registry returned: {len(ticker_news_er)} ticker articles")
    
    ticker_news = pd.concat([ticker_news_newsapi, ticker_news_er], ignore_index=True)
    logger.info(f"📊 Total ticker news (before dedup): {len(ticker_news)} articles")

    # Save data
    logger.info("💾 Saving collected data...")

    # Save stock data
    if not stock_data.empty:
        stock_file = processor.save_data(stock_data, "stock_data")
        logger.info(f"📁 Saved stock data: {stock_file} ({len(stock_data)} rows)")
    else:
        logger.warning("⚠️ No stock data to save")

    # Save Reddit data
    if not reddit_posts.empty:
        reddit_file = processor.save_data(reddit_posts, "reddit_posts")
        logger.info(f"📁 Saved Reddit posts: {reddit_file} ({len(reddit_posts)} rows)")

    if not ticker_mentions.empty:
        mentions_file = processor.save_data(ticker_mentions, "ticker_mentions")
        logger.info(f"📁 Saved ticker mentions: {mentions_file} ({len(ticker_mentions)} rows)")
    
    if not broad_market_reddit_posts.empty:
        broad_reddit_file = processor.save_data(broad_market_reddit_posts, "broad_market_reddit_posts")
        logger.info(f"📁 Saved broad market Reddit posts: {broad_reddit_file} ({len(broad_market_reddit_posts)} rows)")

    # Save news data
    if not financial_news.empty:
        # Remove duplicates before saving
        financial_news_deduped = financial_news.drop_duplicates(subset=['url'], keep='first')
        logger.info(f"🔄 Removed {len(financial_news) - len(financial_news_deduped)} duplicate financial news articles")
        
        news_file = processor.save_data(financial_news_deduped, "financial_news")
        logger.info(f"📁 Saved financial news: {news_file} ({len(financial_news_deduped)} rows)")
    else:
        logger.warning("⚠️ No broad market financial news collected - check API keys/quota")

    if not ticker_news.empty:
        # Remove duplicates before saving
        ticker_news_deduped = ticker_news.drop_duplicates(subset=['url'], keep='first')
        logger.info(f"🔄 Removed {len(ticker_news) - len(ticker_news_deduped)} duplicate ticker news articles")
        
        ticker_news_file = processor.save_data(ticker_news_deduped, "ticker_news")
        logger.info(f"📁 Saved ticker news: {ticker_news_file} ({len(ticker_news_deduped)} rows)")
    else:
        logger.warning("⚠️ No ticker-specific news collected")

    # Combine for sentiment analysis
    logger.info("🔗 Combining data sources...")
    
    # Combine all Reddit data
    reddit_dataframes = [df for df in [reddit_posts, ticker_mentions, broad_market_reddit_posts] if not df.empty]
    combined_reddit = pd.concat(reddit_dataframes, ignore_index=True) if reddit_dataframes else pd.DataFrame()
    logger.info(f"📊 Combined Reddit data: {len(combined_reddit)} total entries")

    # Combine all news data
    news_dataframes = [df for df in [financial_news, ticker_news] if not df.empty]
    combined_news = pd.concat(news_dataframes, ignore_index=True) if news_dataframes else pd.DataFrame()
    logger.info(f"📊 Combined news data: {len(combined_news)} total articles")

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
            logger.info(f"📁 Saved combined text data: {text_file} ({len(text_data)} rows)")
            
            # Log final category distribution
            if 'category' in text_data.columns:
                logger.info("📋 Final category distribution:")
                for category, count in text_data['category'].value_counts().items():
                    logger.info(f"  • {category}: {count} entries")
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
    logger.info("="*60)
    logger.info("📊 COLLECTION SUMMARY:")
    logger.info(f"  • Stock data: {len(stock_data) if not stock_data.empty else 0} rows")
    logger.info(f"  • Reddit posts: {len(combined_reddit) if not combined_reddit.empty else 0} posts")
    logger.info(f"  • News articles: {len(combined_news) if not combined_news.empty else 0} articles")
    logger.info(f"  • Total text entries: {len(text_data) if not combined_reddit.empty or not combined_news.empty else 0}")
    logger.info("="*60)
