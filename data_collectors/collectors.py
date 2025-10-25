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
    etf_tickers = Config.BROAD_MARKET_ETFS 
    processor = DataProcessor()

    # Initialize collectors
    yf_collector = YFinanceCollector(tickers + etf_tickers)
    reddit_collector = RedditCollector()
    
    # Primary: StockNewsAPI, Backup: NewsAPI
    stock_news_collector = NewsCollector(source="stocknewsapi")
    newsapi_collector = NewsCollector(source="newsapi")

    # ========== STOCK DATA ==========
    logger.info("📊 Collecting stock data...")
    stock_data = yf_collector.get_stock_data()
    logger.info(f"✅ Stock data collected: {len(stock_data)} rows")

    # ========== REDDIT DATA ==========
    logger.info("📱 Collecting Reddit data...")
    reddit_posts = reddit_collector.collect_posts_last_month()
    logger.info(f"✅ Reddit posts collected: {len(reddit_posts)} posts")
    
    ticker_mentions = reddit_collector.search_tickers_last_month(tickers)
    logger.info(f"✅ Ticker mentions collected: {len(ticker_mentions)} mentions")
    
    broad_market_reddit_posts = reddit_collector.collect_broad_market_posts_last_month()
    logger.info(f"✅ Broad market Reddit posts collected: {len(broad_market_reddit_posts)} posts")

    # ========== ETF NEWS (Individual ETFs) ==========
    logger.info("📰 Collecting ETF-specific news...")
    etf_news = stock_news_collector.collect_etf_news(
        etf_tickers, 
        days_back=30, 
        max_results_per_etf=50
    )
    logger.info(f"✅ ETF news collected: {len(etf_news)} articles")
    
    if not etf_news.empty and 'category' in etf_news.columns:
        logger.info("📋 ETF news category breakdown:")
        for category, count in etf_news['category'].value_counts().items():
            logger.info(f"  • {category}: {count} articles")

    # ========== BROAD MARKET NEWS ==========
    logger.info("📰 Collecting broad market news...")
    
    # Primary: StockNewsAPI
    broad_market_news_stock = stock_news_collector.collect_financial_news(
        days_back=30, 
        max_results=500
    )
    logger.info(f"  ✅ StockNewsAPI returned: {len(broad_market_news_stock)} articles")
    
    # Backup: NewsAPI (optional, for diversity)
    try:
        broad_market_news_newsapi = newsapi_collector.collect_financial_news(max_results=100)
        logger.info(f"  ✅ NewsAPI backup returned: {len(broad_market_news_newsapi)} articles")
    except Exception as e:
        logger.warning(f"  ⚠️ NewsAPI backup failed: {e}")
        broad_market_news_newsapi = pd.DataFrame()
    
    # Combine broad market news sources
    broad_market_news = pd.concat(
        [broad_market_news_stock, broad_market_news_newsapi], 
        ignore_index=True
    )
    logger.info(f"📊 Total broad market news: {len(broad_market_news)} articles")

    # ========== TICKER-SPECIFIC NEWS ==========
    logger.info("📰 Collecting ticker-specific news...")
    ticker_news = stock_news_collector.collect_ticker_news(
        tickers, 
        max_results=50
    )
    logger.info(f"✅ Ticker news collected: {len(ticker_news)} articles")

    # ========== COMBINE ALL FINANCIAL NEWS ==========
    financial_news = pd.concat([etf_news, broad_market_news], ignore_index=True)
    logger.info(f"📊 Total financial news (ETF + broad market): {len(financial_news)} articles")

    # ========== SAVE DATA ==========
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

    # Save financial news (ETF + broad market)
    if not financial_news.empty:
        financial_news_deduped = financial_news.drop_duplicates(subset=['url'], keep='first')
        logger.info(f"🔄 Removed {len(financial_news) - len(financial_news_deduped)} duplicate financial news articles")
        
        news_file = processor.save_data(financial_news_deduped, "financial_news")
        logger.info(f"📁 Saved financial news: {news_file} ({len(financial_news_deduped)} rows)")
    else:
        logger.warning("⚠️ No financial news collected")

    # Save ticker news
    if not ticker_news.empty:
        ticker_news_deduped = ticker_news.drop_duplicates(subset=['url'], keep='first')
        logger.info(f"🔄 Removed {len(ticker_news) - len(ticker_news_deduped)} duplicate ticker news articles")
        
        ticker_news_file = processor.save_data(ticker_news_deduped, "ticker_news")
        logger.info(f"📁 Saved ticker news: {ticker_news_file} ({len(ticker_news_deduped)} rows)")
    else:
        logger.warning("⚠️ No ticker-specific news collected")

    # ========== COMBINE FOR SENTIMENT ANALYSIS ==========
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
    
    # ========== FINAL SUMMARY ==========
    logger.info("✅ Data collection completed!")
    logger.info("="*60)
    logger.info("📊 COLLECTION SUMMARY:")
    logger.info(f"  • Stock data: {len(stock_data) if not stock_data.empty else 0} rows")
    logger.info(f"  • Reddit posts: {len(combined_reddit) if not combined_reddit.empty else 0} posts")
    logger.info(f"  • ETF news: {len(etf_news) if not etf_news.empty else 0} articles")
    logger.info(f"  • Broad market news: {len(broad_market_news) if not broad_market_news.empty else 0} articles")
    logger.info(f"  • Ticker news: {len(ticker_news) if not ticker_news.empty else 0} articles")
    logger.info(f"  • Total text entries: {len(text_data) if 'text_data' in locals() and not text_data.empty else 0}")
    logger.info("="*60)
