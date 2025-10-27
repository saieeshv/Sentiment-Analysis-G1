import logging
import pandas as pd
from datetime import datetime, timedelta, date 
from config.config import Config
from data_collectors.reddit_collector import RedditCollector
from data_collectors.news_collector import NewsCollector
from utils.data_processor import DataProcessor

logger = logging.getLogger(__name__)


def collect_all_data():
    """Main data collection function with per-ticker date range alignment"""
    logger.info("🚀 Starting data collection...")
    
    days_back = Config.DEFAULT_NEWS_DAYS_BACK
    tickers = Config.DEFAULT_TICKERS
    etf_tickers = Config.BROAD_MARKET_ETFS 
    processor = DataProcessor()

    # Initialize collectors
    stock_news_collector = NewsCollector(source="stocknewsapi")
    newsapi_collector = NewsCollector(source="newsapi")
    reddit_collector = RedditCollector()

    # ========== COLLECT NEWS FIRST ==========
    logger.info("📰 Collecting news data first to determine date ranges...")
    
    # Collect ETF news
    logger.info(f"📰 Collecting ETF-specific news...")
    etf_news = stock_news_collector.collect_etf_news_with_min_days(
    etf_tickers, 
    min_trading_days=40,
    max_results_per_ticker=200
    )
    logger.info(f"✅ ETF news collected: {len(etf_news)} articles")
    
    # Collect broad market news
    logger.info(f"📰 Collecting broad market news...")
    broad_market_news_stock = stock_news_collector.collect_financial_news(
        days_back=days_back,
        max_results=500
    )
    logger.info(f"  ✅ StockNewsAPI returned: {len(broad_market_news_stock)} articles")
    
    # Optional NewsAPI backup
    try:
        broad_market_news_newsapi = newsapi_collector.collect_financial_news(
            days_back=days_back,
            max_results=100
        )
        logger.info(f"  ✅ NewsAPI backup returned: {len(broad_market_news_newsapi)} articles")
    except Exception as e:
        logger.warning(f"  ⚠️ NewsAPI backup failed: {e}")
        broad_market_news_newsapi = pd.DataFrame()
    
    broad_market_news = pd.concat(
        [broad_market_news_stock, broad_market_news_newsapi], 
        ignore_index=True
    )
    
    # Collect ticker news
    logger.info(f"📰 Collecting ticker-specific news...")
    ticker_news = stock_news_collector.collect_ticker_news_with_min_days(
    tickers,
    min_trading_days=40,
    max_results_per_ticker=200
    )
    logger.info(f"✅ Ticker news collected: {len(ticker_news)} articles")

    # ========== DETERMINE PER-TICKER DATE RANGES ==========
    ticker_date_ranges = {}
    
    # Get date ranges from ticker-specific news
    if not ticker_news.empty:
        ticker_news['published_at'] = pd.to_datetime(
            ticker_news['published_at'], 
            format='mixed',
            utc=True
        )
        
        for ticker in tickers:
            ticker_articles = ticker_news[ticker_news['ticker'] == ticker]
            if not ticker_articles.empty:
                ticker_date_ranges[ticker] = {
                    'start': ticker_articles['published_at'].min().date(),
                    'end': ticker_articles['published_at'].max().date(),
                    'count': len(ticker_articles)
                }
                logger.info(f"📅 {ticker} news range: {ticker_date_ranges[ticker]['start']} to {ticker_date_ranges[ticker]['end']} ({ticker_date_ranges[ticker]['count']} articles)")
    
    # Get date ranges from ETF news
    if not etf_news.empty:
        etf_news['published_at'] = pd.to_datetime(
            etf_news['published_at'], 
            format='mixed',
            utc=True
        )
        
        for ticker in etf_tickers:
            ticker_articles = etf_news[etf_news['ticker'] == ticker]
            if not ticker_articles.empty:
                ticker_date_ranges[ticker] = {
                    'start': ticker_articles['published_at'].min().date(),
                    'end': ticker_articles['published_at'].max().date(),
                    'count': len(ticker_articles)
                }
                logger.info(f"📅 {ticker} news range: {ticker_date_ranges[ticker]['start']} to {ticker_date_ranges[ticker]['end']} ({ticker_date_ranges[ticker]['count']} articles)")
    
    # Fallback for tickers with no news
    default_end = datetime.now().date()
    default_start = (datetime.now() - timedelta(days=days_back)).date()
    
    for ticker in tickers + etf_tickers:
        if ticker not in ticker_date_ranges:
            ticker_date_ranges[ticker] = {
                'start': default_start,
                'end': default_end,
                'count': 0
            }
            logger.warning(f"⚠️ {ticker}: No news found, using default range")

    # ========== COLLECT STOCK DATA PER TICKER ==========
    logger.info(f"📊 Collecting stock data with per-ticker date ranges...")

    all_stock_data = []

    for ticker in tickers + etf_tickers:
        date_range = ticker_date_ranges[ticker]
        
        # Add retry logic with backoff
        max_retries = 3
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    import time
                    wait_time = 2 ** attempt  # 2s, 4s, 8s
                    logger.info(f"⏳ Waiting {wait_time}s before retry {attempt + 1}")
                    time.sleep(wait_time)
                
                logger.info(f"📈 Fetching {ticker} data from {date_range['start']} to {date_range['end']}")
                
                import yfinance as yf
                stock = yf.Ticker(ticker)
                hist = stock.history(
                    start=date_range['start'].strftime('%Y-%m-%d'),
                    end=date_range['end'].strftime('%Y-%m-%d'),
                    interval='1d'
                )
                
                if hist.empty:
                    if attempt < max_retries - 1:
                        logger.warning(f"⚠️ No data for {ticker}, retrying...")
                        continue
                    else:
                        logger.warning(f"⚠️ No price data for {ticker} after {max_retries} attempts")
                        break
                
                # Success!
                hist = hist.reset_index()
                hist['ticker'] = ticker
                hist['category'] = Config.get_sector(ticker)
                hist['news_count'] = date_range['count']
                
                all_stock_data.append(hist)
                logger.info(f"✅ Collected {len(hist)} price points for {ticker}")
                break  # Success, exit retry loop
                
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"⚠️ Error fetching {ticker} (attempt {attempt + 1}): {e}")
                else:
                    logger.error(f"❌ Failed to fetch {ticker} after {max_retries} attempts: {e}")

    # Combine all stock data
    if all_stock_data:
        stock_data = pd.concat(all_stock_data, ignore_index=True)
        stock_data.columns = stock_data.columns.str.lower()
        logger.info(f"📊 Total stock data: {len(stock_data)} rows across {len(tickers) + len(etf_tickers)} tickers")
    else:
        stock_data = pd.DataFrame()
        logger.warning("⚠️ No stock data collected")

    # ========== REDDIT DATA ==========
    logger.info("📱 Collecting Reddit data...")
    reddit_posts = reddit_collector.collect_posts_last_month()
    logger.info(f"✅ Reddit posts collected: {len(reddit_posts)} posts")
    
    ticker_mentions = reddit_collector.search_tickers_last_month(tickers)
    logger.info(f"✅ Ticker mentions collected: {len(ticker_mentions)} mentions")
    
    broad_market_reddit_posts = reddit_collector.collect_broad_market_posts_last_month()
    logger.info(f"✅ Broad market Reddit posts collected: {len(broad_market_reddit_posts)} posts")

    # ========== COMBINE FINANCIAL NEWS ==========
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

    # Save financial news
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
    logger.info("📊 COLLECTION SUMMARY (Per-Ticker Alignment):")
    
    for ticker in tickers + etf_tickers:
        if ticker in ticker_date_ranges:
            dr = ticker_date_ranges[ticker]
            stock_count = len(stock_data[stock_data['ticker'] == ticker]) if not stock_data.empty else 0
            logger.info(f"  • {ticker}: {dr['start']} to {dr['end']} | {dr['count']} news | {stock_count} price points")
    
    logger.info(f"\n  • Total stock data: {len(stock_data) if not stock_data.empty else 0} rows")
    logger.info(f"  • Total news articles: {len(combined_news) if not combined_news.empty else 0}")
    logger.info(f"  • Total text entries: {len(text_data) if 'text_data' in locals() and not text_data.empty else 0}")
    logger.info("="*60)
