import logging
import pandas as pd
import time
from datetime import datetime, timedelta, date 
from config.config import Config
from data_collectors.reddit_collector import RedditCollector
from data_collectors.news_collector import NewsCollector
from utils.data_processor import DataProcessor

logger = logging.getLogger(__name__)

def collect_all_data():
    logger.info('🚀 Starting data collection...')
    days_back = Config.DEFAULT_NEWS_DAYS_BACK
    tickers = Config.DEFAULT_TICKERS
    etf_tickers = Config.BROAD_MARKET_ETFS

    # Dynamically set archive directory name with today's date
    today_str = datetime.now().strftime('%Y%m%d')
    archive_dir = f'/content/drive/MyDrive/archive{today_str}_archive/raw_data'

    processor = DataProcessor(
        data_dir='/content/drive/MyDrive/Outputs',
        archive_dir=archive_dir
    )

    # Initialize collectors
    stock_news_collector = NewsCollector(source="stocknewsapi")
    reddit_collector = RedditCollector()
    


    # ========== COLLECT TRENDING STOCKS ==========
    logger.info("📈 Collecting trending stocks...")
    trending_stocks = stock_news_collector.collect_trending_stocks(
        date_range="last7days",
        min_mentions=50,
        max_results=20
    )


    if not trending_stocks.empty:
        trending_file = processor.save_data(trending_stocks, "trending_stocks")
        logger.info(f"📁 Saved trending stocks: {trending_file} ({len(trending_stocks)} stocks)")
        if len(trending_stocks) > 0:
            logger.info(f"   • Most mentioned: {trending_stocks.iloc[0]['ticker']} ({trending_stocks.iloc[0]['total_mentions']} mentions)")
    else:
        logger.warning("⚠️ No trending stocks collected")


    # ========== COLLECT ALL NEWS ARTICLES ==========
    logger.info("📰 Collecting all news articles from category endpoint...")
    all_news_articles = stock_news_collector.collect_all_news(
        items_per_page=100,
        max_pages=20,
        section="alltickers"
    )


    if not all_news_articles.empty:
        all_news_file = processor.save_data(all_news_articles, "all_news_articles")
        logger.info(f"📁 Saved all news articles: {all_news_file} ({len(all_news_articles)} articles)")
        logger.info(f"   • Unique sources: {all_news_articles['source_name'].nunique()}")
        logger.info(f"   • Date range: {all_news_articles['date'].min()} to {all_news_articles['date'].max()}")
    else:
        logger.warning("⚠️ No all news articles collected")


    # ========== COLLECT NEWS FOR CORRELATION ANALYSIS ==========
    logger.info("📰 Collecting news for correlation (MINIMUM 1 year AND 50 articles per ticker)...")
    
    # Collect ETF news
    logger.info(f"📰 Collecting ETF-specific news...")
    etf_news = stock_news_collector.collect_etf_news_for_correlation(
        etf_tickers, 
        min_articles=50,
        min_days_back=365
    )
    logger.info(f"✅ ETF news collected: {len(etf_news)} articles")
    
    # Collect broad market news
    logger.info(f"📰 Collecting broad market news...")
    broad_market_news = stock_news_collector.collect_financial_news(
        days_back=days_back,
        max_results=500
    )
    logger.info(f"  ✅ StockNewsAPI returned: {len(broad_market_news)} articles")
    
    # Collect ticker news
    logger.info(f"📰 Collecting ticker-specific news...")
    ticker_news = stock_news_collector.collect_ticker_news_for_correlation(
        tickers,
        min_articles=50,
        min_days_back=365
    )
    logger.info(f"✅ Ticker news collected: {len(ticker_news)} articles")


    # ========== DETERMINE PER-TICKER DATE RANGES FROM NEWS ==========
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
                date_span_days = (ticker_articles['published_at'].max() - ticker_articles['published_at'].min()).days
                ticker_date_ranges[ticker] = {
                    'start': ticker_articles['published_at'].min().date(),
                    'end': ticker_articles['published_at'].max().date(),
                    'count': len(ticker_articles),
                    'span_days': date_span_days
                }
                logger.info(
                    f"📅 {ticker} news range: {ticker_date_ranges[ticker]['start']} to "
                    f"{ticker_date_ranges[ticker]['end']} "
                    f"({ticker_date_ranges[ticker]['span_days']} days, {ticker_date_ranges[ticker]['count']} articles)"
                )
    
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
                date_span_days = (ticker_articles['published_at'].max() - ticker_articles['published_at'].min()).days
                ticker_date_ranges[ticker] = {
                    'start': ticker_articles['published_at'].min().date(),
                    'end': ticker_articles['published_at'].max().date(),
                    'count': len(ticker_articles),
                    'span_days': date_span_days
                }
                logger.info(
                    f"📅 {ticker} news range: {ticker_date_ranges[ticker]['start']} to "
                    f"{ticker_date_ranges[ticker]['end']} "
                    f"({ticker_date_ranges[ticker]['span_days']} days, {ticker_date_ranges[ticker]['count']} articles)"
                )
    
    # Fallback for tickers with no news (use 1 year default)
    default_end = datetime.now().date()
    default_start = (datetime.now() - timedelta(days=365)).date()
    
    for ticker in tickers + etf_tickers:
        if ticker not in ticker_date_ranges:
            ticker_date_ranges[ticker] = {
                'start': default_start,
                'end': default_end,
                'count': 0,
                'span_days': 365
            }
            logger.warning(f"⚠️ {ticker}: No news found, using 1-year default range")


    # ========== COLLECT STOCK DATA MATCHING NEWS DATE RANGES ==========
    logger.info(f"📊 Collecting stock data aligned with news date ranges...")
    all_stock_data = []


    for ticker in tickers + etf_tickers:
        date_range = ticker_date_ranges[ticker]
        
        # Add retry logic with backoff
        max_retries = 3
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    wait_time = 2 ** attempt  # 2s, 4s, 8s
                    logger.info(f"⏳ Waiting {wait_time}s before retry {attempt + 1}")
                    time.sleep(wait_time)
                
                logger.info(f"📈 Fetching {ticker} prices from {date_range['start']} to {date_range['end']}")
                
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
    
    try:
        reddit_posts = reddit_collector.collect_posts_last_month()
        logger.info(f"✅ Reddit posts collected: {len(reddit_posts)} posts")
    except Exception as e:
        logger.error(f"❌ Error collecting Reddit posts: {e}")
        reddit_posts = pd.DataFrame()
    
    try:
        ticker_mentions = reddit_collector.search_tickers_last_month(tickers)
        logger.info(f"✅ Ticker mentions collected: {len(ticker_mentions)} mentions")
    except Exception as e:
        logger.error(f"❌ Error collecting ticker mentions: {e}")
        ticker_mentions = pd.DataFrame()
    
    try:
        broad_market_reddit_posts = reddit_collector.collect_broad_market_posts_last_month()
        logger.info(f"✅ Broad market Reddit posts collected: {len(broad_market_reddit_posts)} posts")
    except Exception as e:
        logger.error(f"❌ Error collecting broad market Reddit posts: {e}")
        broad_market_reddit_posts = pd.DataFrame()


    # ========== COMBINE FINANCIAL NEWS ==========
    logger.info("🔗 Combining financial news sources...")
    financial_news_dfs = [df for df in [etf_news, broad_market_news] if not df.empty]
    financial_news = pd.concat(financial_news_dfs, ignore_index=True) if financial_news_dfs else pd.DataFrame()
    logger.info(f"📊 Total financial news (ETF + broad market): {len(financial_news)} articles")


    # ========== SAVE DATA ==========
    logger.info("💾 Saving collected data...")


    # Save stock data
    if not stock_data.empty:
        processor.save_data(stock_data, "stock_data")
        logger.info(f"📁 Saved: stock_data.csv ({len(stock_data)} rows)")
    else:
        logger.warning("⚠️ No stock data to save")


    # Save Reddit data
    if not reddit_posts.empty:
        processor.save_data(reddit_posts, "reddit_posts")
        logger.info(f"📁 Saved: reddit_posts.csv ({len(reddit_posts)} rows)")
    else:
        logger.warning("⚠️ No reddit posts to save")

    if not ticker_mentions.empty:
        processor.save_data(ticker_mentions, "ticker_mentions")
        logger.info(f"📁 Saved: ticker_mentions.csv ({len(ticker_mentions)} rows)")
    else:
        logger.warning("⚠️ No ticker mentions to save")
    
    if not broad_market_reddit_posts.empty:
        processor.save_data(broad_market_reddit_posts, "broad_market_reddit_posts")
        logger.info(f"📁 Saved: broad_market_reddit_posts.csv ({len(broad_market_reddit_posts)} rows)")
    else:
        logger.warning("⚠️ No broad market Reddit posts to save")


    # Save financial news
    if not financial_news.empty:
        financial_news_deduped = processor.clean_data(financial_news)
        logger.info(f"🔄 Deduped financial news: {len(financial_news_deduped)} articles")
        
        processor.save_data(financial_news_deduped, "financial_news")
        logger.info(f"📁 Saved: financial_news.csv ({len(financial_news_deduped)} rows)")
    else:
        logger.warning("⚠️ No financial news to save")


    # Save ticker news
    if not ticker_news.empty:
        ticker_news_deduped = processor.clean_data(ticker_news)
        logger.info(f"🔄 Deduped ticker news: {len(ticker_news_deduped)} articles")
        
        processor.save_data(ticker_news_deduped, "ticker_news")
        logger.info(f"📁 Saved: ticker_news.csv ({len(ticker_news_deduped)} rows)")
    else:
        logger.warning("⚠️ No ticker-specific news to save")


    # ========== COMBINE TEXT DATA FOR SENTIMENT ANALYSIS ==========
    logger.info("🔗 Combining data sources for sentiment analysis...")
    
    # Combine all Reddit data
    reddit_dataframes = [df for df in [reddit_posts, ticker_mentions, broad_market_reddit_posts] if not df.empty]
    combined_reddit = pd.concat(reddit_dataframes, ignore_index=True) if reddit_dataframes else pd.DataFrame()
    
    if not combined_reddit.empty:
        logger.info(f"📊 Combined Reddit data: {len(combined_reddit)} total entries")
    else:
        logger.warning("⚠️ No Reddit data to combine")


    # Combine all news data
    news_dataframes = [df for df in [financial_news, ticker_news] if not df.empty]
    combined_news = pd.concat(news_dataframes, ignore_index=True) if news_dataframes else pd.DataFrame()
    
    if not combined_news.empty:
        logger.info(f"📊 Combined news data: {len(combined_news)} total articles")
    else:
        logger.warning("⚠️ No news data to combine")


    # Combine text data and add categories
    if not combined_reddit.empty or not combined_news.empty:
        text_data = processor.combine_text_data(combined_reddit, combined_news)
        
        if not text_data.empty:
            # Clean duplicates
            text_data = processor.clean_data(text_data)
            logger.info(f"🧹 Cleaned text data: {len(text_data)} entries")
            
            # Add category column
            if 'ticker' in text_data.columns:
                text_data['category'] = text_data['ticker'].apply(
                    lambda x: Config.get_sector(x) if pd.notna(x) else 'General'
                )
            else:
                text_data['category'] = 'General'
            
            # Save combined text data
            processor.save_data(text_data, "combined_text_data")
            logger.info(f"📁 Saved: combined_text_data.csv ({len(text_data)} rows)")
            
            # Log final category distribution
            if 'category' in text_data.columns:
                logger.info("📋 Final category distribution:")
                for category, count in text_data['category'].value_counts().items():
                    logger.info(f"  • {category}: {count} entries")
        else:
            logger.warning("⚠️ Combined text data is empty after processing")
    else:
        logger.warning("⚠️ No text data to combine")


    # ========== FINAL SUMMARY ==========
    logger.info("✅ Data collection completed!")
    logger.info("="*70)
    logger.info("📊 COLLECTION SUMMARY (MINIMUM 1 Year + 50 Articles per Ticker):")
    logger.info("="*70)
    
    if not trending_stocks.empty:
        logger.info(f"  • Trending stocks: {len(trending_stocks)} stocks")
    
    if not all_news_articles.empty:
        logger.info(f"  • All news articles collected: {len(all_news_articles)} articles")
    
    logger.info(f"\n  📰 NEWS COLLECTION:")
    for ticker in sorted(tickers + etf_tickers):
        if ticker in ticker_date_ranges:
            dr = ticker_date_ranges[ticker]
            stock_count = len(stock_data[stock_data['ticker'] == ticker]) if not stock_data.empty else 0
            status = "✅" if dr['count'] >= 50 else "⚠️"
            logger.info(
                f"  {status} {ticker}: {dr['start']} to {dr['end']} ({dr['span_days']} days) | "
                f"{dr['count']} news | {stock_count} price points"
            )
    
    logger.info(f"\n  📊 DATA SUMMARY:")
    logger.info(f"  • Total stock data: {len(stock_data) if not stock_data.empty else 0} rows")
    logger.info(f"  • Total news articles: {len(combined_news) if not combined_news.empty else 0}")
    logger.info(f"  • Total Reddit entries: {len(combined_reddit) if not combined_reddit.empty else 0}")
    logger.info(f"  • Total text entries: {len(text_data) if 'text_data' in locals() and not text_data.empty else 0}")
    logger.info("="*70)


if __name__ == "__main__":
    collect_all_data()
