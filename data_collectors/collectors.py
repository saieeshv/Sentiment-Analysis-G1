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
    company_news = yf_collector.get_company_news()

    logger.info("📱 Collecting Reddit data...")
    reddit_posts = reddit_collector.collect_posts_last_month()
    ticker_mentions = reddit_collector.search_tickers_last_month(tickers)
    broad_market_reddit_posts = reddit_collector.collect_broad_market_posts_last_month()

    logger.info("📰 Collecting news data...")
    financial_news_newsapi = news_collector_newsapi.collect_financial_news()
    financial_news_er = news_collector_er.collect_financial_news(max_results=100)
    financial_news = pd.concat([financial_news_newsapi, financial_news_er], ignore_index=True)

    logger.info("📰 Collecting ticker news data...")
    ticker_news_newsapi = news_collector_newsapi.collect_ticker_news(tickers)
    ticker_news_er = news_collector_er.collect_ticker_news(tickers, max_results=10)
    ticker_news = pd.concat([ticker_news_newsapi, ticker_news_er], ignore_index=True)


    # Save data
    logger.info("💾 Saving collected data...")

    if not stock_data.empty:
        stock_file = processor.save_data(stock_data, "stock_data")
        logger.info(f"📁 Saved stock data: {stock_file}")

    if not company_news.empty:
        company_news_file = processor.save_data(company_news, "company_news")
        logger.info(f"📁 Saved company news: {company_news_file}")

    if not reddit_posts.empty:
        reddit_file = processor.save_data(reddit_posts, "reddit_posts")
        logger.info(f"📁 Saved Reddit posts: {reddit_file}")

    if not ticker_mentions.empty:
        mentions_file = processor.save_data(ticker_mentions, "ticker_mentions")
        logger.info(f"📁 Saved ticker mentions: {mentions_file}")
    
    if not broad_market_reddit_posts.empty:
        broad_reddit_file = processor.save_data(broad_market_reddit_posts, "broad_market_reddit_posts")
        logger.info(f"📁 Saved broad market Reddit posts: {broad_reddit_file}")

    if not financial_news.empty:
        news_file = processor.save_data(financial_news, "financial_news")
        logger.info(f"📁 Saved financial news (NewsAPI + Event Registry): {news_file}")

    if not ticker_news.empty:
        ticker_news_file = processor.save_data(ticker_news, "ticker_news")
        logger.info(f"📁 Saved ticker news (NewsAPI + Event Registry): {ticker_news_file}")
    


    # Combine for sentiment analysis
    combined_reddit = (
        pd.concat([reddit_posts, ticker_mentions, broad_market_reddit_posts], ignore_index=True)
        if not reddit_posts.empty and not ticker_mentions.empty and not broad_market_reddit_posts.empty
        else reddit_posts if not reddit_posts.empty else (ticker_mentions if not ticker_mentions.empty else broad_market_reddit_posts)
    )

    combined_news = (
        pd.concat([financial_news, ticker_news], ignore_index=True)
        if not financial_news.empty and not ticker_news.empty 
        else financial_news if not financial_news.empty 
        else ticker_news if not ticker_news.empty 
        else pd.DataFrame()
    )


    if not combined_reddit.empty or not combined_news.empty:
        text_data = processor.combine_text_data(combined_reddit, combined_news)
        
        if not text_data.empty:
            # ADD CATEGORY HERE ⬇️
            if 'ticker' in text_data.columns:
                text_data['category'] = text_data['ticker'].apply(lambda x: Config.get_sector(x) if pd.notna(x) else 'General')
            
            text_file = processor.save_data(text_data, "combined_text_data")
            logger.info(f"📁 Saved combined text: {text_file}")


    NewsCollector.clear_cache_dir("cache")  
    logger.info("✅ Data collection completed!")
