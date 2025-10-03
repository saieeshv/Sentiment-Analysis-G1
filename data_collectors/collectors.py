import logging
import pandas as pd
from config.config import Config
# from data_collectors.yfinance_collector import YFinanceCollector
from data_collectors.reddit_collector import RedditCollector
# from data_collectors.news_collector import NewsCollector
from utils.data_processor import DataProcessor

logger = logging.getLogger(__name__)

def collect_all_data():
    """Main data collection function"""
    logger.info("🚀 Starting data collection...")

    tickers = Config.DEFAULT_TICKERS
    processor = DataProcessor()

    # Initialize collectors
    # yf_collector = YFinanceCollector(tickers)
    reddit_collector = RedditCollector()
    
    # Two separate NewsCollector instances
    # news_collector_newsapi = NewsCollector(source="newsapi")
    # news_collector_er = NewsCollector(source="eventregistry")

    # Collect data
    logger.info("📊 Collecting stock data...")
    # stock_data = yf_collector.get_stock_data()
    # company_news = yf_collector.get_company_news()

    logger.info("📱 Collecting Reddit data...")
    reddit_posts = reddit_collector.collect_posts_last_month()
    ticker_mentions = reddit_collector.search_tickers_last_month(tickers)

    # logger.info("📰 Collecting news data...")
    # financial_news_newsapi = news_collector_newsapi.collect_financial_news()
    # financial_news_er = news_collector_er.collect_financial_news(max_results=1000)
    # financial_news = pd.concat([financial_news_newsapi, financial_news_er], ignore_index=True)

    # logger.info("📰 Collecting ticker news data...")
    # ticker_news_newsapi = news_collector_newsapi.collect_ticker_news(tickers)
    # ticker_news_er = news_collector_er.collect_ticker_news(tickers, max_results=500)
    # ticker_news = pd.concat([ticker_news_newsapi, ticker_news_er], ignore_index=True)

    # Save data
    logger.info("💾 Saving collected data...")

    if not reddit_posts.empty:
        reddit_file = processor.save_data(reddit_posts, "reddit_posts")
        logger.info(f"📁 Saved Reddit posts: {reddit_file}")

    # if not ticker_mentions.empty:
    #     mentions_file = processor.save_data(ticker_mentions, "ticker_mentions")
    #     logger.info(f"📁 Saved ticker mentions: {mentions_file}")

    # if not financial_news.empty:
    #     news_file = processor.save_data(financial_news, "financial_news")
    #     logger.info(f"📁 Saved financial news (NewsAPI + Event Registry): {news_file}")

    # if not ticker_news.empty:
    #     ticker_news_file = processor.save_data(ticker_news, "ticker_news")
    #     logger.info(f"📁 Saved ticker news (NewsAPI + Event Registry): {ticker_news_file}")

    # # Combine for sentiment analysis
    # combined_reddit = (
    #     pd.concat([reddit_posts, ticker_mentions], ignore_index=True)
    #     if not reddit_posts.empty and not ticker_mentions.empty
    #     else reddit_posts if not reddit_posts.empty else ticker_mentions
    # )

    # combined_news = (
    #     pd.concat([financial_news, ticker_news], ignore_index=True)
    #     if not financial_news.empty and not ticker_news.empty
    #     else financial_news if not financial_news.empty else ticker_news
    # )

    # if not combined_reddit.empty or not combined_news.empty:
    #     text_data = processor.combine_text_data(combined_reddit, combined_news)
    #     if not text_data.empty:
    #         text_file = processor.save_data(text_data, "combined_text_data")
    #         logger.info(f"📁 Saved combined text data: {text_file}")
    #         logger.info(f"📈 Total text samples for sentiment analysis: {len(text_data)}")

    logger.info("✅ Data collection completed!")
