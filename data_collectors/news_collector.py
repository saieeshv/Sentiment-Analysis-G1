import requests
import pandas as pd
from typing import List, Optional
from datetime import datetime, timedelta
import logging
from config.config import Config
from newspaper import Article
from bs4 import BeautifulSoup
import re
import nltk

# Ensure required NLTK corpora are downloaded
nltk.download('punkt')
nltk.download('punkt_tab')

class NewsCollector:
    def __init__(self):
        self.api_key = Config.NEWSAPI_KEY
        self.base_url = "https://newsapi.org/v2"
        self.logger = logging.getLogger(__name__)

    def test_connection(self):
        """Test NewsAPI connection"""
        url = f"{self.base_url}/top-headlines"
        params = {'apiKey': self.api_key, 'country': 'us', 'pageSize': 1}
        try:
            response = requests.get(url, params=params)
            if response.status_code == 200:
                self.logger.info("✅ NewsAPI connection successful")
                return True
            else:
                self.logger.error(f"❌ NewsAPI returned status: {response.status_code}")
                return False
        except Exception as e:
            self.logger.error(f"❌ NewsAPI connection failed: {str(e)}")
            return False

    def _fetch_summary(self, url: str) -> Optional[str]:
        """Fetch summary using Newspaper3k; fallback to None on failure"""
        try:
            article = Article(url)
            article.download()
            article.parse()
            article.nlp()
            return article.summary or article.text
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to fetch article summary from {url}: {str(e)}")
            return None

    def _clean_fallback_content(self, raw_text: str) -> str:
        """Clean NewsAPI fallback content for NLP"""
        if not raw_text:
            return ""

        # Remove HTML entities
        text = BeautifulSoup(raw_text, "html.parser").get_text()

        # Remove JavaScript snippets like { window.open(...) }
        text = re.sub(r'\{.*?\}', '', text)

        # Remove remaining HTML entities
        text = re.sub(r'&[a-z]+;', ' ', text)

        # Remove common boilerplate phrases
        boilerplate_patterns = [
            r'Read more', r'Click here', r'Subscribe', r'Advertisement', r'Related:', r'Continue reading'
        ]
        for pattern in boilerplate_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)

        # Remove multiple line breaks and normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def collect_financial_news(self, days_back: int = Config.DEFAULT_NEWS_DAYS_BACK) -> pd.DataFrame:
        """Collect general financial news with cleaned title + content"""
        # Clamp to MAX_DAYS_BACK because NewsAPI free plan only allows up to 30 days
        days_back = min(days_back, Config.MAX_DAYS_BACK)
        from_date = datetime.now() - timedelta(days=days_back)

        url = f"{self.base_url}/everything"
        params = {
            'apiKey': self.api_key,
            'q': 'stock market OR financial news OR earnings',
            'from': from_date.strftime('%Y-%m-%d'),
            'language': 'en',
            'sortBy': 'publishedAt',
            'pageSize': 100
        }
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            articles = response.json().get('articles', [])
            news_data = []

            for article in articles:
                content = self._fetch_summary(article.get('url'))
                if not content:
                    content = self._clean_fallback_content(article.get('content'))

                news_data.append({
                    'title': article.get('title'),
                    'content': content,
                    'source': article.get('source', {}).get('name'),
                    'published_at': article.get('publishedAt'),
                    'url': article.get('url'),
                    'collected_at': datetime.now()
                })

            self.logger.info(f"📰 Collected {len(news_data)} financial news articles from the past {days_back} days")
            return pd.DataFrame(news_data)

        except Exception as e:
            self.logger.error(f"❌ Error collecting financial news: {str(e)}")
            return pd.DataFrame()


    def collect_ticker_news(self, tickers: List[str]) -> pd.DataFrame:
        """Collect ticker-specific news with cleaned title + content"""
        all_news = []
        for ticker in tickers:
            url = f"{self.base_url}/everything"
            params = {
                'apiKey': self.api_key,
                'q': f'"{ticker}" OR "${ticker}"',
                'language': 'en',
                'sortBy': 'publishedAt',
                'pageSize': 20
            }
            try:
                response = requests.get(url, params=params)
                response.raise_for_status()
                articles = response.json().get('articles', [])

                for article in articles:
                    content = self._fetch_summary(article.get('url'))
                    if not content:
                        content = self._clean_fallback_content(article.get('content'))

                    all_news.append({
                        'ticker': ticker,
                        'title': article.get('title'),
                        'content': content,
                        'source': article.get('source', {}).get('name'),
                        'published_at': article.get('publishedAt'),
                        'url': article.get('url'),
                        'collected_at': datetime.now()
                    })

                self.logger.info(f"📰 Collected {len(articles)} articles for {ticker}")

            except Exception as e:
                self.logger.error(f"❌ Error collecting news for {ticker}: {str(e)}")

        return pd.DataFrame(all_news)
