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
    def __init__(self, source: str = "newsapi"):
        """
        Initialize NewsCollector with a specific source.
        source: "newsapi" or "eventregistry"
        """
        self.source = source.lower()
        if self.source == "newsapi":
            self.api_key = Config.NEWSAPI_KEY
            self.base_url = "https://newsapi.org/v2"
        elif self.source == "eventregistry":
            self.api_key = Config.EVENTREGISTRY_KEY
            self.base_url = "https://eventregistry.org/api/v1/article/getArticles"
        else:
            raise ValueError("Unsupported source. Choose 'newsapi' or 'eventregistry'.")

        self.logger = logging.getLogger(__name__)

    def test_connection(self):
        """Test API connection"""
        if self.source == "newsapi":
            url = f"{self.base_url}/top-headlines"
            params = {'apiKey': self.api_key, 'country': 'us', 'pageSize': 1}
        else:
            url = self.base_url
            params = {'apiKey': self.api_key, 'keyword': 'test', 'articlesCount': 1}

        try:
            response = requests.get(url, params=params)
            if response.status_code == 200:
                self.logger.info(f"✅ {self.source} connection successful")
                return True
            else:
                self.logger.error(f"❌ {self.source} returned status: {response.status_code}")
                return False
        except Exception as e:
            self.logger.error(f"❌ {self.source} connection failed: {str(e)}")
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
        """Clean fallback content"""
        if not raw_text:
            return ""

        text = BeautifulSoup(raw_text, "html.parser").get_text()
        text = re.sub(r'\{.*?\}', '', text)
        text = re.sub(r'&[a-z]+;', ' ', text)

        boilerplate_patterns = [
            r'Read more', r'Click here', r'Subscribe',
            r'Advertisement', r'Related:', r'Continue reading'
        ]
        for pattern in boilerplate_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)

        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def collect_financial_news(self, days_back: int = Config.DEFAULT_NEWS_DAYS_BACK, max_results: int = 1000) -> pd.DataFrame:
        days_back = min(days_back, Config.MAX_DAYS_BACK)
        from_date = datetime.now() - timedelta(days=days_back)

        all_articles = []

        if self.source == "newsapi":
            # NewsAPI unchanged
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
                for article in articles:
                    content = self._fetch_summary(article.get('url')) or self._clean_fallback_content(article.get('content'))
                    all_articles.append({
                        'title': article.get('title'),
                        'content': content,
                        'source': article.get('source', {}).get('name'),
                        'published_at': article.get('publishedAt'),
                        'url': article.get('url'),
                        'collected_at': datetime.now()
                    })
                self.logger.info(f"📰 Collected {len(articles)} NewsAPI articles")
            except Exception as e:
                self.logger.error(f"❌ Error collecting financial news from NewsAPI: {str(e)}")

        elif self.source == "eventregistry":
            # Event Registry autopagination
            page = 1
            collected = 0
            while collected < max_results:
                params = {
                    "apiKey": self.api_key,
                    "keyword": "stock market OR financial news OR earnings",
                    "lang": "eng",
                    "resultType": "articles",
                    "articlesPage": page,
                    "articlesCount": min(100, max_results - collected),  # max 100 per page
                    "articlesSortBy": "date",
                    "includeArticleSentiment": True
                }
                try:
                    response = requests.get(self.base_url, params=params)
                    response.raise_for_status()
                    articles = response.json().get("articles", {}).get("results", [])
                    if not articles:
                        break  # no more pages

                    for article in articles:
                        content = self._fetch_summary(article.get('url')) or self._clean_fallback_content(article.get('body'))
                        all_articles.append({
                            'title': article.get('title'),
                            'content': content,
                            'source': article.get('source', {}).get('title'),
                            'published_at': article.get('dateTimePub'),
                            'url': article.get('url'),
                            'sentiment': article.get('sentiment'),
                            'collected_at': datetime.now()
                        })

                    collected += len(articles)
                    self.logger.info(f"📰 Collected {len(articles)} Event Registry articles from page {page}")
                    page += 1

                except Exception as e:
                    self.logger.error(f"❌ Error collecting financial news from Event Registry: {str(e)}")
                    break

        return pd.DataFrame(all_articles)


    def collect_ticker_news(self, tickers: List[str], max_results: int = 100) -> pd.DataFrame:
        """Collect ticker-specific news with Event Registry autopagination"""
        all_news = []

        for ticker in tickers:
            if self.source == "newsapi":
                url = f"{self.base_url}/everything"
                params = {
                    'apiKey': self.api_key,
                    'q': f'"{ticker}" OR "${ticker}"',
                    'language': 'en',
                    'sortBy': 'publishedAt',
                    'pageSize': 100
                }
                try:
                    response = requests.get(url, params=params)
                    response.raise_for_status()
                    articles = response.json().get('articles', [])
                    for article in articles:
                        content = self._fetch_summary(article.get('url')) or self._clean_fallback_content(article.get('content'))
                        all_news.append({
                            'ticker': ticker,
                            'title': article.get('title'),
                            'content': content,
                            'source': article.get('source', {}).get('name'),
                            'published_at': article.get('publishedAt'),
                            'url': article.get('url'),
                            'collected_at': datetime.now()
                        })
                    self.logger.info(f"📰 Collected {len(articles)} articles for {ticker} (NewsAPI)")
                except Exception as e:
                    self.logger.error(f"❌ Error collecting news for {ticker} (NewsAPI): {str(e)}")

            elif self.source == "eventregistry":
                page = 1
                collected = 0
                while collected < max_results:
                    params = {
                        "apiKey": self.api_key,
                        "keyword": ticker,
                        "lang": "eng",
                        "resultType": "articles",
                        "articlesPage": page,
                        "articlesCount": min(100, max_results - collected),
                        "articlesSortBy": "date",
                        "includeArticleSentiment": True
                    }
                    try:
                        response = requests.get(self.base_url, params=params)
                        response.raise_for_status()
                        articles = response.json().get("articles", {}).get("results", [])
                        if not articles:
                            break

                        for article in articles:
                            content = self._fetch_summary(article.get('url')) or self._clean_fallback_content(article.get('body'))
                            all_news.append({
                                'ticker': ticker,
                                'title': article.get('title'),
                                'content': content,
                                'source': article.get('source', {}).get('title'),
                                'published_at': article.get('dateTimePub'),
                                'url': article.get('url'),
                                'sentiment': article.get('sentiment'),
                                'collected_at': datetime.now()
                            })

                        collected += len(articles)
                        self.logger.info(f"📰 Collected {len(articles)} Event Registry articles for {ticker} from page {page}")
                        page += 1

                    except Exception as e:
                        self.logger.error(f"❌ Error collecting news for {ticker} (Event Registry): {str(e)}")
                        break

        return pd.DataFrame(all_news)
