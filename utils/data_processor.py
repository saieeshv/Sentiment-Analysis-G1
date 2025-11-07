import pandas as pd
import re
import os
import shutil
from datetime import datetime
import logging


class DataProcessor:
    def __init__(self, data_dir="data", archive_dir="data_archive"):
        """
        Initialize DataProcessor
        
        Args:
            data_dir: Directory to store current data files
            archive_dir: Directory to store archived files with timestamps
        """
        self.logger = logging.getLogger(__name__)
        self.data_dir = data_dir
        self.archive_dir = archive_dir
        
        # Create directories if they don't exist
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.archive_dir, exist_ok=True)
        
        self.logger.info(f"📁 Data directory: {self.data_dir}")
        self.logger.info(f"📁 Archive directory: {self.archive_dir}")
    
    
    def _archive_existing_file(self, filename: str) -> bool:
        """
        Move existing file to archive with timestamp
        
        Args:
            filename: Name of the file to archive
        
        Returns:
            bool: True if file was archived, False if file didn't exist
        """
        file_path = os.path.join(self.data_dir, filename)
        
        if not os.path.exists(file_path):
            return False
        
        # Generate timestamped filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name, ext = os.path.splitext(filename)
        archived_filename = f"{name}_{timestamp}{ext}"
        archived_path = os.path.join(self.archive_dir, archived_filename)
        
        # Move file to archive
        try:
            shutil.move(file_path, archived_path)
            self.logger.info(f"📦 Archived: {filename} → {archived_filename}")
            return True
        except Exception as e:
            self.logger.error(f"❌ Failed to archive {filename}: {e}")
            return False
    
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Clean text for sentiment analysis"""
        if not isinstance(text, str):
            return ""
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+', '', text)
        # Remove mentions and hashtags for now
        text = re.sub(r'@\w+|#\w+', '', text)
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text.strip()
    
    
    def save_data(self, df: pd.DataFrame, file_type: str) -> str:
        """
        Save DataFrame WITHOUT timestamp, archiving previous version if it exists
        
        Args:
            df: DataFrame to save
            file_type: Type of data (e.g., 'stock_data', 'reddit_posts', 'combined_text_data')
        
        Returns:
            str: Path to the saved file
        """
        if df.empty:
            self.logger.warning(f"⚠️ DataFrame for {file_type} is empty, skipping save")
            return ""
        
        # Generate filename WITHOUT timestamp
        filename = f"{file_type}.csv"
        file_path = os.path.join(self.data_dir, filename)
        
        # Archive existing file if it exists
        if self._archive_existing_file(filename):
            self.logger.info(f"🔄 Previous version archived before saving new {file_type} data")
        
        # Save new file
        try:
            df.to_csv(file_path, index=False)
            self.logger.info(f"✅ Saved {file_type}: {filename} ({len(df)} rows)")
            return file_path
        except Exception as e:
            self.logger.error(f"❌ Failed to save {file_type}: {e}")
            return ""
    
    
    def combine_text_data(self, reddit_df: pd.DataFrame, news_df: pd.DataFrame) -> pd.DataFrame:
        """
        Combine Reddit and news data for sentiment analysis
        
        Args:
            reddit_df: DataFrame with Reddit posts
            news_df: DataFrame with news articles
        
        Returns:
            pd.DataFrame: Combined standardized data
        """
        combined_data = []
        
        # Process Reddit data
        if not reddit_df.empty:
            reddit_processed = self._process_reddit_data(reddit_df)
            combined_data.append(reddit_processed)
            self.logger.info(f"📱 Processed {len(reddit_processed)} Reddit entries")
        
        # Process news data
        if not news_df.empty:
            news_processed = self._process_news_data(news_df)
            combined_data.append(news_processed)
            self.logger.info(f"📰 Processed {len(news_processed)} news entries")
        
        # Combine
        if combined_data:
            result = pd.concat(combined_data, ignore_index=True)
            self.logger.info(f"🔗 Combined into {len(result)} total entries")
            return result
        else:
            self.logger.warning("⚠️ No data to combine")
            return pd.DataFrame()
    
    
    def _process_reddit_data(self, reddit_df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize Reddit data structure
        
        Args:
            reddit_df: Raw Reddit data
        
        Returns:
            pd.DataFrame: Processed Reddit data with standardized columns
        """
        processed_data = []
        
        for _, row in reddit_df.iterrows():
            # Clean text from title and body/text
            title = self.clean_text(str(row.get('title', '')))
            body = self.clean_text(str(row.get('text', row.get('body', row.get('selftext', '')))))
            
            # Combine title and body
            full_text = f"{title} {body}".strip()
            
            if not full_text:
                continue
            
            # Get ticker info
            ticker = row.get('ticker')
            if isinstance(ticker, list) and len(ticker) > 0:
                ticker = ticker[0]
            ticker = str(ticker) if pd.notna(ticker) else 'general'
            
            # Parse date
            date_val = None
            if 'created_utc' in row:
                try:
                    date_val = pd.to_datetime(row['created_utc'], unit='s', utc=True)
                except:
                    date_val = pd.to_datetime(row.get('date', datetime.now()))
            elif 'date' in row:
                try:
                    date_val = pd.to_datetime(row['date'])
                except:
                    date_val = datetime.now()
            else:
                date_val = datetime.now()
            
            processed_data.append({
                'title': title,
                'text': full_text,
                'source': 'Reddit',
                'source_name': row.get('subreddit', 'unknown'),
                'ticker': ticker,
                'date': date_val,
                'url': row.get('url', ''),
                'score': row.get('score', 0),
                'num_comments': row.get('num_comments', 0)
            })
        
        return pd.DataFrame(processed_data)
    
    
    def _process_news_data(self, news_df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize news data structure
        
        Args:
            news_df: Raw news data
        
        Returns:
            pd.DataFrame: Processed news data with standardized columns
        """
        processed_data = []
        
        for _, row in news_df.iterrows():
            # Clean text from title and description/content
            title = self.clean_text(str(row.get('title', '')))
            description = self.clean_text(str(row.get('description', row.get('content', ''))))
            
            # Combine title and description
            full_text = f"{title} {description}".strip()
            
            if not full_text:
                continue
            
            # Get ticker info
            ticker = row.get('ticker')
            if isinstance(ticker, list) and len(ticker) > 0:
                ticker = ticker[0]
            ticker = str(ticker) if pd.notna(ticker) else 'general'
            
            # Get source name
            source_name = row.get('source_name', row.get('source', 'unknown'))
            
            # Parse date
            date_val = None
            if 'published_at' in row:
                try:
                    date_val = pd.to_datetime(row['published_at'])
                except:
                    date_val = pd.to_datetime(row.get('date', datetime.now()))
            elif 'date' in row:
                try:
                    date_val = pd.to_datetime(row['date'])
                except:
                    date_val = datetime.now()
            else:
                date_val = datetime.now()
            
            processed_data.append({
                'title': title,
                'text': full_text,
                'source': 'News',
                'source_name': source_name,
                'ticker': ticker,
                'date': date_val,
                'url': row.get('url', ''),
                'sentiment': row.get('sentiment', None)
            })
        
        return pd.DataFrame(processed_data)
    
    
    def clean_data(self, df: pd.DataFrame, remove_duplicates: bool = True, 
                   remove_empty: bool = True) -> pd.DataFrame:
        """
        Clean and deduplicate dataframe
        
        Args:
            df: DataFrame to clean
            remove_duplicates: Whether to remove duplicate rows
            remove_empty: Whether to remove rows with empty text
        
        Returns:
            pd.DataFrame: Cleaned dataframe
        """
        if df.empty:
            return df
        
        original_len = len(df)
        
        # Remove rows with empty text
        if remove_empty and 'text' in df.columns:
            df = df[df['text'].str.strip() != '']
        
        # Remove duplicates by URL if available
        if remove_duplicates and 'url' in df.columns:
            df = df.drop_duplicates(subset=['url'], keep='first')
        
        # Remove duplicates by text as fallback
        elif remove_duplicates and 'text' in df.columns:
            df = df.drop_duplicates(subset=['text'], keep='first')
        
        removed = original_len - len(df)
        if removed > 0:
            self.logger.info(f"🧹 Cleaned data: removed {removed} rows ({removed/original_len*100:.1f}%)")
        
        return df.reset_index(drop=True)
    
    
    def get_ticker_info(self, df: pd.DataFrame) -> dict:
        """
        Get statistics about ticker distribution in dataframe
        
        Args:
            df: DataFrame to analyze
        
        Returns:
            dict: Ticker statistics
        """
        if df.empty or 'ticker' not in df.columns:
            return {}
        
        ticker_counts = df['ticker'].value_counts().to_dict()
        return {
            'total_tickers': len(ticker_counts),
            'ticker_distribution': ticker_counts,
            'top_ticker': max(ticker_counts, key=ticker_counts.get) if ticker_counts else None,
            'top_ticker_count': max(ticker_counts.values()) if ticker_counts else 0
        }
    
    
    def validate_data(self, df: pd.DataFrame) -> dict:
        """
        Validate data quality and return report
        
        Args:
            df: DataFrame to validate
        
        Returns:
            dict: Validation report
        """
        report = {
            'total_rows': len(df),
            'null_values': df.isnull().sum().to_dict(),
            'columns': list(df.columns),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2
        }
        
        # Log validation
        self.logger.info(f"📊 Data validation report:")
        self.logger.info(f"  • Total rows: {report['total_rows']}")
        self.logger.info(f"  • Memory usage: {report['memory_usage_mb']:.2f} MB")
        
        if any(report['null_values'].values()):
            self.logger.warning(f"  ⚠️ Null values detected:")
            for col, count in report['null_values'].items():
                if count > 0:
                    self.logger.warning(f"    - {col}: {count} nulls")
        
        return report
