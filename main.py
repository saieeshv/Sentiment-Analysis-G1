
import logging
from utils.api_tests import test_apis
from data_collectors.collectors import collect_all_data

logger = logging.getLogger(__name__)

def main():
    print("🤖 Financial Sentiment Analysis Pipeline")
    print("=" * 50)
    
    if not test_apis():
        print("❌ API tests failed. Please check your .env file and API keys.")
        return
    
    try:
        collect_all_data()
        print("\n🎉 Setup complete! Your data collection is working.")
        print("Next steps:")
        print("1. Check the 'data' folder for collected CSV files")
        print("2. Set up FinBERT for sentiment analysis")
        print("3. Build the real-time processing pipeline")
    except Exception as e:
        logger.error(f"❌ Error during data collection: {e}")


if __name__ == "__main__":
    main()