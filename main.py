
# import logging
# from utils.api_tests import test_apis
# from data_collectors.collectors import collect_all_data

# logger = logging.getLogger(__name__)

# def main():
#     print("🤖 Financial Sentiment Analysis Pipeline")
#     print("=" * 50)
    
#     if not test_apis():
#         print("❌ API tests failed. Please check your .env file and API keys.")
#         return
    
#     try:
#         collect_all_data()
#         print("\n🎉 Setup complete! Your data collection is working.")
#         print("Next steps:")
#         print("1. Check the 'data' folder for collected CSV files")
#         print("2. Set up FinBERT for sentiment analysis")
#         print("3. Build the real-time processing pipeline")
#     except Exception as e:
#         logger.error(f"❌ Error during data collection: {e}")


# if __name__ == "__main__":
#     main()

import logging
from google.colab import drive
import os
from utils.api_tests import test_apis
from data_collectors.collectors import collect_all_data

logger = logging.getLogger(__name__)

def main():
    print("🤖 Financial Sentiment Analysis Pipeline")
    print("=" * 50)
    
    # Mount Google Drive
    print("\n📂 Mounting Google Drive...")
    drive.mount('/content/drive')
    
    # Setup output folder
    output_folder = '/content/drive/MyDrive/Outputs/Data'
    os.makedirs(output_folder, exist_ok=True)
    print(f"✅ Files will be saved to: IS484_FYP/Google Colab/Outputs/Data\n")
    
    if not test_apis():
        print("❌ API tests failed. Please check your .env file and API keys.")
        return
    
    try:
        # Pass the output folder to the data collector
        collect_all_data(output_folder=output_folder)
        print("\n🎉 Setup complete! Your data collection is working.")
        print("Next steps:")
        print("1. Check Google Drive > IS484_FYP > Google Colab > Outputs for CSV files")
        print("2. Set up FinBERT for sentiment analysis")
        print("3. Build the real-time processing pipeline")
    except Exception as e:
        logger.error(f"❌ Error during data collection: {e}")

if __name__ == "__main__":
    main()
