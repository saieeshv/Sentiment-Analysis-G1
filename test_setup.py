import os
from config.config import Config

def test_setup():
    """Test your basic setup"""
    print("🔍 Testing setup...")
    
    # Check if .env file exists
    if os.path.exists('.env'):
        print("✅ .env file found")
    else:
        print("❌ .env file missing")
        return False
    
    # Check API keys
    try:
        Config.validate_config()
        print("✅ API keys configured")
    except Exception as e:
        print(f"❌ API key issue: {e}")
        return False
    
    # Check directories
    required_dirs = ['data', 'config', 'data_collectors', 'utils']
    for directory in required_dirs:
        if os.path.exists(directory):
            print(f"✅ {directory}/ directory exists")
        else:
            print(f"❌ {directory}/ directory missing")
    
    print("🎉 Basic setup looks good!")
    return True

if __name__ == "__main__":
    test_setup()
