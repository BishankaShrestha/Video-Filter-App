#!/usr/bin/env python3
"""
Test script for VideoVision app to verify all modules can be imported correctly.
"""

import sys
import os

def test_imports():
    """Test if all modules can be imported correctly."""
    try:
        # Test utils imports
        from utils.frame_utils import extract_frame, get_video_info, bgr_to_rgb
        print("✅ Utils module imports successfully")
        
        # Test filters imports
        from filters.basic_filters import apply_grayscale, apply_blur, apply_sharpen
        print("✅ Basic filters module imports successfully")
        
        from filters.edge_filters import apply_canny, apply_sobel
        print("✅ Edge filters module imports successfully")
        
        # Test main app import
        import app
        print("✅ Main app module imports successfully")
        
        print("\n🎉 All modules imported successfully!")
        print("The VideoVision app is ready to run!")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please install the required packages:")
        print("pip install -r requirements.txt")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing VideoVision app imports...")
    print("=" * 50)
    
    success = test_imports()
    
    if success:
        print("\n🚀 To run the app, use:")
        print("streamlit run app.py")
    else:
        print("\n💡 Please fix the import issues before running the app.")

