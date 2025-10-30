"""
Frame extraction and conversion utilities for VideoVision app.
"""

import cv2
import numpy as np
from typing import Tuple, Optional
import tempfile
import os


def extract_frame(video_path: str, frame_number: int) -> Optional[np.ndarray]:
    """
    Extract a specific frame from a video file.
    
    Args:
        video_path: Path to the video file
        frame_number: Frame number to extract (0-indexed)
        
    Returns:
        Extracted frame as numpy array (BGR format) or None if error
    """
    try:
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            return None
            
        # Set frame position
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            return frame
        else:
            return None
            
    except Exception as e:
        print(f"Error extracting frame: {e}")
        return None


def get_video_info(video_path: str) -> Tuple[int, int, int, float]:
    """
    Get video information including total frames, width, height, and fps.
    
    Args:
        video_path: Path to the video file
        
    Returns:
        Tuple of (total_frames, width, height, fps)
    """
    try:
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            return 0, 0, 0, 0.0
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        cap.release()
        
        return total_frames, width, height, fps
        
    except Exception as e:
        print(f"Error getting video info: {e}")
        return 0, 0, 0, 0.0


def bgr_to_rgb(frame: np.ndarray) -> np.ndarray:
    """
    Convert BGR frame to RGB for display in Streamlit.
    
    Args:
        frame: Frame in BGR format
        
    Returns:
        Frame in RGB format
    """
    if frame is None:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def rgb_to_bgr(frame: np.ndarray) -> np.ndarray:
    """
    Convert RGB frame to BGR for OpenCV processing.
    
    Args:
        frame: Frame in RGB format
        
    Returns:
        Frame in BGR format
    """
    if frame is None:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


def resize_frame(frame: np.ndarray, max_width: int = 800, max_height: int = 600) -> np.ndarray:
    """
    Resize frame while maintaining aspect ratio.
    
    Args:
        frame: Input frame
        max_width: Maximum width
        max_height: Maximum height
        
    Returns:
        Resized frame
    """
    if frame is None:
        return None
        
    height, width = frame.shape[:2]
    
    # Calculate scaling factor
    scale_w = max_width / width
    scale_h = max_height / height
    scale = min(scale_w, scale_h, 1.0)  # Don't upscale
    
    if scale < 1.0:
        new_width = int(width * scale)
        new_height = int(height * scale)
        return cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    return frame


def save_frame_as_temp_file(frame: np.ndarray, prefix: str = "frame") -> str:
    """
    Save frame as a temporary file and return the path.
    
    Args:
        frame: Frame to save
        prefix: Prefix for the temporary file
        
    Returns:
        Path to the temporary file
    """
    if frame is None:
        return None
        
    # Convert BGR to RGB for saving
    rgb_frame = bgr_to_rgb(frame)
    
    # Create temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg', prefix=prefix)
    temp_path = temp_file.name
    temp_file.close()
    
    # Save frame
    cv2.imwrite(temp_path, cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR))
    
    return temp_path


def cleanup_temp_file(file_path: str) -> None:
    """
    Clean up temporary file.
    
    Args:
        file_path: Path to the temporary file to delete
    """
    try:
        if file_path and os.path.exists(file_path):
            os.unlink(file_path)
    except Exception as e:
        print(f"Error cleaning up temp file: {e}")


def validate_video_file(uploaded_file) -> bool:
    """
    Validate if uploaded file is a valid video.
    
    Args:
        uploaded_file: Streamlit uploaded file object
        
    Returns:
        True if valid video, False otherwise
    """
    if uploaded_file is None:
        return False
        
    # Check file extension
    valid_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm']
    file_extension = os.path.splitext(uploaded_file.name)[1].lower()
    
    return file_extension in valid_extensions

