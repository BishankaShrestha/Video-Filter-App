"""
Basic image filters for VideoVision app.
Includes grayscale, blur, and sharpen filters.
"""

import cv2
import numpy as np
from typing import Optional


def apply_grayscale(frame: np.ndarray) -> np.ndarray:
    """
    Apply grayscale filter to the frame.
    
    Args:
        frame: Input frame in BGR format
        
    Returns:
        Grayscale frame
    """
    if frame is None:
        return None
        
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Convert back to 3-channel for consistency
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def apply_blur(frame: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """
    Apply Gaussian blur filter to the frame.
    
    Args:
        frame: Input frame in BGR format
        kernel_size: Size of the blur kernel (must be odd)
        
    Returns:
        Blurred frame
    """
    if frame is None:
        return None
        
    # Ensure kernel size is odd
    if kernel_size % 2 == 0:
        kernel_size += 1
        
    # Apply Gaussian blur
    return cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0)


def apply_sharpen(frame: np.ndarray, intensity: float = 1.0) -> np.ndarray:
    """
    Apply sharpen filter to the frame.
    
    Args:
        frame: Input frame in BGR format
        intensity: Intensity of the sharpen effect (0.0 to 2.0)
        
    Returns:
        Sharpened frame
    """
    if frame is None:
        return None
        
    # Define sharpen kernel
    kernel = np.array([[-1, -1, -1],
                       [-1,  9, -1],
                       [-1, -1, -1]])
    
    # Adjust kernel intensity
    kernel = kernel * intensity
    kernel[1, 1] = kernel[1, 1] + (1 - intensity) * 8
    
    # Apply the kernel
    sharpened = cv2.filter2D(frame, -1, kernel)
    
    # Ensure values are in valid range
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def apply_brightness(frame: np.ndarray, brightness: int = 0) -> np.ndarray:
    """
    Apply brightness adjustment to the frame.
    
    Args:
        frame: Input frame in BGR format
        brightness: Brightness adjustment (-100 to 100)
        
    Returns:
        Brightness adjusted frame
    """
    if frame is None:
        return None
        
    # Apply brightness
    adjusted = cv2.convertScaleAbs(frame, alpha=1.0, beta=brightness)
    
    return adjusted


def apply_contrast(frame: np.ndarray, contrast: float = 1.0) -> np.ndarray:
    """
    Apply contrast adjustment to the frame.
    
    Args:
        frame: Input frame in BGR format
        contrast: Contrast multiplier (0.5 to 2.0)
        
    Returns:
        Contrast adjusted frame
    """
    if frame is None:
        return None
        
    # Apply contrast
    adjusted = cv2.convertScaleAbs(frame, alpha=contrast, beta=0)
    
    return adjusted


def apply_saturation(frame: np.ndarray, saturation: float = 1.0) -> np.ndarray:
    """
    Apply saturation adjustment to the frame.
    
    Args:
        frame: Input frame in BGR format
        saturation: Saturation multiplier (0.0 to 2.0)
        
    Returns:
        Saturation adjusted frame
    """
    if frame is None:
        return None
        
    # Convert to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
    
    # Adjust saturation
    hsv[:, :, 1] = hsv[:, :, 1] * saturation
    
    # Ensure values are in valid range
    hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
    
    # Convert back to BGR
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def apply_sepia(frame: np.ndarray) -> np.ndarray:
    """
    Apply sepia tone filter to the frame.
    
    Args:
        frame: Input frame in BGR format
        
    Returns:
        Sepia toned frame
    """
    if frame is None:
        return None
        
    # Define sepia matrix
    sepia_matrix = np.array([[0.272, 0.534, 0.131],
                            [0.349, 0.686, 0.168],
                            [0.393, 0.769, 0.189]])
    
    # Apply sepia filter
    sepia = cv2.transform(frame, sepia_matrix)
    
    # Ensure values are in valid range
    return np.clip(sepia, 0, 255).astype(np.uint8)


def apply_vintage(frame: np.ndarray) -> np.ndarray:
    """
    Apply vintage filter to the frame.
    
    Args:
        frame: Input frame in BGR format
        
    Returns:
        Vintage filtered frame
    """
    if frame is None:
        return None
        
    # Convert to float for processing
    vintage = frame.astype(np.float32) / 255.0
    
    # Apply vintage effect
    vintage[:, :, 0] = vintage[:, :, 0] * 0.9  # Reduce blue
    vintage[:, :, 1] = vintage[:, :, 1] * 1.1  # Increase green slightly
    vintage[:, :, 2] = vintage[:, :, 2] * 1.2  # Increase red
    
    # Add slight blur for vintage look
    vintage = cv2.GaussianBlur(vintage, (3, 3), 0)
    
    # Convert back to uint8
    return np.clip(vintage * 255, 0, 255).astype(np.uint8)
