"""
Edge detection filters for VideoVision app.
Includes Canny and Sobel edge detection filters.
"""

import cv2
import numpy as np
from typing import Optional, Tuple


def apply_canny(frame: np.ndarray, threshold1: int = 50, threshold2: int = 150) -> np.ndarray:
    """
    Apply Canny edge detection to the frame.
    
    Args:
        frame: Input frame in BGR format
        threshold1: Lower threshold for edge detection
        threshold2: Upper threshold for edge detection
        
    Returns:
        Edge detected frame
    """
    if frame is None:
        return None
        
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply Canny edge detection
    edges = cv2.Canny(blurred, threshold1, threshold2)
    
    # Convert back to 3-channel for consistency
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)


def apply_sobel(frame: np.ndarray, direction: str = 'both') -> np.ndarray:
    """
    Apply Sobel edge detection to the frame.
    
    Args:
        frame: Input frame in BGR format
        direction: Direction of edges to detect ('x', 'y', or 'both')
        
    Returns:
        Edge detected frame
    """
    if frame is None:
        return None
        
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    
    if direction == 'x':
        # Sobel X direction
        sobel = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    elif direction == 'y':
        # Sobel Y direction
        sobel = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    else:
        # Sobel both directions
        sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
        sobel = np.sqrt(sobel_x**2 + sobel_y**2)
    
    # Convert to absolute values and scale to 0-255
    sobel = np.absolute(sobel)
    sobel = np.uint8(sobel / sobel.max() * 255)
    
    # Convert back to 3-channel for consistency
    return cv2.cvtColor(sobel, cv2.COLOR_GRAY2BGR)


def apply_laplacian(frame: np.ndarray) -> np.ndarray:
    """
    Apply Laplacian edge detection to the frame.
    
    Args:
        frame: Input frame in BGR format
        
    Returns:
        Edge detected frame
    """
    if frame is None:
        return None
        
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # Apply Laplacian edge detection
    laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
    
    # Convert to absolute values and scale to 0-255
    laplacian = np.absolute(laplacian)
    laplacian = np.uint8(laplacian / laplacian.max() * 255)
    
    # Convert back to 3-channel for consistency
    return cv2.cvtColor(laplacian, cv2.COLOR_GRAY2BGR)


def apply_prewitt(frame: np.ndarray) -> np.ndarray:
    """
    Apply Prewitt edge detection to the frame.
    
    Args:
        frame: Input frame in BGR format
        
    Returns:
        Edge detected frame
    """
    if frame is None:
        return None
        
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # Define Prewitt kernels
    kernel_x = np.array([[-1, 0, 1],
                         [-1, 0, 1],
                         [-1, 0, 1]])
    
    kernel_y = np.array([[-1, -1, -1],
                         [ 0,  0,  0],
                         [ 1,  1,  1]])
    
    # Apply Prewitt filters
    prewitt_x = cv2.filter2D(blurred, cv2.CV_64F, kernel_x)
    prewitt_y = cv2.filter2D(blurred, cv2.CV_64F, kernel_y)
    
    # Combine gradients
    prewitt = np.sqrt(prewitt_x**2 + prewitt_y**2)
    
    # Convert to absolute values and scale to 0-255
    prewitt = np.absolute(prewitt)
    prewitt = np.uint8(prewitt / prewitt.max() * 255)
    
    # Convert back to 3-channel for consistency
    return cv2.cvtColor(prewitt, cv2.COLOR_GRAY2BGR)


def apply_roberts(frame: np.ndarray) -> np.ndarray:
    """
    Apply Roberts edge detection to the frame.
    
    Args:
        frame: Input frame in BGR format
        
    Returns:
        Edge detected frame
    """
    if frame is None:
        return None
        
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # Define Roberts kernels
    kernel_x = np.array([[1, 0],
                         [0, -1]])
    
    kernel_y = np.array([[0, 1],
                         [-1, 0]])
    
    # Apply Roberts filters
    roberts_x = cv2.filter2D(blurred, cv2.CV_64F, kernel_x)
    roberts_y = cv2.filter2D(blurred, cv2.CV_64F, kernel_y)
    
    # Combine gradients
    roberts = np.sqrt(roberts_x**2 + roberts_y**2)
    
    # Convert to absolute values and scale to 0-255
    roberts = np.absolute(roberts)
    roberts = np.uint8(roberts / roberts.max() * 255)
    
    # Convert back to 3-channel for consistency
    return cv2.cvtColor(roberts, cv2.COLOR_GRAY2BGR)


def apply_edge_enhancement(frame: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """
    Apply edge enhancement to the frame.
    
    Args:
        frame: Input frame in BGR format
        strength: Strength of edge enhancement (0.0 to 2.0)
        
    Returns:
        Edge enhanced frame
    """
    if frame is None:
        return None
        
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply Laplacian for edge detection
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    
    # Convert back to uint8
    laplacian = np.uint8(np.absolute(laplacian))
    
    # Enhance edges
    enhanced = cv2.addWeighted(gray, 1.0, laplacian, strength, 0)
    
    # Convert back to 3-channel for consistency
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)


# def apply_hough_transform(frame: np.ndarray, threshold: int = 100, min_line_length: int = 50, max_line_gap: int = 10) -> np.ndarray:
#     """
#     Apply Hough line transform to detect lines in the frame.
    
#     Args:
#         frame: Input frame in BGR format
#         threshold: Accumulator threshold for line detection
#         min_line_length: Minimum line length
#         max_line_gap: Maximum gap between line segments
        
#     Returns:
#         Frame with detected lines drawn
#     """
#     if frame is None:
#         return None
        
#     # Convert to grayscale
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
#     # Apply edge detection
#     edges = cv2.Canny(gray, 50, 150)
    
#     # Apply Hough line transform
#     lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold, 
#                         minLineLength=min_line_length, maxLineGap=max_line_gap)

    
#     # Create output image
#     result = frame.copy()
    
#     # Draw lines if any are detected
#     if lines is not None:
#         for line in lines:
#             x1, y1, x2, y2 = line[0]
#             cv2.line(result, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
#     return result
def apply_hough_transform(frame: np.ndarray, threshold: int = 120, min_line_length: int = 80, max_line_gap: int = 5) -> np.ndarray:
    if frame is None:
        return None

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.5)

    edges = cv2.Canny(blurred, 50, 150)

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold,
                            minLineLength=min_line_length, maxLineGap=max_line_gap)

    result = frame.copy()
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(result, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return result



def apply_dilation(frame: np.ndarray, kernel_size: int = 5, iterations: int = 1) -> np.ndarray:
    """
    Apply morphological dilation to the frame.
    
    Args:
        frame: Input frame in BGR format
        kernel_size: Size of the dilation kernel
        iterations: Number of iterations
        
    Returns:
        Dilated frame
    """
    if frame is None:
        return None
        
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Create kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    # Apply dilation
    dilated = cv2.dilate(gray, kernel, iterations=iterations)
    
    # Convert back to 3-channel for consistency
    return cv2.cvtColor(dilated, cv2.COLOR_GRAY2BGR)


def apply_erosion(frame: np.ndarray, kernel_size: int = 5, iterations: int = 1) -> np.ndarray:
    """
    Apply morphological erosion to the frame.
    
    Args:
        frame: Input frame in BGR format
        kernel_size: Size of the erosion kernel
        iterations: Number of iterations
        
    Returns:
        Eroded frame
    """
    if frame is None:
        return None
        
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Create kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    # Apply erosion
    eroded = cv2.erode(gray, kernel, iterations=iterations)
    
    # Convert back to 3-channel for consistency
    return cv2.cvtColor(eroded, cv2.COLOR_GRAY2BGR)


def apply_morphological_edges(frame: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    Apply morphological edge detection to the frame.
    
    Args:
        frame: Input frame in BGR format
        kernel_size: Size of the morphological kernel
        
    Returns:
        Morphological edge detected frame
    """
    if frame is None:
        return None
        
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Create kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    # Apply morphological operations
    gradient = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
    
    # Convert back to 3-channel for consistency
    return cv2.cvtColor(gradient, cv2.COLOR_GRAY2BGR)

