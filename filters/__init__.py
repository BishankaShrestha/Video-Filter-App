"""
Filters package for VideoVision app.
"""

from .basic_filters import (
    apply_grayscale,
    apply_blur,
    apply_sharpen,
    apply_brightness,
    apply_contrast,
    apply_saturation,
    apply_sepia,
    apply_vintage
)

from .edge_filters import (
    apply_canny,
    apply_sobel,
    apply_laplacian,
    apply_prewitt,
    apply_roberts,
    apply_edge_enhancement,
    apply_morphological_edges
)

__all__ = [
    # Basic filters
    'apply_grayscale',
    'apply_blur',
    'apply_sharpen',
    'apply_brightness',
    'apply_contrast',
    'apply_saturation',
    'apply_sepia',
    'apply_vintage',
    
    # Edge filters
    'apply_canny',
    'apply_sobel',
    'apply_laplacian',
    'apply_prewitt',
    'apply_roberts',
    'apply_edge_enhancement',
    'apply_morphological_edges'
]
