"""
Image preprocessing components.
"""
from .grayscale import convert_to_grayscale
from .deskew import deskew_image
from .clahe import apply_clahe
from .thresholding import apply_threshold
from .denoising import denoise_image
from .morphology import apply_morphology
from .resize import resize_image
from .orientation import auto_orient_image, detect_orientation, rotate_image  # NEW

__all__ = [
    'convert_to_grayscale',
    'deskew_image',
    'apply_clahe',
    'apply_threshold',
    'denoise_image',
    'apply_morphology',
    'resize_image',
    'auto_orient_image',  # NEW
    'detect_orientation',  # NEW
    'rotate_image'  # NEW
]