# ==================== FILE: OCRImage/components/resize.py ====================
"""
Image resizing module.
"""
import cv2
from OCRImage.exception.exception import OCRImageException
from OCRImage.logging import logger
import sys


def resize_image(image, width=None, height=None, scale=None):
    """
    Resize image to specified dimensions or scale.
    
    Parameters:
    image (numpy.ndarray): Input image
    width (int): Target width (optional)
    height (int): Target height (optional)
    scale (float): Scale factor (optional)
    
    Returns:
    numpy.ndarray: Resized image
    """
    try:
        if scale is not None:
            # Resize by scale factor
            new_width = int(image.shape[1] * scale)
            new_height = int(image.shape[0] * scale)
            resized_image = cv2.resize(image, (new_width, new_height))
            logger.logging.info(f"Image resized by scale factor {scale} to {new_width}x{new_height}")
        
        elif width is not None and height is not None:
            # Resize to specific dimensions
            resized_image = cv2.resize(image, (width, height))
            logger.logging.info(f"Image resized to {width}x{height}")
        
        elif width is not None:
            # Resize maintaining aspect ratio (width specified)
            aspect_ratio = image.shape[0] / image.shape[1]
            new_height = int(width * aspect_ratio)
            resized_image = cv2.resize(image, (width, new_height))
            logger.logging.info(f"Image resized to {width}x{new_height} (aspect ratio maintained)")
        
        elif height is not None:
            # Resize maintaining aspect ratio (height specified)
            aspect_ratio = image.shape[1] / image.shape[0]
            new_width = int(height * aspect_ratio)
            resized_image = cv2.resize(image, (new_width, height))
            logger.logging.info(f"Image resized to {new_width}x{height} (aspect ratio maintained)")
        
        else:
            raise ValueError("Must specify either width, height, or scale")
        
        return resized_image
        
    except Exception as e:
        logger.logging.error(f"Error in resize_image: {e}")
        raise OCRImageException(e, sys)