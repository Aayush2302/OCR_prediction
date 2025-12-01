"""
Deskew module for image rotation correction.
"""
import cv2
import numpy as np
from OCRImage.exception.exception import OCRImageException
from OCRImage.logging import logger
import sys


def deskew_image(image):
    """
    Deskew the input image using minimum area rectangle.
    
    Parameters:
    image (numpy.ndarray): Input image (BGR or grayscale)
    
    Returns:
    numpy.ndarray: Deskewed image
    """
    try:
        # Convert to grayscale if needed
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Find non-zero coordinates
        coords = cv2.findNonZero(cv2.bitwise_not(gray))
        
        if coords is None:
            logger.logging.warning("No non-zero pixels found for deskewing")
            return image
        
        # Calculate rotation angle
        angle = cv2.minAreaRect(coords)[-1]
        angle = -(90 + angle) if angle < -45 else -angle
        
        # Rotate image
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        deskewed = cv2.warpAffine(
            image, M, (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )
        
        logger.logging.info(f"Image deskewed by {angle:.2f} degrees")
        return deskewed
        
    except Exception as e:
        logger.logging.error(f"Error in deskew_image: {e}")
        raise OCRImageException(e, sys)