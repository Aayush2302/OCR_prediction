"""
Grayscale conversion module.
"""
import cv2
from OCRImage.exception.exception import OCRImageException
from OCRImage.logging import logger
import sys


def convert_to_grayscale(image):
    """
    Convert the input image to grayscale.
    
    Parameters:
    image (numpy.ndarray): Input image in BGR format
    
    Returns:
    numpy.ndarray: Grayscale image
    """
    try:
        if len(image.shape) == 2:
            logger.logging.info("Image is already grayscale")
            return image
        
        grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        logger.logging.info("Image converted to grayscale")
        return grayscale_image
        
    except Exception as e:
        logger.logging.error(f"Error in convert_to_grayscale: {e}")
        raise OCRImageException(e, sys)

