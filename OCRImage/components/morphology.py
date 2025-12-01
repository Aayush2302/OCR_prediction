"""
Morphological operations module.
"""
import cv2
from OCRImage.exception.exception import OCRImageException
from OCRImage.logging import logger
from OCRImage.constant.config import PreProcessingConfig as cfg
import sys


def apply_morphology(image, method=None, kernel_size=None, iterations=1):
    """
    Apply morphological operations to binary image.
    
    Parameters:
    image (numpy.ndarray): Input binary image
    method (str): Morphology method ('dilation', 'erosion', 'opening', 'closing')
    kernel_size (int): Size of structuring element
    iterations (int): Number of times to apply operation
    
    Returns:
    numpy.ndarray: Morphologically processed image
    """
    try:
        method = method or cfg.MORPHOLOGY_METHOD
        method = method.lower()
        
        kernel_size = kernel_size or cfg.MORPHOLOGY_KERNEL_SIZE
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (kernel_size, kernel_size)
        )
        
        if method == 'dilation':
            morph_image = cv2.dilate(image, kernel, iterations=iterations)
            logger.logging.info(f"Dilation applied with kernel_size={kernel_size}, iterations={iterations}")
        
        elif method == 'erosion':
            morph_image = cv2.erode(image, kernel, iterations=iterations)
            logger.logging.info(f"Erosion applied with kernel_size={kernel_size}, iterations={iterations}")
        
        elif method == 'opening':
            morph_image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
            logger.logging.info(f"Opening applied with kernel_size={kernel_size}")
        
        elif method == 'closing':
            morph_image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
            logger.logging.info(f"Closing applied with kernel_size={kernel_size}")
        
        else:
            raise ValueError(f"Unsupported morphology method: {method}")
        
        return morph_image
        
    except Exception as e:
        logger.logging.error(f"Error in apply_morphology: {e}")
        raise OCRImageException(e, sys)
