"""
Thresholding module for binary image conversion.
"""
import cv2
import sys
from OCRImage.exception.exception import OCRImageException
from OCRImage.logging import logger
from OCRImage.constant.config import PreProcessingConfig as cfg


def apply_threshold(image, method=None, **kwargs):
    """
    Apply thresholding to convert image to binary.
    
    Parameters:
    image (numpy.ndarray): Input grayscale image
    method (str): Thresholding method ('binary', 'adaptive', 'otsu')
    **kwargs: Additional parameters for specific methods
    
    Returns:
    numpy.ndarray: Binary thresholded image
    """
    try:
        method = (method or cfg.THRESHOLD_METHOD).lower()
        
        if method == 'binary':
            threshold_value = kwargs.get('threshold_value', 127)
            _, thresh_image = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)
            logger.logging.info(f"Binary threshold applied with threshold={threshold_value}")

        elif method == 'adaptive':
            block_size = kwargs.get('block_size', cfg.ADAPTIVE_BLOCK_SIZE)
            C = kwargs.get('C', cfg.ADAPTIVE_C)

            thresh_image = cv2.adaptiveThreshold(
                image, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                block_size, C
            )
            logger.logging.info(f"Adaptive thresholding with block_size={block_size}, C={C}")

        elif method == 'otsu':
            _, thresh_image = cv2.threshold(
                image, 0, 255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            logger.logging.info("Otsu thresholding applied")

        else:
            raise ValueError(f"Unsupported thresholding method: {method}")
        
        return thresh_image

    except Exception as e:
        logger.logging.error(f"Error in thresholding: {e}")
        raise OCRImageException(e, sys)
