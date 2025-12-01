"""
CLAHE (Contrast Limited Adaptive Histogram Equalization) module.
"""
import cv2
from OCRImage.exception.exception import OCRImageException
from OCRImage.logging import logger
from OCRImage.constant.config import PreProcessingConfig as cfg
import sys


def apply_clahe(image, clip_limit=None, tile_grid_size=None):
    """
    Apply CLAHE to enhance image contrast.
    
    Parameters:
    image (numpy.ndarray): Input grayscale image
    clip_limit (float): Threshold for contrast limiting (default: from config)
    tile_grid_size (int): Size of grid for histogram equalization (default: from config)
    
    Returns:
    numpy.ndarray: CLAHE-enhanced image
    """
    try:
        clip_limit = clip_limit or cfg.CLAHE_CLIP_LIMIT
        tile_grid_size = tile_grid_size or cfg.CLAHE_TILE_GRID_SIZE
        
        clahe = cv2.createCLAHE(
            clipLimit=clip_limit,
            tileGridSize=(tile_grid_size, tile_grid_size)
        )
        
        clahe_image = clahe.apply(image)
        logger.logging.info(f"CLAHE applied with clip_limit={clip_limit}, tile_grid_size={tile_grid_size}")
        
        return clahe_image
        
    except Exception as e:
        logger.logging.error(f"Error in apply_clahe: {e}")
        raise OCRImageException(e, sys)
