"""
Denoising module to reduce noise in the image while preserving text edges.
"""
import cv2
import sys
from OCRImage.constant.config import PreProcessingConfig as cfg
from OCRImage.exception.exception import OCRImageException
from OCRImage.logging import logger


def denoise_image(image):
    """
    Apply appropriate denoising algorithm.

    If SMART_DENOISE is enabled → Use Bilateral Filter (preserves edges, best for text).
    Else → fallback to Gaussian or Median based on config.
    """
    try:
        if cfg.SMART_DENOISE:
            logger.logging.info("Using Bilateral Filter for edge-preserving denoising")
            return cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)

        if cfg.USE_GAUSSIAN:
            logger.logging.info("Using Gaussian Blur for denoising")
            return cv2.GaussianBlur(image, cfg.GAUSSIAN_KERNEL, 0)

        logger.logging.info("Using Median Blur for denoising")
        return cv2.medianBlur(image, cfg.MEDIAN_KERNEL)

    except Exception as e:
        logger.logging.error(f"Error in denoise_image: {e}")
        raise OCRImageException(e, sys)
