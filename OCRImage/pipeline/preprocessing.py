import cv2
import sys,os
from OCRImage.exception.exception import OCRImageException
from OCRImage.logging import logger
from OCRImage.constant.config import PreProcessingConfig as cfg
from OCRImage.components.grayscale import convert_to_grayscale
from OCRImage.components.thresholding import apply_threshold
from OCRImage.components.denoising import denoise_image
from OCRImage.components.morphology import apply_morphology
from OCRImage.components.clahe import apply_clahe


def preprocess_image(image):
    """
    Preprocess the input image by applying a series of operations: grayscale conversion,
    CLAHE, thresholding, denoising, and morphological operations.

    Parameters:
    image (numpy.ndarray): The input image.

    Returns:
    numpy.ndarray: The preprocessed image.
    """
    try:
        # Convert to Grayscale
        gray_image = convert_to_grayscale(image)

        # Apply CLAHE
        clahe_image = apply_clahe(gray_image)

        # Apply Thresholding
        thresh_image = apply_threshold(clahe_image)

        # Denoise Image
        denoised_image = denoise_image(thresh_image)

        # Apply Morphological Operations
        morph_image = apply_morphology(denoised_image)

        return morph_image

    except Exception as e:
        logger.logging.error(f"Error in preprocess_image: {e}")
        raise OCRImageException(e, sys)