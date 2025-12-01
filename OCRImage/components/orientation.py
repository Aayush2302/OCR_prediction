"""
Automatic image orientation detection and correction.
Handles vertical images and ensures correct rotation direction.
"""

import cv2
import numpy as np
from OCRImage.exception.exception import OCRImageException
from OCRImage.logging import logger
import sys


def detect_orientation(image):
    """
    Detect if image needs rotation based on aspect ratio and text orientation.
    
    Parameters:
    image (numpy.ndarray): Input image
    
    Returns:
    int: Rotation angle (0, 90, 180, or 270 degrees)
    """
    try:
        height, width = image.shape[:2]
        
        # Method 1: Simple aspect ratio check
        aspect_ratio = width / height
        
        logger.logging.info(f"Image dimensions: {width}x{height}, aspect_ratio: {aspect_ratio:.2f}")
        
        # If width > height, likely horizontal (correct orientation)
        if aspect_ratio > 1.2:
            logger.logging.info("Image is horizontal (landscape) - no rotation needed")
            return 0
        
        # If height > width, likely vertical (needs rotation)
        elif aspect_ratio < 0.8:
            logger.logging.info("Image is vertical (portrait) - needs rotation")
            
            # Determine rotation direction using text detection
            rotation_angle = determine_rotation_direction(image)
            return rotation_angle
        
        # If aspect ratio is close to 1:1, use OCR-based detection
        else:
            logger.logging.info("Image is square-ish - using OCR-based detection")
            rotation_angle = ocr_based_orientation(image)
            return rotation_angle
            
    except Exception as e:
        logger.logging.error(f"Error in detect_orientation: {e}")
        raise OCRImageException(e, sys)


def determine_rotation_direction(image):
    """
    Determine which direction to rotate (90 or 270 degrees).
    Uses edge detection and text line analysis.
    
    Parameters:
    image (numpy.ndarray): Input vertical image
    
    Returns:
    int: 90 or 270 degrees
    """
    try:
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Try both rotations and score them
        rotated_90 = cv2.rotate(gray, cv2.ROTATE_90_CLOCKWISE)
        rotated_270 = cv2.rotate(gray, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        # Score based on horizontal text lines
        score_90 = score_horizontal_lines(rotated_90)
        score_270 = score_horizontal_lines(rotated_270)
        
        logger.logging.info(f"Rotation scores - 90°: {score_90:.2f}, 270°: {score_270:.2f}")
        
        # Choose rotation with higher score
        if score_90 > score_270:
            logger.logging.info("Best rotation: 90° clockwise")
            return 90
        else:
            logger.logging.info("Best rotation: 270° clockwise (90° counter-clockwise)")
            return 270
            
    except Exception as e:
        logger.logging.error(f"Error in determine_rotation_direction: {e}")
        # Default to 270 if error
        return 270


def score_horizontal_lines(image):
    """
    Score image based on horizontal line detection.
    Higher score means more horizontal text lines (correct orientation).
    
    Parameters:
    image (numpy.ndarray): Grayscale image
    
    Returns:
    float: Score (higher is better)
    """
    try:
        # Apply edge detection
        edges = cv2.Canny(image, 50, 150)
        
        # Detect horizontal lines using morphology
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
        horizontal_lines = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        # Count horizontal line pixels
        horizontal_pixels = np.sum(horizontal_lines > 0)
        
        # Detect vertical lines
        kernel_vert = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
        vertical_lines = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_vert)
        vertical_pixels = np.sum(vertical_lines > 0)
        
        # Score is ratio of horizontal to vertical lines
        if vertical_pixels > 0:
            score = horizontal_pixels / vertical_pixels
        else:
            score = horizontal_pixels
        
        return score
        
    except Exception as e:
        logger.logging.error(f"Error in score_horizontal_lines: {e}")
        return 0.0


def ocr_based_orientation(image):
    """
    Use simple OCR to detect orientation for square images.
    Tries all 4 rotations and picks the one with most detected text.
    
    Parameters:
    image (numpy.ndarray): Input image
    
    Returns:
    int: Best rotation angle (0, 90, 180, 270)
    """
    try:
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        rotations = {
            0: gray,
            90: cv2.rotate(gray, cv2.ROTATE_90_CLOCKWISE),
            180: cv2.rotate(gray, cv2.ROTATE_180),
            270: cv2.rotate(gray, cv2.ROTATE_90_COUNTERCLOCKWISE)
        }
        
        scores = {}
        
        for angle, rotated in rotations.items():
            # Score based on horizontal lines
            scores[angle] = score_horizontal_lines(rotated)
        
        # Get best rotation
        best_angle = max(scores, key=scores.get)
        
        logger.logging.info(f"OCR-based orientation scores: {scores}")
        logger.logging.info(f"Best orientation: {best_angle}°")
        
        return best_angle
        
    except Exception as e:
        logger.logging.error(f"Error in ocr_based_orientation: {e}")
        return 0


def rotate_image(image, angle):
    """
    Rotate image by specified angle.
    
    Parameters:
    image (numpy.ndarray): Input image
    angle (int): Rotation angle (0, 90, 180, 270)
    
    Returns:
    numpy.ndarray: Rotated image
    """
    try:
        if angle == 0:
            return image
        elif angle == 90:
            rotated = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        elif angle == 180:
            rotated = cv2.rotate(image, cv2.ROTATE_180)
        elif angle == 270:
            rotated = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        else:
            raise ValueError(f"Invalid rotation angle: {angle}")
        
        logger.logging.info(f"Image rotated by {angle}°")
        return rotated
        
    except Exception as e:
        logger.logging.error(f"Error in rotate_image: {e}")
        raise OCRImageException(e, sys)


def auto_orient_image(image):
    """
    Automatically detect and correct image orientation.
    Main function to use in preprocessing pipeline.
    
    Parameters:
    image (numpy.ndarray): Input image (any orientation)
    
    Returns:
    numpy.ndarray: Correctly oriented image (horizontal/landscape)
    """
    try:
        logger.logging.info("Starting automatic orientation detection...")
        
        # Detect required rotation
        rotation_angle = detect_orientation(image)
        
        # Rotate if needed
        if rotation_angle != 0:
            corrected_image = rotate_image(image, rotation_angle)
            logger.logging.info(f"✓ Image orientation corrected ({rotation_angle}° rotation)")
            return corrected_image
        else:
            logger.logging.info("✓ Image already in correct orientation")
            return image
            
    except Exception as e:
        logger.logging.error(f"Error in auto_orient_image: {e}")
        raise OCRImageException(e, sys)
