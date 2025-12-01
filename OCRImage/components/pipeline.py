"""
Optimized preprocessing pipeline for your shipping labels.
Key: Proper order and ALL steps enabled!
"""
import cv2
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple
import json
import numpy as np

from OCRImage.constant.config import PreProcessingConfig as cfg


from OCRImage.exception.exception import OCRImageException
from OCRImage.logging import logger
from OCRImage.components import (
    convert_to_grayscale,
    deskew_image,
    apply_clahe,
    apply_threshold,
    denoise_image,
    apply_morphology,
    resize_image
)
import sys


class PreprocessingPipeline:
    """
    Manages the complete image preprocessing workflow with artifact storage.
    """
    
    def __init__(self, config, artifacts_dir: str = "artifacts/preprocessing"):
        """
        Initialize the preprocessing pipeline.
        
        Args:
            config: Configuration object (PreProcessingConfig)
            artifacts_dir: Directory to store intermediate results
        """
        self.config = config
        self.artifacts_dir = Path(artifacts_dir)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        
        # Store processing metadata
        self.metadata = {
            "processing_steps": [],
            "timestamp": None,
            "original_image_path": None,
            "image_shape": None
        }
    
    def _save_artifact(self, image, step_name: str, session_id: str):
        """Save intermediate processing result."""
        if self.config.SAVE_INTERMEDIATE_STEPS:
            session_dir = self.artifacts_dir / session_id
            session_dir.mkdir(exist_ok=True)
            
            filename = f"{len(self.metadata['processing_steps']):02d}_{step_name}.png"
            filepath = session_dir / filename
            cv2.imwrite(str(filepath), image)
            
            self.metadata['processing_steps'].append({
                "step": step_name,
                "artifact_path": str(filepath),
                "shape": image.shape
            })
    
    def _apply_step(self, image, step_func, step_name: str, session_id: str, **kwargs):
        """Apply a preprocessing step and save artifact."""
        try:
            processed = step_func(image, **kwargs)
            self._save_artifact(processed, step_name, session_id)
            return processed
        except Exception as e:
            logger.logging.error(f"Error in {step_name}: {e}")
            raise OCRImageException(e, sys)
    
    def process(self, image, image_name: str = "unknown") -> Tuple[any, Dict]:
        """
        Run complete preprocessing pipeline.
        OPTIMIZED ORDER: Resize → Grayscale → Denoise → CLAHE → Threshold → Morphology
        
        Args:
            image: Input image (numpy array)
            image_name: Original image filename
            
        Returns:
            Tuple of (preprocessed_image, metadata_dict)
        """
        try:
            # Generate unique session ID
            session_id = f"{image_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Initialize metadata
            self.metadata = {
                "processing_steps": [],
                "timestamp": datetime.now().isoformat(),
                "original_image_path": image_name,
                "image_shape": image.shape,
                "session_id": session_id
            }
            
            logger.logging.info(f"Starting preprocessing pipeline for {image_name}")
            
            # Save original image
            self._save_artifact(image, "00_original", session_id)
            
            current_image = image
            
            # ========================================
            # STEP 1: Resize FIRST (upscale for better OCR)
            # ========================================
            if hasattr(self.config, 'TARGET_SIZE') and self.config.TARGET_SIZE:
                current_image = self._apply_step(
                    current_image,
                    resize_image,
                    "01_resized",
                    session_id,
                    width=self.config.TARGET_SIZE
                )
                logger.logging.info(f"Resized to width={self.config.TARGET_SIZE}")
            
            # ========================================
            # STEP 2: Deskew (skip if disabled)
            # ========================================
            if self.config.ENABLE_DESKEW and len(current_image.shape) == 3:
                current_image = self._apply_step(
                    current_image,
                    deskew_image,
                    "02_deskewed",
                    session_id
                )
            
            # ========================================
            # STEP 3: Convert to Grayscale
            # ========================================
            if len(current_image.shape) == 3:
                current_image = self._apply_step(
                    current_image,
                    convert_to_grayscale,
                    "03_grayscale",
                    session_id
                )
            
            # ========================================
            # STEP 4: Denoise BEFORE CLAHE
            # Remove noise before enhancing contrast!
            # ========================================
            # current_image = self._apply_step(
            #     current_image,
            #     denoise_image,
            #     "04_denoised",
            #     session_id
            # )
            # logger.logging.info(f"Denoising: {self.config.DENOISING_METHOD}")
            
            # ========================================
            # STEP 5: CLAHE (enhance contrast on clean image)
            # ========================================
            if self.config.USE_CLAHE:
                current_image = self._apply_step(
                    current_image,
                    apply_clahe,
                    "05_clahe",
                    session_id
                )
                logger.logging.info(f"CLAHE: clip={self.config.CLAHE_CLIP_LIMIT}")
            
            # ========================================
            # STEP 6: Thresholding (convert to binary)
            # ========================================
            current_image = self._apply_step(
                current_image,
                apply_threshold,
                "06_threshold",
                session_id
            )
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            current_image = cv2.filter2D(current_image, -1, kernel)
            current_image = cv2.equalizeHist(current_image)
            self._save_artifact(current_image, "06b_sharpened", session_id)
            logger.logging.info(f"Threshold: {self.config.THRESHOLD_METHOD}")
            
            # ========================================
            # STEP 7: Morphology (clean up binary image)
            # ========================================
            current_image = self._apply_step(
                current_image,
                apply_morphology,
                "07_morphology",
                session_id
            )
            logger.logging.info(f"Morphology: {self.config.MORPHOLOGY_METHOD}")

            
            
            # Save metadata
            self._save_metadata(session_id)
            
            logger.logging.info(f"Preprocessing pipeline completed for {image_name}")
            
            return current_image, self.metadata
            
        except Exception as e:
            logger.logging.error(f"Error in preprocessing pipeline: {e}")
            raise OCRImageException(e, sys)
    
    def _save_metadata(self, session_id: str):
        """Save processing metadata to JSON file."""
        try:
            session_dir = self.artifacts_dir / session_id
            metadata_path = session_dir / "metadata.json"
            
            with open(metadata_path, 'w') as f:
                json.dump(self.metadata, f, indent=2)
            
            logger.logging.info(f"Metadata saved to {metadata_path}")
            
        except Exception as e:
            logger.logging.error(f"Error saving metadata: {e}")
