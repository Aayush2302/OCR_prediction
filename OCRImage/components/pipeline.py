"""
Updated preprocessing pipeline with automatic orientation correction.
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
    resize_image,
    auto_orient_image  # NEW IMPORT
)
import sys


class PreprocessingPipeline:
    """
    Manages the complete image preprocessing workflow with artifact storage.
    NOW INCLUDES: Automatic orientation correction!
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
        self.result_path = Path(artifacts_dir) / "results"
        self.result_path.mkdir(parents=True, exist_ok=True)
        
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
        
        NEW ORDER: 
        0. Auto-Orient (NEW!) → 1. Resize → 2. Grayscale → 3. CLAHE → 4. Threshold → 5. Sharpen → 6. Morphology
        
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
            # STEP 0: AUTO-ORIENT (NEW!)
            # Detect and correct vertical images
            # ========================================
            if hasattr(self.config, 'AUTO_ORIENT') and self.config.AUTO_ORIENT:
                current_image = self._apply_step(
                    current_image,
                    auto_orient_image,
                    "01_oriented",
                    session_id
                )
                logger.logging.info("Auto-orientation applied")
            
            # ========================================
            # STEP 1: Resize (upscale for better OCR)
            # ========================================
            if hasattr(self.config, 'TARGET_SIZE') and self.config.TARGET_SIZE:
                current_image = self._apply_step(
                    current_image,
                    resize_image,
                    "02_resized",
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
                    "03_deskewed",
                    session_id
                )
            
            # ========================================
            # STEP 3: Convert to Grayscale
            # ========================================
            if len(current_image.shape) == 3:
                current_image = self._apply_step(
                    current_image,
                    convert_to_grayscale,
                    "04_grayscale",
                    session_id
                )
            
            # ========================================
            # STEP 4: CLAHE (enhance contrast)
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
            # STEP 5: Thresholding (convert to binary)
            # ========================================
            current_image = self._apply_step(
                current_image,
                apply_threshold,
                "06_threshold",
                session_id
            )
            
            # ========================================
            # STEP 6: Sharpening (your custom step)
            # ========================================
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            current_image = cv2.filter2D(current_image, -1, kernel)
            current_image = cv2.equalizeHist(current_image)
            self._save_artifact(current_image, "07_sharpened", session_id)
            logger.logging.info("Sharpening applied")
            
            # ========================================
            # STEP 7: Morphology (clean up binary image)
            # ========================================
            # current_image = self._apply_step(
            #     current_image,
            #     apply_morphology,
            #     "08_morphology",
            #     session_id
            # )
            # logger.logging.info(f"Morphology: {self.config.MORPHOLOGY_METHOD}")
            
            # Save metadata
            self._save_metadata(session_id)
            
            logger.logging.info(f"Preprocessing pipeline completed for {image_name}")

            # save image as the preprocessing result 
            result_path = self.result_path / f"{image_name}_preprocessed.png"
            cv2.imwrite(str(result_path), current_image)
            logger.logging.info(f"Preprocessed image saved to {result_path}")
            
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
