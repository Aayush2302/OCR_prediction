"""
Updated OCR Engine with improved text extraction.
"""

import re
from typing import List, Tuple, Optional, Dict
import cv2
import numpy as np
from pathlib import Path
from OCRImage.exception.exception import OCRImageException
from OCRImage.logging import logger
from OCRImage.components.text_extraction import ImprovedTextExtractor


class OCREngine:
    """OCR Engine wrapper for multiple backends."""
    
    def __init__(self, engine_type: str = 'easyocr', lang: List[str] = ['en']):
        self.engine_type = engine_type.lower()
        self.lang = lang
        
        if self.engine_type == 'easyocr':
            import easyocr
            self.reader = easyocr.Reader(lang, gpu=True)
        elif self.engine_type == 'paddleocr':
            from paddleocr import PaddleOCR
            self.reader = PaddleOCR(use_angle_cls=True, lang='en')
        elif self.engine_type == 'tesseract':
            import pytesseract
            self.reader = pytesseract
        else:
            raise ValueError(f"Unsupported OCR engine: {engine_type}")
    
    def extract_text(self, image) -> List[Tuple[str, float, List]]:
        """Extract text from image."""
        if self.engine_type == 'easyocr':
            results = self.reader.readtext(image)
            return [(text.strip(), conf, bbox) for bbox, text, conf in results]
        
        elif self.engine_type == 'paddleocr':
            results = self.reader.ocr(image, cls=True)
            extracted = []
            if results and results[0]:
                for line in results[0]:
                    bbox, (text, conf) = line
                    extracted.append((text.strip(), conf, bbox))
            return extracted
        
        elif self.engine_type == 'tesseract':
            data = self.reader.image_to_data(image, output_type=self.reader.Output.DICT)
            extracted = []
            for i in range(len(data['text'])):
                if int(data['conf'][i]) > 0:
                    text = data['text'][i].strip()
                    if text:
                        x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                        bbox = [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
                        extracted.append((text, float(data['conf'][i]) / 100.0, bbox))
            return extracted
        
        return []


class ShippingLabelOCR:
    """Complete OCR pipeline for shipping labels with improved extraction."""
    
    def __init__(
        self, 
        ocr_engine_type: str = 'easyocr',
        similarity_threshold: float = 0.75,
        save_visualizations: bool = False
    ):
        self.ocr_engine = OCREngine(engine_type=ocr_engine_type)
        self.text_extractor = ImprovedTextExtractor(similarity_threshold=similarity_threshold)
        self.save_viz = save_visualizations
    
    def process_image(
        self, 
        image, 
        expected_id: str, 
        visualize: bool = False,
        verbose: bool = True
    ) -> Dict:
        """
        Process shipping label image with improved extraction.
        
        Args:
            image: Preprocessed image
            expected_id: Expected ID to match
            visualize: Create visualization
            verbose: Print detailed output
            
        Returns:
            Dictionary with results
        """
        # Extract text with OCR
        ocr_results = self.ocr_engine.extract_text(image)
        
        if verbose:
            print("\n📜 Raw OCR Output:")
            for text, conf, _ in ocr_results:
                print(f"  ➤ {text}  (conf: {conf:.2f})")
        
        # Find best match
        best_match = self.text_extractor.extract_best_match(ocr_results, expected_id)
        
        # Get all candidates
        all_candidates = self.text_extractor.find_all_candidates(ocr_results)
        
        # Prepare result
        result = {
            'expected_id': expected_id,
            'best_match': best_match,
            'all_candidates': all_candidates,
            'total_detections': len(ocr_results),
            'success': best_match is not None
        }
        
        if best_match:
            result['extracted_text'] = best_match['text']
            result['confidence'] = best_match['confidence']
            result['similarity'] = best_match['similarity']
            result['match_type'] = best_match['match_type']
        else:
            result['extracted_text'] = None
            result['confidence'] = 0.0
            result['similarity'] = 0.0
            result['match_type'] = 'no_match'
        
        return result
