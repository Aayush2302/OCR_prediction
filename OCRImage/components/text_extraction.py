"""
Improved text extraction with fuzzy matching and pattern recognition.
Handles missing underscores and partial matches with 75% threshold.
"""

import re
from typing import List, Tuple, Optional, Dict
from difflib import SequenceMatcher
import numpy as np


def calculate_similarity(str1: str, str2: str) -> float:
    """
    Calculate similarity between two strings (0.0 to 1.0).
    Uses sequence matching algorithm.
    
    Args:
        str1: First string
        str2: Second string
        
    Returns:
        float: Similarity score (0.0 = no match, 1.0 = perfect match)
    """
    return SequenceMatcher(None, str1.lower(), str2.lower()).ratio()


def normalize_text(text: str) -> str:
    """
    Normalize text for better matching.
    - Remove extra spaces
    - Standardize separators
    - Keep alphanumeric and underscores
    
    Args:
        text: Input text
        
    Returns:
        str: Normalized text
    """
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # Replace common OCR errors
    replacements = {
        '|': '1',
        'l': '1',  # lowercase L to 1
        'O': '0',  # uppercase O to 0
        'o': '0',  # lowercase o to 0
        'I': '1',  # uppercase I to 1
        ' ': '',   # Remove spaces
    }
    
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    return text


def extract_digits_and_position(text: str) -> Tuple[str, int]:
    """
    Extract the longest sequence of digits and its position.
    
    Args:
        text: Input text
        
    Returns:
        Tuple of (digit_sequence, position)
    """
    digit_sequences = []
    current_seq = ""
    start_pos = -1
    
    for i, char in enumerate(text):
        if char.isdigit():
            if not current_seq:
                start_pos = i
            current_seq += char
        else:
            if current_seq:
                digit_sequences.append((current_seq, start_pos))
                current_seq = ""
    
    if current_seq:
        digit_sequences.append((current_seq, start_pos))
    
    if not digit_sequences:
        return "", -1
    
    # Return longest sequence
    longest = max(digit_sequences, key=lambda x: len(x[0]))
    return longest


def reconstruct_with_underscore(text: str, expected_id: str) -> Optional[str]:
    """
    Try to reconstruct the ID by intelligently adding underscores.
    
    Expected format: NNNNNNNNNNNNNNNNNN_1_XXX
                     (18 digits)_(1)_(alphanumeric)
    
    Args:
        text: OCR detected text
        expected_id: Expected ID format
        
    Returns:
        Reconstructed text or None
    """
    # Remove all separators first
    clean_text = re.sub(r'[_\-\s]', '', text)
    
    # Try to match expected pattern: 18 digits + "1" + alphanumeric
    # Expected: 163278430531063296_1_XXX
    
    # Pattern 1: Find 18+ consecutive digits followed by 1
    match = re.search(r'(\d{18,})1([a-zA-Z0-9]{0,10})', clean_text)
    if match:
        main_id = match.group(1)[:18]  # Take first 18 digits
        suffix = match.group(2)
        
        # Reconstruct with underscores
        if suffix:
            reconstructed = f"{main_id}_1_{suffix}"
        else:
            reconstructed = f"{main_id}_1"
        
        return reconstructed
    
    # Pattern 2: Look for expected_id digits in text (without underscores)
    expected_digits = expected_id.split('_')[0] if '_' in expected_id else expected_id
    
    if expected_digits in clean_text:
        # Found the digit part, try to reconstruct
        idx = clean_text.index(expected_digits)
        remaining = clean_text[idx + len(expected_digits):]
        
        # Check if next char is '1'
        if remaining and remaining[0] == '1':
            if len(remaining) > 1:
                reconstructed = f"{expected_digits}_1_{remaining[1:]}"
            else:
                reconstructed = f"{expected_digits}_1"
            return reconstructed
    
    return None


def fuzzy_match_with_threshold(detected: str, expected: str, threshold: float = 0.75) -> bool:
    """
    Check if detected text matches expected with fuzzy matching.
    
    Args:
        detected: OCR detected text
        expected: Expected text
        threshold: Minimum similarity (0.0 to 1.0)
        
    Returns:
        bool: True if similarity >= threshold
    """
    if not detected or not expected:
        return False
    
    # Normalize both strings
    norm_detected = normalize_text(detected)
    norm_expected = normalize_text(expected)
    
    # Calculate similarity
    similarity = calculate_similarity(norm_detected, norm_expected)
    
    return similarity >= threshold


class ImprovedTextExtractor:
    """
    Enhanced text extractor with fuzzy matching and reconstruction.
    """
    
    def __init__(self, similarity_threshold: float = 0.75):
        """
        Initialize extractor.
        
        Args:
            similarity_threshold: Minimum similarity for fuzzy matching (default: 0.75)
        """
        self.threshold = similarity_threshold
        
        # Patterns for shipping label IDs
        self.patterns = [
            # Pattern 1: 18 digits_1_alphanumeric
            re.compile(r'\b(\d{18})[-_\s]*1[-_\s]*([a-zA-Z0-9]{1,10})\b', re.IGNORECASE),
            
            # Pattern 2: 18 digits_1 (without suffix)
            re.compile(r'\b(\d{18})[-_\s]*1\b', re.IGNORECASE),
            
            # Pattern 3: Long digit sequence with "1" somewhere
            re.compile(r'\b(\d{12,20})[-_\s]*1[-_\s]*([a-zA-Z0-9]*)\b', re.IGNORECASE),
            
            # Pattern 4: Just the number part (fallback)
            re.compile(r'\b(\d{18})\b'),
        ]
    
    def extract_best_match(
        self, 
        ocr_results: List[Tuple[str, float, List]], 
        expected_id: str
    ) -> Optional[Dict]:
        """
        Extract best matching text with multiple strategies.
        
        Args:
            ocr_results: List of (text, confidence, bbox) tuples
            expected_id: Expected ID to match against
            
        Returns:
            Dictionary with match details or None
        """
        if not ocr_results:
            return None
        
        best_match = None
        best_score = 0.0
        
        # Strategy 1: Exact match
        for text, conf, bbox in ocr_results:
            if text.strip() == expected_id:
                return {
                    'text': text.strip(),
                    'confidence': conf,
                    'similarity': 1.0,
                    'match_type': 'exact',
                    'bbox': bbox
                }
        
        # Strategy 2: Fuzzy match with normalization
        for text, conf, bbox in ocr_results:
            similarity = calculate_similarity(normalize_text(text), normalize_text(expected_id))
            
            if similarity >= self.threshold:
                if similarity > best_score:
                    best_score = similarity
                    best_match = {
                        'text': text.strip(),
                        'confidence': conf,
                        'similarity': similarity,
                        'match_type': 'fuzzy',
                        'bbox': bbox
                    }
        
        if best_match:
            return best_match
        
        # Strategy 3: Pattern-based extraction with reconstruction
        for text, conf, bbox in ocr_results:
            # Try to reconstruct with underscores
            reconstructed = reconstruct_with_underscore(text, expected_id)
            
            if reconstructed:
                similarity = calculate_similarity(reconstructed, expected_id)
                
                if similarity >= self.threshold:
                    if similarity > best_score:
                        best_score = similarity
                        best_match = {
                            'text': reconstructed,
                            'original_text': text.strip(),
                            'confidence': conf,
                            'similarity': similarity,
                            'match_type': 'reconstructed',
                            'bbox': bbox
                        }
        
        if best_match:
            return best_match
        
        # Strategy 4: Digit-based matching (extract main number part)
        expected_digits = expected_id.split('_')[0] if '_' in expected_id else expected_id[:18]
        
        for text, conf, bbox in ocr_results:
            # Extract longest digit sequence
            digit_seq, _ = extract_digits_and_position(text)
            
            if digit_seq and len(digit_seq) >= 15:  # At least 15 digits
                # Check if it matches expected digits
                similarity = calculate_similarity(digit_seq, expected_digits)
                
                if similarity >= self.threshold:
                    if similarity > best_score:
                        best_score = similarity
                        
                        # Try to reconstruct full ID
                        reconstructed = reconstruct_with_underscore(text, expected_id)
                        if reconstructed:
                            final_text = reconstructed
                        else:
                            final_text = f"{digit_seq}_1"
                        
                        best_match = {
                            'text': final_text,
                            'original_text': text.strip(),
                            'confidence': conf,
                            'similarity': similarity,
                            'match_type': 'digit_based',
                            'bbox': bbox
                        }
        
        if best_match:
            return best_match
        
        # Strategy 5: Partial substring match
        for text, conf, bbox in ocr_results:
            # Check if expected_id is substring of text (or vice versa)
            if expected_id in text or text in expected_id:
                similarity = calculate_similarity(text, expected_id)
                
                if similarity > best_score:
                    best_score = similarity
                    best_match = {
                        'text': text.strip(),
                        'confidence': conf,
                        'similarity': similarity,
                        'match_type': 'substring',
                        'bbox': bbox
                    }
        
        return best_match if best_score >= self.threshold else None
    
    def find_all_candidates(
        self, 
        ocr_results: List[Tuple[str, float, List]]
    ) -> List[Dict]:
        """
        Find all potential ID candidates in OCR results.
        
        Args:
            ocr_results: List of (text, confidence, bbox) tuples
            
        Returns:
            List of candidate dictionaries
        """
        candidates = []
        
        for text, conf, bbox in ocr_results:
            # Check against all patterns
            for i, pattern in enumerate(self.patterns):
                match = pattern.search(text)
                if match:
                    candidates.append({
                        'text': text.strip(),
                        'matched_pattern': i,
                        'confidence': conf,
                        'bbox': bbox,
                        'groups': match.groups()
                    })
                    break
            
            # Also check for long digit sequences
            digit_seq, pos = extract_digits_and_position(text)
            if digit_seq and len(digit_seq) >= 15:
                candidates.append({
                    'text': text.strip(),
                    'digit_sequence': digit_seq,
                    'confidence': conf,
                    'bbox': bbox
                })
        
        # Sort by confidence
        candidates.sort(key=lambda x: x['confidence'], reverse=True)
        
        return candidates
