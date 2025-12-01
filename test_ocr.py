from pathlib import Path
import cv2
from OCRImage.components.OCR_Engine import OCREngine, ShippingLabelOCR
from OCRImage.logging import logger
from OCRImage.components.text_extraction import ImprovedTextExtractor



def extract_expected_id(filename: str) -> str:
    """Extract expected ID from filename."""
    name_without_ext = filename.split(".")[0]
    parts = name_without_ext.split("-")
    raw_part = parts[1] if len(parts) > 1 else ""
    
    if "_preprocessed" in raw_part:
        raw_part = raw_part.replace("_preprocessed", "")
    
    return raw_part.strip()


def batch_test_improved_ocr(folder_path: str, similarity_threshold: float = 0.75):
    """
    Batch test OCR with improved extraction.
    
    Args:
        folder_path: Path to preprocessed images
        similarity_threshold: Minimum similarity for matching (0.75 = 75%)
    """
    print("=" * 80)
    print("🔬 IMPROVED OCR BATCH TESTING WITH FUZZY MATCHING")
    print(f"📊 Similarity Threshold: {similarity_threshold * 100}%")
    print("=" * 80)
    
    images_path = Path(folder_path)
    if not images_path.exists():
        print(f"❌ Folder not found: {images_path}")
        return
    
    image_files = list(images_path.glob("*.png")) + list(images_path.glob("*.jpg"))
    if not image_files:
        print("⚠ No image files found for testing.")
        return
    
    # Initialize OCR system
    ocr_system = ShippingLabelOCR(
        ocr_engine_type='easyocr',
        similarity_threshold=similarity_threshold
    )
    
    # Statistics
    stats = {
        'total': len(image_files),
        'exact_match': 0,
        'fuzzy_match': 0,
        'reconstructed': 0,
        'digit_based': 0,
        'failed': 0,
        'results': []
    }
    
    # Process each image
    for img_file in image_files:
        print("\n" + "-" * 80)
        print(f"🖼 Processing: {img_file.name}")
        
        expected_id = extract_expected_id(img_file.name)
        print(f"🎯 Expected ID: {expected_id}")
        
        # Load image
        image = cv2.imread(str(img_file))
        if image is None:
            print("❌ Error: Couldn't load image")
            stats['failed'] += 1
            continue
        
        # Process with improved OCR
        result = ocr_system.process_image(image, expected_id, verbose=False)
        
        # Display result
        if result['success']:
            match = result['best_match']
            print(f"✅ MATCH FOUND!")
            print(f"   📌 Extracted: {match['text']}")
            print(f"   🎯 Match Type: {match['match_type']}")
            print(f"   📊 Similarity: {match['similarity']*100:.1f}%")
            print(f"   🔍 OCR Confidence: {match['confidence']*100:.1f}%")
            
            # Update stats
            if match['match_type'] == 'exact':
                stats['exact_match'] += 1
            elif match['match_type'] == 'fuzzy':
                stats['fuzzy_match'] += 1
            elif match['match_type'] == 'reconstructed':
                stats['reconstructed'] += 1
            elif match['match_type'] == 'digit_based':
                stats['digit_based'] += 1
            
            stats['results'].append({
                'file': img_file.name,
                'expected': expected_id,
                'extracted': match['text'],
                'similarity': match['similarity'],
                'match_type': match['match_type']
            })
        else:
            print(f"❌ NO MATCH (Similarity < {similarity_threshold*100}%)")
            stats['failed'] += 1
            
            # Show best candidates
            if result['all_candidates']:
                print("   🔍 Top Candidates:")
                for i, cand in enumerate(result['all_candidates'][:3], 1):
                    print(f"      {i}. {cand['text']} (conf: {cand['confidence']*100:.1f}%)")
    
    # Print summary
    print("\n" + "=" * 80)
    print("📊 FINAL TEST SUMMARY")
    print("=" * 80)
    print(f"🧪 Total Images Tested    : {stats['total']}")
    print(f"✅ Total Successful       : {stats['exact_match'] + stats['fuzzy_match'] + stats['reconstructed'] + stats['digit_based']}")
    print(f"   ├─ Exact Match         : {stats['exact_match']}")
    print(f"   ├─ Fuzzy Match         : {stats['fuzzy_match']}")
    print(f"   ├─ Reconstructed       : {stats['reconstructed']}")
    print(f"   └─ Digit-Based         : {stats['digit_based']}")
    print(f"❌ Failed                 : {stats['failed']}")
    print(f"\n📈 Accuracy: {((stats['total'] - stats['failed']) / stats['total'] * 100):.1f}%")
    print("=" * 80)
    
    return stats

if __name__ == "__main__":
    # Test with your data
    FOLDER_PATH = r"D:\\projects\\ML\\OCR_prediction\\artifacts\\preprocessing\\results"
    
    # Test with 75% threshold (default)
    stats = batch_test_improved_ocr(FOLDER_PATH, similarity_threshold=0.75)
    