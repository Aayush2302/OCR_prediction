# ocr_batch_pipeline.py
from pathlib import Path
import cv2
from OCRImage.components.OCR_Engine import ShippingLabelOCR
from OCRImage.components.text_extraction import ImprovedTextExtractor
from OCRImage.logging import logger


def extract_expected_id(filename: str) -> str:
    """Extract expected ID from filename."""
    name_without_ext = filename.split(".")[0]
    parts = name_without_ext.split("-")
    raw_part = parts[1] if len(parts) > 1 else ""
    if "_preprocessed" in raw_part:
        raw_part = raw_part.replace("_preprocessed", "")
    return raw_part.strip()


def run_ocr_batch(
    preprocessed_folder: str = r"artifacts/preprocessing/results",
    similarity_threshold: float = 0.75,
    ocr_engine_type: str = "easyocr",
    verbose: bool = True,
):
    """
    Run OCR batch on all preprocessed images in folder.
    Returns stats dict with results list.
    """
    folder = Path(preprocessed_folder)
    if not folder.exists():
        print(f"❌ Folder not found: {folder.resolve()}")
        return {}

    # collect images
    image_files = sorted([p for p in folder.glob("*.*") if p.suffix.lower() in [".png", ".jpg", ".jpeg"]])
    if not image_files:
        print(f"⚠ No preprocessed images found in {folder.resolve()}")
        return {}

    ocr_system = ShippingLabelOCR(ocr_engine_type=ocr_engine_type, similarity_threshold=similarity_threshold)

    stats = {
        "total": len(image_files),
        "exact_match": 0,
        "fuzzy_match": 0,
        "reconstructed": 0,
        "digit_based": 0,
        "failed": 0,
        "results": [],
    }

    for img_file in image_files:
        print("\n" + "-" * 80)
        print(f"🖼 Processing: {img_file.name}")
        expected_id = extract_expected_id(img_file.name)
        print(f"🎯 Expected ID: {expected_id}")

        img = cv2.imread(str(img_file))
        if img is None:
            print("  ❌ Unable to load image.")
            stats["failed"] += 1
            continue

        try:
            result = ocr_system.process_image(img, expected_id, visualize=False, verbose=False)

            # Result structure assumed: best_match dict or None, all_candidates list, similarity float etc.
            if result.get("success"):
                best = result["best_match"]
                sim = best.get("similarity", result.get("similarity", 0.0))
                conf = best.get("confidence", result.get("confidence", 0.0))
                mtype = best.get("match_type", result.get("match_type", "fuzzy"))
                text = best.get("text")

                print("  ✅ MATCH FOUND!")
                print(f"      Extracted: {text}")
                print(f"      Match Type: {mtype}")
                print(f"      Similarity: {sim*100:.1f}%")
                print(f"      OCR Confidence: {conf*100:.1f}%")

                # update stats
                if mtype == "exact":
                    stats["exact_match"] += 1
                elif mtype == "fuzzy":
                    stats["fuzzy_match"] += 1
                elif mtype == "reconstructed":
                    stats["reconstructed"] += 1
                elif mtype == "digit_based":
                    stats["digit_based"] += 1

                stats["results"].append({
                    "file": img_file.name,
                    "expected": expected_id,
                    "extracted": text,
                    "similarity": sim,
                    "confidence": conf,
                    "match_type": mtype,
                })
            else:
                print(f"  ❌ NO MATCH (similarity < {similarity_threshold*100:.1f}%)")
                stats["failed"] += 1
                # show top candidates if available
                candidates = result.get("all_candidates", [])
                if candidates:
                    print("   🔍 Top Candidates:")
                    for i, cand in enumerate(candidates[:3], start=1):
                        print(f"      {i}. {cand['text']} (conf: {cand['confidence']*100:.1f}%)")
        except Exception as e:
            print(f"  ❌ OCR Error: {e}")
            logger.exception(e)
            stats["failed"] += 1

    # summary
    total_success = stats["exact_match"] + stats["fuzzy_match"] + stats["reconstructed"] + stats["digit_based"]
    accuracy = (total_success / stats["total"]) * 100 if stats["total"] > 0 else 0.0

    print("\n" + "=" * 80)
    print("📊 FINAL TEST SUMMARY")
    print("=" * 80)
    print(f"🧪 Total Images Tested : {stats['total']}")
    print(f"✅ Total Successful    : {total_success}")
    print(f"   ├─ Exact Match      : {stats['exact_match']}")
    print(f"   ├─ Fuzzy Match      : {stats['fuzzy_match']}")
    print(f"   ├─ Reconstructed    : {stats['reconstructed']}")
    print(f"   └─ Digit-Based      : {stats['digit_based']}")
    print(f"❌ Failed              : {stats['failed']}")
    print(f"\n📈 Accuracy: {accuracy:.1f}%")
    print("=" * 80)

    return stats


if __name__ == "__main__":
    run_stats = run_ocr_batch()
