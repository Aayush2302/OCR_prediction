"""
Main execution script for processing all images inside 'data' folder
using the optimized preprocessing pipeline.
"""

import cv2
import sys
from pathlib import Path

from OCRImage.exception.exception import OCRImageException
from OCRImage.logging import logger
from OCRImage.constant.config import PreProcessingConfig
from OCRImage.components.pipeline import PreprocessingPipeline


def process_single_image(image_path: Path, output_dir: Path, pipeline):
    """
    Process a single image using the given pipeline
    """
    try:
        print(f"\n🔍 Processing: {image_path.name}")
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"⚠️ Skipping (invalid image): {image_path}")
            return

        # Run pipeline
        processed_image, metadata = pipeline.process(img, image_path.stem)

        # Save output
        output_path = output_dir / f"{image_path.stem}_processed.png"
        cv2.imwrite(str(output_path), processed_image)

        print(f"   ✔ Saved: {output_path}")
        print(f"   🧪 Artifacts: artifacts/optimized/{metadata['session_id']}/")

    except Exception as e:
        print(f"❌ Failed to process {image_path}: {e}")


def main():
    data_dir = Path("data")       # 📌 Folder where your images are stored
    output_dir = Path("results")  # 📁 Output folder

    if not data_dir.exists():
        print("❌ 'data' folder does not exist! Make sure images are inside it.")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load optimized configuration
    config = PreProcessingConfig()
    pipeline = PreprocessingPipeline(config=config, artifacts_dir="artifacts/optimized")

    print("\n============================================")
    print("📦 Batch Preprocessing: All images in 'data/'")
    print("============================================")

    # Supported image extensions
    image_extensions = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

    # Process each image inside data folder
    image_files = list(data_dir.glob("*"))
    if not image_files:
        print("❌ No files found in 'data' folder.")
        return

    for image_path in image_files:
        if image_path.suffix.lower() in image_extensions:
            process_single_image(image_path, output_dir, pipeline)
        else:
            print(f"⏭️ Skipping non-image file: {image_path.name}")

    print("\nBatch processing completed!")
    print(f"All results saved in: {output_dir}")
    print("For OCR, use the thresholded artifact images.")


if __name__ == "__main__":
    main()
