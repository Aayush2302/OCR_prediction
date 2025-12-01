# preprocess_pipeline.py
from pathlib import Path
import cv2
from OCRImage.components.orientation import auto_orient_image
from OCRImage.components.pipeline import PreprocessingPipeline
from OCRImage.constant.config import PreProcessingConfig
from OCRImage.logging import logger


def run_preprocessing_on_folder(
    data_folder: str = "data",
    output_folder: str = r"artifacts/preprocessing/results",
    visualize: bool = False,
):
    """
    Run orientation correction and preprocessing pipeline on all images in `data_folder`.
    Saves preprocessed output images into `output_folder` with suffix `_preprocessed.png`.
    Returns list of saved file paths.
    """
    data_path = Path(data_folder)
    out_path = Path(output_folder)
    out_path.mkdir(parents=True, exist_ok=True)

    if not data_path.exists():
        print(f"❌ Data folder not found: {data_path.resolve()}")
        return []

    image_files = sorted([p for p in data_path.glob("*.*") if p.suffix.lower() in [".png", ".jpg", ".jpeg", ".tif", ".tiff"]])
    if not image_files:
        print(f"⚠ No image files found in {data_path.resolve()}")
        return []

    print(f"Found {len(image_files)} images in {data_path.resolve()}")
    config = PreProcessingConfig()
    pipeline = PreprocessingPipeline(config)

    saved_files = []
    for img_path in image_files:
        print("\n" + "-" * 70)
        print(f"Processing: {img_path.name}")
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  ❌ Unable to load: {img_path.name}")
            continue

        try:
            # Step 1: orientation correction
            oriented = auto_orient_image(img)
            print(f"  Orientation corrected: {img.shape} -> {oriented.shape}")

            # Step 2: preprocessing pipeline (returns processed_image, metadata)
            processed, metadata = pipeline.process(oriented, img_path.stem)
            # If your pipeline returns differently, adjust above accordingly.

            # Save processed image
            out_name = f"{img_path.stem}_preprocessed.png"
            out_file = out_path.joinpath(out_name)
            # Convert if needed to BGR uint8
            if processed is None:
                print("  ⚠ Pipeline returned None for processed image, skipping save.")
                continue

            cv2.imwrite(str(out_file), processed)
            print(f"  ✅ Saved preprocessed image: {out_file}")
            saved_files.append(out_file)
        except Exception as e:
            print(f"  ❌ Error processing {img_path.name}: {e}")
            logger.exception(e)

    print("\nPreprocessing completed.")
    return saved_files


if __name__ == "__main__":
    run_preprocessing_on_folder()
