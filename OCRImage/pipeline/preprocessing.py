import cv2
from pathlib import Path
from OCRImage.components.orientation import auto_orient_image
from OCRImage.components.pipeline import PreprocessingPipeline
from OCRImage.constant.config import PreProcessingConfig


def test_pipeline_on_data_folder():
    """
    Test orientation correction and preprocessing pipeline
    on all image files in the 'data' directory.
    """

    print("=" * 70)
    print(" BATCH TESTING: ORIENTATION + PREPROCESSING PIPELINE")
    print("=" * 70)

    data_folder = Path(PreProcessingConfig.DATA_FOLDER)
    if not data_folder.exists():
        print(f" Folder not found: {data_folder.resolve()}")
        return

    image_files = list(data_folder.glob("*.*"))

    if not image_files:
        print(" No image files found in 'data/' folder.")
        return

    print(f" Found {len(image_files)} image(s) in '{data_folder.resolve()}'")

    # Initialize preprocessing pipeline
    config = PreProcessingConfig()
    pipeline = PreprocessingPipeline(config)

    for img_path in image_files:
        print("\n" + "-" * 70)
        print(f" Processing Image: {img_path.name}")

        image = cv2.imread(str(img_path))
        if image is None:
            print(f" Unable to load image: {img_path.name}")
            continue

        print("🔧 Step 1: Orientation Correction...")
        oriented_image = auto_orient_image(image)
        print(f"    Shape: {image.shape} → {oriented_image.shape}")

        print("🔄 Step 2: Running preprocessing pipeline...")
        try:
            result, metadata = pipeline.process(oriented_image, img_path.stem)
            print(f"    Pipeline completed – {len(metadata['processing_steps'])} steps executed")
            print(f"    Artifacts saved under: 'artifacts/preprocessing/{img_path.stem}_*/'")
        except Exception as e:
            print(f"    Pipeline error: {e}")
            continue

    print("\n" + "=" * 70)
    print("🎉 TESTING FINISHED!")

    return True


if __name__ == "__main__":
    test_pipeline_on_data_folder()
