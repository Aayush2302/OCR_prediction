# main.py
import argparse
from OCRImage.pipeline.preprocessing import test_pipeline_on_data_folder as run_preprocessing_on_folder
from OCRImage.pipeline.ocr import batch_test_improved_ocr as run_ocr_batch
from OCRImage.constant.config import PreProcessingConfig, OCRConfig


def main():
    parser = argparse.ArgumentParser(description="Preprocess + OCR batch pipeline")
    parser.add_argument("--data-folder", type=str, default="data", help="Folder containing raw images")
    parser.add_argument("--preprocessed-folder", type=str, default=r"artifacts/preprocessing/results", help="Where preprocessed images are saved/read")
    parser.add_argument("--similarity-threshold", type=float, default=0.75, help="Similarity threshold for match (0-1)")
    parser.add_argument("--step", choices=["all", "preprocess", "ocr"], default="all", help="Which step to run")
    args = parser.parse_args()

    if args.step in ["all", "preprocess"]:
        print("\n==== RUNNING PREPROCESSING ====\n")
        saved = run_preprocessing_on_folder()
        print(f"\nPreprocessing saved processed files to {PreProcessingConfig.OUTPUT_FOLDER}")

    if args.step in ["all", "ocr"]:
        print("\n==== RUNNING OCR BATCH ====\n")
        stats = run_ocr_batch(folder_path=OCRConfig.PREPROCESSED_FOLDER, similarity_threshold=args.similarity_threshold)
        print("\nOCR batch completed.")
    
    if args.step == "all":
        print("\n==== RUNNING FULL PIPELINE ====\n")
        saved = run_preprocessing_on_folder()
        print(f"\nPreprocessing saved processed files to {PreProcessingConfig.OUTPUT_FOLDER}")
        stats = run_ocr_batch(folder_path=OCRConfig.PREPROCESSED_FOLDER, similarity_threshold=args.similarity_threshold)
        print("\nOCR batch completed.")


    print("\nPipeline finished.")


if __name__ == "__main__":
    main()
