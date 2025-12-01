# main.py
import argparse
from OCRImage.pipeline.preprocessing import run_preprocessing_on_folder
from OCRImage.pipeline.ocr import run_ocr_batch


def main():
    parser = argparse.ArgumentParser(description="Preprocess + OCR batch pipeline")
    parser.add_argument("--data-folder", type=str, default="data", help="Folder containing raw images")
    parser.add_argument("--preprocessed-folder", type=str, default=r"artifacts/preprocessing/results", help="Where preprocessed images are saved/read")
    parser.add_argument("--similarity-threshold", type=float, default=0.75, help="Similarity threshold for match (0-1)")
    parser.add_argument("--step", choices=["all", "preprocess", "ocr"], default="all", help="Which step to run")
    args = parser.parse_args()

    if args.step in ["all", "preprocess"]:
        print("\n==== RUNNING PREPROCESSING ====\n")
        saved = run_preprocessing_on_folder(data_folder=args.data_folder, output_folder=args.preprocessed_folder)
        print(f"\nPreprocessing saved {len(saved)} processed files to {args.preprocessed_folder}")

    if args.step in ["all", "ocr"]:
        print("\n==== RUNNING OCR BATCH ====\n")
        stats = run_ocr_batch(preprocessed_folder=args.preprocessed_folder, similarity_threshold=args.similarity_threshold)
        print("\nOCR batch completed.")

    print("\nPipeline finished.")


if __name__ == "__main__":
    main()
