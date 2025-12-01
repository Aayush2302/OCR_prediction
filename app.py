# app.py
import streamlit as st
import cv2
import numpy as np
from pathlib import Path

from OCRImage.components.pipeline import PreprocessingPipeline
from OCRImage.constant.config import PreProcessingConfig
from OCRImage.components.OCR_Engine import ShippingLabelOCR



# ----------------------------
# Utility Functions
# ----------------------------

def extract_expected_id_from_filename(filename: str) -> str:
    """
    Example filename:
        reverseWaybill-163233702292313922_1_preprocessed.png
    Should return:
        163233702292313922_1
    """
    name = filename.split(".")[0]
    parts = name.split("-")

    if len(parts) < 2:
        return ""     # fallback

    id_part = parts[1]

    if "_preprocessed" in id_part:
        id_part = id_part.replace("_preprocessed", "")

    return id_part.strip()


def preprocess_image(image_np):
    """Runs ONLY the preprocessing pipeline (no orientation)."""
    config = PreProcessingConfig()
    pipeline = PreprocessingPipeline(config)

    processed_img, metadata = pipeline.process(image_np, "uploaded_img")
    return processed_img



# ----------------------------
# Streamlit UI
# ----------------------------

st.set_page_config(page_title="Shipping Label OCR Demo", layout="wide")
st.title("📦 Shipping Label OCR –")
st.write("Upload an image → Preprocess → Extract the target waybill/order ID.")


# -------------------------------------
# 1) IMAGE UPLOAD
# -------------------------------------
uploaded_file = st.file_uploader("Upload Shipping Label Image", type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    file_bytes = uploaded_file.read()
    np_img = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)

    st.image(np_img, caption="Uploaded Image", channels="BGR", use_column_width=True)

    # -------------------------------------
    # Extract Expected ID automatically
    # -------------------------------------
    expected_id = extract_expected_id_from_filename(uploaded_file.name)

    if expected_id:
        st.success(f"🔍 **Auto-detected Expected ID:** `{expected_id}`")
    else:
        st.warning("⚠ Could not detect expected ID from filename. OCR will still run but similarity may be low.")

    # -------------------------------------
    # 2) RUN PREPROCESSING PIPELINE
    # -------------------------------------
    st.subheader("🧹 Preprocessing Options")

    apply_preprocessing = st.checkbox("Apply Preprocessing Pipeline", value=True)

    if apply_preprocessing:
        st.write("Running preprocessing pipeline...")
        processed_image = preprocess_image(np_img)
        # Ensure the preprocessed image is valid for Streamlit
    if len(processed_image.shape) == 2:
        # Grayscale → BGR
        processed_image = cv2.cvtColor(processed_image, cv2.COLOR_GRAY2BGR)

    elif processed_image.shape[2] == 4:
        # BGRA → BGR
        processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGRA2BGR)

        st.image(processed_image, caption="Preprocessed Image", channels="BGR", use_column_width=True)

    else:
        processed_image = np_img


    # -------------------------------------
    # 3) OCR EXTRACTION
    # -------------------------------------
    st.subheader("🔍 OCR Processing")

    similarity_threshold = st.slider(
        "Minimum Similarity Threshold (%)",
        min_value=50,
        max_value=95,
        value=75,
        step=1
    ) / 100.0

    run_ocr = st.button("Run OCR Extraction")

    if run_ocr:
        st.write("Running OCR…")

        # Initialize OCR System
        ocr_system = ShippingLabelOCR(
            ocr_engine_type="easyocr",
            similarity_threshold=similarity_threshold
        )

        # OCR
        result = ocr_system.process_image(
            processed_image,
            expected_id,
            visualize=False,
            verbose=False
        )

        st.subheader("📤 OCR Output")

        # ------------------------------
        # SUCCESS CASE
        # ------------------------------
        if result.get("success"):
            best = result["best_match"]

            extracted_text = best['text']
            confidence = best['confidence']
            similarity = best['similarity']
            match_type = best['match_type']

            st.success(f"🎯 **Extracted Target:** `{extracted_text}`")

            st.markdown(
                f"""
                **Match Type:** `{match_type}`  
                **Similarity:** `{similarity*100:.2f}%`  
                **OCR Confidence:** `{confidence*100:.2f}%`
                """
            )

            # --------------------------------
            # Show ALL OCR text detections
            # --------------------------------
            st.subheader("📝 All OCR Detections")
            for cand in result["all_candidates"]:
                if cand["text"] == extracted_text:
                    st.markdown(f"✅ **{cand['text']}** — `{cand['confidence']*100:.1f}%` (extracted)")
                else:
                    st.markdown(f"- {cand['text']} — `{cand['confidence']*100:.1f}%`")

        else:
            # FAILURE CASE
            st.error("No match found. Try lowering threshold or re-uploading.")

            st.subheader("📝 Candidate Texts")
            for cand in result.get("all_candidates", []):
                st.markdown(f"- {cand['text']} — `{cand['confidence']*100:.1f}%`")
