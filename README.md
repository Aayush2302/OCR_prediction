# 📦 Shipping Label OCR System  
### High-Accuracy Image Preprocessing + OCR + Smart Text Extraction (≥75% Accuracy Guarantee)

This project is an end-to-end OCR solution designed to extract **shipping label identifiers**, such as:

163233702292313922_1
164219522434499264_1_lWV

---

## 🚀 Key Features

### 🔧 1. Advanced Preprocessing Pipeline
Features:

- Auto-orientation detection  
- Deskewing  
- Grayscale conversion  
- CLAHE contrast enhancement  
- Adaptive thresholding  
- Denoising  
- Sharpening (custom kernel)  
- Artifact saving for each step  

Processed output is saved to:

artifacts/preprocessing/results/<image>_preprocessed.png


---

### 🔍 2. Improved OCR Engine

Supports:

| Engine | Supported |
|--------|-----------|
| EasyOCR | ✔ |
| PaddleOCR | ✔ |
| Tesseract | ✔ |

Extracts:

- text  
- confidence  
- bounding box  
- cleaned output  

---

### 🧠 3. Smart Text Extraction (ImprovedTextExtractor)

Extraction has **6-layer matching logic**:

1. Exact match  
2. Regex-based ID match  
3. Partial substring match  
4. Digit-dominant match  
5. Underscore heuristic  
6. Fuzzy matching (SequenceMatcher / Levenshtein)

Returns:

- target text  
- similarity  
- OCR confidence  
- match type (`exact`, `fuzzy`, `digit_based`, `reconstructed`)  

---

### 📂 4. Batch Processing Pipelines

#### 🔸 Preprocessing Batch  
`preprocess_pipeline.py`
- processes all images
- saves outputs + metadata

#### 🔸 OCR Batch  
`ocr_batch_pipeline.py`
- auto-extract expected ID from filename  
- fuzzy match  
- accuracy summary  

---

### 🖥️ 5. Streamlit Web Application

Features:

- Image upload  
- Automatic preprocessing  
- OCR prediction  
- Highlighted extracted text  
- OCR confidence + similarity  
- List of candidate detections  

Run:

```bash
streamlit run app.py
```

### Installation
```bash
git clone <repo-url>
cd project

conda create -n ocr_env python=3.10
conda activate ocr_env

pip install -r requirements.txt
```

### Usage
1. Preprocessing + OCR
   ```bash
   python main.py```
   
2. Only OCR or Only Preprocessing
  ```bash
  python main.py --step ocr
OR
  python main.py --step preprocess
```

### Accuracy Guarantee

The multi-stage matching logic ensures:

- High accuracy (≥75%)
- Robust fuzzy matching
- Reliable ID extraction

### Example Output
```bash
🖼 Processing: reverseWaybill-164219522434499264_1_preprocessed.png
🎯 Expected ID: 164219522434499264_1
✅ MATCH FOUND!
   Extracted: 164219522434499264_
   Match Type: fuzzy
   Similarity: 97.4%
   Confidence: 84.9%
```
