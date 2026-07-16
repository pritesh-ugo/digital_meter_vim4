import os
import sys
import time  # Imported for tracking times

import cv2
import numpy as np
from paddleocr import PaddleOCR

# Pre Processing the image 

# Tunable parameters – adjust these if your images change
ADAPTIVE_BLOCK_SIZE     = 51    # Local neighborhood size for adaptive threshold (must be odd)
ADAPTIVE_C              = 5     # How much darker a pixel must be than its neighborhood
MIN_COMPONENT_AREA      = 20    # Drop blobs smaller than this (noise specks)
MAX_COMPONENT_AREA_FRAC = 0.4   # Drop blobs larger than this fraction of image (LCD frame)
PADDING_SIZE            = 150   # Extra black border around digits (helps OCR framing)


def clean_image(image_path: str) -> np.ndarray:
    """
    Load a raw LCD/7-segment photo from *image_path*, clean it, and return
    the cleaned image as a BGR numpy array (ready for PaddleOCR).
    """
    # -- 1. Load ---------------------------------------------------------------
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: '{image_path}'")

    # -- 2. Denoise ------------------------------------------------------------
    denoised = cv2.bilateralFilter(img, d=9, sigmaColor=50, sigmaSpace=50)

    # -- 3. Adaptive threshold -------------------------------------------------
    binary = cv2.adaptiveThreshold(
        denoised, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        ADAPTIVE_BLOCK_SIZE, ADAPTIVE_C
    )

    # -- 4. Morphological cleanup ----------------------------------------------
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # -- 5. Connected-component filter -----------------------------------------
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    clean_mask = np.zeros_like(binary)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] > MIN_COMPONENT_AREA:
            clean_mask[labels == i] = 255

    # -- 6. Render: red glow on black background with padding ------------------
    h, w = clean_mask.shape
    padded_h = h + 2 * PADDING_SIZE
    padded_w = w + 2 * PADDING_SIZE

    output_bgr = np.zeros((padded_h, padded_w, 3), dtype=np.uint8)

    # Soft glow in the red channel
    glow = cv2.GaussianBlur(clean_mask, (15, 15), 0)
    output_bgr[PADDING_SIZE:PADDING_SIZE + h, PADDING_SIZE:PADDING_SIZE + w, 2] = glow

    # Solid bright-red pixels on actual digit locations (sharp core)
    roi = output_bgr[PADDING_SIZE:PADDING_SIZE + h, PADDING_SIZE:PADDING_SIZE + w]
    roi[clean_mask > 0] = [50, 50, 255]

    return output_bgr   # BGR ndarray – drop-in for cv2.imread() output


# Running the OCR Model

print("Loading the OCR Model")
ocr = PaddleOCR(
    rec_model_dir='./meter_ocr_model',
    rec_char_dict_path='./meter_ocr_model/en_dict.txt',
    rec_algorithm='SVTR_LCNet',
    use_angle_cls=False,
    use_gpu=False,      # Set True if you have NVIDIA GPU + CUDA
    show_log=False
)
print("Model loaded.\n")


def run_pipeline(image_path: str) -> list[tuple[str, float]]:
    """
    Full pipeline:
      raw image path → clean → OCR → list of (text, confidence) tuples.

    Returns an empty list if nothing is detected.
    """
    if not os.path.exists(image_path):
        print(f"File not found: '{image_path}'")
        return []

    print(f"Input image : {image_path}")
    
    pipeline_start = time.perf_counter()

    # ── Stage 1: clean (Preprocessing) ────────────────────────────────────────
    print("Cleaning image...")
    preprocess_start = time.perf_counter()
    try:
        cleaned_frame = clean_image(image_path)
    except FileNotFoundError as e:
        print(f"{e}")
        return []
    preprocess_time = time.perf_counter() - preprocess_start

    # ── Stage 2: OCR (Inference) ──────────────────────────────────────────────
    print("Running OCR inference...")
    inference_start = time.perf_counter()
    try:
        result = ocr.ocr(cleaned_frame, det=True, cls=False)
    except Exception as e:
        print(f"OCR pipeline error: {e}")
        return []
    inference_time = time.perf_counter() - inference_start

    # ── Stage 3: Parse results (Postprocessing) ─────────────────────────────────
    postprocess_start = time.perf_counter()
    readings: list[tuple[str, float]] = []

    if result and result[0]:
        print(f"\nDetection Results for {os.path.basename(image_path)} ---")
        for idx, line in enumerate(result[0], 1):
            box        = line[0]            # 4-corner bounding box
            pred_text  = line[1][0]         # recognised string
            confidence = line[1][1]         # confidence score 0–1

            readings.append((pred_text, confidence))
            print(f"  [{idx}] Text: '{pred_text}'  |  Confidence: {confidence:.2f}")
            print(f"       Box: {box}")
        print("----------------------------------------------------\n")
    else:
        print("No 7-segment digits detected in this image.\n")
    postprocess_time = time.perf_counter() - postprocess_start
    
    pipeline_total_time = time.perf_counter() - pipeline_start

    # ── Print Performance Breakdown ───────────────────────────────────────────
    print("⏱️  Execution Time Breakdown:")
    print(f"  • Preprocessing : {preprocess_time * 1000:.2f} ms")
    print(f"  • OCR Inference : {inference_time * 1000:.2f} ms")
    print(f"  • Postprocessing: {postprocess_time * 1000:.2f} ms")
    print(f"  • Total Pipeline: {pipeline_total_time * 1000:.2f} ms")
    print("----------------------------------------------------\n")

    return readings


# Loading Image 

if __name__ == "__main__":
    # Pass the target image as a CLI argument, or fall back to the default
    if len(sys.argv) > 1:
        target = sys.argv[1]
    else:
        target = "2.png"          # ← change this default to your usual test image

    readings = run_pipeline(target)

    if readings:
        print("Final readings:")
        for text, conf in readings:
            print(f"   → '{text}'  (confidence: {conf:.2%})")
    else:
        print("~No readings extracted.")
