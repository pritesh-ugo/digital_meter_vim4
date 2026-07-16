import os
import sys
import time  # Added for precision profiling
import cv2
import numpy as np
from ksnn.api import KSNN

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEBUG_DIR = os.path.join(BASE_DIR, "debug")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# --- Pre-Processing Tunable Parameters ---
ADAPTIVE_BLOCK_SIZE     = 51    
ADAPTIVE_C              = 5     
MIN_COMPONENT_AREA      = 20    

# --- 1. Load the Dictionary ---
with open('en_dict.txt', 'r', encoding='utf-8') as f:
    char_list = [line.strip('\n') for line in f.readlines()]

character_dict = ['blank'] + char_list + [' '] 
dict_size = len(character_dict)

# --- 2. Initialize the VIM4 NPU ---
print("Loading Model to NPU...")
ocr_model = KSNN('VIM4')
ocr_model.nn_init(library="./libnn_ppocr_rec.so", model="./inference_int16.adla", level=0)

os.makedirs(DEBUG_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def clean_and_format_image(image_path: str, save_debug_img: bool = True) -> np.ndarray:
    """Applies preprocessing steps and formats the result as a raw FLOAT32 CHW tensor."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: '{image_path}'")

    denoised = cv2.bilateralFilter(img, d=9, sigmaColor=50, sigmaSpace=50)

    binary = cv2.adaptiveThreshold(
        denoised, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        ADAPTIVE_BLOCK_SIZE, ADAPTIVE_C
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    clean_mask = np.zeros_like(binary)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] > MIN_COMPONENT_AREA:
            clean_mask[labels == i] = 255

    h, w = clean_mask.shape
    output_bgr = np.zeros((h, w, 3), dtype=np.uint8)

    glow = cv2.GaussianBlur(clean_mask, (15, 15), 0)
    output_bgr[:, :, 2] = glow
    output_bgr[clean_mask > 0] = [50, 50, 255]

    final_img_bgr = cv2.resize(output_bgr, (320, 48))

    if save_debug_img:
        cv2.imwrite(os.path.join(DEBUG_DIR, "processed_input_320x48.jpg"), final_img_bgr)

    output_rgb = cv2.cvtColor(final_img_bgr, cv2.COLOR_BGR2RGB)
    final_img_tensor = output_rgb.astype(np.float32)
    final_img_tensor = final_img_tensor.transpose(2, 0, 1)

    return final_img_tensor


def load_preprocessed_image(image_path: str, save_debug_img: bool = True) -> np.ndarray:
    """Loads an already-preprocessed 320x48 RGB/BGR image and formats it as RAW CHW tensor."""
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: '{image_path}'")

    if img.shape[1] != 320 or img.shape[0] != 48:
        img = cv2.resize(img, (320, 48))

    if save_debug_img:
        cv2.imwrite(os.path.join(DEBUG_DIR, "processed_input_320x48.jpg"), img)

    output_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    final_img_tensor = output_rgb.astype(np.float32)
    final_img_tensor = final_img_tensor.transpose(2, 0, 1)

    return final_img_tensor


# --- 3. Execute Pipeline with Timing Profiles ---
if __name__ == "__main__":
    args = [arg for arg in sys.argv[1:] if not arg.startswith("--")]
    target_image = args[0] if len(args) > 0 else "a.jpg"
    preprocessed_input = "--preprocessed" in sys.argv

    # ==========================================
    # PHASE 1: PRE-PROCESSING TIME
    # ==========================================
    start_preprocess = time.perf_counter()
    try:
        if preprocessed_input:
            input_data = load_preprocessed_image(target_image, save_debug_img=True)
        else:
            input_data = clean_and_format_image(target_image, save_debug_img=True)
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        sys.exit(1)
    end_preprocess = time.perf_counter()
    preprocess_time = (end_preprocess - start_preprocess) * 1000.0  # Convert to ms

    # ==========================================
    # PHASE 2: NPU INFERENCE TIME
    # ==========================================
    start_inference = time.perf_counter()
    outputs = ocr_model.nn_inference(
        input_data, 
        (3, 48, 320),          
        "RAW",                 
        [(1, 40, dict_size)],  
        "FLOAT"                
    )
    end_inference = time.perf_counter()
    inference_time = (end_inference - start_inference) * 1000.0  # Convert to ms

    # ==========================================
    # PHASE 3: POST-PROCESSING (CTC DECODE) TIME
    # ==========================================
    start_postprocess = time.perf_counter()
    raw_output = np.array(outputs[0]).reshape(1, -1, dict_size)
    predicted_indices = np.argmax(raw_output[0], axis=1)

    decoded_word = []
    for i in range(len(predicted_indices)):
        if predicted_indices[i] != 0 and not (i > 0 and predicted_indices[i - 1] == predicted_indices[i]):
            decoded_word.append(character_dict[predicted_indices[i]])

    final_text = "".join(decoded_word)
    end_postprocess = time.perf_counter()
    postprocess_time = (end_postprocess - start_postprocess) * 1000.0  # Convert to ms

    total_pipeline_time = preprocess_time + inference_time + postprocess_time

    # ==========================================
    # PRINT PERFORMANCE SUMMARY
    # ==========================================
    print("\n" + "="*40)
    print("      VIM4 NPU PIPELINE BENCHMARK      ")
    print("="*40)
    print(f"PRE-PROCESS TIME  : {preprocess_time:7.2f} ms  (CV2 Filters & Cast)")
    print(f"NPU INFERENCE TIME: {inference_time:7.2f} ms  (INT16 Silicon Execution)")
    print(f"POST-PROCESS TIME : {postprocess_time:7.2f} ms  (CTC Dictionary Decode)")
    print("-" * 40)
    print(f"TOTAL TIME        : {total_pipeline_time:7.2f} ms")
    print(f"ESTIMATED MAX FPS : {1000.0 / total_pipeline_time:7.2f} frames/sec")
    print("="*40)
    print(f"PREDICTED TEXT    : '{final_text}'")
    print("="*40 + "\n")

    with open(os.path.join(RESULTS_DIR, "ocr_result.txt"), "w", encoding="utf-8") as result_file:
        result_file.write(final_text + "\n")
        result_file.write(f"preprocess_ms={preprocess_time:.2f}\n")
        result_file.write(f"inference_ms={inference_time:.2f}\n")
        result_file.write(f"postprocess_ms={postprocess_time:.2f}\n")
        result_file.write(f"total_ms={total_pipeline_time:.2f}\n")
