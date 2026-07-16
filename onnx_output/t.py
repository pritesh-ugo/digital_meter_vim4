import cv2
import numpy as np
from ksnn.api import KSNN
import sys

# --- 1. Load the Dictionary ---
# Read your en_dict.txt to map NPU numbers back to English letters
with open('en_dict.txt', 'r', encoding='utf-8') as f:
    char_list = [line.strip('\n') for line in f.readlines()]

# CTC decoding in PaddleOCR standardly adds a 'blank' token at index 0 and a space at the end
character_dict = ['blank'] + char_list + [' '] 
dict_size = len(character_dict)

# --- 2. Initialize the VIM4 NPU ---
print("Loading Model to NPU...")
ocr_model = KSNN('VIM4')
ocr_model.nn_init(library="./libnn_ppocr_rec.so", model="./inference_int16.adla", level=0)

# --- 3. Prepare the Image ---
print("Processing Image 'a-2.jpg'...")
img = cv2.imread('a-2.jpg')

# Safety check in case the image path is wrong
if img is None:
    print("Error: Could not load 'a.jpg'. Please ensure the file is in the same directory.")
    sys.exit(1)

# OpenCV loads in BGR, we need standard RGB
#img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
#img = cv2.resize(img, (320, 48))  # Force to our fixed shape

# Fix 1: Convert the array to 32-bit floats
img = img.astype(np.float32)

# Fix 2: Rearrange from HWC (48, 320, 3) to CHW (3, 48, 320) as ONNX demands
img = img.transpose(2, 0, 1)

# Fix 3: Wrap it in a list. The API expects a list of inputs.
input_data = img

# --- 4. Run NPU Inference ---
print("Running NPU Inference...")
# Passing arguments positionally to satisfy the KSNN 1.4.0.2 API requirements
outputs = ocr_model.nn_inference(
    input_data, 
    (3, 48, 320),        # input_shape: Wrapped in a list to match the input_data list
    "RAW",             # input_type: Explicitly telling the driver we are passing floats
    [(1, 40, dict_size)],  # output_shape: Batch, Sequence Length, Dictionary Size
    "FLOAT"              # output_type: Returns raw probabilities to Python as decimals
)

# --- 5. Decode the Output (CTC Decoding) ---
# Reshape the flat output array back into a 3D matrix
raw_output = np.array(outputs[0]).reshape(1, -1, dict_size)

# Find the index of the highest probability character in each column (time step)
predicted_indices = np.argmax(raw_output[0], axis=1)

# CTC Decode logic: ignore blanks (index 0) and ignore consecutive duplicate letters
decoded_word = []
for i in range(len(predicted_indices)):
    if predicted_indices[i] != 0 and not (i > 0 and predicted_indices[i - 1] == predicted_indices[i]):
        decoded_word.append(character_dict[predicted_indices[i]])

final_text = "".join(decoded_word)

# --- 6. Print Results ---
print("-" * 30)
print(f"RAW NPU MATRIX SHAPE: {raw_output.shape}")
print(f"PREDICTED TEXT: '{final_text}'")
print("-" * 30)
