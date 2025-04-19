import cv2
import paddleocr
import os
import time
import numpy as np

# --- Configuration ---
IMAGE_PATH = 'sample.png' # Use the generated test image!
# LANGUAGE = 'en' # We set language later
USE_GPU = True

# --- Initialize PaddleOCR ---
print("Initializing PaddleOCR (Using Multilingual_v3 detector, English_v4 recognizer)...")

# Define model paths
ml_det_dir = os.path.expanduser('~/.paddleocr/whl/det/ml/Multilingual_PP-OCRv3_det_infer')
en_rec_dir = os.path.expanduser('~/.paddleocr/whl/rec/en/en_PP-OCRv4_rec_infer') # English v4 Recognizer

ocr_engine = paddleocr.PaddleOCR(
    use_angle_cls=False,
    # *** Specify 'ml' for detection language, but keep 'en' for recognition if needed ***
    # The 'lang' parameter often sets defaults for both, overriding might be needed
    # Let's try setting lang='ml' first, it might still use the specific rec_model_dir
    lang='en',
    use_gpu=USE_GPU,
    show_log=True,
    # --- Point to the correct models ---
    det_model_dir=ml_det_dir,   # Use the Multilingual Detector
    rec_model_dir=en_rec_dir,   # Use the English v4 Recognizer
    # --- Use slightly higher thresholds, maybe v3 needs it ---
    det_db_thresh=0.3, # Back to default
    det_db_box_thresh=0.6, # Back to default
)
print("PaddleOCR Initialized.")

# --- Load Image ---
# ... (rest of image loading code) ...
img = cv2.imread(IMAGE_PATH)
# ... (error checking) ...
print(f"Image loaded successfully: {IMAGE_PATH}, Shape: {img.shape}")


# --- Perform OCR ---
print("Running OCR...")
start_time = time.time()
# Need to explicitly pass 'en' to recognition? Check docs. Usually rec_model_dir overrides.
results = ocr_engine.ocr(img, cls=False)
end_time = time.time()
print(f"OCR finished in {end_time - start_time:.4f} seconds.")

# --- Process and Print Results ---
# ... (rest of the processing loop) ...
print(f"Final OCR Results variable: {results}")
# ... (drawing and saving) ...
output_filename = 'output/paddleocr_output_ml_det.png'
# ... (save code) ...
print(f"Output image saved to {output_filename}")

print("Script finished.")