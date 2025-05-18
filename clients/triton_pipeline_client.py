import tritonclient.http as httpclient
from tritonclient.utils import np_to_triton_dtype
import numpy as np
import cv2
import time
import os
import math
import yaml
from shapely.geometry import Polygon
import pyclipper
import argparse

# === OCR Configuration ===
OCR_DET_MODEL_NAME = "ocr_ml_detector"    # Name in Triton model repo
OCR_DET_INPUT_NAME = "x"                  # Check Netron for your ocr_ml_detector ONNX
OCR_DET_OUTPUT_NAME = "fetch_name_0"      # Check Netron for your ocr_ml_detector ONNX

LANG_CONFIG = {
    'en': {
        'rec_model_name': 'ocr_en_recogniser', # Your English recognizer in Triton
        'rec_input_name': 'x',                # Check Netron
        'rec_output_name': 'fetch_name_0',     # Check Netron
        'dict_path': 'dicts/en_dict.txt',
        'rec_img_h': 48
    },
    'ja': {
        'rec_model_name': 'ocr_ja_recogniser', # Your Japanese recognizer in Triton
        'rec_input_name': 'x',                # Check Netron
        'rec_output_name': 'fetch_name_0',     # Check Netron
        'dict_path': 'dicts/japan_dict.txt',
        'rec_img_h': 48
    },
    'ko': {
        'rec_model_name': 'ocr_ko_recogniser', # Your Korean recognizer in Triton
        'rec_input_name': 'x',                # Check Netron
        'rec_output_name': 'fetch_name_0',     # Check Netron
        'dict_path': 'dicts/korean_dict.txt',
        'rec_img_h': 48
    }
}

# OCR Preprocessing
DET_MAX_SIDE_LEN = 960
DET_IMG_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
DET_IMG_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
REC_IMG_MEAN_OCR = 0.5 # For OCR recognizer
REC_IMG_STD_OCR = 0.5  # For OCR recognizer
# OCR Postprocessing
DET_DB_THRESH = 0.3
DET_DB_BOX_THRESH = 0.6 # Lower this if detector misses faint text
DET_DB_UNCLIP_RATIO = 1.5
MIN_BOX_AREA_OCR = 10
MIN_BOX_SCORE_OCR = 0.5

# === LaMa Inpainting Configuration ===
LAMA_MODEL_NAME_PY = "lama_inpainter" # Name of Python backend model in Triton
LAMA_INPUT_IMAGE_NAME_PY = "IMAGE_IN"    # Matches input name in lama_inpainter_py/config.pbtxt
LAMA_INPUT_MASK_NAME_PY = "MASK_IN"      # Matches input name in lama_inpainter_py/config.pbtxt
LAMA_OUTPUT_NAME_PY = "INPAINTED_OUT"    # Matches output name in lama_inpainter_py/config.pbtxt
LAMA_TARGET_SIZE = 512 # Resize image/mask to this square size for LaMa input (or 0 for no resize)


# === Global Triton URL ===
TRITON_URL_DEFAULT = "localhost:8000"


def load_character_dict(dict_path):
    char_list = []
    with open(dict_path, "r", encoding="utf-8") as f:
        for line in f:
            char_list.append(line.strip())
    char_list = ['<blank>'] + char_list
    char_map = {char: i for i, char in enumerate(char_list)}
    print(f"Loaded dictionary: {len(char_list)} chars, first 10: {char_list[:10]}")
    print(char_map)
    return char_list, char_map


def preprocess_det_image(img, max_side_len=DET_MAX_SIDE_LEN):
    h, w, _ = img.shape
    if max(h, w) > max_side_len: ratio = float(max_side_len) / max(h, w)
    else: ratio = 1.0
    resize_h, resize_w = int(h * ratio), int(w * ratio)
    resize_h = max(int(round(resize_h / 32) * 32), 32)
    resize_w = max(int(round(resize_w / 32) * 32), 32)
    resized_img = cv2.resize(img, (resize_w, resize_h))
    img_normalized = (resized_img.astype(np.float32) / 255.0 - DET_IMG_MEAN) / DET_IMG_STD
    img_transposed = img_normalized.transpose((2, 0, 1))
    return np.expand_dims(img_transposed, axis=0), (ratio, h, w)


def preprocess_rec_image_ocr(img_crop, rec_img_h):
    h, w, _ = img_crop.shape
    ratio = rec_img_h / float(h)
    resize_w = int(w * ratio)
    resize_w = max(32, int(round(resize_w / 4) * 4))
    resized_img = cv2.resize(img_crop, (resize_w, rec_img_h))
    img_normalized = (resized_img.astype(np.float32) / 255.0 - REC_IMG_MEAN_OCR) / REC_IMG_STD_OCR
    img_transposed = img_normalized.transpose((2, 0, 1))
    return np.expand_dims(img_transposed, axis=0)


def postprocess_recognition(rec_output_raw, char_list):
    if rec_output_raw.ndim == 3 and rec_output_raw.shape[0] == 1: preds = rec_output_raw[0]
    elif rec_output_raw.ndim == 2: preds = rec_output_raw
    else: return "Decoding Error", 0.0
    pred_indices = np.argmax(preds, axis=1)
    pred_probs = np.max(preds, axis=1)
    decoded_indices, decoded_probs_list = [], []
    last_idx = -1
    for i, idx in enumerate(pred_indices):
        if idx != 0 and idx != last_idx:
            decoded_indices.append(idx)
            decoded_probs_list.append(pred_probs[i])
        last_idx = idx
    text = "".join([char_list[i] for i in decoded_indices if 0 < i < len(char_list)])
    confidence = np.mean(decoded_probs_list) if decoded_probs_list else 0.0
    return text, float(confidence)


class DBPostProcess:
    def __init__(self, thresh=DET_DB_THRESH, box_thresh=DET_DB_BOX_THRESH, max_candidates=1000,
                 unclip_ratio=DET_DB_UNCLIP_RATIO, min_size=MIN_BOX_AREA_OCR, box_type='quad'): # Use OCR specific min_size
        self.thresh = thresh
        self.box_thresh = box_thresh
        self.max_candidates = max_candidates
        self.unclip_ratio = unclip_ratio
        self.min_size = min_size
        self.box_type = box_type
    
    def polygons_from_bitmap(self, pred, bitmap, dest_width, dest_height):
        contours, _ = cv2.findContours((bitmap * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        num_contours = min(len(contours), self.max_candidates); boxes = []; scores = []
        for contour in contours[:num_contours]:
            points, sside = self.get_mini_boxes(contour)
            if sside < self.min_size: continue
            score = self.box_score_fast(pred, points.reshape(-1, 2)) # Use pred here
            if self.box_thresh > score: continue
            box = self.unclip(points).reshape(-1, 2) # unclip returns (N,2)
            if box.shape[0] != 4: continue # Ensure it's a quad after unclip logic
            box[:, 0] = np.clip(np.round(box[:, 0]), 0, dest_width - 1)
            box[:, 1] = np.clip(np.round(box[:, 1]), 0, dest_height - 1)
            boxes.append(box); scores.append(score)
        return boxes, scores

    def box_score_fast(self, bitmap, _box):
        h, w = bitmap.shape; box = _box.copy()
        xmin = np.clip(np.floor(box[:, 0].min()).astype(int), 0, w - 1)
        xmax = np.clip(np.ceil(box[:, 0].max()).astype(int), 0, w - 1)
        ymin = np.clip(np.floor(box[:, 1].min()).astype(int), 0, h - 1)
        ymax = np.clip(np.ceil(box[:, 1].max()).astype(int), 0, h - 1)
        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        box[:, 0] -= xmin; box[:, 1] -= ymin
        cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)
        return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]

    def unclip(self, box):
        poly = Polygon(box); distance = poly.area * self.unclip_ratio / poly.length
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        expanded = offset.Execute(distance)
        if not expanded: return cv2.boxPoints(cv2.minAreaRect(box)) # Fallback
        # Find largest expanded polygon (simplified, assumes one dominant)
        # expanded_poly = max(expanded, key=lambda p: cv2.contourArea(np.array(p).reshape(-1,1,2)))
        expanded_poly = np.array(expanded[0]).reshape(-1, 2)
        return cv2.boxPoints(cv2.minAreaRect(expanded_poly))

    def get_mini_boxes(self, contour):
        bounding_box = cv2.minAreaRect(contour)
        return cv2.boxPoints(bounding_box), min(bounding_box[1])

    def __call__(self, det_output_raw, resize_info):
        if det_output_raw.ndim == 4 and det_output_raw.shape[0:2] == (1, 1): pred = det_output_raw[0, 0, :, :]
        else: return []
        bitmap = pred > self.thresh
        resize_ratio, orig_h, orig_w = resize_info
        dest_height, dest_width = pred.shape
        boxes, scores = self.polygons_from_bitmap(pred, bitmap, dest_width, dest_height) # Pass pred
        final_boxes = []
        if boxes:
            boxes_np = np.array(boxes)
            if resize_ratio == 0: return []
            boxes_np /= resize_ratio
            boxes_np[:, :, 0] = np.clip(boxes_np[:, :, 0], 0, orig_w - 1)
            boxes_np[:, :, 1] = np.clip(boxes_np[:, :, 1], 0, orig_h - 1)
            for i, box in enumerate(boxes_np):
                 if scores[i] >= MIN_BOX_SCORE_OCR:
                    final_boxes.append(box.tolist())
        return final_boxes


def order_points_clockwise(pts):
    pts_np = np.array(pts, dtype="float32")
    xSorted = pts_np[np.argsort(pts_np[:, 0]), :]
    leftMost = xSorted[:2, :]; rightMost = xSorted[2:, :]
    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]; (tl, bl) = leftMost
    D = np.linalg.norm(rightMost - tl, axis=1)
    rightMost = rightMost[np.argsort(D)[::-1], :]; (br, tr) = rightMost
    return np.array([tl, tr, br, bl], dtype="float32")


def get_rotate_crop_image(img, points):
    ordered_points = order_points_clockwise(points)
    width_A = np.linalg.norm(ordered_points[0] - ordered_points[1])
    width_B = np.linalg.norm(ordered_points[3] - ordered_points[2])
    img_crop_width = int(max(width_A, width_B))
    height_A = np.linalg.norm(ordered_points[0] - ordered_points[3])
    height_B = np.linalg.norm(ordered_points[1] - ordered_points[2])
    img_crop_height = int(max(height_A, height_B))
    if img_crop_width <= 0 or img_crop_height <= 0: return None
    pts_std = np.float32([[0, 0], [img_crop_width, 0], [img_crop_width, img_crop_height], [0, img_crop_height]])
    M = cv2.getPerspectiveTransform(ordered_points, pts_std)
    return cv2.warpPerspective(img, M, (img_crop_width, img_crop_height), borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_CUBIC)


# === LaMa Preprocessing and Postprocessing (Client Side) ===
def preprocess_lama_inputs_for_triton_py(image_bgr, mask_gray, target_size=LAMA_TARGET_SIZE):
    orig_h, orig_w = image_bgr.shape[:2]
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    current_target_size = target_size if target_size and target_size > 0 else None
    if current_target_size:
        img_resized = cv2.resize(img_rgb, (current_target_size, current_target_size), interpolation=cv2.INTER_AREA)
        mask_resized = cv2.resize(mask_gray, (current_target_size, current_target_size), interpolation=cv2.INTER_NEAREST)
    else:
        img_resized = img_rgb; mask_resized = mask_gray
    img_normalized_neg1_pos1 = (img_resized.astype(np.float32) / 255.0) * 2.0 - 1.0
    img_chw = img_normalized_neg1_pos1.transpose(2, 0, 1)
    img_batch = np.expand_dims(img_chw, axis=0)
    mask_hole_is_1 = (mask_resized > 127).astype(np.float32)
    mask_1_channel_hw = mask_hole_is_1
    mask_chw = np.expand_dims(mask_1_channel_hw, axis=0)
    mask_batch = np.expand_dims(mask_chw, axis=0)
    return img_batch, mask_batch, (orig_h, orig_w)


def postprocess_lama_output_from_triton_py(inpainted_batch, original_shape, target_size_used):
    if inpainted_batch.ndim == 4 and inpainted_batch.shape[0] == 1: inpainted_chw = inpainted_batch[0]
    else: inpainted_chw = inpainted_batch
    inpainted_hwc = inpainted_chw.transpose(1, 2, 0)
    inpainted_0_1 = (inpainted_hwc + 1.0) / 2.0
    inpainted_0_255 = (np.clip(inpainted_0_1, 0, 1) * 255.0).astype(np.uint8)
    inpainted_bgr = cv2.cvtColor(inpainted_0_255, cv2.COLOR_RGB2BGR)
    current_target_size_used = target_size_used if target_size_used and target_size_used > 0 else None
    if current_target_size_used:
        orig_h, orig_w = original_shape
        return cv2.resize(inpainted_bgr, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
    return inpainted_bgr


# === Main Pipeline Execution ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Triton OCR + LaMa Inpainting Client")
    parser.add_argument('--image', type=str, required=True, help='Path to the input image file.')
    parser.add_argument('--lang', type=str, required=True, choices=LANG_CONFIG.keys(), help='Target language code.')
    parser.add_argument('--output_dir', type=str, default='output_pipeline', help='Directory for all output images.')
    parser.add_argument('--triton_url', type=str, default=TRITON_URL_DEFAULT, help='Triton server URL.')
    parser.add_argument('--lama_target_size', type=int, default=LAMA_TARGET_SIZE, help='Target size for LaMa input (0 for no resize).')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    IMAGE_PATH = args.image
    TARGET_LANG = args.lang
    TRITON_URL = args.triton_url
    CURRENT_LAMA_TARGET_SIZE = args.lama_target_size if args.lama_target_size > 0 else None

    # --- Get language specific OCR config ---
    if TARGET_LANG not in LANG_CONFIG:
        print(f"Error: Unsupported language '{TARGET_LANG}'. Supported: {list(LANG_CONFIG.keys())}")
        exit()
    ocr_lang_config = LANG_CONFIG[TARGET_LANG]
    OCR_REC_MODEL_NAME = ocr_lang_config['rec_model_name']
    OCR_REC_INPUT_NAME = ocr_lang_config['rec_input_name']
    OCR_REC_OUTPUT_NAME = ocr_lang_config['rec_output_name']
    OCR_CHARACTER_DICT_PATH = ocr_lang_config['dict_path']
    OCR_REC_IMG_H = ocr_lang_config['rec_img_h']

    # Load original image
    img_orig = cv2.imread(IMAGE_PATH)
    if img_orig is None:
        print(f"Error: Could not load image {IMAGE_PATH}"); exit()
    print(f"Pipeline: Loaded original image: {IMAGE_PATH}, Shape: {img_orig.shape}")

    # Load OCR character dictionary
    if not os.path.exists(OCR_CHARACTER_DICT_PATH):
        print(f"Error: OCR Character dictionary not found: {OCR_CHARACTER_DICT_PATH}"); exit()
    ocr_char_list, _ = load_character_dict(OCR_CHARACTER_DICT_PATH)

    # Initialize OCR DB Postprocessor
    ocr_db_postprocessor = DBPostProcess()

     # Create Triton Client and Check All Models
    try:
        triton_client = httpclient.InferenceServerClient(url=TRITON_URL, verbose=False)
        print(f"Pipeline: Checking Triton server Liveness ({TRITON_URL})...")
        if not triton_client.is_server_live(): print(f"Error: Triton server is not live"); exit()
        
        print("Pipeline: Checking Model Readiness...")
        models_to_check = [OCR_DET_MODEL_NAME, OCR_REC_MODEL_NAME, LAMA_MODEL_NAME_PY]
        for model_name_check in models_to_check:
            if not triton_client.is_model_ready(model_name_check):
                print(f"Error: Model '{model_name_check}' is not ready on Triton server."); exit()
        print(f"Pipeline: Triton client connected. All models ({', '.join(models_to_check)}) are ready.")
    except Exception as e:
        print(f"Error creating Triton client or checking models: {e}"); exit()

    
    # === 1. OCR PHASE ===
    print("\n--- OCR Phase ---")
    # Preprocess for detection
    ocr_det_input_image, ocr_resize_info = preprocess_det_image(img_orig)
    
    # Detection inference
    ocr_det_inputs = [httpclient.InferInput(OCR_DET_INPUT_NAME, ocr_det_input_image.shape, np_to_triton_dtype(ocr_det_input_image.dtype))]
    ocr_det_inputs[0].set_data_from_numpy(ocr_det_input_image)
    ocr_det_outputs = [httpclient.InferRequestedOutput(OCR_DET_OUTPUT_NAME)]
    
    print(f"Sending OCR detection request to '{OCR_DET_MODEL_NAME}'...")
    ocr_det_results_triton = triton_client.infer(model_name=OCR_DET_MODEL_NAME, inputs=ocr_det_inputs, outputs=ocr_det_outputs)
    ocr_det_output_raw = ocr_det_results_triton.as_numpy(OCR_DET_OUTPUT_NAME)
    
    # Postprocess detection
    detected_boxes_poly = ocr_db_postprocessor(ocr_det_output_raw, ocr_resize_info)
    print(f"OCR Detection: Found {len(detected_boxes_poly)} potential text boxes.")
    cv2.imwrite(os.path.join(args.output_dir, f"ocr_detection_viz_{TARGET_LANG}.png"), 
                cv2.polylines(img_orig.copy(), [np.array(p, dtype=np.int32) for p in detected_boxes_poly], True, (0,255,0), 2))


    # Recognition inference
    all_ocr_results_list = [] # To store {'box': ..., 'text': ..., 'confidence': ...}
    img_for_ocr_drawing = img_orig.copy()

    if detected_boxes_poly:
        for i, box_poly_list in enumerate(detected_boxes_poly):
            box_poly_np = np.array(box_poly_list, dtype=np.float32)
            if box_poly_np.shape != (4,2): continue # Skip invalid shapes from DB postprocess

            img_crop_ocr = get_rotate_crop_image(img_orig, box_poly_np)
            if img_crop_ocr is None or img_crop_ocr.shape[0] < 8 or img_crop_ocr.shape[1] < 8: continue

            ocr_rec_input_image = preprocess_rec_image_ocr(img_crop_ocr, OCR_REC_IMG_H)
            
            ocr_rec_inputs = [httpclient.InferInput(OCR_REC_INPUT_NAME, ocr_rec_input_image.shape, np_to_triton_dtype(ocr_rec_input_image.dtype))]
            ocr_rec_inputs[0].set_data_from_numpy(ocr_rec_input_image)
            ocr_rec_outputs = [httpclient.InferRequestedOutput(OCR_REC_OUTPUT_NAME)]
            
            ocr_rec_results_triton = triton_client.infer(model_name=OCR_REC_MODEL_NAME, inputs=ocr_rec_inputs, outputs=ocr_rec_outputs)
            ocr_rec_output_raw = ocr_rec_results_triton.as_numpy(OCR_REC_OUTPUT_NAME)
            
            text, confidence = postprocess_recognition(ocr_rec_output_raw, ocr_char_list)
            all_ocr_results_list.append({'box': box_poly_list, 'text': text, 'confidence': confidence})
            print(f"  OCR Box {i}: Text='{text}', Conf={confidence:.3f}")
            
            # Draw on OCR visualization image
            cv2.polylines(img_for_ocr_drawing, [box_poly_np.astype(np.int32)], True, (0,0,255), 2)
            cv2.putText(img_for_ocr_drawing, f"{text} ({confidence:.2f})", (int(box_poly_np[0,0]), int(box_poly_np[0,1])-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)
        cv2.imwrite(os.path.join(args.output_dir, f"ocr_recognition_viz_{TARGET_LANG}.png"), img_for_ocr_drawing)


    # === 2. MASK GENERATION PHASE ===
    print("\n--- Mask Generation Phase ---")
    inpainting_mask = np.zeros(img_orig.shape[:2], dtype=np.uint8)
    if detected_boxes_poly:
        for box_poly_list in detected_boxes_poly:
            polygon_points = np.array(box_poly_list, dtype=np.int32)
            cv2.fillPoly(inpainting_mask, [polygon_points], (255)) # White for hole

        dilation_kernel_size = 15 # Adjust as needed
        kernel = np.ones((dilation_kernel_size, dilation_kernel_size), np.uint8)
        inpainting_mask_dilated = cv2.dilate(inpainting_mask, kernel, iterations=1)
        cv2.imwrite(os.path.join(args.output_dir, f"inpainting_mask_{TARGET_LANG}.png"), inpainting_mask_dilated)
        print(f"Inpainting mask generated and saved.")
    else:
        print("No text detected by OCR, using an empty mask (no inpainting).")
        inpainting_mask_dilated = inpainting_mask # Empty mask

    
     # === 3. LAMA INPAINTING PHASE ===
    print("\n--- LaMa Inpainting Phase ---")
    # Preprocess original image and generated mask for LaMa
    # lama_input_img_tensor: original image, normalized to [-1,1]
    # lama_input_mask_tensor: mask where hole is 1, float32
    lama_input_img_tensor, lama_input_mask_tensor, lama_original_shape = preprocess_lama_inputs_for_triton_py(img_orig, inpainting_mask_dilated, CURRENT_LAMA_TARGET_SIZE)
    lama_inputs = []
    lama_inputs.append(httpclient.InferInput(LAMA_INPUT_IMAGE_NAME_PY, lama_input_img_tensor.shape, np_to_triton_dtype(lama_input_img_tensor.dtype)))
    lama_inputs[0].set_data_from_numpy(lama_input_img_tensor)
    lama_inputs.append(httpclient.InferInput(LAMA_INPUT_MASK_NAME_PY, lama_input_mask_tensor.shape, np_to_triton_dtype(lama_input_mask_tensor.dtype)))
    lama_inputs[1].set_data_from_numpy(lama_input_mask_tensor)
    lama_outputs = [httpclient.InferRequestedOutput(LAMA_OUTPUT_NAME_PY)]

    # Run LaMa Inference
    print(f"Sending request to Triton for LaMa model '{LAMA_MODEL_NAME_PY}'...")
    start_time_lama = time.time()

    try:
        lama_results_triton = triton_client.infer(model_name=LAMA_MODEL_NAME_PY, inputs=lama_inputs, outputs=lama_outputs)
        lama_output_raw = lama_results_triton.as_numpy(LAMA_OUTPUT_NAME_PY)
        print("Received inpainted result from Triton LaMa.")
    except Exception as e:
        print(f"Error during LaMa inference: {e}"); lama_output_raw = None

    infer_lama_time = time.time() - start_time_lama
    print(f"Triton LaMa inference time: {infer_lama_time:.4f}s")

    if lama_output_raw is not None:
        final_inpainted_image = postprocess_lama_output_from_triton_py(lama_output_raw, lama_original_shape, CURRENT_LAMA_TARGET_SIZE)
        final_output_path = os.path.join(args.output_dir, f"final_inpainted_image_{TARGET_LANG}.png")
        cv2.imwrite(final_output_path, final_inpainted_image)
        print(f"Final inpainted image saved to: {final_output_path}")
    else:
        print("LaMa inpainting failed or produced no output. Saving original image instead for this step.")
        final_output_path = os.path.join(args.output_dir, f"final_inpainted_image_FAILED_{TARGET_LANG}.png")
        cv2.imwrite(final_output_path, img_orig) # Save original if inpainting fails

    print("\nPipeline finished.")
    