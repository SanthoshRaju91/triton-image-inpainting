import tritonclient.http as httpclient
from tritonclient.utils import np_to_triton_dtype, triton_to_np_dtype
import numpy as np
import cv2
import time
import os
import math
import yaml
from shapely.geometry import Polygon
import pyclipper


# --- configuration
TRITON_URL = "localhost:8000"
DET_MODEL_NAME = "ocr_ml_detector"
REC_MODEL_NAME = "ocr_en_recogniser"

DET_INPUT_NAME = "x"
DET_OUTPUT_NAME = "fetch_name_0"
REC_INPUT_NAME = "x"
REC_OUTPUT_NAME = "fetch_name_0"

IMAGE_PATH = "quiz.jpg"
CHARACTER_DICT_PATH = "en_dict.txt"

# --- Preprocessing parameters
DET_MAX_SIDE_LEN = 960
REC_IMG_H = 48
DET_IMG_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
DET_IMG_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
REC_IMG_MEAN = 0.5
REC_IMG_STD = 0.5

# --- Postprocessing parameters
DET_DB_THRESH = 0.3
DET_DB_BOX_THRESH = 0.6
DET_DB_UNCLIP_RATIO = 1.5
DET_BOX_TYPE = 'quad'
MIN_BOX_AREA = 10
MIN_BOX_SCORE = 0.5


def load_character_dict(dict_path):
    char_list = []
    with open(dict_path, "r", encoding="utf-8") as f:
        for line in f:
            char_list.append(line.strip())
    char_list = ['<blank>'] + char_list
    char_map = {char: i for i, char in enumerate(char_list)}
    print(f"Loaded dictionary: {len(char_list)} chars, first 10: {char_list[:10]}")
    return char_list, char_map

def preprocess_det_image(img, max_side_len=DET_MAX_SIDE_LEN):
    h, w, _ = img.shape
    if max(h, w) > max_side_len:
        ratio = float(max_side_len) / max(h, w)
    else:
        ratio = 1.0
    resize_h = int(h * ratio)
    resize_w = int(w * ratio)
    resize_h = max(int(round(resize_h / 32) * 32), 32)
    resize_w = max(int(round(resize_w / 32) * 32), 32)

    try:
        resized_img = cv2.resize(img, (resize_w, resize_h))
    except:
        print(f"Error resizing image to {(resize_w, resize_h)}")
        return None, None
    
    img_normalized = (resized_img.astype(np.float32) / 255.0 - DET_IMG_MEAN) / DET_IMG_STD
    img_transposed = img_normalized.transpose((2,0,1))
    img_batch = np.expand_dims(img_transposed, axis=0)
    return img_batch, (ratio, h, w)


def preprocess_rec_image(img_crop, rec_img_h=REC_IMG_H):
    h,w,_ = img_crop.shape
    ratio = rec_img_h / float(h)
    resize_w = int(w* ratio)
    resize_w = max(32, int(round(resize_w / 4) * 4))
    resized_img = cv2.resize(img_crop, (resize_w, rec_img_h))
    img_normalized = (resized_img.astype(np.float32) / 255.0 - REC_IMG_MEAN) / REC_IMG_STD
    img_transposed = img_normalized.transpose((2,0,1))
    img_batch = np.expand_dims(img_transposed, axis=0)
    return img_batch


def postprocess_recognition(rec_output_raw, char_list):
    if rec_output_raw.ndim == 3 and rec_output_raw.shape[0] == 1:
        preds = rec_output_raw[0]
    elif rec_output_raw.ndim == 2:
        preds = rec_output_raw
    else:
        print(f"Error: Unexpected recognition output shape: {rec_output_raw.shape}")
        return "Decoding Error", 0.0
    
    pred_indices = np.argmax(preds, axis=1)
    pred_probs = np.max(preds, axis=1)

    decoded_indices = []
    decoded_probs = []
    last_idx = -1
    for i, idx in enumerate(pred_indices):
        if idx != 0 and idx != last_idx:
            decoded_indices.append(idx)
            decoded_probs.append(pred_probs[i])
        last_idx = idx
    
    text = "".join([char_list[i] for i in decoded_indices if 0 < i < len(char_list)])
    confidence = np.mean(decoded_probs) if decoded_probs else 0.0

    return text, float(confidence)

class DBPostProcess:
    def __init__(self,
                 thresh=DET_DB_THRESH,
                 box_thresh=DET_DB_BOX_THRESH,
                 max_candidates=1000,
                 unclip_ratio=DET_DB_UNCLIP_RATIO,
                 min_size=MIN_BOX_AREA,
                 box_type='quad'):
        self.thresh = thresh
        self.box_thresh = box_thresh
        self.max_candidates = max_candidates
        self.unclip_ratio = unclip_ratio
        self.min_size = min_size
        self.box_type = box_type

    def polygons_from_bitmap(self, pred, bitmap, dest_width, dest_height):
        assert len(bitmap.shape) == 2
        height, width = bitmap.shape
        contours, _ = cv2.findContours((bitmap * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        num_contours = min(len(contours), self.max_candidates)
        boxes = []
        scores = []

        for index in range(num_contours):
            contour = contours[index]
            points, sside = self.get_mini_boxes(contour)
            if sside < self.min_size:
                continue
            score = self.box_score_fast(pred, points.reshape(-1, 2))
            if self.box_thresh > score:
                continue

            if self.box_type == 'poly':
                box = points
            elif self.box_type == 'quad':
                box = self.unclip(points).reshape(-1, 1, 2)
                if len(box) != 4:
                    continue
                box = box.reshape(4, 2)
            else:
                raise ValueError(f"Unsupported box_type: {self.box_type}")
            
            box[:, 0] = np.clip(np.round(box[:, 0]), 0, width - 1)
            box[:, 1] = np.clip(np.round(box[:, 1]), 0, height - 1)

            boxes.append(box)
            scores.append(score)

        return boxes, scores
    

    def box_score_fast(self, bitmap, _box):
        h, w = bitmap.shape
        box = _box.copy()
        xmin = np.clip(np.floor(box[:, 0].min()).astype(int), 0, w - 1)
        xmax = np.clip(np.ceil(box[:, 0].max()).astype(int), 0, w - 1)
        ymin = np.clip(np.floor(box[:, 1].min()).astype(int), 0, h - 1)
        ymax = np.clip(np.ceil(box[:, 1].max()).astype(int), 0, h - 1)

        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        box[:, 0] = box[:, 0] - xmin
        box[:, 1] = box[:, 1] - ymin
        cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)
        return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]
    
    def unclip(self, box):
        poly = Polygon(box)
        distance = poly.area * self.unclip_ratio / poly.length
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        expanded = offset.Execute(distance)

        if not expanded or len(expanded) > 1 :           
            print(f"Warning: Unclip resulted in {len(expanded)} polygons. Using minimum area rect of original.")
            points = cv2.minAreaRect(box)
            return cv2.boxPoints(points)
        
        expanded_poly = np.array(expanded[0]).reshape(-1, 2)
        rect = cv2.minAreaRect(expanded_poly)
        points = cv2.boxPoints(rect)
        return points
    
    def get_mini_boxes(self, contour):
        bounding_box = cv2.minAreaRect(contour)
        points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

        index_1, index_2, index_3, index_4 = 0, 1, 2, 3
        if points[1][1] > points[0][1]:
            index_1 = 0
            index_4 = 1
        else:
            index_1 = 1
            index_4 = 0
        if points[3][1] > points[2][1]:
            index_2 = 2
            index_3 = 3
        else:
            index_2 = 3
            index_3 = 2

        box = [points[index_1], points[index_2], points[index_3], points[index_4]]
        return np.array(box), min(bounding_box[1])
    
    def __call__(self, det_output_raw, resize_info):
        if det_output_raw.ndim == 4 and det_output_raw.shape[0:2] == (1, 1):
            pred = det_output_raw[0, 0, :, :]
        else:
            print(f"Error: Unexpected detection output shape for postprocessing: {det_output_raw.shape}")
            return []
        
        bitmap = pred > self.thresh
        resize_ratio, orig_h, orig_w = resize_info
        dest_height, dest_width = pred.shape
        boxes, scores = self.polygons_from_bitmap(pred, bitmap, dest_width, dest_height)

        final_boxes = []
        if boxes:
            boxes_np = np.array(boxes)
            if resize_ratio == 0:
                print("Error: Zero resize_ratio during postprocessing")
                return []
            boxes_np /= resize_ratio

            boxes_np[:, :, 0] = np.clip(boxes_np[:, :, 0], 0, orig_w - 1)
            boxes_np[:, :, 1] = np.clip(boxes_np[:, :, 1], 0, orig_h - 1)

            for i, box in enumerate(boxes_np):
                if scores[i] >= MIN_BOX_SCORE:
                    final_boxes.append(box.tolist())
        return final_boxes
    

def get_rotate_crop_image(img, points):
    assert len(points) == 4, "Points number must be 4"
    img_crop_width = int(max(np.linalg.norm(points[0] - points[1]),
                             np.linalg.norm(points[2] - points[3])))
    img_crop_height = int(max(np.linalg.norm(points[0] - points[3]),
                              np.linalg.norm(points[1] - points[2])))
    pts_std = np.float32([[0, 0], [img_crop_width, 0], [img_crop_width, img_crop_height], [0, img_crop_height]])
    points_float = np.array(points, dtype=np.float32)
    M = cv2.getPerspectiveTransform(points_float, pts_std)
    dst_img = cv2.warpPerspective(img, M, (img_crop_width, img_crop_height), borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_CUBIC)
    dst_img_h, dst_img_w = dst_img.shape[0:2]
    if dst_img_h * 1.0 / dst_img_w >= 1.5:
        dst_img = np.rot90(dst_img, k=3)
    return dst_img


if __name__ == "__main__":

    # --- Load image
    img_orig = cv2.imread(IMAGE_PATH)
    if img_orig is None:
        print(f"Error loading image: {IMAGE_PATH}")
        exit()
    print(f"Original image shape: {img_orig.shape}")


    # --- Load character dictionary
    if not os.path.exists(CHARACTER_DICT_PATH):
        print(f"Error: Character dictionary not found at {CHARACTER_DICT_PATH}")
        exit()
    char_list = char_map = load_character_dict(CHARACTER_DICT_PATH)

    # --- Initialise DB Postprocessor
    db_postprocessor = DBPostProcess()

    # --- Create triton client
    try:
        triton_client = httpclient.InferenceServerClient(url=TRITON_URL, verbose=False)
        print(f"Checking Triton server liveness ({TRITON_URL})")

        if not triton_client.is_server_live:
            print(f"Error: Triton server is not live at {TRITON_URL}")
            exit()
        
        if not triton_client.is_model_ready(DET_MODEL_NAME):
            print(f"Error: Detection mode '{DET_MODEL_NAME}' is not ready.")
            exit()
        
        if not triton_client.is_model_ready(REC_MODEL_NAME):
            print(f"Error: Recognition model '{REC_MODEL_NAME}' is not ready.")
            exit()

        print("Triton client created and models are ready")
    except Exception as e:
        print(f"Error creating triton client or checking models: {e}")
        exit()


    # --- 1. Preprocess image for detection
    print("\n--- Running Detection ---")
    start_time = time.time()
    det_input_image, resize_info = preprocess_det_image(img_orig)
    if det_input_image is None:
        print("Detection preprocessing failed.")
        exit()
    prep_det_time = time.time() - start_time
    print(f"Detection preprocessing time: {prep_det_time:.4f}s")

    # ---- 2. Prepare Triton Request for Detection
    det_inputs = []
    det_outputs = []
    det_inputs.append(httpclient.InferInput(
        DET_INPUT_NAME,
        det_input_image.shape,
        np_to_triton_dtype(det_input_image.dtype)
    ))
    det_inputs[0].set_data_from_numpy(det_input_image)
    det_outputs.append(httpclient.InferRequestedOutput(DET_OUTPUT_NAME))


    # --- 3. Run Detection Inference via Triton
    print(f"Sending request to Triton for model '{DET_MODEL_NAME}'...")
    start_time = time.time()

    try:
        det_results = triton_client.infer(model_name=DET_MODEL_NAME, inputs=det_inputs, outputs=det_outputs)
        det_output_raw = det_results.as_numpy(DET_OUTPUT_NAME)
        print(f"Received detection output with shape: {det_output_raw.shape}")
    except Exception as e:
        print(f"Error during detection inference: {e}")
        exit()

    infer_det_time = time.time() - start_time
    print(f"Triton detection inference time: {infer_det_time:.4f}s")


    # --- 4. Postprocess Detection Results (Using accurate DBPostProcess)
    start_time = time.time()
    detected_boxes_poly = db_postprocessor(det_output_raw, resize_info)
    post_det_time = time.time() - start_time
    print(f"Detection postprocessin time: {post_det_time:.4f}s. Found {len(detected_boxes_poly)} boxes.")

    # --- 5. Run Recognition for each detected box
    print("\n--- Running Recognition ---")
    all_ocr_results = []
    output_image_final = img_orig.copy()

    if not detected_boxes_poly:
        print("No text boxes detected after postprocessing.")
    else:
        total_rec_prep_time = 0
        total_rec_infer_time = 0
        total_rec_post_time = 0

        for i, box_poly_list in enumerate(detected_boxes_poly):
            box_poly = np.array(box_poly_list).astype(np.float32)
            if box_poly.shape != (4, 2):
                print(f"Warning: Skipping box {i} due to unexpected shape {box_poly.shape} after postprocessing.")
                continue
            
            print(f"\n--- Processing Box {i} ---")
            img_crop = get_rotate_crop_image(img_orig, box_poly)

            if img_crop is None or img_crop.shape[0] < 8 or img_crop.shape[1] < 8:
                print(f"Warning: Skipping box {i} due to invalid crop (shape: {img_crop.shape if img_crop is not None else 'None'}).")
                continue
            
            print(f"Cropped image shape: {img_crop.shape}")

            # Preprocess crop for Recognition
            start_time = time.time()
            rec_input_image = preprocess_rec_image(img_crop)
            prep_rec_time = time.time() - start_time
            total_rec_prep_time += prep_rec_time
            rec_inputs = []
            rec_outputs = []
            rec_inputs.append(httpclient.InferInput(REC_INPUT_NAME, rec_input_image.shape, np_to_triton_dtype(rec_input_image.dtype)))
            rec_inputs[0].set_data_from_numpy(rec_input_image)
            rec_outputs.append(httpclient.InferRequestedOutput(REC_OUTPUT_NAME))

            # Run Recognition Inference via Triton
            print(f"Sending request to Triton for model '{REC_MODEL_NAME}'...")
            start_time = time.time()
            try:
                rec_results = triton_client.infer(model_name=REC_MODEL_NAME, inputs=rec_inputs, outputs=rec_outputs)
                rec_output_raw = rec_results.as_numpy(REC_OUTPUT_NAME)
            except Exception as e:
                print(f"Error during recognition inference for box {i}: {e}")
                continue
            infer_rec_time = time.time() - start_time
            total_rec_infer_time += infer_rec_time

            print(f"Triton recognition inference time: {infer_rec_time:.4f}s")

            # Postprocess Recognition Results (CTC Decode)
            start_time = time.time()
            text, confidence = postprocess_recognition(rec_output_raw, char_list)
            post_rec_time = time.time() - start_time
            total_rec_post_time += post_rec_time
            print(f"Recognition postprocessing time: {post_rec_time:.4f}s")
            print(f"Result Box {i}: Text='{text}', Confidence={confidence:.4f}")

            all_ocr_results.append({
                'box': box_poly_list, # Keep as list of lists [x, y]
                'text': text,
                'confidence': confidence
            })

            # Draw on output image
            try:
                cv2.polylines(output_image_final, [box_poly.astype(np.int32)], isClosed=True, color=(0, 0, 255), thickness=2)
                label_pos = (int(box_poly[0][0]), int(box_poly[0][1]) - 10)
                cv2.putText(output_image_final, f"{text} ({confidence:.2f})", label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            except Exception as draw_e:
                print(f"Error drawing results for box {i}: {draw_e}")
    
        print("\n--- Timing Summary ---")
        print(f"Detection Prep : {prep_det_time:.4f}s")
        print(f"Detection Infer: {infer_det_time:.4f}s")
        print(f"Detection Post : {post_det_time:.4f}s")
        print(f"Avg Rec Prep   : {total_rec_prep_time / len(detected_boxes_poly) if detected_boxes_poly else 0:.4f}s")
        print(f"Avg Rec Infer  : {total_rec_infer_time / len(detected_boxes_poly) if detected_boxes_poly else 0:.4f}s")
        print(f"Avg Rec Post   : {total_rec_post_time / len(detected_boxes_poly) if detected_boxes_poly else 0:.4f}s")
        print(f"Processed {len(all_ocr_results)} text instances via Triton.")
    
    # Save the final image
    output_filename = 'output/triton_ocr_final_output.png'
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    save_success = cv2.imwrite(output_filename, output_image_final)
    if save_success:
        print(f"\nOutput image with Triton results saved to {output_filename}")
    else:
        print(f"\nError saving final output image to {output_filename}")

    print("\nScript finished.")

