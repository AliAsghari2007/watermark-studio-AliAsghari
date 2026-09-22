import io
import os
import sys
import cv2
import numpy as np
import streamlit as st
import torch
from PIL import Image
from ultralytics import YOLO
from ultralytics.engine.results import Boxes

# =========================================================
# ۱. تنظیمات صفحه و استایل UI
# =========================================================
st.set_page_config(
    page_title="Watermark Studio AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Vazirmatn:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] {
        font-family: 'Vazirmatn', sans-serif;
    }
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        text-align: center;
        margin-top: -15px;
        margin-bottom: 5px;
        background: linear-gradient(90deg, #1E88E5, #00C853);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .sub-title {
        text-align: center;
        color: #78909C;
        font-size: 1.0rem;
        margin-bottom: 20px;
    }
    .method-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 12px;
        padding: 12px;
        text-align: center;
        margin-bottom: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("<div class='main-title'>🛡️ Watermark Studio AI Platform</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>سامانه هوشمند تشخیص، بازسازی و مقایسه سه متد حذف واترمارک</div>", unsafe_allow_html=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "watermarks_s_yolov8_v1.pt")

if BASE_DIR not in sys.path and os.path.exists(BASE_DIR):
    sys.path.insert(0, BASE_DIR)

# =========================================================
# ۲. بارگذاری مدل‌ها
# =========================================================
@st.cache_resource
def load_yolo_model(model_path):
    try:
        if os.path.exists(model_path):
            return YOLO(model_path)
    except Exception as e:
        st.sidebar.error(f"خطا در مدل YOLO: {e}")
    return None

@st.cache_resource
def load_mobile_sam_model(repo_dir):
    try:
        ckpt_path = os.path.join(BASE_DIR, "mobile_sam.pt")
        if not os.path.exists(ckpt_path):
            return None, f"فایل {ckpt_path} یافت نشد."
        
        from mobile_sam import sam_model_registry, SamPredictor
        device = "cuda" if torch.cuda.is_available() else "cpu"
        mobile_sam = sam_model_registry["vit_t"](checkpoint=ckpt_path)
        mobile_sam.to(device=device)
        mobile_sam.eval()
        predictor = SamPredictor(mobile_sam)
        return predictor, None
    except Exception as e:
        return None, str(e)

model = load_yolo_model(MODEL_PATH)

# =========================================================
# ۳. پایپ‌لاین تشخیص واترمارک (YOLO & Pattern Mining)
# =========================================================
def run_kernel_1(raw_test_img, model_instance):
    h, w = raw_test_img.shape[:2]
    if model_instance is None:
        custom_tensor = torch.tensor([[int(w*0.1), int(h*0.1), int(w*0.9), int(h*0.9), 0.90, 0.0]], dtype=torch.float32)
        from types import SimpleNamespace
        res = SimpleNamespace(boxes=Boxes(custom_tensor, orig_shape=(h, w)))
        return [res], "Fallback Box Scan"

    results = model_instance.predict(source=raw_test_img, conf=0.25, verbose=False)
    if len(results[0].boxes) > 0:
        detected_stage = "Stage 1: YOLO Standard"
    else:
        results = model_instance.predict(source=raw_test_img, conf=0.08, augment=True, verbose=False)
        if len(results[0].boxes) > 0:
            detected_stage = "Stage 2: YOLO High-Sensitivity"
        else:
            gray = cv2.cvtColor(raw_test_img, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            grad_x = cv2.Sobel(enhanced, cv2.CV_32F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(enhanced, cv2.CV_32F, 0, 1, ksize=3)
            magnitude = cv2.magnitude(grad_x, grad_y)
            _, std_val = cv2.meanStdDev(magnitude)
            _, binary_edges = cv2.threshold(magnitude, 0.5 * std_val[0][0], 255, cv2.THRESH_BINARY)
            binary_edges = binary_edges.astype(np.uint8)

            k_size = int(max(w, h) * 0.05) | 1
            edge_density = cv2.GaussianBlur(binary_edges, (k_size, k_size), 0)
            _, density_mask = cv2.threshold(edge_density, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(density_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            candidate_boxes = []
            total_area = h * w
            for cnt in contours:
                bx, by, bw, bh = cv2.boundingRect(cnt)
                if (total_area * 0.04) < (bw * bh) < (total_area * 0.85):
                    candidate_boxes.append([bx, by, bx + bw, by + bh])

            if len(candidate_boxes) > 0:
                c_arr = np.array(candidate_boxes)
                custom_tensor = torch.tensor([[int(np.min(c_arr[:, 0])), int(np.min(c_arr[:, 1])),
                                               int(np.max(c_arr[:, 2])), int(np.max(c_arr[:, 3])), 0.90, 0.0]], dtype=torch.float32)
                results[0].boxes = Boxes(custom_tensor, orig_shape=(h, w))
                detected_stage = "Stage 3: Analytical Transparent Scan"
            else:
                detected_stage = "Result: No Watermark Found"

    return results, detected_stage


def run_kernel_2(raw_img_rgb, results):
    img_h, img_w, _ = raw_img_rgb.shape
    if results[0].boxes is None or len(results[0].boxes) == 0:
        return np.zeros((0, 4)), []

    yolo_boxes = results[0].boxes.xyxy.cpu().numpy()
    yolo_confs = results[0].boxes.conf.cpu().numpy()

    if len(yolo_boxes) == 0:
        return yolo_boxes, []

    sorted_indices = np.argsort(-yolo_confs)
    top_indices = sorted_indices[: min(3, len(sorted_indices))]

    gray_img = cv2.cvtColor(raw_img_rgb, cv2.COLOR_RGB2GRAY)
    grad_x = cv2.Sobel(gray_img, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray_img, cv2.CV_32F, 0, 1, ksize=3)
    texture_activity = np.std(cv2.magnitude(grad_x, grad_y))
    sim_threshold = np.clip(0.22 + (texture_activity - 60.0) * 0.0007, 0.22, 0.29)

    max_pad_w, max_pad_h = 0, 0
    templates_data = []

    for idx in top_indices:
        box = yolo_boxes[idx].astype(int)
        bx1, by1 = max(0, box[0]), max(0, box[1])
        bx2, by2 = min(img_w, box[2]), min(img_h, box[3])
        tmpl = raw_img_rgb[by1:by2, bx1:bx2]
        th, tw, _ = tmpl.shape
        if th >= 10 and tw >= 10:
            templates_data.append((tmpl, th, tw))
            max_pad_w = max(max_pad_w, tw // 2)
            max_pad_h = max(max_pad_h, th // 2)

    pad_gray = cv2.copyMakeBorder(gray_img, max_pad_h, max_pad_h, max_pad_w, max_pad_w, borderType=cv2.BORDER_REFLECT)
    pad_grad = cv2.Sobel(pad_gray, cv2.CV_32F, 1, 1, ksize=3)

    all_new_candidates, all_new_scores = [], []
    for tmpl, th, tw in templates_data:
        tmpl_grad = cv2.Sobel(cv2.cvtColor(tmpl, cv2.COLOR_RGB2GRAY), cv2.CV_32F, 1, 1, ksize=3)
        similarity_map = cv2.matchTemplate(pad_grad, tmpl_grad, cv2.TM_CCOEFF_NORMED)
        map_h, map_w = similarity_map.shape

        for yb in yolo_boxes:
            x1, y1, x2, y2 = yb.astype(int)
            similarity_map[max(0, y1 + max_pad_h - th//2):min(map_h, y2 + max_pad_h),
                           max(0, x1 + max_pad_w - tw//2):min(map_w, x2 + max_pad_w)] = -1.0

        loc = np.where(similarity_map >= sim_threshold)
        for pt in zip(*loc[::-1]):
            all_new_candidates.append([pt[0] - max_pad_w, pt[1] - max_pad_h, tw, th])
            all_new_scores.append(float(similarity_map[pt[1], pt[0]]))

    discovered_boxes = []
    if len(all_new_candidates) > 0:
        indices = cv2.dnn.NMSBoxes(bboxes=all_new_candidates, scores=all_new_scores, score_threshold=float(sim_threshold), nms_threshold=0.25)
        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = all_new_candidates[i]
                discovered_boxes.append([max(0, x), max(0, y), min(img_w, x + w), min(img_h, y + h)])

    return yolo_boxes, discovered_boxes


def run_kernel_3(raw_img_rgb, yolo_boxes, discovered_boxes):
    original_yolo = [list(map(int, b)) for b in yolo_boxes]
    sim_candidates = [list(map(int, b)) for b in discovered_boxes]

    accepted_new_boxes = []
    for cand in sim_candidates:
        c_area = (cand[2] - cand[0]) * (cand[3] - cand[1])
        overlap = False
        for ref in original_yolo + accepted_new_boxes:
            xA, yA = max(cand[0], ref[0]), max(cand[1], ref[1])
            xB, yB = min(cand[2], ref[2]), min(cand[3], ref[3])
            inter = max(0, xB - xA) * max(0, yB - yA)
            if inter / float(max(1, c_area)) > 0.15:
                overlap = True
                break
        if not overlap:
            accepted_new_boxes.append(cand)

    final_clean_boxes = original_yolo + accepted_new_boxes
    vis_img = raw_img_rgb.copy()
    for b in original_yolo:
        cv2.rectangle(vis_img, (b[0], b[1]), (b[2], b[3]), (0, 140, 255), 2)
    for b in accepted_new_boxes:
        cv2.rectangle(vis_img, (b[0], b[1]), (b[2], b[3]), (0, 140, 255), 3)

    stats = {
        "yolo_count": len(original_yolo),
        "recovered_count": len(accepted_new_boxes),
        "total_count": len(final_clean_boxes),
    }
    return vis_img, final_clean_boxes, stats

# =========================================================
# ۴. متد ۱: WatermarkDirector
# =========================================================
class WatermarkDirector:
    def __init__(self, image_rgb: np.ndarray):
        self.image_rgb = image_rgb
        self.h, self.w = image_rgb.shape[:2]

        self.stroke_mask = np.zeros((self.h, self.w), dtype=np.uint8)
        self.background_mask = np.zeros((self.h, self.w), dtype=np.uint8)
        self.mask = np.zeros((self.h, self.w), dtype=np.uint8)

        self.boxes = []
        self.processed_boxes = []
        self.is_repeated_pattern = False
        self.execution_stats = {"STROKE": 0, "BACKGROUND": 0, "SKIPPED": 0}

    def set_boxes(self, boxes):
        self.boxes = [list(map(int, b)) for b in boxes]
        self.processed_boxes = []

    def _check_repeated_pattern(self, image_gray: np.ndarray) -> bool:
        valid_rois = []
        box_dims = []

        for b in self.boxes:
            x1 = max(0, min(self.w - 1, b[0]))
            y1 = max(0, min(self.h - 1, b[1]))
            x2 = max(0, min(self.w, b[2]))
            y2 = max(0, min(self.h, b[3]))

            bw, bh = x2 - x1, y2 - y1
            if bw >= 8 and bh >= 8:
                roi = image_gray[y1:y2, x1:x2]
                roi_resized = cv2.resize(roi, (48, 48))
                roi_norm = cv2.normalize(roi_resized, None, 0, 255, cv2.NORM_MINMAX)
                valid_rois.append(roi_norm)
                box_dims.append((bw, bh, bw / float(bh)))

        n = len(valid_rois)
        if n < 2:
            return False

        if n >= 3:
            ratios = [d[2] for d in box_dims]
            areas = [d[0] * d[1] for d in box_dims]
            ratio_std = np.std(ratios) / (np.mean(ratios) + 1e-5)
            area_std = np.std(areas) / (np.mean(areas) + 1e-5)
            if ratio_std < 0.35 and area_std < 0.65:
                return True

        similar_pairs = 0
        for i in range(n):
            for j in range(i + 1, n):
                res = cv2.matchTemplate(valid_rois[i], valid_rois[j], cv2.TM_CCOEFF_NORMED)
                sim_score = res[0][0]
                ar_diff = abs(box_dims[i][2] - box_dims[j][2])
                if sim_score > 0.40 and ar_diff < 0.30:
                    similar_pairs += 1
                    if similar_pairs >= 1:
                        return True

        return False

    def process_rois(self):
        self.stroke_mask.fill(0)
        self.background_mask.fill(0)
        self.mask.fill(0)

        self.execution_stats = {"STROKE": 0, "BACKGROUND": 0, "SKIPPED": 0}
        self.processed_boxes = []

        image_bgr = cv2.cvtColor(self.image_rgb, cv2.COLOR_RGB2BGR)
        image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

        self.is_repeated_pattern = self._check_repeated_pattern(image_gray)

        for (x1, y1, x2, y2) in self.boxes:
            x1 = max(0, min(self.w - 1, x1))
            y1 = max(0, min(self.h - 1, y1))
            x2 = max(0, min(self.w, x2))
            y2 = max(0, min(self.h, y2))

            if (x2 - x1) < 4 or (y2 - y1) < 4:
                self.execution_stats["SKIPPED"] += 1
                continue

            roi_bgr = image_bgr[y1:y2, x1:x2]
            roi_gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)

            bg_local = None
            if not self.is_repeated_pattern:
                bg_local = self._detect_background_overlay(roi_bgr, roi_gray)

            if bg_local is not None:
                current = self.background_mask[y1:y2, x1:x2]
                self.background_mask[y1:y2, x1:x2] = np.maximum(current, bg_local)
                self.execution_stats["BACKGROUND"] += 1
                self.processed_boxes.append((x1, y1, x2, y2, "BACKGROUND"))
            else:
                stroke_local = self._safe_complex_morph(roi_bgr, roi_gray)
                current = self.stroke_mask[y1:y2, x1:x2]
                self.stroke_mask[y1:y2, x1:x2] = np.maximum(current, stroke_local)
                self.execution_stats["STROKE"] += 1
                self.processed_boxes.append((x1, y1, x2, y2, "STROKE"))

        self.mask = cv2.bitwise_or(self.background_mask, self.stroke_mask)
        return self.mask

    def _detect_background_overlay(self, roi_bgr: np.ndarray, roi_gray: np.ndarray):
        h, w = roi_gray.shape
        if h * w < 64:
            return None

        mean_val = float(np.mean(roi_gray))
        median_val = float(np.median(roi_gray))
        dark_ratio = float(np.mean(roi_gray < 95))
        std_val = float(np.std(roi_gray))

        is_dark = (dark_ratio > 0.35) or (median_val < 105) or (mean_val < 105)
        is_relatively_flat = std_val < 55

        if is_dark and is_relatively_flat:
            banner_mask = np.full((h, w), 255, dtype=np.uint8)
            if h > 2 and w > 2:
                banner_mask[0, :] = 0
                banner_mask[-1, :] = 0
                banner_mask[:, 0] = 0
                banner_mask[:, -1] = 0
            return banner_mask
        return None

    def _safe_complex_morph(self, roi_bgr: np.ndarray, roi_gray: np.ndarray) -> np.ndarray:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

        top_hat = cv2.morphologyEx(roi_gray, cv2.MORPH_TOPHAT, kernel)
        _, top_mask = cv2.threshold(top_hat, 22, 255, cv2.THRESH_BINARY)

        black_hat = cv2.morphologyEx(roi_gray, cv2.MORPH_BLACKHAT, kernel)
        _, black_mask = cv2.threshold(black_hat, 22, 255, cv2.THRESH_BINARY)

        stroke_candidates = cv2.bitwise_or(top_mask, black_mask)

        grad = cv2.morphologyEx(roi_gray, cv2.MORPH_GRADIENT, kernel)
        _, grad_mask = cv2.threshold(grad, 28, 255, cv2.THRESH_BINARY)

        combined = cv2.bitwise_or(stroke_candidates, grad_mask)

        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)
        combined = cv2.dilate(combined, kernel, iterations=1)

        coverage = float(np.mean(combined > 0))
        if coverage > 0.35:
            combined = top_mask.copy()

        if float(np.mean(combined > 0)) > 0.40:
            combined = np.zeros_like(combined)

        return combined


def inpaint_watermark_advanced(image_bgr: np.ndarray, mask: np.ndarray, inpaint_radius: int = 3, dark_banner_alpha: float = 0.65, normal_alpha: float = 0.20):
    if len(mask.shape) == 3:
        mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    else:
        mask_gray = mask.copy()
        
    _, mask_binary = cv2.threshold(mask_gray, 10, 255, cv2.THRESH_BINARY)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_binary)
    
    img_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    res_bgr = image_bgr.copy().astype(np.float32)
    final_text_mask = np.zeros_like(mask_binary)

    for i in range(1, num_labels):
        comp_mask = (labels == i).astype(np.uint8) * 255
        x, y, w, h, area = stats[i]
        solidity = area / float(w * h) if (w * h) > 0 else 0

        if solidity > 0.70 and area > 500:
            alpha = np.clip(dark_banner_alpha, 0.05, 0.85)
            res_bgr[comp_mask > 0] = res_bgr[comp_mask > 0] / (1.0 - alpha)
            roi_gray = img_gray[y:y+h, x:x+w]
            text_thresh = np.percentile(roi_gray, 85)
            comp_text_mask = np.zeros_like(mask_binary)
            comp_text_mask[(comp_mask > 0) & (img_gray > text_thresh)] = 255
            if np.count_nonzero(comp_text_mask) == 0:
                comp_text_mask = cv2.morphologyEx(comp_mask, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
            final_text_mask = cv2.bitwise_or(final_text_mask, comp_text_mask)
        else:
            alpha = np.clip(normal_alpha, 0.0, 0.50)
            if alpha > 0.0:
                res_bgr[comp_mask > 0] = res_bgr[comp_mask > 0] / (1.0 - alpha)
            final_text_mask = cv2.bitwise_or(final_text_mask, comp_mask)

    res_bgr = np.clip(res_bgr, 0, 255).astype(np.uint8)
    
    if np.count_nonzero(final_text_mask) == 0:
        restored_telea = res_bgr
    else:
        restored_telea = cv2.inpaint(res_bgr, final_text_mask, inpaintRadius=inpaint_radius, flags=cv2.INPAINT_TELEA)

    return restored_telea, final_text_mask


def run_method_1(orig_bgr, final_boxes, raw_img_rgb):
    director = WatermarkDirector(raw_img_rgb)
    director.set_boxes(final_boxes)
    mask = director.process_rois()
    telea_bgr, final_mask = inpaint_watermark_advanced(orig_bgr, mask, inpaint_radius=2, dark_banner_alpha=0.65, normal_alpha=0.1)
    telea_rgb = cv2.cvtColor(telea_bgr, cv2.COLOR_BGR2RGB)
    return telea_rgb, telea_bgr, final_mask

# =========================================================
# ۵. متد ۲: Hybrid Opacity Adjustment
# =========================================================
def extract_closed_region_mask(roi_bgr: np.ndarray, min_area: int = 10, max_area_ratio: float = 0.65, close_kernel_size: int = 3) -> np.ndarray:
    h, w = roi_bgr.shape[:2]
    total_area = h * w
    roi_gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.bilateralFilter(roi_gray, d=7, sigmaColor=50, sigmaSpace=50)
    v = np.median(blurred)
    lower = int(max(0, (1.0 - 0.33) * v))
    upper = int(min(255, (1.0 + 0.33) * v))
    edges = cv2.Canny(blurred, lower, upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel_size, close_kernel_size))
    closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, hierarchy = cv2.findContours(closed_edges, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    filled_mask = np.zeros((h, w), dtype=np.uint8)
    if contours and hierarchy is not None:
        hierarchy = hierarchy[0]
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area or area > (total_area * max_area_ratio):
                continue
            x, y, cw, ch = cv2.boundingRect(cnt)
            aspect_ratio = cw / float(ch + 1e-5)
            if 0.08 < aspect_ratio < 12.0:
                cv2.drawContours(filled_mask, [cnt], -1, 255, thickness=-1)
    clean_mask = cv2.dilate(filled_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
    return clean_mask


def hybrid_watermark_mask(roi_bgr: np.ndarray, watermark_color=(255, 255, 255), threshold: float = 0.15):
    b_channel = cv2.medianBlur(roi_bgr, 15)
    diff = roi_bgr.astype(np.float32) - b_channel.astype(np.float32)
    W = np.array(watermark_color, dtype=np.float32)
    B = b_channel.astype(np.float32)
    denom = W - B
    denom[np.abs(denom) < 1e-5] = 1.0 
    alpha_map = diff / denom
    alpha_gray = np.mean(alpha_map, axis=2)
    alpha_gray = np.clip(alpha_gray, 0.0, 1.0)
    alpha_mask = (alpha_gray > threshold).astype(np.uint8) * 255
    structure_mask = extract_closed_region_mask(roi_bgr, min_area=10, close_kernel_size=3)
    final_mask = cv2.bitwise_and(alpha_mask, structure_mask)
    return final_mask, alpha_mask, structure_mask


def run_method_2(orig_bgr, final_boxes):
    img_h, img_w = orig_bgr.shape[:2]
    full_mask = np.zeros((img_h, img_w), dtype=np.uint8)

    for box in final_boxes:
        x1, y1, x2, y2 = map(int, box)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img_w, x2), min(img_h, y2)
        if (x2 - x1) < 5 or (y2 - y1) < 5:
            continue
        roi_bgr = orig_bgr[y1:y2, x1:x2]
        roi_m, _, _ = hybrid_watermark_mask(roi_bgr, watermark_color=(255, 255, 255), threshold=0.15)
        full_mask[y1:y2, x1:x2] = cv2.bitwise_or(full_mask[y1:y2, x1:x2], roi_m)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    continuous_mask = cv2.morphologyEx(full_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    robust_mask = cv2.dilate(continuous_mask, kernel, iterations=1)

    telea_bgr = cv2.inpaint(orig_bgr, robust_mask, 5, cv2.INPAINT_TELEA)
    telea_rgb = cv2.cvtColor(telea_bgr, cv2.COLOR_BGR2RGB)
    return telea_rgb, telea_bgr, robust_mask

# =========================================================
# ۶. متد ۳: MobileSAM Hybrid Masking
# =========================================================
def build_hybrid_sam_mask(image_rgb, boxes, sam_predictor, grad_thresh=25, min_blob_area=15, dilate_iter=2, max_coverage=0.7):
    h, w = image_rgb.shape[:2]
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(grad_x, grad_y)
    grad = cv2.normalize(grad, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    sam_predictor.set_image(image_rgb)
    full_mask = np.zeros((h, w), dtype=np.uint8)

    for box in boxes:
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        bw, bh = x2 - x1, y2 - y1
        if bw <= 2 or bh <= 2:
            continue

        masks, scores, _ = sam_predictor.predict(
            box=np.array([x1, y1, x2, y2], dtype=np.float32),
            multimask_output=True
        )
        sam_mask = (masks[np.argmax(scores)] * 255).astype(np.uint8)

        roi_sam = sam_mask[y1:y2, x1:x2]
        roi_grad = grad[y1:y2, x1:x2]
        text_like = (roi_grad > grad_thresh).astype(np.uint8) * 255

        combined = (roi_sam > 0) & (text_like > 0)
        combined = (combined * 255).astype(np.uint8)

        num, labels, stats, _ = cv2.connectedComponentsWithStats(combined)
        roi_clean = np.zeros_like(combined)
        for i in range(1, num):
            if stats[i, cv2.CC_STAT_AREA] >= min_blob_area:
                roi_clean[labels == i] = 255

        coverage = roi_clean.mean() / 255.0
        if coverage > max_coverage or coverage < 0.01:
            full_mask[y1:y2, x1:x2] = text_like
        else:
            full_mask[y1:y2, x1:x2] = roi_clean

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    return cv2.dilate(full_mask, kernel, iterations=dilate_iter)


def run_method_3(orig_bgr, raw_img_rgb, final_boxes, sam_predictor):
    final_mask_sam = build_hybrid_sam_mask(raw_img_rgb, final_boxes, sam_predictor)
    telea_bgr = cv2.inpaint(orig_bgr, final_mask_sam, inpaintRadius=2, flags=cv2.INPAINT_TELEA)
    telea_rgb = cv2.cvtColor(telea_bgr, cv2.COLOR_BGR2RGB)
    return telea_rgb, telea_bgr, final_mask_sam

#==================================================
# Add watermark 

import random

def rotate_bound_rgba(image_rgba, angle):
    """چرخش تصویر RGBA با محاسبه ابعاد جدید کادر تا هیچ پیکسلی بریده نشود."""
    h, w = image_rgba.shape[:2]
    cx, cy = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    nw = int((h * sin) + (w * cos))
    nh = int((h * cos) + (w * sin))
    M[0, 2] += (nw / 2) - cx
    M[1, 2] += (nh / 2) - cy
    return cv2.warpAffine(image_rgba, M, (nw, nh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

def build_watermark_patch(text, logo_bytes=None, font_scale=1.0, thickness=2, text_color=(255, 255, 255)):
    """ساخت پچ واترمارک به صورت 4 کاناله RGBA از لوگو و/یا متن."""
    elements = []
    
    # پردازش لوگو در صورت آپلود
    if logo_bytes is not None:
        logo_arr = np.frombuffer(logo_bytes, np.uint8)
        logo_img = cv2.imdecode(logo_arr, cv2.IMREAD_UNCHANGED)
        if logo_img is not None:
            if logo_img.ndim == 2:
                logo_img = cv2.cvtColor(logo_img, cv2.COLOR_GRAY2BGRA)
            elif logo_img.shape[2] == 3:
                logo_img = cv2.cvtColor(logo_img, cv2.COLOR_BGR2BGRA)
            # تغییر مقیاس هوشمند در صورت بزرگ بودن بیش از حد
            if logo_img.shape[1] > 220:
                scale = 220.0 / logo_img.shape[1]
                logo_img = cv2.resize(logo_img, (220, int(logo_img.shape[0] * scale)), interpolation=cv2.INTER_AREA)
            elements.append(logo_img)

    # ساخت پچ متن در صورت پر بودن
    if text.strip():
        font = cv2.FONT_HERSHEY_DUPLEX
        (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        pad = 8
        t_patch = np.zeros((th + baseline + pad * 2, tw + pad * 2, 4), dtype=np.uint8)
        cv2.putText(t_patch, text, (pad, th + pad), font, font_scale, (*text_color, 255), thickness, cv2.LINE_AA)
        elements.append(t_patch)

    if not elements:
        return None

    if len(elements) == 1:
        return elements[0]

    # ترکیب عمودی لوگو و متن
    max_w = max(elements[0].shape[1], elements[1].shape[1])
    total_h = elements[0].shape[0] + elements[1].shape[0] + 10
    comb = np.zeros((total_h, max_w, 4), dtype=np.uint8)

    # چسباندن لوگو در بالا (وسط‌چین)
    l_w, l_h = elements[0].shape[1], elements[0].shape[0]
    ox1 = (max_w - l_w) // 2
    comb[0:l_h, ox1:ox1+l_w] = elements[0]

    # چسباندن متن در پایین (وسط‌چین)
    t_w, t_h = elements[1].shape[1], elements[1].shape[0]
    ox2 = (max_w - t_w) // 2
    comb[l_h+10:l_h+10+t_h, ox2:ox2+t_w] = elements[1]

    return comb

def overlay_patch(bg_bgr, patch_rgba, x, y, opacity):
    """چسباندن پچ RGBA روی تصویر BGR با آلفا بلندیگ و بررسی دقیق مرزها."""
    h_bg, w_bg = bg_bgr.shape[:2]
    h_p, w_p = patch_rgba.shape[:2]

    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(w_bg, x + w_p), min(h_bg, y + h_p)

    if x1 >= x2 or y1 >= y2:
        return bg_bgr

    px1, py1 = x1 - x, y1 - y
    px2, py2 = px1 + (x2 - x1), py1 + (y2 - y1)

    crop_patch = patch_rgba[py1:py2, px1:px2]
    alpha = (crop_patch[:, :, 3] / 255.0) * opacity
    alpha_3d = np.repeat(alpha[:, :, np.newaxis], 3, axis=2)

    bg_roi = bg_bgr[y1:y2, x1:x2].astype(np.float32)
    fg_roi = crop_patch[:, :, :3].astype(np.float32)

    blended = fg_roi * alpha_3d + bg_roi * (1.0 - alpha_3d)
    bg_bgr[y1:y2, x1:x2] = np.clip(blended, 0, 255).astype(np.uint8)
    return bg_bgr

# =========================================================
# ۷. منوی کناری (Sidebar)
# =========================================================
if "current_tab" not in st.session_state:
    st.session_state.current_tab = "detection"

st.sidebar.markdown("### 🧭 ناوبری سامانه")

if st.sidebar.button("🔍 ۱. تشخیص واترمارک (Detection)", use_container_width=True, type="primary" if st.session_state.current_tab == "detection" else "secondary"):
    st.session_state.current_tab = "detection"
    st.rerun()

if st.sidebar.button("🧹 ۲. مقایسه حذف (3-Method Telea)", use_container_width=True, type="primary" if st.session_state.current_tab == "removal" else "secondary"):
    st.session_state.current_tab = "removal"
    st.rerun()

if st.sidebar.button("➕ ۳. درج واترمارک (Add)", use_container_width=True, type="primary" if st.session_state.current_tab == "add" else "secondary"):
    st.session_state.current_tab = "add"
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("##### 📁 بارگذاری تصویر")
uploaded_file = st.sidebar.file_uploader("تصویر را انتخاب یا اینجا بکشید:", type=["jpg", "jpeg", "png", "webp"])

# =========================================================
# ۸. اجرای تب‌ها
# =========================================================
if uploaded_file is not None:
    file_bytes = np.frombuffer(uploaded_file.getvalue(), np.uint8)
    orig_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if orig_bgr is None:
        st.error("❌ فایل تصویر بارگذاری‌شده معتبر نیست.")
        st.stop()

    raw_img_rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)

    # ------------------ تب ۱: تشخیص ------------------
    if st.session_state.current_tab == "detection":
        st.markdown("### 🔍 ماژول تشخیص خودکار واترمارک")
        st.image(raw_img_rgb, caption="تصویر ورودی", use_container_width=True)

        if st.button("🚀 اجرای شناسایی باکس‌ها", type="primary", use_container_width=True):
            with st.spinner("در حال پایش تصویر..."):
                r1, stage_name = run_kernel_1(orig_bgr, model)
                yb, db = run_kernel_2(raw_img_rgb, r1)
                vis, f_boxes, stats = run_kernel_3(raw_img_rgb, yb, db)

            st.success(f"✔️ پردازش به پایان رسید | {stage_name}")
            c1, c2, c3 = st.columns(3)
            c1.metric("باکس‌های YOLO", f"{stats['yolo_count']} عدد")
            c2.metric("باکس‌های بازیابی تطبیق الگو", f"{stats['recovered_count']} عدد")
            c3.metric("مجموع نهایی واترمارک‌ها", f"{stats['total_count']} عدد")
            st.image(vis, caption="خروجی نهایی تشخیص", use_container_width=True)

    # ------------------ تب ۲: مقایسه حذف (3-Method Telea Comparison) ------------------
    elif st.session_state.current_tab == "removal":
        st.markdown("### 🧹 مقایسه ۳ متد اصلی بازسازی تصویر با الگوریتم Telea")
        st.caption("مقایسه خروجی Telea Inpainting متد ۱ (WatermarkDirector)، متد ۲ (Hybrid Opacity) و متد ۳ (MobileSAM Hybrid)")

        with st.expander("👁️ پیش‌نمایش تصویر بارگذاری‌شده", expanded=False):
            st.image(raw_img_rgb, use_container_width=True)

        if st.button("🚀 اجرای بازسازی و مقایسه ۳ متد (Telea)", type="primary", use_container_width=True):
            with st.spinner("در حال استخراج باکس‌ها و اجرای ۳ الگوریتم..."):
                r1, _ = run_kernel_1(orig_bgr, model)
                yb, db = run_kernel_2(raw_img_rgb, r1)
                _, final_boxes, _ = run_kernel_3(raw_img_rgb, yb, db)

                # متد ۱
                m1_rgb, m1_bgr, m1_mask = run_method_1(orig_bgr, final_boxes, raw_img_rgb)
                
                # متد ۲
                m2_rgb, m2_bgr, m2_mask = run_method_2(orig_bgr, final_boxes)

                # متد ۳
                sam_predictor, sam_err = load_mobile_sam_model(BASE_DIR)
                if sam_predictor is not None:
                    m3_rgb, m3_bgr, m3_mask = run_method_3(orig_bgr, raw_img_rgb, final_boxes, sam_predictor)
                else:
                    st.warning(f"⚠️ مدل MobileSAM بارگذاری نشد ({sam_err})؛ برای متد ۳ از ماسک متد ۱ استفاده شد.")
                    m3_rgb, m3_bgr, m3_mask = m1_rgb, m1_bgr, m1_mask

                st.session_state["m1_rgb"] = m1_rgb
                st.session_state["m1_bgr"] = m1_bgr
                st.session_state["m1_mask"] = m1_mask

                st.session_state["m2_rgb"] = m2_rgb
                st.session_state["m2_bgr"] = m2_bgr
                st.session_state["m2_mask"] = m2_mask

                st.session_state["m3_rgb"] = m3_rgb
                st.session_state["m3_bgr"] = m3_bgr
                st.session_state["m3_mask"] = m3_mask

        if "m1_rgb" in st.session_state:
            st.markdown("---")
            
            p1, p2, p3 = st.columns(3)

            with p1:
                st.markdown("<div class='method-card'><b>1. WatermarkDirector</b><br><small>Telea + Two-Mask Architecture</small></div>", unsafe_allow_html=True)
                st.image(st.session_state["m1_rgb"], use_container_width=True)
                with st.expander("👁️ نمایش ماسک"):
                    st.image(st.session_state["m1_mask"], use_container_width=True)
                _, buf_m1 = cv2.imencode(".png", st.session_state["m1_bgr"])
                st.download_button("📥 دانلود خروجی متد ۱", data=buf_m1.tobytes(), file_name="telea_watermark_director.png", mime="image/png", use_container_width=True)

            with p2:
                st.markdown("<div class='method-card'><b>2. Hybrid Opacity</b><br><small>Telea + Alpha Physics & Structure</small></div>", unsafe_allow_html=True)
                st.image(st.session_state["m2_rgb"], use_container_width=True)
                with st.expander("👁️ نمایش ماسک"):
                    st.image(st.session_state["m2_mask"], use_container_width=True)
                _, buf_m2 = cv2.imencode(".png", st.session_state["m2_bgr"])
                st.download_button("📥 دانلود خروجی متد ۲", data=buf_m2.tobytes(), file_name="telea_hybrid_opacity.png", mime="image/png", use_container_width=True)

            with p3:
                st.markdown("<div class='method-card'><b>3. MobileSAM Hybrid</b><br><small>Telea + SAM Segmentation & Gradient</small></div>", unsafe_allow_html=True)
                st.image(st.session_state["m3_rgb"], use_container_width=True)
                with st.expander("👁️ نمایش ماسک"):
                    st.image(st.session_state["m3_mask"], use_container_width=True)
                _, buf_m3 = cv2.imencode(".png", st.session_state["m3_bgr"])
                st.download_button("📥 دانلود خروجی متد ۳", data=buf_m3.tobytes(), file_name="telea_mobilesam_hybrid.png", mime="image/png", use_container_width=True)

    # ------------------ تب ۳: درج واترمارک ------------------
    # ------------------ تب ۳: درج واترمارک ------------------
    # ------------------ تب ۳: درج واترمارک ------------------
    elif st.session_state.current_tab == "add":
        st.markdown("### ➕ درج واترمارک سفارشی پیشرفته")
        
        # ۱. انتخاب منبع واترمارک
        c_src1, c_src2 = st.columns(2)
        with c_src1:
            source_type = st.radio("📌 منبع واترمارک:", ["نوشتن متن", "آپلود لوگو (همراه با متن)"], horizontal=True)
        with c_src2:
            wm_type = st.selectbox("📐 نوع چیدمان واترمارک:", ["تکی (Single)", "تکرارشونده (Repeated)", "باکس پس‌زمینه (Background Box)"])

        # پارامترهای محتوایی
        col_m1, col_m2 = st.columns(2)
        wm_text = ""
        logo_bytes = None

        if source_type == "نوشتن متن":
            with col_m1:
                wm_text = st.text_input("متن واترمارک:", value="Protected by AI")
            with col_m2:
                text_color_hex = st.color_picker("رنگ متن:", "#FFFFFF")
        else:
            with col_m1:
                uploaded_logo = st.file_uploader("فایل لوگو را انتخاب کنید (PNG شفاف یا JPG):", type=["png", "jpg", "jpeg"])
                if uploaded_logo is not None:
                    logo_bytes = uploaded_logo.getvalue()
            with col_m2:
                wm_text = st.text_input("متن همراه لوگو (اختیاری):", value="")
                text_color_hex = st.color_picker("رنگ متن:", "#FFFFFF")

        # تبدیل رنگ متن به BGR
        tc = text_color_hex.lstrip("#")
        text_bgr = tuple(int(tc[i:i+2], 16) for i in (4, 2, 0))

        # پارامترهای استایل و وضعیت چیدمان
        st.markdown("---")
        c_p1, c_p2, c_p3 = st.columns(3)
        with c_p1:
            wm_opacity = st.slider("شفافیت واترمارک (Opacity):", 0.05, 1.0, 0.60, 0.05)
        with c_p2:
            is_random = st.checkbox("🎲 زاویه و موقعیت رندوم", value=True)
        with c_p3:
            angle_manual = st.slider("زاویه چرخش دستی:", -90, 90, 0, 5, disabled=is_random)

        # تنظیمات رنگ باکس در حالت Background Box
        bg_bar_bgr = (0, 0, 0)
        box_opacity = 0.5
        if wm_type == "باکس پس‌زمینه (Background Box)":
            c_bg1, c_bg2 = st.columns(2)
            with c_bg1:
                bg_color_hex = st.color_picker("🎨 رنگ باکس پس‌زمینه:", "#000000")
                bc = bg_color_hex.lstrip("#")
                bg_bar_bgr = tuple(int(bc[i:i+2], 16) for i in (4, 2, 0))
            with c_bg2:
                box_opacity = st.slider("شفافیت باکس پس‌زمینه:", 0.1, 1.0, 0.5, 0.05)

        # دکمه اجرا
        if st.button("🚀 اعمال واترمارک", type="primary", use_container_width=True):
            if source_type == "آپلود لوگو (همراه با متن)" and logo_bytes is None and not wm_text.strip():
                st.warning("⚠️ لطفاً فایل لوگو را آپلود کرده یا یک متن وارد نمایید.")
            else:
                out_img = orig_bgr.copy()
                h_img, w_img = out_img.shape[:2]

                # ساخت پچ واترمارک
                patch = build_watermark_patch(
                    text=wm_text,
                    logo_bytes=logo_bytes,
                    font_scale=1.2,
                    thickness=2,
                    text_color=text_bgr
                )

                if patch is not None:
                    # تعیین زاویه
                    chosen_angle = float(random.choice([-45, -30, -15, 0, 15, 30, 45])) if is_random else float(angle_manual)
                    rotated_patch = rotate_bound_rgba(patch, chosen_angle)
                    ph, pw = rotated_patch.shape[:2]

                    # ۱. حالت تکی (Single)
                    if wm_type == "تکی (Single)":
                        if pw > w_img or ph > h_img:
                            scale_factor = min(w_img / float(pw), h_img / float(ph)) * 0.85
                            rotated_patch = cv2.resize(rotated_patch, (int(pw * scale_factor), int(ph * scale_factor)), interpolation=cv2.INTER_AREA)
                            ph, pw = rotated_patch.shape[:2]

                        max_x = max(0, w_img - pw)
                        max_y = max(0, h_img - ph)
                        pos_x = random.randint(0, max_x) if is_random and max_x > 0 else (w_img - pw) // 2
                        pos_y = random.randint(0, max_y) if is_random and max_y > 0 else (h_img - ph) // 2

                        out_img = overlay_patch(out_img, rotated_patch, pos_x, pos_y, wm_opacity)

                    # ۲. حالت تکرارشونده (Repeated)
                    elif wm_type == "تکرارشونده (Repeated)":
                        step_x = max(pw + 60, 100)
                        step_y = max(ph + 60, 100)
                        offset_x = random.randint(0, step_x // 2) if is_random else 0
                        offset_y = random.randint(0, step_y // 2) if is_random else 0

                        for y in range(-ph + offset_y, h_img + ph, step_y):
                            for x in range(-pw + offset_x, w_img + pw, step_x):
                                out_img = overlay_patch(out_img, rotated_patch, x, y, wm_opacity)

                    # ۳. حالت باکس پس‌زمینه (Background Box)
                    elif wm_type == "باکس پس‌زمینه (Background Box)":
                        pad = 18
                        if pw + 2 * pad > w_img or ph + 2 * pad > h_img:
                            s_f = min((w_img - 2 * pad) / float(pw), (h_img - 2 * pad) / float(ph)) * 0.90
                            rotated_patch = cv2.resize(rotated_patch, (int(pw * s_f), int(ph * s_f)), interpolation=cv2.INTER_AREA)
                            ph, pw = rotated_patch.shape[:2]

                        box_w = pw + 2 * pad
                        box_h = ph + 2 * pad
                        max_x = max(0, w_img - box_w)
                        max_y = max(0, h_img - box_h)

                        bx1 = random.randint(0, max_x) if is_random and max_x > 0 else (w_img - box_w) // 2
                        by1 = random.randint(0, max_y) if is_random and max_y > 0 else (h_img - box_h) // 2
                        bx2 = bx1 + box_w
                        by2 = by1 + box_h

                        # رسم باکس دور واترمارک
                        box_layer = out_img.copy()
                        cv2.rectangle(box_layer, (bx1, by1), (bx2, by2), bg_bar_bgr, -1)
                        out_img = cv2.addWeighted(box_layer, box_opacity, out_img, 1.0 - box_opacity, 0)

                        # قرار دادن متن/لوگو داخل باکس
                        out_img = overlay_patch(out_img, rotated_patch, bx1 + pad, by1 + pad, wm_opacity)

                    # ذخیره در session_state تا با هر کلیک یا رفرش پاک نشود
                    st.session_state["watermarked_result"] = out_img
                else:
                    st.error("خطا در ایجاد واترمارک. متن یا لوگو را بررسی کنید.")

        # نمایش تصویر نهایی خارج از بلاک if تا پایدار بماند
        if "watermarked_result" in st.session_state and st.session_state["watermarked_result"] is not None:
            res_img = st.session_state["watermarked_result"]
            st.image(cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB), caption="خروجی واترمارک اعمال‌شده", use_container_width=True)
            _, buf = cv2.imencode(".png", res_img)
            st.download_button(
                "📥 دانلود تصویر با واترمارک",
                data=buf.tobytes(),
                file_name="watermarked_custom.png",
                mime="image/png",
                use_container_width=True
            )



else:
    st.info("👈 لطفاً از پنل سمت چپ یک تصویر بارگذاری کنید.")
