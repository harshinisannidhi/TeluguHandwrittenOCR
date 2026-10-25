
import re
import cv2
import numpy as np
from skimage.morphology import skeletonize

def decode_gray(data):
    # bytes of an image file -> uint8 gray, transparent pixels become WHITE.
    # Used by BOTH training and the Flask app (no train/serve skew).
    try:
        img = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    except Exception:
        return None
    if img is None:
        return None
    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if img.ndim == 3 and img.shape[2] == 4:
        alpha = img[:, :, 3:4].astype(np.float32) / 255.0
        rgb = img[:, :, :3].astype(np.float32)
        img = (rgb * alpha + 255.0 * (1.0 - alpha)).astype(np.uint8)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def load_image_gray(path):
    # robust loader (handles unicode/space paths and RGBA images); returns uint8 gray or None
    try:
        return decode_gray(np.fromfile(path, dtype=np.uint8).tobytes())
    except Exception:
        return None

def crop_to_ink(img, margin=0.15):
    # img: dark ink on light paper. Crop to the ink bounding box (+margin) and return a SQUARE canvas,
    # so the character always fills the frame however much white space the uploaded image has.
    h0, w0 = img.shape[:2]
    bg = int(np.percentile(img, 90))
    blur = cv2.GaussianBlur(img, (3, 3), 0)
    _, ink = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    x0, y0, x1, y1 = 0, 0, w0, h0                      # fallback = whole image
    if (ink > 0).mean() < 0.6:                         # >60% "ink" means a bad threshold -> keep whole image
        n, _, st, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
        min_area = max(3, int(0.0002 * h0 * w0))       # drop dust, but keep small dots (anusvara / visarga)
        keep = [i for i in range(1, n) if st[i, cv2.CC_STAT_AREA] >= min_area]
        if keep:
            x0 = min(st[i, cv2.CC_STAT_LEFT] for i in keep)
            y0 = min(st[i, cv2.CC_STAT_TOP] for i in keep)
            x1 = max(st[i, cv2.CC_STAT_LEFT] + st[i, cv2.CC_STAT_WIDTH] for i in keep)
            y1 = max(st[i, cv2.CC_STAT_TOP] + st[i, cv2.CC_STAT_HEIGHT] for i in keep)
    crop = img[y0:y1, x0:x1]
    ch, cw = crop.shape[:2]
    side = max(int(round(max(ch, cw) * (1 + 2 * margin))), 8)
    canvas = np.full((side, side), bg, dtype=np.uint8)
    oy, ox = (side - ch) // 2, (side - cw) // 2
    canvas[oy:oy + ch, ox:ox + cw] = crop
    return canvas

def preprocess_gray(img, size=96, margin=0.15):
    # 1) polarity -> dark ink on light paper
    if np.median(img) < 127:
        img = 255 - img
    # 2) crop to the strokes, centre on a square canvas, resize
    img = crop_to_ink(img, margin)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    # 3) noise removal
    img = cv2.fastNlMeansDenoising(img, None, h=10, templateWindowSize=5, searchWindowSize=15)
    # 4) CLAHE contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    return clahe.apply(img)

def compute_stroke_maps(gray):
    # 3-channel stroke descriptor: skeleton, stroke-width (distance transform), edge magnitude
    _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    skel = skeletonize(ink > 0).astype(np.float32)
    dist = cv2.distanceTransform(ink, cv2.DIST_L2, 3)
    dist = dist / max(float(dist.max()), 1e-6)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    mag = mag / max(float(mag.max()), 1e-6)
    return np.stack([skel, dist, mag]).astype(np.float32)

# ---------- Telugu text helpers (unicode units + dictionary correction) ----------
def telugu_units(text):
    # split a string into aksharas: base char + following signs; a virama glues the next consonant
    units, cur, glue = [], "", False
    for ch in text:
        cp = ord(ch)
        is_sign = (0x0C3E <= cp <= 0x0C56) or (0x0C00 <= cp <= 0x0C03)
        if cur and (is_sign or glue):
            cur += ch
            glue = (cp == 0x0C4D)
        else:
            if cur:
                units.append(cur)
            cur, glue = ch, False
    if cur:
        units.append(cur)
    return units

def edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        curr = [i]
        for j, y in enumerate(b, 1):
            curr.append(min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = curr
    return prev[-1]

def correct_word(word, dictionary_units, max_dist=None):
    # dictionary_units: dict {word: list_of_units}. returns (best_word, distance)
    u = telugu_units(word)
    if not u:
        return word, 0
    if max_dist is None:
        max_dist = 1 if len(u) <= 4 else 2
    best, best_d = word, max_dist + 1
    for w, wu in dictionary_units.items():
        if abs(len(wu) - len(u)) > max_dist:
            continue
        d = edit_distance(u, wu)
        if d < best_d:
            best, best_d = w, d
            if d == 0:
                break
    return (best, best_d) if best_d <= max_dist else (word, -1)
