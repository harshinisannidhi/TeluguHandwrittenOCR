import os
import io
import json
import csv
import re
import glob
import uuid
import time
import warnings
from datetime import datetime
from xml.sax.saxutils import escape as xml_escape
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont, features as pil_features
from flask import Flask, render_template, request, redirect, url_for, session, send_file, abort
from werkzeug.utils import secure_filename
import mysql.connector
from mysql.connector import Error

# ReportLab is used for the downloadable PDF report (pip install reportlab).
try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
                                    Image as RLImage, KeepTogether)
    REPORTLAB_OK = True
except ImportError:
    REPORTLAB_OK = False
    print("⚠️  reportlab not installed: PDF report download disabled (pip install reportlab)")

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models import mobilenet_v3_small

# Same helper files that the notebook wrote (CELL 2 / CELL 3.1): preprocessing, stroke maps,
# akshara splitting, dictionary correction and class-name -> Telugu Unicode resolution.
# Using the identical code here means there is no train/serve skew.
import ocr_utils
from ocr_utils import load_image_gray, preprocess_gray, compute_stroke_maps, telugu_units, correct_word
from telugu_unicode import resolve_unicode, TELUGU_RE, VIRAMA

warnings.filterwarnings('ignore')

# The retrained model expects images that were cropped to the ink bounding box (crop_to_ink) before resizing.
# An old ocr_utils.py silently skips that step and accuracy drops, so fail loudly instead.
if not hasattr(ocr_utils, 'crop_to_ink'):
    raise RuntimeError("ocr_utils.py is outdated (no crop_to_ink). Replace it with the ocr_utils.py written by the "
                       "latest notebook run, otherwise predictions will not match how the model was trained.")

# ====================== INIT APP ======================
app = Flask(__name__)
app.secret_key = 'telugu-ocr-secret-2026-change-this'
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024      # 32 MB upload limit (word = many character images)

# ====================== MYSQL CONFIG ======================
# Same users table / login flow as before (database "BBR").
try:
    mydb = mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        port="3306",
        database="BBR"
    )
    mycursor = mydb.cursor(dictionary=True)
    print("✅ MySQL Connected")
except Error as e:
    print(f"❌ DB Error: {e}")
    exit(1)


# ====================== DB HELPERS ======================
def execute_query(query, values=None):
    try:
        if values:
            mycursor.execute(query, values)
        else:
            mycursor.execute(query)
        mydb.commit()
    except Error as e:
        mydb.rollback()
        print(f"Query Error: {e}")
        raise


def fetch_all(query, values=None):
    try:
        if values:
            mycursor.execute(query, values)
        else:
            mycursor.execute(query)
        return mycursor.fetchall()
    except Error as e:
        print(f"Fetch Error: {e}")
        return []


# ====================== CONFIG ======================
# Anchored to this file's own folder — NOT the process's current working
# directory — so model loading works no matter where `python app.py` is run from.
#
# Expected project layout (matches the notebook's /kaggle/working outputs):
#     app.py , ocr_utils.py , telugu_unicode.py
#     telugu_words.txt                          (optional, one Telugu word per line, UTF-8)
#     reports/                                  (read by /model_performance)
#         model_comparison.csv
#         classification_report_Model2_SGF_MobileNetV3_Proposed.csv
#         accuracy_per_group.csv
#         class_unicode_mapping.csv             (optional)
#         upload_simulation.csv                 (optional, robustness test shown on /model_performance)
#         metrics_summary.json                  (optional)
#     saved_models/                             <- copy from /kaggle/working/models/
#         best_model.pth  +  best_model_meta.json      (preferred, safe to load)
#         (or best_model_checkpoint.pt, which already contains the metadata)
#     static/fonts/NotoSansTelugu-Regular.ttf   (optional, only needed for Telugu text inside the PDF report)
#     templates/ , static/
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
ARTIFACT_DIR   = os.path.join(BASE_DIR, 'saved_models')
REPORT_DIR     = os.path.join(BASE_DIR, 'reports')
UPLOAD_DIR     = os.path.join(BASE_DIR, 'uploads')
RESULT_DIR     = os.path.join(BASE_DIR, 'static', 'results')
REPORT_OUT_DIR = os.path.join(BASE_DIR, 'report_cache')     # generated PDF reports (not web-public)
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(REPORT_OUT_DIR, exist_ok=True)

WEIGHT_FILENAMES = ['best_model.pth', 'best_model_checkpoint.pt']   # first one found is loaded
META_FILENAME    = 'best_model_meta.json'
DICT_FILENAMES   = [os.path.join(BASE_DIR, 'telugu_words.txt'),
                    os.path.join(ARTIFACT_DIR, 'telugu_words.txt')]
ALLOWED_IMAGE_EXTENSIONS = {'jpg', 'jpeg', 'png', 'bmp', 'webp', 'tif', 'tiff'}
MAX_CHARACTERS = 40                                         # max character images in one word

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ====================== IMAGE / MODEL CONSTANTS ======================
# Mirrors the notebook's config (CELL 1 / CELL 7). IMG_SIZE, CLASS_KEYS, IDX2UNI,
# MODEL_NAME and USES_STROKE are overwritten at startup from the saved metadata.
IMG_SIZE = 96
IM_MEAN, IM_STD = 0.449, 0.226                              # notebook CELL 7 (gray ImageNet statistics)
MODEL_NAME  = 'Model2_SGF_MobileNetV3_Proposed'             # model selected in notebook CELL 11
MODEL_SHORT = {'Model1_MobileNetV3Small_Baseline': 'Baseline',
               'Model2_SGF_MobileNetV3_Proposed': 'SGF-MobileNetV3'}
USES_STROKE = True
GROUPS = ['Guninthamulu', 'achulu', 'hallulu', 'othulu']    # source groups of the dataset (used for the per-group report)
CLASS_KEYS = []                                             # "<type>:<folder>" per class index, e.g. "cons:ka", "vowel:aa", "sub:c"
# Classes are now merged by Unicode: one class per Telugu character, whichever dataset group it came from.
TYPE_ALIASES = {'Guninthamulu': 'cons', 'hallulu': 'cons', 'achulu': 'vowel', 'othulu': 'sub'}   # also reads older "<group>/<folder>" keys
TYPE_LABELS  = {'cons': 'Consonant', 'vowel': 'Vowel', 'sub': 'Conjunct (othulu)'}
TYPE_TO_GROUP = {'cons': 'hallulu', 'vowel': 'achulu', 'sub': 'othulu'}                          # only for the Unicode fallback
IDX2UNI = {}                                                # class index -> Telugu Unicode string

# ---- Prediction display settings ---------------------------------------------
TOP_K          = 3        # alternatives listed for every recognised character
MIN_CONFIDENCE = 0.50     # below this top-1 probability a character is flagged as uncertain (same 0.5 as the notebook)

GROUP_DESCRIPTIONS = {
    'achulu':       'Achulu (vowels): అ ఆ ఇ ఈ ఉ ఊ ఋ ఎ ఏ ఐ ఒ ఓ ఔ ం ః',
    'hallulu':      'Hallulu (consonants): క ఖ గ ఘ ఙ చ ఛ జ ఝ ఞ ట ఠ డ ఢ ణ త థ ద ధ న ప ఫ బ భ మ య ర ల వ శ ష స హ ళ క్ష ఱ',
    'Guninthamulu': 'Guninthamulu: a consonant combined with a vowel sign (కా కి కీ కు కూ ...).',
    'othulu':       'Othulu (vattulu): conjunct subscript consonants, stored as VIRAMA + consonant (్క ్ఖ ్గ ...).',
}

# Fallback snapshot of the notebook's latest run (sections 3.1, 9, 9.1, 10, 11): 292,752 images in 121 folders ->
# merged into 76 classes, capped to 73,463 images and split 51,423 / 11,020 / 11,020. load_reports_from_disk()
# below replaces these with the live CSV / JSON files when they are in ./reports.
NUM_CLASSES_SNAPSHOT = 76
TEST_SPLIT = {'train': 51423, 'val': 11020, 'test': 11020}
DATASET_INFO = {'raw_images': 292752, 'raw_folders': 121, 'classes': 76, 'merged_characters': 33,
                'images_used': 73463, 'cap_per_group_class': 1000}

MODEL_COMPARISON = [
    {'model': 'SGF-MobileNetV3', 'accuracy': 0.9590, 'top3_accuracy': 0.9948, 'precision': 0.9533,
     'recall': 0.9571, 'f1': 0.9534, 'weighted_f1': 0.9589, 'cpu_ms': 18.2209, 'params_m': 1.4065,
     'size_mb': 5.7855, 'best_epoch': 29, 'train_minutes': 22.6105},
    {'model': 'Baseline', 'accuracy': 0.9074, 'top3_accuracy': 0.9861, 'precision': 0.8997,
     'recall': 0.9231, 'f1': 0.9069, 'weighted_f1': 0.9089, 'cpu_ms': 14.7804, 'params_m': 0.1942,
     'size_mb': 0.8590, 'best_epoch': 30, 'train_minutes': 17.7028},
]

# Per-group test accuracy (notebook section 9): {group: {model short name: accuracy}}
GROUP_ACCURACY = [
    {'group': 'Guninthamulu', 'accuracies': {'Baseline': 0.8911, 'SGF-MobileNetV3': 0.9434}},
    {'group': 'achulu',       'accuracies': {'Baseline': 0.9560, 'SGF-MobileNetV3': 0.9784}},
    {'group': 'hallulu',      'accuracies': {'Baseline': 0.8602, 'SGF-MobileNetV3': 0.9491}},
    {'group': 'othulu',       'accuracies': {'Baseline': 0.9566, 'SGF-MobileNetV3': 0.9834}},
]

# Robustness test (notebook section 9.1): test images re-saved the way users upload them (extra white margins, random size,
# JPEG compression, blur) and pushed through the same preprocessing as this app.
UPLOAD_SIM = [
    {'model': 'SGF-MobileNetV3', 'clean': 0.9607, 'upload': 0.9063,
     'groups': {'Guninthamulu': 0.8777, 'achulu': 0.9390, 'hallulu': 0.9021, 'othulu': 0.9392}},
    {'model': 'Baseline', 'clean': 0.9067, 'upload': 0.8510,
     'groups': {'Guninthamulu': 0.8300, 'achulu': 0.8829, 'hallulu': 0.8070, 'othulu': 0.9146}},
]

# Metrics of the deployed model (first row of MODEL_COMPARISON is replaced by the real deployed row on load).
TEST_SET_METRICS = {
    'accuracy':        0.9590,
    'top3_accuracy':   0.9948,
    'macro_precision': 0.9533,
    'macro_recall':    0.9571,
    'macro_f1':        0.9534,
    'weighted_f1':     0.9589,
}

# Per-class report and weakest classes are only available from the exported CSV (one row per class).
TEST_SET_CLASSIFICATION_REPORT = []
TEST_SET_REPORT_AVERAGES = []
WEAKEST_CLASSES = []
LATENCY = {'preprocessing_ms': 10.48, 'stroke_extraction_ms': 0.40}
# NOTE: the notebook only plots the confusion matrix; it does not export it as a CSV,
# so none is shown in the web app.


def split_key(key):
    """'cons:ka' -> ('cons', 'ka').  Older 'hallulu/ka' keys are mapped to the same three types."""
    m = re.split(r'[:/]', key, maxsplit=1)
    t, name = (m[0], m[1]) if len(m) == 2 else ('', key)
    return TYPE_ALIASES.get(t, t), name


def class_type_label(key):
    return TYPE_LABELS.get(split_key(key)[0], split_key(key)[0] or '-')


def _num(row, key, cast=float, default=None):
    try:
        return cast(float(row[key]))
    except (KeyError, ValueError, TypeError):
        return default


def load_reports_from_disk():
    """Overrides the static snapshot above with the real files exported by the
    notebook (CELL 12 / CELL 14 / CELL 15), when they are present in ./reports. If a
    file is missing or unreadable the hard-coded snapshot is kept."""
    global TEST_SET_METRICS, TEST_SPLIT, MODEL_COMPARISON, GROUP_ACCURACY, LATENCY
    global TEST_SET_CLASSIFICATION_REPORT, TEST_SET_REPORT_AVERAGES, WEAKEST_CLASSES, UPLOAD_SIM

    # ---- model_comparison.csv (first column = short model name; index of the notebook DataFrame)
    try:
        with open(os.path.join(REPORT_DIR, 'model_comparison.csv'), newline='', encoding='utf-8-sig') as f:
            rows = list(csv.DictReader(f))
        name_col = list(rows[0].keys())[0]
        MODEL_COMPARISON = [
            {'model': r[name_col], 'accuracy': _num(r, 'accuracy'), 'top3_accuracy': _num(r, 'top3_accuracy'),
             'precision': _num(r, 'precision_macro'), 'recall': _num(r, 'recall_macro'),
             'f1': _num(r, 'f1_macro'), 'weighted_f1': _num(r, 'f1_weighted'),
             'cpu_ms': _num(r, 'cpu_end2end_ms'), 'params_m': _num(r, 'params_M'),
             'size_mb': _num(r, 'size_MB'), 'best_epoch': _num(r, 'best_epoch', int),
             'train_minutes': _num(r, 'train_time_min')}
            for r in rows
        ]
        MODEL_COMPARISON.sort(key=lambda m: m['f1'] or 0, reverse=True)
        print("✅ Loaded model_comparison.csv")
    except Exception as e:
        print(f"⚠️  Using built-in model comparison snapshot ({e})")

    # ---- metrics of the deployed model = its row in the comparison table
    deployed = next((m for m in MODEL_COMPARISON if m['model'] == MODEL_SHORT.get(MODEL_NAME)), None)
    if deployed:
        TEST_SET_METRICS.update({
            'accuracy': deployed['accuracy'], 'top3_accuracy': deployed['top3_accuracy'],
            'macro_precision': deployed['precision'], 'macro_recall': deployed['recall'],
            'macro_f1': deployed['f1'], 'weighted_f1': deployed['weighted_f1'],
        })

    # ---- classification_report_<model>.csv (pandas DataFrame.T of sklearn output_dict)
    try:
        with open(os.path.join(REPORT_DIR, f'classification_report_{MODEL_NAME}.csv'),
                  newline='', encoding='utf-8-sig') as f:
            rows = list(csv.DictReader(f))
        name_col = list(rows[0].keys())[0]
        summary_names = ('accuracy', 'macro avg', 'weighted avg')
        TEST_SET_CLASSIFICATION_REPORT = [
            {'class': r[name_col], 'group': class_type_label(r[name_col]),
             'precision': float(r['precision']), 'recall': float(r['recall']),
             'f1': float(r['f1-score']), 'support': int(float(r['support']))}
            for r in rows if r[name_col] not in summary_names
        ]
        TEST_SET_REPORT_AVERAGES = [
            {'label': r[name_col], 'precision': float(r['precision']), 'recall': float(r['recall']),
             'f1': float(r['f1-score']), 'support': int(float(r['support']))}
            for r in rows if r[name_col] in ('macro avg', 'weighted avg')
        ]
        WEAKEST_CLASSES = sorted(TEST_SET_CLASSIFICATION_REPORT, key=lambda c: c['f1'])[:15]
        print(f"✅ Loaded classification_report_{MODEL_NAME}.csv")
    except Exception as e:
        print(f"⚠️  Per-class report not available ({e})")

    # ---- accuracy_per_group.csv (columns: model, group, accuracy, n)
    try:
        with open(os.path.join(REPORT_DIR, 'accuracy_per_group.csv'), newline='', encoding='utf-8-sig') as f:
            rows = list(csv.DictReader(f))
        by_group = {}
        for r in rows:
            by_group.setdefault(r['group'], {})[r['model']] = float(r['accuracy'])
        GROUP_ACCURACY = [{'group': g, 'accuracies': by_group[g]} for g in GROUPS if g in by_group]
        print("✅ Loaded accuracy_per_group.csv")
    except Exception as e:
        print(f"⚠️  Using built-in per-group accuracy snapshot ({e})")

    # ---- upload_simulation.csv (index = model; clean_acc, upload_acc, upload_<group> ...)
    try:
        with open(os.path.join(REPORT_DIR, 'upload_simulation.csv'), newline='', encoding='utf-8-sig') as f:
            rows = list(csv.DictReader(f))
        name_col = list(rows[0].keys())[0]
        UPLOAD_SIM = [{'model': r[name_col], 'clean': float(r['clean_acc']), 'upload': float(r['upload_acc']),
                       'groups': {k[len('upload_'):]: float(v) for k, v in r.items()
                                  if k.startswith('upload_') and k != 'upload_acc' and v not in ('', None)}}
                      for r in rows]
        UPLOAD_SIM.sort(key=lambda m: m['upload'], reverse=True)
        print("✅ Loaded upload_simulation.csv")
    except Exception as e:
        print(f"⚠️  Using built-in upload-simulation snapshot ({e})")

    # ---- metrics_summary.json (split sizes + preprocessing / stroke timing)
    try:
        with open(os.path.join(REPORT_DIR, 'metrics_summary.json'), encoding='utf-8') as f:
            ms = json.load(f)
        TEST_SPLIT = ms.get('split', TEST_SPLIT)
        LATENCY = {'preprocessing_ms': ms.get('preprocessing_ms', LATENCY['preprocessing_ms']),
                   'stroke_extraction_ms': ms.get('stroke_extraction_ms', LATENCY['stroke_extraction_ms'])}
        print("✅ Loaded metrics_summary.json")
    except Exception as e:
        print(f"⚠️  Using built-in split / timing snapshot ({e})")


# (load_reports_from_disk() is called once below, after the model metadata is read, so the
#  deployed model's row can be matched in the comparison table.)


# ====================== MODEL ARCHITECTURE (INFERENCE-ONLY) ======================
# Mirrors the notebook's CELL 7 (BaselineMNV3, StrokeBranch, SGFMobileNetV3) EXACTLY, so the
# saved state dict loads into identical module shapes. ImageNet weights are NOT downloaded
# here (weights=None) — they come from the checkpoint.

def build_backbone():
    return mobilenet_v3_small(weights=None)


def to_rgb(x):                      # (B,1,H,W) in [0,1] -> normalised (B,3,H,W)
    return ((x - IM_MEAN) / IM_STD).expand(-1, 3, -1, -1)


class BaselineMNV3(nn.Module):
    """Reduced-capacity MobileNetV3-Small: early stages only (features[:9] -> 96-d)."""
    def __init__(self, n_classes):
        super().__init__()
        m = build_backbone()
        self.features = nn.Sequential(*list(m.features.children())[:9])
        self.pool = nn.AdaptiveAvgPool2d(1)
        with torch.no_grad():
            feat_dim = self.features(torch.zeros(1, 3, IMG_SIZE, IMG_SIZE)).shape[1]
        self.feat_dim = feat_dim
        self.head = nn.Sequential(nn.Dropout(0.35), nn.Linear(feat_dim, n_classes))

    def forward(self, x, s=None, return_gate=False):
        f = self.pool(self.features(to_rgb(x))).flatten(1)
        return self.head(f)


class StrokeBranch(nn.Module):
    def __init__(self, out_dim=128):
        super().__init__()

        def blk(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, 2, 1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))

        self.net = nn.Sequential(blk(3, 16), blk(16, 32), blk(32, 64), blk(64, out_dim), nn.AdaptiveAvgPool2d(1))

    def forward(self, s):
        return self.net(s).flatten(1)


class SGFMobileNetV3(nn.Module):
    """Full MobileNetV3-Small + stroke branch + gated feature fusion."""
    def __init__(self, n_classes, fused_dim=256):
        super().__init__()
        m = build_backbone()
        self.features, self.pool = m.features, nn.AdaptiveAvgPool2d(1)    # full 576-d backbone
        self.stroke = StrokeBranch(128)
        self.proj_app = nn.Sequential(nn.Linear(576, fused_dim), nn.BatchNorm1d(fused_dim), nn.ReLU(inplace=True))
        self.proj_stroke = nn.Sequential(nn.Linear(128, fused_dim), nn.BatchNorm1d(fused_dim), nn.ReLU(inplace=True))
        self.gate = nn.Sequential(nn.Linear(576 + 128, fused_dim), nn.Sigmoid())
        self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(fused_dim, n_classes))

    def forward(self, x, s, return_gate=False):
        fa = self.pool(self.features(to_rgb(x))).flatten(1)
        fs = self.stroke(s)
        g = self.gate(torch.cat([fa, fs], 1))
        fused = g * self.proj_app(fa) + (1 - g) * self.proj_stroke(fs)
        logits = self.head(fused)
        return (logits, g) if return_gate else logits


MODEL_CLASSES = {
    'Model1_MobileNetV3Small_Baseline': (BaselineMNV3, False),
    'Model2_SGF_MobileNetV3_Proposed':  (SGFMobileNetV3, True),
}

# Same manual fixes as the notebook's MANUAL_UNICODE, keyed by the new "<type>:<folder>" class names.
# Only used by the fallback below (when the saved metadata has no idx_to_unicode).
MANUAL_UNICODE = {
    "cons:ana": "ఙ", "cons:khh": "ఖ", "vowel:ao": "ఔ", "cons:jna": "ఞ", "cons:ks": "క్ష",
    "sub:an": VIRAMA + "ఙ", "sub:z": VIRAMA + "ళ",
}

# Optional manual corrections applied AFTER the saved mapping, keyed by class name -> Telugu Unicode.
# The notebook auto-resolved "cons:thah" to థః (థ + visarga) and asked you to verify it against the sample images it prints.
# If those images show a plain థ, un-comment the line below; duplicate Unicode is handled by build_unicode_groups().
UNICODE_OVERRIDES = {
    # "cons:thah": "థ",
}


def build_idx2uni(class_keys, saved=None):
    """Class index -> Telugu Unicode. Uses the mapping saved by the notebook when present,
    otherwise re-resolves the folder names with telugu_unicode.resolve_unicode()."""
    out = {}
    for i, key in enumerate(class_keys):
        if saved and str(i) in saved and TELUGU_RE.search(saved[str(i)]):
            out[i] = saved[str(i)]
        else:
            t, name = split_key(key)
            name = re.sub(r'_[0-9a-f]+$', '', name)            # strip the "_<hex>" the notebook adds to disambiguate names
            u, _ = resolve_unicode(name, TYPE_TO_GROUP.get(t, ''))
            out[i] = MANUAL_UNICODE.get(f'{t}:{name}', u)
        if key in UNICODE_OVERRIDES:
            out[i] = UNICODE_OVERRIDES[key]
    return out


# Several classes can end up with the same Unicode (only if you add UNICODE_OVERRIDES, or load an older un-merged model).
# Their probabilities are summed so the character is judged as one class, not split across look-alike folders.
UNI_LIST, UNI_MEMBERS, AGG = [], {}, None


def build_unicode_groups():
    global UNI_LIST, UNI_MEMBERS, AGG
    UNI_MEMBERS = {}
    for i in range(len(CLASS_KEYS)):
        UNI_MEMBERS.setdefault(IDX2UNI[i], []).append(i)
    UNI_LIST = list(UNI_MEMBERS)
    AGG = np.zeros((len(UNI_LIST), len(CLASS_KEYS)), dtype=np.float32)
    for r, u in enumerate(UNI_LIST):
        AGG[r, UNI_MEMBERS[u]] = 1.0


def codepoints(text):
    return ' '.join(f'U+{ord(c):04X}' for c in text)


# ====================== LOAD DEPLOYMENT ARTIFACTS ======================
# Loads what the notebook's section 12 (export_model) saved: best_model.pth (state_dict) and
# best_model_meta.json (model_name, uses_stroke, img_size, class_keys, idx_to_unicode, ...).
# If only best_model_checkpoint.pt is present, the metadata is read from inside it.
FUSION_MODEL = None
ARTIFACTS_READY = False

try:
    weights_path = next((os.path.join(ARTIFACT_DIR, fn) for fn in WEIGHT_FILENAMES
                         if os.path.exists(os.path.join(ARTIFACT_DIR, fn))), None)
    if weights_path is None:
        raise FileNotFoundError(f"none of {WEIGHT_FILENAMES} found in {ARTIFACT_DIR}")

    meta, state = None, None
    meta_path = os.path.join(ARTIFACT_DIR, META_FILENAME)
    if os.path.isfile(meta_path):
        with open(meta_path, encoding='utf-8') as f:
            meta = json.load(f)
    if weights_path.endswith('.pth'):
        state = torch.load(weights_path, map_location=device, weights_only=True)
    else:
        # full checkpoint holds numpy scalars in its metadata, so weights_only=True cannot read it;
        # only load a checkpoint you created yourself.
        obj = torch.load(weights_path, map_location=device, weights_only=False)
        state, meta = obj['state_dict'], (meta or obj.get('meta'))
    if meta is None:
        raise FileNotFoundError(f"{META_FILENAME} not found in {ARTIFACT_DIR}")

    MODEL_NAME = meta['model_name']
    if MODEL_NAME not in MODEL_CLASSES:
        raise ValueError(f"unknown model '{MODEL_NAME}' (expected one of {list(MODEL_CLASSES)})")
    model_cls, USES_STROKE = MODEL_CLASSES[MODEL_NAME]
    IMG_SIZE   = int(meta.get('img_size', IMG_SIZE))
    CLASS_KEYS = list(meta['class_keys'])
    IDX2UNI    = build_idx2uni(CLASS_KEYS, meta.get('idx_to_unicode'))
    build_unicode_groups()

    FUSION_MODEL = model_cls(len(CLASS_KEYS)).to(device)
    FUSION_MODEL.load_state_dict(state)
    FUSION_MODEL.eval()

    ARTIFACTS_READY = True
    print(f"✅ Loaded {MODEL_NAME} | classes: {len(CLASS_KEYS)} | image size: {IMG_SIZE} "
          f"| stroke branch: {USES_STROKE} | file: {os.path.basename(weights_path)} | device: {device}")
except Exception as e:
    print(f"❌ Could not load deployment artifacts from {ARTIFACT_DIR}: {e}")
    print("⚠️  Predictions will be unavailable until this is fixed.")

# Load the exported reports now that the deployed model name is known.
load_reports_from_disk()


# ====================== DICTIONARY (WORD CORRECTION) ======================
# Notebook CELL 13: akshara-level dictionary correction. Put your own list in telugu_words.txt
# (one Telugu word per line, UTF-8); the small built-in list below is used otherwise.
BUILTIN_WORDS = """తెలుగు అమ్మ నాన్న అన్న అక్క తమ్ముడు చెల్లి నీరు పాలు పండు పూలు చెట్టు ఇల్లు బడి పుస్తకం కలము
రాముడు కృష్ణుడు భారతదేశం ఆంధ్ర తల్లి తండ్రి స్నేహితుడు గురువు విద్యార్థి పాఠశాల ఆవు గుర్రం పిల్లి కుక్క చేపలు
పక్షి సూర్యుడు చంద్రుడు నక్షత్రం ఆకాశం భూమి గాలి నిప్పు వర్షం కథ పాట నది కొండ అడవి పట్టణం గ్రామం""".split()


def load_dictionary():
    for path in DICT_FILENAMES:
        if os.path.isfile(path):
            try:
                with open(path, encoding='utf-8-sig') as f:
                    words = [w.strip() for w in f if w.strip()]
                print(f"✅ Loaded dictionary: {len(words)} words ({os.path.basename(path)})")
                return words
            except Exception as e:
                print(f"⚠️  Dictionary unreadable ({path}): {e}")
    print(f"ℹ️  No telugu_words.txt found — using the {len(BUILTIN_WORDS)}-word built-in dictionary.")
    return BUILTIN_WORDS


DICT_UNITS = {w: telugu_units(w) for w in dict.fromkeys(load_dictionary())}


# ====================== INFERENCE-ONLY PREPROCESSING ======================
# Mirrors the notebook exactly: load_image_gray -> preprocess_gray (polarity fix, square pad,
# resize, denoise, CLAHE) -> compute_stroke_maps (skeleton, stroke width, edges).
def allowed_image(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_IMAGE_EXTENSIONS


def prepare_character(path):
    """Returns the preprocessed IMG_SIZE x IMG_SIZE uint8 image, or None if unreadable."""
    try:
        raw = load_image_gray(path)
        return None if raw is None else preprocess_gray(raw, IMG_SIZE)
    except Exception:
        return None


# ====================== INFERENCE + STROKE VISUALISATION ======================
@torch.no_grad()
def infer_characters(grays):
    """grays: list of preprocessed uint8 images. Returns (probabilities (N, C), gate (N,) or None, stroke maps)."""
    FUSION_MODEL.eval()
    strokes = [compute_stroke_maps(g) for g in grays]
    x = torch.from_numpy(np.stack(grays)).float().div(255.).unsqueeze(1).to(device)
    gate = None
    if USES_STROKE:
        s = torch.from_numpy(np.stack(strokes)).to(device)
        logits, g = FUSION_MODEL(x, s, return_gate=True)
        gate = g.float().mean(1).cpu().numpy()            # 1 = appearance features, 0 = stroke features
    else:
        logits = FUSION_MODEL(x)
    p = F.softmax(logits.float(), dim=1).cpu().numpy()
    return p, gate, strokes


def _save_gray(path, img):
    cv2.imwrite(path, img)


def stroke_montage(gray, stroke):
    """gray | skeleton | stroke width | edge magnitude, side by side (what the stroke branch sees)."""
    panels = [cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)]
    for ch in stroke:
        panels.append(cv2.applyColorMap((np.clip(ch, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_MAGMA))
    m = np.hstack(panels)
    return cv2.resize(m, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_NEAREST)


def cleanup_old_results(max_age_seconds=3600):
    """Deletes generated visualisation images older than max_age_seconds."""
    now = time.time()
    for fn in os.listdir(RESULT_DIR):
        fp = os.path.join(RESULT_DIR, fn)
        try:
            if os.path.isfile(fp) and now - os.path.getmtime(fp) > max_age_seconds:
                os.remove(fp)
        except OSError:
            pass


def predict_images(image_paths, filenames, mode):
    """Runs the full OCR pipeline (notebook section 13) on one character image (mode='character')
    or on several character images ordered left -> right (mode='word').

    Keys returned (for prediction.html):
        mode, text_raw, text_final, correction {applied, matched, distance, dictionary_size},
        characters [{position, filename, class_key, group, unicode, codepoints, confidence_score (%),
                     status ('detected' / 'uncertain'), top_k [{class_key, unicode, probability %}],
                     appearance_share (%) or None, images {input, stroke}}],
        n_characters, mean_confidence (%), min_confidence (%), status, uncertain_positions,
        (group = Consonant / Vowel / Conjunct (othulu); class_key = '<type>:<folder>', e.g. 'cons:ka')
        model {name, short, uses_stroke, img_size, num_classes}, inference_ms, note, warning
        (images use url_for('static', filename=...))
    """
    t0 = time.perf_counter()
    grays = []
    for path, name in zip(image_paths, filenames):
        g = prepare_character(path)
        if g is None:
            raise ValueError(f"Could not read the image '{name}'.")
        grays.append(g)

    p, gate, strokes = infer_characters(grays)
    base = uuid.uuid4().hex[:12]

    characters = []
    for i, (g, st, probs) in enumerate(zip(grays, strokes, p)):
        agg = AGG @ probs                                   # probability per distinct Telugu character
        order = np.argsort(-agg)
        conf = float(agg[order[0]])
        members = UNI_MEMBERS[UNI_LIST[int(order[0])]]
        pred = members[int(np.argmax(probs[members]))]      # most likely class among those giving this character
        key = CLASS_KEYS[pred]
        files = {'input': f'{base}_{i}_input.png', 'stroke': f'{base}_{i}_stroke.png'}
        _save_gray(os.path.join(RESULT_DIR, files['input']), cv2.resize(g, None, fx=1.5, fy=1.5))
        _save_gray(os.path.join(RESULT_DIR, files['stroke']), stroke_montage(g, st))
        characters.append({
            'position': i + 1,
            'filename': filenames[i],
            'class_key': key,
            'group': class_type_label(key),
            'unicode': IDX2UNI[pred],
            'codepoints': codepoints(IDX2UNI[pred]),
            'confidence_score': round(conf * 100, 2),
            'status': 'detected' if conf >= MIN_CONFIDENCE else 'uncertain',
            'top_k': [{'class_key': CLASS_KEYS[UNI_MEMBERS[UNI_LIST[int(r)]][int(np.argmax(probs[UNI_MEMBERS[UNI_LIST[int(r)]]]))]],
                       'unicode': UNI_LIST[int(r)],
                       'probability': round(float(agg[r]) * 100, 2)}
                      for r in order[:TOP_K]],
            'appearance_share': round(float(gate[i]) * 100, 1) if gate is not None else None,
            'images': {k: f'results/{v}' for k, v in files.items()},
        })

    raw_text = ''.join(c['unicode'] for c in characters)
    correction = {'applied': False, 'matched': False, 'distance': -1, 'dictionary_size': len(DICT_UNITS)}
    final_text = raw_text
    if mode == 'word' and DICT_UNITS:
        fixed, dist = correct_word(raw_text, DICT_UNITS)
        correction.update(matched=dist >= 0, distance=dist, applied=(dist >= 0 and fixed != raw_text))
        if dist >= 0:
            final_text = fixed

    confs = [c['confidence_score'] for c in characters]
    uncertain = [c['position'] for c in characters if c['status'] == 'uncertain']
    note = None
    if uncertain:
        note = ('Low confidence for character(s) ' + ', '.join(map(str, uncertain)) +
                '. Check the alternatives listed for them, or re-upload a clearer image.')
    elif mode == 'word' and not correction['matched']:
        note = 'No dictionary word was close enough, so the raw prediction is shown unchanged.'

    return {
        'mode': mode,
        'text_raw': raw_text,
        'text_final': final_text,
        'correction': correction,
        'characters': characters,
        'n_characters': len(characters),
        'mean_confidence': round(float(np.mean(confs)), 2),
        'min_confidence': round(float(np.min(confs)), 2),
        'status': 'uncertain' if uncertain else 'detected',
        'uncertain_positions': uncertain,
        'model': {'name': MODEL_NAME, 'short': MODEL_SHORT.get(MODEL_NAME, MODEL_NAME),
                  'uses_stroke': USES_STROKE, 'img_size': IMG_SIZE, 'num_classes': len(CLASS_KEYS)},
        'inference_ms': round((time.perf_counter() - t0) * 1000, 1),
        'note': note,
        'warning': 'Handwritten-character recogniser: the output is a machine prediction and may contain errors.',
    }


def warm_up_pipeline():
    """Runs one throwaway prediction through the loaded model at startup.

    The model is already loaded eagerly (at import time), so this isn't about
    *loading* it again. It's about the one-time cost of the *first* forward pass
    (CUDA/cuDNN kernel init, OpenCV / skimage first calls). Left alone, whichever
    user submits the first real image pays for all of that in one request. Running
    it once here, before app.run() starts serving requests, means every real
    request afterwards runs at steady-state speed.
    """
    try:
        rng = np.random.RandomState(0)
        dummy = [preprocess_gray((rng.rand(64, 64) * 255).astype(np.uint8), IMG_SIZE)]
        infer_characters(dummy)
        print("✅ Warm-up prediction complete — model is hot and ready.")
    except Exception as e:
        print(f"⚠️  Warm-up prediction failed (predictions may be slow on first use): {e}")


if ARTIFACTS_READY:
    warm_up_pipeline()


# ====================== DOWNLOADABLE PDF REPORT ======================
# ReportLab's built-in fonts have no Telugu glyphs, and ReportLab does not shape complex scripts.
# So Telugu text is drawn with Pillow (which shapes it when libraqm is installed) and embedded as an
# image. Without a Telugu font / libraqm the report falls back to transliteration + Unicode code points.
TELUGU_FONT_CANDIDATES = [
    os.path.join(BASE_DIR, 'static', 'fonts', 'NotoSansTelugu-Regular.ttf'),
    r'C:\Windows\Fonts\gautami.ttf',
    r'C:\Windows\Fonts\Nirmala.ttc',
    '/usr/share/fonts/truetype/noto/NotoSansTelugu-Regular.ttf',
    '/usr/share/fonts/truetype/lohit-telugu/Lohit-Telugu.ttf',
    '/Library/Fonts/Telugu MN.ttc',
    '/System/Library/Fonts/Supplemental/Telugu MN.ttc',
]


def find_telugu_font():
    paths = TELUGU_FONT_CANDIDATES + sorted(glob.glob(os.path.join(BASE_DIR, 'static', 'fonts', '*.ttf')))
    return next((p for p in paths if os.path.isfile(p)), None)


TELUGU_FONT = find_telugu_font()
try:
    TELUGU_SHAPING_OK = bool(TELUGU_FONT) and pil_features.check('raqm')
except Exception:
    TELUGU_SHAPING_OK = False
if REPORTLAB_OK and not TELUGU_SHAPING_OK:
    print("ℹ️  PDF report: Telugu text will be listed as transliteration + Unicode code points "
          "(needs a Telugu .ttf in static/fonts/ and Pillow with libraqm to draw real Telugu).")


def telugu_png(text, size=44, pad=6):
    """Renders Telugu text to a PNG (BytesIO) and returns (buffer, width_px, height_px), or None."""
    if not (TELUGU_SHAPING_OK and text):
        return None
    try:
        font = ImageFont.truetype(TELUGU_FONT, size, layout_engine=ImageFont.Layout.RAQM)
        probe = ImageDraw.Draw(Image.new('RGB', (1, 1)))
        l, t, r, b = probe.textbbox((0, 0), text, font=font)
        img = Image.new('RGB', (r - l + 2 * pad, b - t + 2 * pad), 'white')
        ImageDraw.Draw(img).text((pad - l, pad - t), text, font=font, fill='black')
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        buf.seek(0)
        return buf, img.width, img.height
    except Exception:
        return None


LEVEL_HEX = {'detected': '#16a34a', 'uncertain': '#d97706'}


def _esc(value):
    return xml_escape(str(value))


def cleanup_old_files(directory, max_age_seconds):
    now = time.time()
    for fn in os.listdir(directory):
        fp = os.path.join(directory, fn)
        try:
            if os.path.isfile(fp) and now - os.path.getmtime(fp) > max_age_seconds:
                os.remove(fp)
        except OSError:
            pass


def build_report_pdf(path, user, result):
    """Writes a PDF report for one recognition (character or word)."""
    base = getSampleStyleSheet()
    H1 = ParagraphStyle('H1', parent=base['Title'], fontSize=20, leading=24, alignment=0, spaceAfter=2)
    H2 = ParagraphStyle('H2', parent=base['Heading2'], fontSize=13, leading=16, spaceBefore=14, spaceAfter=6,
                        textColor=colors.HexColor('#0f3a5f'))
    BODY = ParagraphStyle('B', parent=base['BodyText'], fontSize=9.5, leading=13)
    SMALL = ParagraphStyle('S', parent=BODY, fontSize=8, leading=10.5, textColor=colors.HexColor('#555555'))
    CELL = ParagraphStyle('C', parent=BODY, fontSize=9, leading=11.5)
    HEAD = ParagraphStyle('Hd', parent=CELL, fontName='Helvetica-Bold', textColor=colors.white)

    is_word = result['mode'] == 'word'
    status_color = colors.HexColor(LEVEL_HEX.get(result['status'], '#0f3a5f'))
    W = A4[0] - 36 * mm

    def grid(rows, widths, header=True):
        data = [[Paragraph(_esc(c), HEAD if (header and i == 0) else CELL) for c in r] for i, r in enumerate(rows)]
        t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
        style = [('GRID', (0, 0), (-1, -1), 0.4, colors.HexColor('#cbd5e1')),
                 ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                 ('TOPPADDING', (0, 0), (-1, -1), 4), ('BOTTOMPADDING', (0, 0), (-1, -1), 4)]
        if header:
            style.append(('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0f3a5f')))
        else:
            style.append(('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#eef2f7')))
        t.setStyle(TableStyle(style))
        return t

    def telugu_flowable(text, size=44, max_h=16 * mm):
        r = telugu_png(text, size=size)
        if not r:
            return Paragraph(_esc(codepoints(text)), CELL)
        buf, w, h = r
        scale = min(max_h / h, 1.0 * (W * 0.9) / w)
        return RLImage(buf, width=w * scale, height=h * scale)

    story = []

    # ----- title block
    story.append(Paragraph('Telugu Handwritten OCR – Recognition Report', H1))
    story.append(Paragraph(
        f'Generated {datetime.now():%d %b %Y, %H:%M} &nbsp;|&nbsp; User: {_esc(user)} &nbsp;|&nbsp; '
        f"Model: {_esc(result['model']['short'])} &nbsp;|&nbsp; "
        f"Input: {'word (%d characters)' % result['n_characters'] if is_word else 'single character'}", SMALL))
    story.append(Spacer(1, 8))

    # ----- headline
    left = [Paragraph('<font size="9" color="#555555">RECOGNISED TEXT</font>', BODY),
            telugu_flowable(result['text_final'])]
    head = Table([[left, Paragraph(
        f'<font size="9" color="#555555">STATUS</font><br/>'
        f'<font size="14" color="{LEVEL_HEX.get(result["status"], "#0f3a5f")}">'
        f'<b>{_esc(result["status"].capitalize())}</b></font><br/>'
        f'<font size="9">mean confidence {result["mean_confidence"]:.1f}%</font>', BODY)]],
        colWidths=[W * 0.65, W * 0.35])
    head.setStyle(TableStyle([('BOX', (0, 0), (-1, -1), 1.2, status_color),
                              ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f8fafc')),
                              ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                              ('LEFTPADDING', (0, 0), (-1, -1), 10), ('TOPPADDING', (0, 0), (-1, -1), 8),
                              ('BOTTOMPADDING', (0, 0), (-1, -1), 8)]))
    story.append(head)
    story.append(Spacer(1, 8))

    rows = [['Item', 'Value'],
            ['Final text (Unicode)', codepoints(result['text_final'])],
            ['Raw model prediction (Unicode)', codepoints(result['text_raw'])],
            ['Characters recognised', str(result['n_characters'])],
            ['Mean / minimum confidence', f"{result['mean_confidence']:.1f}% / {result['min_confidence']:.1f}%"],
            ['Inference time', f"{result['inference_ms']} ms"]]
    if is_word:
        c = result['correction']
        rows.append(['Dictionary correction',
                     ('applied, edit distance %d' % c['distance']) if c['applied'] else
                     ('exact dictionary match' if c['matched'] else 'no close dictionary word – raw text kept')])
    story.append(grid(rows, [W * 0.4, W * 0.6]))
    if result.get('note'):
        story.append(Spacer(1, 6))
        story.append(Paragraph(f'<b>Note:</b> {_esc(result["note"])}', BODY))

    # ----- character images
    story.append(Paragraph('Character images (preprocessed: denoised + CLAHE)', H2))
    cells, per_row = [], 8
    for c in result['characters']:
        p = os.path.join(RESULT_DIR, os.path.basename(c['images']['input']))
        if os.path.isfile(p):
            cells.append([RLImage(p, width=18 * mm, height=18 * mm), Paragraph(f"#{c['position']}", SMALL)])
    for i in range(0, len(cells), per_row):
        chunk = cells[i:i + per_row]
        t = Table([[x[0] for x in chunk], [x[1] for x in chunk]], colWidths=[W / per_row] * len(chunk))
        t.setStyle(TableStyle([('ALIGN', (0, 0), (-1, -1), 'CENTER')]))
        story.append(t)
        story.append(Spacer(1, 4))

    # ----- per-character predictions
    story.append(Paragraph('Per-character predictions', H2))
    rows = [['#', 'Class (folder)', 'Group', 'Unicode', 'Confidence', 'Alternatives']]
    for c in result['characters']:
        alts = '; '.join(f"{split_key(a['class_key'])[1]} {a['probability']:.1f}%" for a in c['top_k'][1:])
        rows.append([c['position'], c['class_key'], c['group'], c['codepoints'],
                     f"{c['confidence_score']:.1f}%" + (' (low)' if c['status'] == 'uncertain' else ''), alts])
    story.append(grid(rows, [W * 0.06, W * 0.22, W * 0.14, W * 0.17, W * 0.15, W * 0.26]))
    story.append(Spacer(1, 4))
    if result['characters'][0]['appearance_share'] is not None:
        story.append(Paragraph(
            'The stroke-gated fusion model weighs appearance features against stroke features (skeleton, '
            'stroke width, edges) per character: ' + ', '.join(
                f"#{c['position']} {c['appearance_share']:.0f}% appearance" for c in result['characters'][:12]) +
            ('…' if result['n_characters'] > 12 else '') + '.', SMALL))

    story.append(Spacer(1, 12))
    story.append(Paragraph(_esc(result['warning']) + ' Verify important text manually.', SMALL))

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont('Helvetica', 8)
        canvas.setFillColor(colors.HexColor('#777777'))
        canvas.drawString(18 * mm, 10 * mm, 'Telugu Handwritten OCR · SGF-MobileNetV3')
        canvas.drawRightString(A4[0] - 18 * mm, 10 * mm, f'Page {doc.page}')
        canvas.restoreState()

    SimpleDocTemplate(path, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=16 * mm,
                      bottomMargin=18 * mm, title='Telugu Handwritten OCR Report',
                      author='Telugu OCR').build(story, onFirstPage=footer, onLaterPages=footer)


def make_report(**ctx):
    """Builds the PDF and returns its id, or None if reports are unavailable / generation fails."""
    if not REPORTLAB_OK:
        return None
    try:
        cleanup_old_files(REPORT_OUT_DIR, 24 * 3600)
        report_id = uuid.uuid4().hex
        build_report_pdf(os.path.join(REPORT_OUT_DIR, report_id + '.pdf'), **ctx)
        return report_id
    except Exception as e:
        print(f'⚠️ Report generation failed: {e}')
        return None


@app.template_filter('nice')
def nice_filter(value):
    """Rounds numeric strings from CSV reports to 4 decimals for display."""
    try:
        f = float(value)
        return f'{f:.4f}'.rstrip('0').rstrip('.') if '.' in str(value) else str(value)
    except (ValueError, TypeError):
        return value


# ====================== ROUTES ======================
@app.route('/register', methods=['GET', 'POST'])
def register():
    message = None
    if request.method == 'POST':
        email      = request.form.get('email')
        password   = request.form.get('password')
        c_password = request.form.get('c_password')
        if not email or not password:
            message = "All fields are required."
        elif password != c_password:
            message = "Passwords do not match."
        elif fetch_all("SELECT email FROM users WHERE email=%s", (email,)):
            message = "Email already registered."
        else:
            execute_query("INSERT INTO users (email, password) VALUES (%s, %s)", (email, password))
            return redirect(url_for('login'))
    return render_template("register.html", message=message)


@app.route('/login', methods=['GET', 'POST'])
def login():
    message = None
    if request.method == 'POST':
        email    = request.form.get('email')
        password = request.form.get('password')
        if fetch_all("SELECT * FROM users WHERE email=%s AND password=%s", (email, password)):
            session['user'] = email
            return redirect(url_for('home'))
        message = "Invalid credentials."
    return render_template("login.html", message=message)


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))


@app.route('/')
def index():
    return render_template("index.html")


@app.route('/home')
def home():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template("home.html", user=session['user'])


@app.route('/model_performance')
def model_performance():
    return render_template(
        "model_performance.html",
        model_name=MODEL_NAME,
        model_short=MODEL_SHORT.get(MODEL_NAME, MODEL_NAME),
        uses_stroke=USES_STROKE,
        img_size=IMG_SIZE,
        num_classes=len(CLASS_KEYS) or NUM_CLASSES_SNAPSHOT,
        groups=GROUPS,
        group_descriptions=GROUP_DESCRIPTIONS,
        min_confidence=MIN_CONFIDENCE,
        artifacts_ready=ARTIFACTS_READY,
        dictionary_size=len(DICT_UNITS),
        test_metrics=TEST_SET_METRICS,
        test_split=TEST_SPLIT,
        model_comparison=MODEL_COMPARISON,
        group_accuracy=GROUP_ACCURACY,
        classification_report=TEST_SET_CLASSIFICATION_REPORT,
        report_averages=TEST_SET_REPORT_AVERAGES,
        weakest_classes=WEAKEST_CLASSES,
        latency=LATENCY,
        upload_sim=UPLOAD_SIM,
        dataset_info=DATASET_INFO,
    )


def _natural_key(name):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', name)]


def _detect_prediction_mode(files):
    """One image -> single character; several images -> a word (ordered left -> right)."""
    provided = request.form.get('prediction_mode', 'auto').strip().lower()
    if provided not in ('auto', 'character', 'word'):
        raise ValueError('Invalid prediction mode.')
    if not files:
        raise ValueError('Please upload a Telugu character image (.jpg, .jpeg, .png, .bmp, .webp or .tif).')
    if provided == 'auto':
        provided = 'character' if len(files) == 1 else 'word'
    if provided == 'character' and len(files) != 1:
        raise ValueError('Character mode expects exactly one image. Use word mode for several images.')
    if len(files) > MAX_CHARACTERS:
        raise ValueError(f'At most {MAX_CHARACTERS} character images can be recognised at once.')
    return provided


@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    if 'user' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        saved_paths = []
        try:
            files = [f for f in request.files.getlist('image_files') if f and f.filename]
            if not files:                                       # single-input form (field name image_file)
                single = request.files.get('image_file')
                files = [single] if single is not None and single.filename else []
            mode = _detect_prediction_mode(files)

            if not ARTIFACTS_READY:
                raise RuntimeError(
                    'No Telugu OCR model is loaded. Place best_model.pth and best_model_meta.json '
                    '(or best_model_checkpoint.pt) in saved_models/ and check the startup logs.'
                )

            for f in files:
                if not allowed_image(f.filename):
                    raise ValueError('Only .jpg, .jpeg, .png, .bmp, .webp and .tif images are supported.')
            # word mode: browsers send files in selection order; optionally sort by file name (natural order)
            if mode == 'word' and request.form.get('sort_by_name'):
                files.sort(key=lambda f: _natural_key(f.filename))

            names = []
            for f in files:
                filename = secure_filename(f.filename)
                if not filename:
                    raise ValueError('Please upload valid image filenames.')
                path = os.path.join(UPLOAD_DIR, f'{uuid.uuid4().hex[:8]}_{filename}')
                f.save(path)
                saved_paths.append(path)
                names.append(filename)

            cleanup_old_results()
            latest = predict_images(saved_paths, names, mode)
            report_id = make_report(user=session['user'], result=latest)
            return render_template('prediction.html', latest=latest, prediction_mode=mode,
                                   report_id=report_id, group_descriptions=GROUP_DESCRIPTIONS)
        except Exception as e:
            return render_template('prediction.html', error=str(e), latest=None,
                                   prediction_mode=request.form.get('prediction_mode', 'auto'))
        finally:
            for p in saved_paths:
                if os.path.exists(p):
                    os.remove(p)

    return render_template('prediction.html', latest=None, prediction_mode='auto')


@app.route('/download_report/<report_id>')
def download_report(report_id):
    """Sends the PDF report generated for a prediction."""
    if 'user' not in session:
        return redirect(url_for('login'))
    if not re.fullmatch(r'[0-9a-f]{32}', report_id):
        abort(404)
    path = os.path.join(REPORT_OUT_DIR, report_id + '.pdf')
    if not os.path.isfile(path):
        abort(404)
    return send_file(path, mimetype='application/pdf', as_attachment=True,
                     download_name=f'TeluguOCR_Report_{datetime.now():%Y%m%d_%H%M%S}.pdf')


@app.route('/about')
def about():
    return render_template("about.html")


# ====================== RUN ======================
if __name__ == "__main__":
    app.run(debug=True)
