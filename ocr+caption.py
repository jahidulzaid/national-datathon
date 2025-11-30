# ============================================================
# Strong OCR (Bangla+English) + Image Captioning Pipeline
# Works on ~15GB VRAM (BLIP large or base)
# ============================================================

from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image, ImageOps
import torch
from tqdm import tqdm
import easyocr
from transformers import pipeline
import zipfile
import os
import cv2

torch.set_grad_enabled(False)

# =======================
# 1. Unzip dataset
# =======================
# zip_path = "upload.zip"          # adjust if needed
# extract_to = "/content/"         # root folder

# with zipfile.ZipFile(zip_path, 'r') as zip_ref:
#     zip_ref.extractall(extract_to)

# print("Done extracting!")

# =======================
# 2. Find BASE_DIR (Train/Train.csv, Test/Test.csv)
# =======================
default_base = Path('/content/upload')
candidates = [default_base] + [p for p in default_base.iterdir() if p.is_dir()]
BASE_DIR = None
for candidate in candidates:
    train_csv = candidate / 'Train' / 'Train.csv'
    test_csv = candidate / 'Test' / 'Test.csv'
    if train_csv.exists() and test_csv.exists():
        BASE_DIR = candidate
        break

if BASE_DIR is None:
    raise FileNotFoundError(
        'Could not find a dataset folder containing Train/Train.csv and Test/Test.csv under /content/upload. '
        'Please set BASE_DIR manually.'
    )

TRAIN_CSV = BASE_DIR / 'Train' / 'Train.csv'
TEST_CSV = BASE_DIR / 'Test' / 'Test.csv'
TRAIN_IMG_DIR = BASE_DIR / 'Train' / 'Image'
TEST_IMG_DIR = BASE_DIR / 'Test' / 'Image'

print("BASE_DIR:", BASE_DIR)

# Output files
OUT_TRAIN = Path('ocr_caption_train.csv')
OUT_TEST = Path('ocr_caption_test.csv')

# =======================
# 3. Initialize models
# =======================

print("Initializing EasyOCR (Bangla + English)...")
reader = easyocr.Reader(
    ['bn', 'en'],
    gpu=torch.cuda.is_available()
)

print("Initializing caption model (BLIP)...")
device = 0 if torch.cuda.is_available() else -1

# For 15GB GPU you can try the LARGE model first:
CAPTION_MODEL = "Salesforce/blip-image-captioning-large"
# If you get CUDA OOM, switch to:
# CAPTION_MODEL = "Salesforce/blip-image-captioning-base"

captioner = pipeline(
    "image-to-text",
    model=CAPTION_MODEL,
    device=device,
    max_new_tokens=32
)

print('Using device for captioner:', 'cuda' if torch.cuda.is_available() else 'cpu')
print('Caption model:', CAPTION_MODEL)

# =======================
# 4. Image preprocessing utilities
# =======================

def pil_to_cv2(img: Image.Image):
    """Convert PIL Image (RGB) to OpenCV BGR numpy array."""
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def cv2_to_pil(arr: np.ndarray):
    """Convert OpenCV BGR numpy array to PIL Image (RGB)."""
    return Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))

def upscale_image(img: Image.Image, min_side: int = 1024) -> Image.Image:
    """
    Upscale image so that the longer side is at least min_side.
    Keeps aspect ratio.
    """
    w, h = img.size
    max_side = max(w, h)
    if max_side >= min_side:
        return img

    scale = min_side / max_side
    new_w, new_h = int(w * scale), int(h * scale)
    return img.resize((new_w, new_h), Image.BICUBIC)

def make_variants_for_ocr(img: Image.Image):
    """
    Create a few variants of the input image for OCR:
      - upscaled color
      - upscaled grayscale
      - upscaled high-contrast (thresholded)
    """
    img = upscale_image(img)

    # Color variant (as-is)
    color = img

    # Grayscale
    gray = ImageOps.grayscale(img)

    # Thresholded (use OpenCV for adaptive threshold)
    cv_img = pil_to_cv2(img)
    gray_cv = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

    # Adaptive threshold can help for white meme text etc.
    thr = cv2.adaptiveThreshold(
        gray_cv,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        10
    )
    thresh_pil = cv2_to_pil(cv2.cvtColor(thr, cv2.COLOR_GRAY2BGR))

    return [color, gray, thresh_pil]

# =======================
# 5. OCR and caption functions
# =======================

def extract_ocr_text(img_path: Path) -> str:
    """
    Robust OCR over multiple preprocessed variants.
    Uses EasyOCR with Bangla + English.
    """
    try:
        img = Image.open(img_path).convert("RGB")
    except Exception as exc:
        print(f"[WARN] Failed to open image {img_path}: {exc}")
        return ""

    variants = make_variants_for_ocr(img)

    all_texts = []

    for idx, v in enumerate(variants):
        try:
            arr = np.array(v)
            # detail=0 -> just text; paragraph=False -> keep all detections
            results = reader.readtext(
                arr,
                detail=0,
                paragraph=False
            )
            # Clean results
            texts = [
                t.strip()
                for t in results
                if isinstance(t, str) and t.strip()
            ]
            all_texts.extend(texts)
        except Exception as exc:
            print(f"[WARN] OCR variant {idx} failed for {img_path.name}: {exc}")

    # De-duplicate while preserving order
    seen = set()
    merged = []
    for t in all_texts:
        norm = t.lower()
        if norm not in seen:
            seen.add(norm)
            merged.append(t)

    return " ".join(merged).strip()

def generate_caption(img_path: Path) -> str:
    """
    Generate a short caption describing the image using BLIP.
    """
    try:
        img = Image.open(img_path).convert('RGB')
        outputs = captioner(img)
        # BLIP pipeline returns list of dicts with `"generated_text"`
        caption = outputs[0].get("generated_text", "").strip()
    except Exception as exc:
        caption = ""
        print(f"[WARN] Caption failed for {img_path.name}: {exc}")
    return caption

# =======================
# 6. Dataset loop
# =======================

def process_split(csv_path: Path, img_dir: Path, out_path: Path):
    df = pd.read_csv(csv_path)
    records = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        name = row['Image_name']
        img_path = img_dir / name

        if not img_path.exists():
            print(f"[WARN] Image not found: {img_path}")
            ocr_text = ""
            caption = ""
        else:
            ocr_text = extract_ocr_text(img_path)
            caption = generate_caption(img_path)

        records.append({
            'Image_name': name,
            'ocr_text': ocr_text,
            'caption': caption
        })

    out_df = pd.DataFrame(records)
    out_df.to_csv(out_path, index=False)
    print(f'Wrote {len(out_df)} rows to {out_path}')
    return out_df

# =======================
# 7. Run extraction
# =======================

print("Processing TRAIN split...")
train_meta = process_split(TRAIN_CSV, TRAIN_IMG_DIR, OUT_TRAIN)

print("Processing TEST split...")
test_meta = process_split(TEST_CSV, TEST_IMG_DIR, OUT_TEST)

print("Done.")
print(train_meta.head())
print(test_meta.head())
