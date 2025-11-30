from pathlib import Path
import json
import pandas as pd
from PIL import Image
import numpy as np
import torch
from tqdm import tqdm
import easyocr
from transformers import pipeline
import zipfile
import os

torch.set_grad_enabled(False)

# =======================
# 1. Unzip
# =======================
zip_path = "upload.zip"          # adjust if needed
extract_to = "/content/"         # root folder

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_to)

print("Done extracting!")

# =======================
# 2. Find BASE_DIR
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

# Output files
OUT_TRAIN = Path('ocr_caption_train.csv')
OUT_TEST = Path('ocr_caption_test.csv')

# =======================
# 3. Models: EasyOCR + BLIP
# =======================
print("Initializing EasyOCR...")
reader = easyocr.Reader(['bn', 'en'], gpu=torch.cuda.is_available())

print("Initializing caption model (BLIP)...")
device = 0 if torch.cuda.is_available() else -1
captioner = pipeline(
    "image-to-text",
    model="Salesforce/blip-image-captioning-base",
    device=device
)
print('Using device:', 'cuda' if torch.cuda.is_available() else 'cpu')

# =======================
# 4. OCR + caption utils
# =======================
def extract_ocr_text(img_path: Path) -> str:
    """
    Run EasyOCR and return concatenated text lines for Bangla + English.
    Includes simple preprocessing + upscaling for small images.
    """
    try:
        img = Image.open(img_path).convert("RGB")
        
        # Upscale small images
        w, h = img.size
        target_min_side = 1024
        if max(w, h) < target_min_side:
            scale = target_min_side / max(w, h)
            new_w, new_h = int(w * scale), int(h * scale)
            img = img.resize((new_w, new_h), Image.BICUBIC)

        img_np = np.array(img)
        results = reader.readtext(img_np, detail=0, paragraph=False)

        texts = [r.strip() for r in results if isinstance(r, str) and r.strip()]
        text = " ".join(texts).strip()
    except Exception as exc:
        text = ""
        print(f"[WARN] OCR failed for {img_path.name}: {exc}")
    return text


def generate_caption(img_path: Path) -> str:
    """Generate a short caption describing the image."""
    try:
        outputs = captioner(Image.open(img_path).convert('RGB'))
        caption = outputs[0].get("generated_text", "").strip()
    except Exception as exc:
        caption = ""
        print(f"[WARN] Caption failed for {img_path.name}: {exc}")
    return caption


def process_split(csv_path: Path, img_dir: Path, out_path: Path):
    df = pd.read_csv(csv_path)
    records = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        name = row['Image_name']
        img_path = img_dir / name

        # Skip if the image file is missing
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
# 5. Run extraction
# =======================
train_meta = process_split(TRAIN_CSV, TRAIN_IMG_DIR, OUT_TRAIN)
test_meta = process_split(TEST_CSV, TEST_IMG_DIR, OUT_TEST)

display(train_meta.head())
display(test_meta.head())
