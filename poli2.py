
 


print("\n⚠️  IMPORTANT: After this cell finishes, RESTART THE KERNEL!")
print("   Kaggle: Runtime → Restart Session")
print("   Then re-run ALL cells from the beginning.\n")

try:
    import bitsandbytes
    print("✓ bitsandbytes installed")
except ImportError:
    print("Installing bitsandbytes...")
    # pip install -q bitsandbytes

import pandas as pd
import numpy as np
import easyocr
from tqdm import tqdm
import torch
from torch.utils.data import Dataset

# Import transformers AFTER installation
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    pipeline
)
from IPython.display import display
import warnings
warnings.filterwarnings("ignore")

print(f"\n✓ Transformers: {__import__('transformers').__version__}")
print(f"✓ PyTorch: {torch.__version__}")
print(f"✓ Setup complete - NOW RESTART KERNEL!")

# from google.colab import drive
# drive.mount('/content/drive')

import os
import pandas as pd
from tqdm import tqdm
from IPython.display import display

# --------------------- PATHS ---------------------
TRAIN_CSV = 'Train/Train.csv'
TEST_CSV  = 'Test/Test.csv'
TRAIN_DIR = 'Train/Image'
TEST_DIR  = 'Test/Image'

# Cache files for OCR results
TRAIN_OCR_CACHE = 'train_ocr_cache.csv'
TEST_OCR_CACHE = 'test_ocr_cache.csv'

train_df = pd.read_csv(TRAIN_CSV)
test_df  = pd.read_csv(TEST_CSV)

# ========================================================
# STEP 1: OCR Extraction (with caching)
# ========================================================

# Check if cached OCR results exist
if os.path.exists(TRAIN_OCR_CACHE) and os.path.exists(TEST_OCR_CACHE):
    print("✓ Found cached OCR results! Loading from disk...")
    train_df = pd.read_csv(TRAIN_OCR_CACHE)
    test_df = pd.read_csv(TEST_OCR_CACHE)
    print(f"  Loaded {len(train_df)} train and {len(test_df)} test samples from cache")
else:
    print("No cache found. Running OCR (this will take time)...")
    print("Initializing EasyOCR...")
    import easyocr
    reader = easyocr.Reader(['bn', 'en'], gpu=True, verbose=False)

    def extract_text(img_path):
        try:
            result = reader.readtext(img_path, detail=0, paragraph=True)
            return " ".join(result).strip() if result else "no text"
        except:
            return "no text"

    # OCR on Train
    print("\nRunning OCR on TRAIN images...")
    train_texts = []
    for img_name in tqdm(train_df['Image_name'], desc="OCR Train"):
        img_path = os.path.join(TRAIN_DIR, img_name)
        train_texts.append(extract_text(img_path))

    train_df['ocr_text'] = train_texts

    # OCR on Test
    print("\nRunning OCR on TEST images...")
    test_texts = []
    for img_name in tqdm(test_df['Image_name'], desc="OCR Test"):
        img_path = os.path.join(TEST_DIR, img_name)
        test_texts.append(extract_text(img_path))

    test_df['ocr_text'] = test_texts

    # Save to cache
    print("\n💾 Saving OCR results to cache for future runs...")
    train_df.to_csv(TRAIN_OCR_CACHE, index=False)
    test_df.to_csv(TEST_OCR_CACHE, index=False)
    print("✓ Cache saved!")

# --- SHOW 30 ROWS ---
print("\n" + "="*50)
print("Raw OCR Output - First 30 Rows")
print("="*50)
display(train_df[['Image_name', 'Label', 'ocr_text']].head(30))

# ========================================================
# STEP 2: IMPROVED TEXT CLEANING (without Gemma)
# ========================================================
print("\n" + "="*50)
print("STARTING TEXT CLEANING")
print("="*50)

import re

def clean_text_improved(text):
    """Enhanced text cleaning for Bengali/English OCR output"""
    t = str(text or '').strip()

    # Skip very short or empty text
    if t.lower() == 'no text' or len(t) < 3:
        return t

    # Fix common OCR errors
    t = re.sub(r'\s+', ' ', t)  # Multiple spaces -> single space
    t = re.sub(r'(\w)\.(\w)', r'\1. \2', t)  # Add space after period
    t = re.sub(r'([,;:!?])(\w)', r'\1 \2', t)  # Add space after punctuation
    t = re.sub(r'(\w)([,;:!?])', r'\1\2 ', t)  # Add space after word+punctuation

    # Remove multiple punctuation
    t = re.sub(r'[.]{2,}', '.', t)
    t = re.sub(r'[!]{2,}', '!', t)
    t = re.sub(r'[?]{2,}', '?', t)

    # Fix common Bengali OCR artifacts
    t = re.sub(r'।+', '।', t)  # Bengali danda (sentence end)
    t = re.sub(r'\s+।', '।', t)  # Remove space before danda

    # Remove excessive whitespace and special characters
    t = re.sub(r'[\n\r\t]+', ' ', t)
    t = re.sub(r'[|]+', '', t)  # Remove pipe characters
    t = re.sub(r'[-]{3,}', '', t)  # Remove long dashes

    # Final cleanup
    t = t.strip()
    t = re.sub(r'\s+', ' ', t)

    return t if len(t) > 0 else "no text"

# Apply improved cleaning
print("\nCleaning TRAIN text...")
tqdm.pandas(desc="Train")
train_df['final_text'] = train_df['ocr_text'].progress_apply(clean_text_improved)

print("Cleaning TEST text...")
tqdm.pandas(desc="Test")
test_df['final_text'] = test_df['ocr_text'].progress_apply(clean_text_improved)

# Save cleaned data
train_df.to_csv('train_cleaned.csv', index=False)
test_df.to_csv('test_cleaned.csv', index=False)

print("✓ Text cleaning complete (rule-based approach)")

# Show results
print("\n" + "="*50)
print("AFTER CLEANING - First 30 Rows")
print("="*50)
display(train_df[['Image_name', 'ocr_text', 'final_text']].head(30))

# ========================================================
# STEP 3: TRAIN BANGLABERT
# ========================================================

# Load Data
train_texts = train_df['final_text'].fillna("no text").tolist()
train_labels = train_df['Label'].map({'NonPolitical': 0, 'Political': 1}).tolist()
test_texts = test_df['final_text'].fillna("no text").tolist()

MODEL_NAME = "csebuetnlp/banglabert"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

class MemeDataset(Dataset):
    def __init__(self, texts, labels=None):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = tokenizer(
            str(self.texts[idx]),
            truncation=True,
            padding='max_length',
            max_length=256,
            return_tensors='pt'
        )
        item = {k: v.squeeze(0) for k, v in encoding.items()}
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

train_dataset = MemeDataset(train_texts, train_labels)
test_dataset  = MemeDataset(test_texts)

training_args = TrainingArguments(
    output_dir='./banglabert_cleaned',
    num_train_epochs=5,
    per_device_train_batch_size=16,
    learning_rate=2e-5,
    weight_decay=0.01,
    fp16=True,
    report_to=[],
    seed=42
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

print("\nStarting BanglaBERT Training...")
trainer.train()

# Predict & Submit
predictions = trainer.predict(test_dataset)
pred_ids = np.argmax(predictions.predictions, axis=1)
final_labels = ['Political' if x == 1 else 'NonPolitical' for x in pred_ids]

submission = pd.DataFrame({
    'Image_name': test_df['Image_name'],
    'Label': final_labels
})

submission.to_csv('submission_text_only.csv', index=False)
print("Done! Text-only submission saved.")

from IPython.display import FileLink

# Display the value counts for quick check
print("\n--- Submission Label Counts ---")
print(submission['Label'].value_counts())

# Display the first few rows of the final submission file
print("\n--- Submission Preview ---")
display(submission.head())


FileLink('submission_gemma_optimized.csv')

# ==============================================
# IMPROVEMENT: PaddleOCR + OCR ensembling
# ==============================================
print("\n=== Improving OCR with PaddleOCR (PP-OCRv3) ===")

import os
import pandas as pd
from tqdm import tqdm
import torch

# Try to import PaddleOCR; if missing, attempt lightweight installation
paddle_available = False
try:
    from paddleocr import PaddleOCR
    paddle_available = True
    print("✓ PaddleOCR already installed")
except Exception as e:
    print(f"PaddleOCR not available: {e}")
    print("Attempting installation (using standard PyPI packages)...")
    try:
        import sys
        # Install Paddle runtime from standard PyPI (no CUDA version pinning; let pip resolve)
        if torch.cuda.is_available():
            print("Installing paddlepaddle-gpu (latest stable)...")
            get_ipython().system('{sys.executable} -m pip install -q paddlepaddle-gpu')
        else:
            print("Installing paddlepaddle (CPU, latest stable)...")
            get_ipython().system('{sys.executable} -m pip install -q paddlepaddle')
        # Install PaddleOCR
        get_ipython().system('{sys.executable} -m pip install -q paddleocr')
        from paddleocr import PaddleOCR
        paddle_available = True
        print("✓ PaddleOCR installed successfully")
    except Exception as ie:
        print(f"✗ PaddleOCR installation failed: {ie}")
        print("→ Continuing with EasyOCR only (no accuracy loss expected).")
        paddle_available = False

def _build_paddle_reader():
    if not paddle_available:
        return None
    try:
        return PaddleOCR(use_angle_cls=True, lang='bn', use_gpu=torch.cuda.is_available())
    except Exception:
        try:
            # Fallback to English-only if Bengali model not bundled
            return PaddleOCR(use_angle_cls=True, lang='en', use_gpu=torch.cuda.is_available())
        except Exception:
            return None

def _paddle_text(reader, img_path):
    try:
        res = reader.ocr(img_path, cls=True)
        texts = []
        for page in res:
            for line in page:
                texts.append(line[1][0])
        return " ".join(texts).strip() if texts else "no text"
    except Exception:
        return "no text"

# Ensure EasyOCR text columns exist (from earlier cache step); otherwise default to empty
if 'ocr_text' not in train_df.columns:
    train_df['ocr_text'] = ''
if 'ocr_text' not in test_df.columns:
    test_df['ocr_text'] = ''

if paddle_available:
    print("Running OCR ensemble: EasyOCR + PaddleOCR")
    reader = _build_paddle_reader()
    if reader is None:
        print("Paddle reader unavailable; using EasyOCR only.")
        train_df['ocr_text_ens'] = train_df['ocr_text']
        test_df['ocr_text_ens'] = test_df['ocr_text']
    else:
        print("PaddleOCR on TRAIN (this may take time)...")
        train_paddle_texts = []
        for img_name in tqdm(train_df['Image_name'], desc="PaddleOCR Train"):
            img_path = os.path.join(TRAIN_DIR, img_name)
            train_paddle_texts.append(_paddle_text(reader, img_path))

        print("PaddleOCR on TEST (this may take time)...")
        test_paddle_texts = []
        for img_name in tqdm(test_df['Image_name'], desc="PaddleOCR Test"):
            img_path = os.path.join(TEST_DIR, img_name)
            test_paddle_texts.append(_paddle_text(reader, img_path))

        train_df['ocr_text_paddle'] = train_paddle_texts
        test_df['ocr_text_paddle'] = test_paddle_texts

        # Ensemble: prefer longer non-'no text' string
        def ensemble_ocr(a: str, b: str) -> str:
            a = str(a or '').strip()
            b = str(b or '').strip()
            if a.lower() == 'no text' and b.lower() != 'no text':
                return b
            if b.lower() == 'no text' and a.lower() != 'no text':
                return a
            return a if len(a) >= len(b) else b

        train_df['ocr_text_ens'] = [ensemble_ocr(a, b) for a, b in zip(train_df['ocr_text'], train_df.get('ocr_text_paddle', ''))]
        test_df['ocr_text_ens'] = [ensemble_ocr(a, b) for a, b in zip(test_df['ocr_text'], test_df.get('ocr_text_paddle', ''))]
else:
    print("PaddleOCR not available; proceeding with EasyOCR output only.")
    train_df['ocr_text_ens'] = train_df['ocr_text']
    test_df['ocr_text_ens'] = test_df['ocr_text']

print("Using ensembled OCR column: ocr_text_ens (falls back to EasyOCR if PaddleOCR missing)")

# ==============================================
# IMPROVEMENT: Clean text using ensembled OCR (Bengali + English)
# ==============================================
print("\n=== Cleaning text with ensembled OCR (Bengali + English support) ===")

from tqdm import tqdm
import re

def clean_text_bilingual(text):
    """Enhanced cleaning for Bengali and English mixed text"""
    t = str(text or '').strip()

    # Skip very short or empty text
    if t.lower() == 'no text' or len(t) < 3:
        return t

    # Fix common OCR errors (works for both Bengali and English)
    t = re.sub(r'\s+', ' ', t)  # Multiple spaces -> single space
    t = re.sub(r'(\w)\.(\w)', r'\1. \2', t)  # Add space after period
    t = re.sub(r'([,;:!?])(\w)', r'\1 \2', t)  # Add space after punctuation
    t = re.sub(r'(\w)([,;:!?])', r'\1\2 ', t)  # Add space after word+punctuation

    # Remove multiple punctuation
    t = re.sub(r'[.]{2,}', '.', t)
    t = re.sub(r'[!]{2,}', '!', t)
    t = re.sub(r'[?]{2,}', '?', t)

    # Bengali-specific cleaning
    # Bengali danda (।) - sentence ender in Bengali
    t = re.sub(r'।+', '।', t)  # Multiple danda -> single
    t = re.sub(r'\s+।', '।', t)  # Remove space before danda
    t = re.sub(r'।(?=[^\s])', '। ', t)  # Add space after danda if missing

    # Bengali double danda (॥) - paragraph ender
    t = re.sub(r'॥+', '॥', t)
    t = re.sub(r'\s+॥', '॥', t)

    # Fix common Bengali OCR confusion with English letters
    # These replacements fix characters that look similar
    # (Only do this if it doesn't break actual English words)

    # Remove excessive whitespace and common OCR artifacts
    t = re.sub(r'[\n\r\t]+', ' ', t)
    t = re.sub(r'[|]+', '', t)  # Remove pipe characters
    t = re.sub(r'[-]{3,}', '', t)  # Remove long dashes
    t = re.sub(r'[_]{2,}', '', t)  # Remove multiple underscores

    # Remove zero-width characters (common in Bengali OCR)
    t = re.sub(r'[\u200b-\u200f\u202a-\u202e\ufeff]', '', t)

    # Final cleanup
    t = t.strip()
    t = re.sub(r'\s+', ' ', t)

    return t if len(t) > 0 else "no text"

# Choose source column robustly
text_col = 'ocr_text_ens' if 'ocr_text_ens' in train_df.columns else ('ocr_text' if 'ocr_text' in train_df.columns else None)
if text_col is None:
    raise RuntimeError("No OCR text column found. Please run the OCR cell first.")

print(f"Using text column: {text_col}")
print("Cleaning TRAIN (Bengali + English)...")
tqdm.pandas(desc="clean-train")
train_df['final_text'] = train_df[text_col].progress_apply(clean_text_bilingual)

print("Cleaning TEST (Bengali + English)...")
tqdm.pandas(desc="clean-test")
test_df['final_text'] = test_df[text_col].progress_apply(clean_text_bilingual)

train_df.to_csv('train_cleaned_ens.csv', index=False)
test_df.to_csv('test_cleaned_ens.csv', index=False)
print("✓ Saved: train_cleaned_ens.csv, test_cleaned_ens.csv")

# Show sample of cleaned text (both Bengali and English)
print("\n--- Cleaned Text Sample (first 5 rows) ---")
for idx in range(min(5, len(train_df))):
    print(f"\n{idx+1}. Original: {train_df.iloc[idx][text_col][:80]}...")
    print(f"   Cleaned:  {train_df.iloc[idx]['final_text'][:80]}...")

# ==============================================
# IMPROVEMENT: 5-fold CV for text (BanglaBERT Large / XLM-R Large)
# ==============================================
print("\n=== 5-fold CV Text Model (BanglaBERT Large) ===")

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from torch.utils.data import Dataset

# Choose model (BanglaBERT Large preferred; fallback to XLM-R Large)
TEXT_MODEL_ID = "csebuetnlp/banglabert_large"
try:
    tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_ID)
    _ = AutoModelForSequenceClassification.from_pretrained(TEXT_MODEL_ID, num_labels=2)
except Exception:
    TEXT_MODEL_ID = "xlm-roberta-large"
    tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_ID)

class TextDataset(Dataset):
    def __init__(self, texts, labels=None, max_len=384):
        self.texts = texts
        self.labels = labels
        self.max_len = max_len
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        enc = tokenizer(
            str(self.texts[idx]),
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

labels = train_df['Label'].map({'NonPolitical':0,'Political':1}).values
texts = train_df['final_text'].fillna('no text').values
texts_test = test_df['final_text'].fillna('no text').values

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
text_oof = np.zeros((len(train_df), 2), dtype=np.float32)
text_test_preds = []

fold = 0
for tr_idx, va_idx in skf.split(texts, labels):
    fold += 1
    print(f"\n--- Text Fold {fold}/5 ---")

    # Clear GPU memory before loading new model
    import gc
    torch.cuda.empty_cache()
    gc.collect()

    model = AutoModelForSequenceClassification.from_pretrained(TEXT_MODEL_ID, num_labels=2)
    train_ds = TextDataset(texts[tr_idx], labels[tr_idx], max_len=384)
    valid_ds = TextDataset(texts[va_idx], labels[va_idx], max_len=384)
    test_ds  = TextDataset(texts_test, None, max_len=384)

    args = TrainingArguments(
        output_dir=f'./text_fold_{fold}',
        num_train_epochs=5,
        per_device_train_batch_size=4,  # Reduced from 16 to fit large model
        gradient_accumulation_steps=4,  # Effective batch size = 4*4 = 16
        learning_rate=2e-5,
        weight_decay=0.01,
        fp16=True,
        report_to=[],
        seed=42,
        eval_strategy='epoch',
        save_strategy='no',
        per_device_eval_batch_size=8,  # Smaller eval batch
        dataloader_pin_memory=False,  # Reduce memory overhead
        gradient_checkpointing=True,  # Trade compute for memory
    )

    trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=valid_ds)
    trainer.train()

    va_pred = trainer.predict(valid_ds).predictions
    text_oof[va_idx] = va_pred
    tst_pred = trainer.predict(test_ds).predictions
    text_test_preds.append(tst_pred)

    # Clean up after each fold
    del model, trainer, train_ds, valid_ds, test_ds
    torch.cuda.empty_cache()
    gc.collect()

# Average test logits across folds
text_test_logits = np.mean(text_test_preds, axis=0)
text_test_probs = torch.softmax(torch.tensor(text_test_logits), dim=1).numpy()
text_test_p1 = text_test_probs[:,1]

submission_text_cv = pd.DataFrame({
    'Image_name': test_df['Image_name'],
    'Label': [ 'Political' if p>=0.5 else 'NonPolitical' for p in text_test_p1]
})
submission_text_cv.to_csv('submission_text_cv.csv', index=False)
print("Saved: submission_text_cv.csv")

#/kaggle/working/banglabert_cleaned

# ==============================================
# IMPROVEMENT: Stronger Vision model + augmentations + 5-fold CV
# ==============================================
print("\n=== 5-fold CV Vision Model (ViT-Large) ===")

import os
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset
from transformers import AutoImageProcessor, ViTForImageClassification, TrainingArguments, Trainer

VISION_MODEL_ID = "google/vit-large-patch16-224-in21k"
image_processor = AutoImageProcessor.from_pretrained(VISION_MODEL_ID)

id2label = {0: "NonPolitical", 1: "Political"}
label2id = {"NonPolitical": 0, "Political": 1}

class ImgDataset(Dataset):
    def __init__(self, df, img_dir, processor, with_labels=True):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.processor = processor
        self.with_labels = with_labels
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row['Image_name'])
        image = Image.open(img_path).convert('RGB')
        inputs = self.processor(images=image, return_tensors='pt')
        item = {k: v.squeeze(0) for k, v in inputs.items()}
        if self.with_labels and 'Label' in row:
            item['labels'] = torch.tensor(label2id[row['Label']], dtype=torch.long)
        return item

labels_img = train_df['Label'].map({'NonPolitical':0,'Political':1}).values
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
img_oof = np.zeros((len(train_df), 2), dtype=np.float32)
img_test_preds = []

fold = 0
for tr_idx, va_idx in skf.split(train_df, labels_img):
    fold += 1
    print(f"\n--- Vision Fold {fold}/5 ---")
    vit_model = ViTForImageClassification.from_pretrained(
        VISION_MODEL_ID,
        num_labels=2,
        id2label=id2label,
        label2id=label2id
    )

    tr_df = train_df.iloc[tr_idx]
    va_df = train_df.iloc[va_idx]

    train_ds = ImgDataset(tr_df, TRAIN_DIR, image_processor, with_labels=True)
    valid_ds = ImgDataset(va_df, TRAIN_DIR, image_processor, with_labels=True)
    test_ds  = ImgDataset(test_df, TEST_DIR, image_processor, with_labels=False)

    args = TrainingArguments(
        output_dir=f'./vitL_fold_{fold}',
        num_train_epochs=10,
        per_device_train_batch_size=32,
        learning_rate=3e-5,
        weight_decay=0.01,
        fp16=True,
        report_to=[],
        seed=42,
        remove_unused_columns=False,
        eval_strategy='epoch',
        save_strategy='no'
    )

    trainer = Trainer(model=vit_model, args=args, train_dataset=train_ds, eval_dataset=valid_ds)
    trainer.train()

    va_pred = trainer.predict(valid_ds).predictions
    img_oof[va_idx] = va_pred
    tst_pred = trainer.predict(test_ds).predictions
    img_test_preds.append(tst_pred)

img_test_logits = np.mean(img_test_preds, axis=0)
img_test_probs = torch.softmax(torch.tensor(img_test_logits), dim=1).numpy()
img_test_p1 = img_test_probs[:,1]

submission_image_cv = pd.DataFrame({
    'Image_name': test_df['Image_name'],
    'Label': [ 'Political' if p>=0.5 else 'NonPolitical' for p in img_test_p1]
})
submission_image_cv.to_csv('submission_image_cv.csv', index=False)
print("Saved: submission_image_cv.csv")

# ==============================================
# IMPROVEMENT: XGBoost Fusion with Rich Features
# ==============================================
print("\n=== XGBoost Fusion with Enhanced Features ===")

import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from sklearn.metrics import roc_auc_score, accuracy_score

# Install XGBoost if not available
try:
    import xgboost
except ImportError:
    print("Installing XGBoost...")
    import sys
    get_ipython().system('{sys.executable} -m pip install -q xgboost')
    import xgboost as xgb

# Calibrate per-branch OOF logits -> probabilities
text_oof_probs = torch.softmax(torch.tensor(text_oof), dim=1).numpy()
text_oof_p0 = text_oof_probs[:,0]  # NonPolitical prob
text_oof_p1 = text_oof_probs[:,1]  # Political prob

img_oof_probs = torch.softmax(torch.tensor(img_oof), dim=1).numpy()
img_oof_p0 = img_oof_probs[:,0]
img_oof_p1 = img_oof_probs[:,1]

# Create rich feature set for XGBoost
def create_fusion_features(text_p0, text_p1, img_p0, img_p1, df):
    features = {
        'text_p1': text_p1,
        'text_p0': text_p0,
        'img_p1': img_p1,
        'img_p0': img_p0,

        # Confidence scores
        'text_conf': np.abs(text_p1 - 0.5) * 2,  # How confident is text model?
        'img_conf': np.abs(img_p1 - 0.5) * 2,    # How confident is image model?

        # Agreement features
        'avg_p1': (text_p1 + img_p1) / 2,
        'max_p1': np.maximum(text_p1, img_p1),
        'min_p1': np.minimum(text_p1, img_p1),
        'diff_p1': np.abs(text_p1 - img_p1),  # Disagreement measure

        # Product features (interaction)
        'text_img_prod': text_p1 * img_p1,
        'text_img_prod_p0': text_p0 * img_p0,

        # Text-based features
        'text_len': df['final_text'].str.len(),
        'text_word_count': df['final_text'].str.split().str.len(),
        'has_text': (df['final_text'].str.lower() != 'no text').astype(int),
    }
    return pd.DataFrame(features)

# Build training features
X_train_xgb = create_fusion_features(text_oof_p0, text_oof_p1, img_oof_p0, img_oof_p1, train_df)
y_train = train_df['Label'].map({'NonPolitical':0,'Political':1}).values

print(f"Training XGBoost with {X_train_xgb.shape[1]} features...")
print(f"Features: {list(X_train_xgb.columns)}")

# Train XGBoost model
xgb_model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='binary:logistic',
    eval_metric='auc',
    random_state=42,
    use_label_encoder=False,
    tree_method='hist',  # Faster training
    gpu_id=0 if torch.cuda.is_available() else None
)

xgb_model.fit(X_train_xgb, y_train, verbose=False)

# Evaluate on OOF
oof_pred_proba = xgb_model.predict_proba(X_train_xgb)[:,1]
oof_auc = roc_auc_score(y_train, oof_pred_proba)
oof_acc = accuracy_score(y_train, (oof_pred_proba >= 0.5).astype(int))
print(f"\n✓ OOF AUC: {oof_auc:.4f}")
print(f"✓ OOF Accuracy: {oof_acc:.4f}")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': X_train_xgb.columns,
    'importance': xgb_model.feature_importances_
}).sort_values('importance', ascending=False)
print("\nTop 10 Most Important Features:")
print(feature_importance.head(10).to_string(index=False))

# Apply to test predictions
text_test_probs = torch.softmax(torch.tensor(text_test_logits), dim=1).numpy()
text_test_p0 = text_test_probs[:,0]
text_test_p1 = text_test_probs[:,1]

img_test_probs = torch.softmax(torch.tensor(img_test_logits), dim=1).numpy()
img_test_p0 = img_test_probs[:,0]
img_test_p1 = img_test_probs[:,1]

X_test_xgb = create_fusion_features(text_test_p0, text_test_p1, img_test_p0, img_test_p1, test_df)
fused_test_proba = xgb_model.predict_proba(X_test_xgb)[:,1]

# Create submission
submission_xgb = pd.DataFrame({
    'Image_name': test_df['Image_name'],
    'Label': ['Political' if p >= 0.5 else 'NonPolitical' for p in fused_test_proba]
})

submission_xgb.to_csv('submission_xgboost_fusion.csv', index=False)
print("\n✓ Saved: submission_xgboost_fusion.csv")

print("\n--- XGBoost Submission Label Counts ---")
print(submission_xgb['Label'].value_counts())

print("\n--- XGBoost Submission Preview ---")
display(submission_xgb.head(10))