
import subprocess
import sys

def _pip_install(package):
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])

try:
    import bitsandbytes
    import accelerate
    print("bitsandbytes and accelerate are already installed.")
except Exception:
    print("Installing bitsandbytes and accelerate...")
    _pip_install('bitsandbytes')
    _pip_install('accelerate')
    print("Installation complete. Restarting kernel might be required.")

import os
import pandas as pd
import numpy as np
try:
    from paddleocr import PaddleOCR
except Exception:
    print('paddleocr not found, attempting to install...')
    _pip_install('paddleocr')
    from paddleocr import PaddleOCR
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
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
import argparse

# --------------------- PATHS ---------------------
TRAIN_CSV = '/kaggle/input/poli-meme-decode-cuet-cse-fest/PoliMemeDecode/Train/Train.csv'
TEST_CSV  = '/kaggle/input/poli-meme-decode-cuet-cse-fest/PoliMemeDecode/Test/Test.csv'
TRAIN_DIR = '/kaggle/input/poli-meme-decode-cuet-cse-fest/PoliMemeDecode/Train/Image'
TEST_DIR  = '/kaggle/input/poli-meme-decode-cuet-cse-fest/PoliMemeDecode/Test/Image'

train_df = pd.read_csv(TRAIN_CSV)
test_df  = pd.read_csv(TEST_CSV)

# ========================================================
# STEP 1: OCR Extraction (PaddleOCR Bangla + English)
# Notes:
# - Some PaddleOCR releases don't include a prebuilt 'bangla' model. The script
#   attempts a few alias names (e.g., 'bn') and falls back gracefully if none
#   are available.
# - Earlier versions of some PaddleOCR releases accepted 'show_log' argument; newer
#   ones no longer accept it. This script avoids that argument to be compatible
#   with multiple SDL versions.
# ========================================================
print("Initializing PaddleOCR (Bangla + English)...")
# Common language name(s) to attempt. PaddleOCR has limited prebuilt language packs.
# We'll attempt 'bangla' but allow graceful fallback to 'en' or a dummy reader.
OCR_LANGS = ["bangla", "en"]
ocr_readers = []
use_gpu_flag = False
try:
    import torch
    use_gpu_flag = torch.cuda.is_available()
except Exception:
    use_gpu_flag = False

for lang in OCR_LANGS:
    try:
        # Some paddleocr versions don't accept show_log; avoid that argument.
        reader = PaddleOCR(use_angle_cls=True, lang=lang, use_gpu=use_gpu_flag)
        ocr_readers.append(reader)
        print(f"Loaded PaddleOCR lang={lang}")
    except Exception as exc:
        print(f"Skipping {lang} OCR due to error: {exc}")
        # atypical alias fallback for bangla; try bn (some models name it bn)
        if lang.lower().startswith('bang'):
            for alias in ('bn', 'bang', 'bangla_bn'):
                try:
                    reader = PaddleOCR(use_angle_cls=True, lang=alias, use_gpu=use_gpu_flag)
                    ocr_readers.append(reader)
                    print(f"Loaded PaddleOCR lang alias={alias} for request={lang}")
                    break
                except Exception:
                    continue

if not ocr_readers:
    # Fallback: provide a dummy reader that returns no OCR so pipeline can continue.
    print("No PaddleOCR readers initialized. Falling back to no-op OCR reader.")

    class DummyReader:
        def ocr(self, image, cls=True):
            return []

    ocr_readers.append(DummyReader())


def _list_readers():
    readers = []
    for r in ocr_readers:
        try:
            readers.append(getattr(r, 'lang', r.__class__.__name__))
        except Exception:
            readers.append(r.__class__.__name__)
    return readers


def main_test(img_path=None):
    print('OCR readers available:', _list_readers())
    if img_path:
        print('Testing OCR on:', img_path)
        print('Detected:', extract_text(img_path))


def extract_text(img_path):
    texts = []
    for reader in ocr_readers:
        try:
            # some PaddleOCR versions accept img_path; others want cv2 image array.
            # Keep cls=True to get angle classification results (if supported).
            ocr_result = reader.ocr(img_path, cls=True)
            lines = []
            for line in ocr_result:
                # line can be either a list of (bbox, (text, score)) or a list of dicts.
                if isinstance(line, list):
                    for item in line:
                        # tuple-like result
                        if isinstance(item, (list, tuple)) and len(item) >= 2:
                            rec = item[1]
                            if isinstance(rec, (list, tuple)) and len(rec) >= 2:
                                text, score = rec[0], rec[1]
                                try:
                                    if float(score) >= 0.25:
                                        lines.append(str(text))
                                except Exception:
                                    lines.append(str(text))
                        # dict-like result
                        elif isinstance(item, dict):
                            text = item.get('text') or item.get('transcription')
                            score = item.get('confidence') or item.get('score') or 0
                            try:
                                if float(score) >= 0.25 and text:
                                    lines.append(str(text))
                            except Exception:
                                if text:
                                    lines.append(str(text))
                elif isinstance(line, dict):
                    text = line.get('text') or line.get('transcription')
                    score = line.get('confidence') or line.get('score') or 0
                    try:
                        if float(score) >= 0.25 and text:
                            lines.append(str(text))
                    except Exception:
                        if text:
                            lines.append(str(text))
            if lines:
                texts.append(" ".join(lines))
        except Exception:
            continue
    combined = " || ".join(texts)
    return combined if combined else "no text"

# OCR on Train

train_texts = []
for img_name in tqdm(train_df['Image_name'], desc="OCR Train"):
    img_path = os.path.join(TRAIN_DIR, img_name)
    train_texts.append(extract_text(img_path))

train_df['ocr_text'] = train_texts

# OCR on Test

test_texts = []
for img_name in tqdm(test_df['Image_name'], desc="OCR Test"):
    img_path = os.path.join(TEST_DIR, img_name)
    test_texts.append(extract_text(img_path))

test_df['ocr_text'] = test_texts

# --- USER REQUEST: SHOW 30 ROWS BEFORE CLEANING ---
print(" " + "="*50)
print("BEFORE LLM CLEANING (Raw OCR Output) - First 30 Rows")
print("="*50)
display(train_df[['Image_name', 'Label', 'ocr_text']].head(30))

# ========================================================
# STEP 2: GEMMA-2-2B TEXT CLEANING
# ========================================================
print(" " + "="*50)
print("STARTING GEMMA-2-2B TEXT CLEANING")
print("="*50)

# Use Gemma-2-2B-IT (Lightweight & Fast)
model_id = "google/gemma-2-2b-it"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

print(f"Loading LLM: {model_id}...")
try:
    llm_tokenizer = AutoTokenizer.from_pretrained(model_id)
    llm_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto"
    )

    # Create pipeline
    text_generator = pipeline(
        "text-generation",
        model=llm_model,
        tokenizer=llm_tokenizer,
        max_new_tokens=128,
        return_full_text=False
    )

    def clean_text_with_gemma(text):
        if len(str(text)) < 5 or text == "no text":
            return text

        # Gemma Prompt
        messages = [
            {"role": "user", "content": f"Fix the grammar and punctuation of this Bengali/English text. Output ONLY the corrected text:\n\n{text}"}
        ]

        prompt = llm_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        try:
            output = text_generator(prompt)[0]['generated_text']
            # Gemma output cleanup
            cleaned = output.strip()
            return cleaned
        except:
            return text

    # Apply to Train
    print("Cleaning TRAIN text with Gemma...")
    tqdm.pandas()
    train_df['final_text'] = train_df['ocr_text'].progress_apply(clean_text_with_gemma)

    # Apply to Test
    print("Cleaning TEST text with Gemma...")
    test_df['final_text'] = test_df['ocr_text'].progress_apply(clean_text_with_gemma)

    # Save cleaned data
    train_df.to_csv('train_cleaned_gemma.csv', index=False)
    test_df.to_csv('test_cleaned_gemma.csv', index=False)

    # --- USER REQUEST: SHOW 30 ROWS AFTER CLEANING ---
    print("" + "="*50)
    print("AFTER GEMMA CLEANING (Final Training Data) - First 30 Rows")
    print("="*50)
    display(train_df[['Image_name', 'ocr_text', 'final_text']].head(30))

except Exception as e:
    print(f"Gemma Loading Failed: {e}")
    print("Falling back to raw OCR text.")
    train_df['final_text'] = train_df['ocr_text']
    test_df['final_text'] = test_df['ocr_text']

# ========================================================
# STEP 3: TRAIN XLM-R LARGE
# ========================================================

# Load Data
train_texts = train_df['final_text'].fillna("no text").tolist()
train_labels = train_df['Label'].map({'NonPolitical': 0, 'Political': 1}).tolist()
test_texts = test_df['final_text'].fillna("no text").tolist()

MODEL_NAME = "FacebookAI/xlm-roberta-large"
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
    output_dir='./xlmr_cleaned',
    num_train_epochs=4,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    weight_decay=0.01,
    fp16=True,
    warmup_ratio=0.1,
    report_to=[],
    seed=42,
    logging_steps=50,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)


trainer.train()

# Predict & Submit
predictions = trainer.predict(test_dataset)
pred_ids = np.argmax(predictions.predictions, axis=1)
final_labels = ['Political' if x == 1 else 'NonPolitical' for x in pred_ids]

submission = pd.DataFrame({
    'Image_name': test_df['Image_name'],
    'Label': final_labels
})

submission.to_csv('submission_paddleocr_xlmr.csv', index=False)
print("Done! Submission saved.")


# %%
from IPython.display import FileLink

# Display the value counts for quick check
print("--- Submission Label Counts ---")
print(submission['Label'].value_counts())

# Display the first few rows of the final submission file
print("--- Submission Preview ---")
display(submission.head())


FileLink('submission_paddleocr_xlmr.csv')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PoliMeme OCR + XLM-R pipeline runner/test')
    parser.add_argument('--test-ocr', action='store_true', help='Run a quick OCR sanity test')
    parser.add_argument('--img', type=str, default=None, help='Optional image path to test OCR on')
    args = parser.parse_args()
    if args.test_ocr:
        main_test(args.img)


# %%



