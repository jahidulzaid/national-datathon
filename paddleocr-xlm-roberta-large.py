
pip install paddleocr

try:
    import bitsandbytes
    import accelerate
    print("bitsandbytes and accelerate are already installed.")
except ImportError:
    print("Installing bitsandbytes and accelerate...")
    pip install -q -U bitsandbytes accelerate
    print("Installation complete. Restarting kernel might be required.")

import os
import pandas as pd
import numpy as np
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

# --------------------- PATHS ---------------------
TRAIN_CSV = '/kaggle/input/poli-meme-decode-cuet-cse-fest/PoliMemeDecode/Train/Train.csv'
TEST_CSV  = '/kaggle/input/poli-meme-decode-cuet-cse-fest/PoliMemeDecode/Test/Test.csv'
TRAIN_DIR = '/kaggle/input/poli-meme-decode-cuet-cse-fest/PoliMemeDecode/Train/Image'
TEST_DIR  = '/kaggle/input/poli-meme-decode-cuet-cse-fest/PoliMemeDecode/Test/Image'

train_df = pd.read_csv(TRAIN_CSV)
test_df  = pd.read_csv(TEST_CSV)

# ========================================================
# STEP 1: OCR Extraction (PaddleOCR Bangla + English)
# ========================================================
print("Initializing PaddleOCR (Bangla + English)...")
OCR_LANGS = ["bangla", "en"]
ocr_readers = []
for lang in OCR_LANGS:
    try:
        reader = PaddleOCR(
            use_angle_cls=True,
            lang=lang,
            show_log=False,
            use_gpu=True,
        )
        ocr_readers.append(reader)
        print(f"Loaded PaddleOCR lang={lang}")
    except Exception as exc:
        print(f"Skipping {lang} OCR due to error: {exc}")

if not ocr_readers:
    raise RuntimeError("No PaddleOCR readers initialized.")


def extract_text(img_path):
    texts = []
    for reader in ocr_readers:
        try:
            ocr_result = reader.ocr(img_path, cls=True)
            lines = []
            for line in ocr_result:
                for (_, (text, score)) in line:
                    if score >= 0.25:
                        lines.append(text)
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


# %%



