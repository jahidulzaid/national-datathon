
import os
import pandas as pd
import numpy as np
import easyocr
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

# Set CUDA_HOME to prevent DeepSpeed compilation errors
if 'CUDA_HOME' not in os.environ:
    import subprocess
    try:
        nvcc_path = subprocess.check_output(['which', 'nvcc']).decode().strip()
        os.environ['CUDA_HOME'] = os.path.dirname(os.path.dirname(nvcc_path))
    except:
        # Fallback: common CUDA paths
        for cuda_path in ['/usr/local/cuda', '/usr/local/cuda-12', '/usr/local/cuda-11']:
            if os.path.exists(cuda_path):
                os.environ['CUDA_HOME'] = cuda_path
                break

# Disable DeepSpeed to prevent compilation issues
os.environ['ACCELERATE_USE_DEEPSPEED'] = 'false'
os.environ['TRANSFORMERS_NO_DEEPSPEED'] = '1'

# --------------------- PATHS ---------------------
TRAIN_CSV = 'Train/Train.csv'
TEST_CSV  = 'Test/Test.csv'
TRAIN_DIR = 'Train/Image'
TEST_DIR  = 'Test/Image'

train_df = pd.read_csv(TRAIN_CSV)
test_df  = pd.read_csv(TEST_CSV)

# ========================================================
# STEP 1: OCR Extraction
# ========================================================
print("Initializing EasyOCR...")
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

# --- USER REQUEST: SHOW 30 ROWS BEFORE CLEANING ---
print("\n" + "="*50)
print("BEFORE LLM CLEANING (Raw OCR Output) - First 30 Rows")
print("="*50)
display(train_df[['Image_name', 'Label', 'ocr_text']].head(30))

# ========================================================
# STEP 2: GEMMA-2-2B TEXT CLEANING
# ========================================================
print("\n" + "="*50)
print("STARTING gemma-3-4b-it TEXT CLEANING")
print("="*50)

# Use Gemma-2-2B-IT (Lightweight & Fast)
model_id = "google/gemma-3-4b-it"

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
        return_full_text=False,
        device_map="auto",
        batch_size=1  # Process one at a time to avoid memory issues
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
            output = text_generator(
                prompt,
                do_sample=False,  # Use greedy decoding for stability
                temperature=None,  # Disable temperature sampling
                # top_p=None,       # Disable top-p sampling
                pad_token_id=llm_tokenizer.eos_token_id,
            )[0]['generated_text']
            # Gemma output cleanup
            cleaned = output.strip()
            return cleaned
        except Exception as e:
            print(f"Warning: Failed to clean text, using original. Error: {e}")
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
    print("\n" + "="*50)
    print("AFTER GEMMA CLEANING (Final Training Data) - First 30 Rows")
    print("="*50)
    display(train_df[['Image_name', 'ocr_text', 'final_text']].head(30))

    # Clean up Gemma resources to free GPU memory
    print("\nCleaning up Gemma model from GPU...")
    del text_generator
    del llm_model
    del llm_tokenizer
    torch.cuda.empty_cache()
    print("GPU memory cleared.")

except Exception as e:
    print(f"Gemma Loading Failed: {e}")
    print("Falling back to raw OCR text.")
    train_df['final_text'] = train_df['ocr_text']
    test_df['final_text'] = test_df['ocr_text']

# ========================================================
# STEP 3: TRAIN BANGLABERT
# ========================================================

# Clear any lingering CUDA errors and reset GPU state
print("\nResetting CUDA state before training...")
torch.cuda.synchronize()
torch.cuda.empty_cache()
if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()
print("CUDA state reset complete.")

# Load Data
train_texts = train_df['final_text'].fillna("no text").tolist()
train_labels = train_df['Label'].map({'NonPolitical': 0, 'Political': 1}).tolist()
test_texts = test_df['final_text'].fillna("no text").tolist()

# MODEL_NAME = "FacebookAI/xlm-roberta-large"
MODEL_NAME = "microsoft/deberta-v3-large"

print(f"\nLoading model: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

# Move model to GPU explicitly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print(f"Model loaded on: {device}")

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
    output_dir='./banglabert_gemma_cleaned',
    num_train_epochs=5,
    per_device_train_batch_size=16,
    learning_rate=2e-5,
    weight_decay=0.01,
    fp16=True,
    report_to=[],
    seed=42,
    # Disable all features that might trigger DeepSpeed
    fsdp='',
    fsdp_config=None,
    accelerator_config=None,
)

print("\nInitializing Trainer...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

print("\nStarting Training on Gemma-Cleaned Text...")
trainer.train()

# Predict & Submit
predictions = trainer.predict(test_dataset)
pred_ids = np.argmax(predictions.predictions, axis=1)
final_labels = ['Political' if x == 1 else 'NonPolitical' for x in pred_ids]

submission = pd.DataFrame({
    'Image_name': test_df['Image_name'],
    'Label': final_labels
})

submission.to_csv('submission_xlm-roberta-large.csv', index=False)
print("Done! Submission saved.")

# %%
from IPython.display import FileLink

# Display the value counts for quick check
print("\n--- Submission Label Counts ---")
print(submission['Label'].value_counts())

# Display the first few rows of the final submission file
print("\n--- Submission Preview ---")
display(submission.head())


FileLink('submission_xlm-roberta-large.csv')




