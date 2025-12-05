
import os
import pandas as pd
import numpy as np
import re
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, TrainingArguments, Trainer, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from tqdm.auto import tqdm
import easyocr
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
import warnings

# Configuration
warnings.filterwarnings("ignore")
sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.figsize'] = (12, 6)

# --------------------- KAGGLE-SPECIFIC PATHS ---------------------

TRAIN_CSV = 'Train/Train.csv'
TEST_CSV  = 'Test/Test.csv'
TRAIN_DIR = 'Train/Image'
TEST_DIR  = 'Test/Image'

print('Data source import complete.')

# --------------------- STEP 1: OCR EXTRACTION ---------------------
print("="*70)
print("🔍 STEP 1: OCR TEXT EXTRACTION")
print("="*70)

reader = easyocr.Reader(['bn', 'en'], gpu=torch.cuda.is_available())

def extract_text_from_image(img_name, folder_path):
    img_path = os.path.join(folder_path, img_name)
    try:
        result = reader.readtext(img_path, detail=0, paragraph=True)
        return " ".join(result)
    except Exception:
        return ""

# Load Dataframes
train_df = pd.read_csv(TRAIN_CSV)
test_df  = pd.read_csv(TEST_CSV)

print(f"📊 Training samples: {len(train_df)}")
print(f"📊 Test samples: {len(test_df)}")

# Run OCR
if 'ocr_text' not in train_df.columns:
    print(f"\n🔄 Running OCR on {len(train_df)} Training images...")
    tqdm.pandas(desc="Training OCR")
    train_df['ocr_text'] = train_df['Image_name'].progress_apply(
        lambda x: extract_text_from_image(x, TRAIN_DIR)
    )
    train_df.to_csv('train_with_ocr.csv', index=False)
    print("✅ Training OCR completed and saved!")

if 'ocr_text' not in test_df.columns:
    print(f"\n🔄 Running OCR on {len(test_df)} Test images...")
    tqdm.pandas(desc="Test OCR")
    test_df['ocr_text'] = test_df['Image_name'].progress_apply(
        lambda x: extract_text_from_image(x, TEST_DIR)
    )
    test_df.to_csv('test_with_ocr.csv', index=False)
    print("✅ Test OCR completed and saved!")

# --------------------- STEP 2: PREPROCESSING ---------------------
print("\n" + "="*70)
print("🧹 STEP 2: TEXT PREPROCESSING")
print("="*70)

def clean_and_normalize(text):
    if pd.isna(text): return ""
    text = str(text)
    # Keep Bangla, English, and punctuation
    text = re.sub(r'[^ঀ-\u09FFa-zA-Z0-9\s\!\?\।\.,\-]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

train_df['clean_text'] = train_df['ocr_text'].apply(clean_and_normalize)
test_df['clean_text']  = test_df['ocr_text'].apply(clean_and_normalize)

# Show processed data samples
print("\n📋 Sample of Processed Data:")
print("-"*70)
print(train_df[['Image_name', 'Label', 'ocr_text', 'clean_text']].head(10).to_string(index=False))

# Save processed data
train_df.to_csv('train_processed.csv', index=False)
test_df.to_csv('test_processed.csv', index=False)
print("\n✅ Processed data saved to CSV files!")

# --------------------- STEP 3: COMPREHENSIVE VISUALIZATIONS ---------------------
print("\n" + "="*70)
print("📊 STEP 3: DATA ANALYSIS & VISUALIZATION")
print("="*70)

# 1. Class Distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.countplot(x='Label', data=train_df, palette='viridis', ax=axes[0])
axes[0].set_title('Class Distribution', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Label', fontsize=12)
axes[0].set_ylabel('Count', fontsize=12)
for container in axes[0].containers:
    axes[0].bar_label(container)

# Pie chart
train_df['Label'].value_counts().plot.pie(
    autopct='%1.1f%%',
    colors=['#FF6B6B', '#4ECDC4'],
    startangle=90,
    ax=axes[1]
)
axes[1].set_title('Class Proportion', fontsize=14, fontweight='bold')
axes[1].set_ylabel('')

plt.tight_layout()
plt.show()

# 2. Text Length Distribution
train_df['text_length'] = train_df['clean_text'].str.len()
train_df['word_count'] = train_df['clean_text'].str.split().str.len()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for label in ['Political', 'NonPolitical']:
    subset = train_df[train_df['Label'] == label]
    axes[0].hist(subset['text_length'], bins=30, alpha=0.6, label=label)
axes[0].set_title('Text Length Distribution by Class', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Character Count')
axes[0].set_ylabel('Frequency')
axes[0].legend()

for label in ['Political', 'NonPolitical']:
    subset = train_df[train_df['Label'] == label]
    axes[1].hist(subset['word_count'], bins=30, alpha=0.6, label=label)
axes[1].set_title('Word Count Distribution by Class', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Word Count')
axes[1].set_ylabel('Frequency')
axes[1].legend()

plt.tight_layout()
plt.show()

# 3. Word Frequency Analysis - Political Memes
print("\n🔍 Analyzing Word Frequencies in Political Memes...")
political_text = " ".join(train_df[train_df['Label'] == 'Political']['clean_text'].tolist())
non_political_text = " ".join(train_df[train_df['Label'] == 'NonPolitical']['clean_text'].tolist())

political_words = [w for w in political_text.split() if len(w) > 2]
counter = Counter(political_words)
common_words = counter.most_common(25)

word_freq_df = pd.DataFrame(common_words, columns=['Word', 'Count'])
print("\n📌 Top 25 Words in Political Memes:")
print(word_freq_df.to_string(index=False))

# Plot Top Words
plt.figure(figsize=(14, 7))
sns.barplot(x='Count', y='Word', data=word_freq_df, palette='magma')
plt.title('Top 25 Frequent Words in Political Memes', fontsize=16, fontweight='bold')
plt.xlabel('Frequency', fontsize=12)
plt.ylabel('Word', fontsize=12)
plt.tight_layout()
plt.show()

# 4. Word Clouds
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Political Word Cloud
wc_political = WordCloud(
    width=800, height=400,
    background_color='white',
    colormap='Reds',
    regexp=r"[ঀ-\u09FFa-zA-Z]+"
).generate(political_text)

axes[0].imshow(wc_political, interpolation='bilinear')
axes[0].axis('off')
axes[0].set_title('Political Memes - Word Cloud', fontsize=14, fontweight='bold')

# Non-Political Word Cloud
wc_non_political = WordCloud(
    width=800, height=400,
    background_color='white',
    colormap='Blues',
    regexp=r"[ঀ-\u09FFa-zA-Z]+"
).generate(non_political_text)

axes[1].imshow(wc_non_political, interpolation='bilinear')
axes[1].axis('off')
axes[1].set_title('Non-Political Memes - Word Cloud', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.show()

# --------------------- STEP 4: ENHANCED FEATURE EXTRACTION ---------------------
print("\n" + "="*70)
print("🎯 STEP 4: ADVANCED FEATURE ENGINEERING")
print("="*70)

# Expanded keyword lists based on your notes
PERSONS = ["হাসিনা", "খালেদা", "তারেক", "জিয়া", "মোদি", "মুজিব", "শেখ", "রহমান",
           "ইউনুস", "কাদের", "আব্দুল", "বেগম"]

PARTIES = ["আওয়ামী", "বিএনপি", "জামায়াত", "লীগ", "জাপা", "BNP", "BAL", "Jamat",
           "Shibir", "ছাত্রলীগ", "যুবলীগ", "ছাত্রদল"]

ELECTION = ["ভোট", "নির্বাচন", "ইলেকশন", "election", "vote", "ইসি", "ব্যালট",
            "প্রার্থী", "মনোনয়ন", "Mujib", "15 Aug", "মেপ"]

CORRUPTION = ["দুর্নীতি", "লুটপাট", "চুরি", "ঘুষ", "scam", "দালাল", "চাঁদাবাজি",
              "অবৈধ", "দুর্নীতিবাজ"]

VIOLENCE = ["হরতাল", "আন্দোলন", "মিছিল", "পুলিশ", "গুলি", "হামলা", "সংঘর্ষ",
            "বিক্ষোভ", "অগ্নিসংযোগ", "রাজপথ", "July", "domination", "স্বৈরাচার"]

GOVT = ["সরকার", "প্রধানমন্ত্রী", "মন্ত্রী", "government", "PM", "রাষ্ট্র",
        "শাসন", "প্রশাসন", "interim", "Playing", "জনী মাকে"]

BANGLISH = ["hasina", "khaleda", "tarek", "dalal", "modi", "bnp", "awami",
            "joy bangla", "জয় বাংলা", "জুলাই"]

SLANG = ["চামচা", "দালাল", "লুটেরা", "গদি", "বাহিনী", "সুবিধাবাদী",
         "domination", "মিথ্যাচার"]

SENTIMENT_NEG = ["না", "নেই", "কেন", "কোনো", "খারাপ", "বাজে", "ভয়াবহ",
                 "অপমান", "ব্যর্থ", "অন্যায়", "অত্যাচার"]

SYMBOLS = ["গণতন্ত্র", "স্বাধীনতা", "মুক্তিযুদ্ধ", "শহীদ", "পতাকা",
           "মানবিকতা", "আইন"]

def extract_features(text):
    if pd.isna(text): text = ""
    text_lower = text.lower()
    words = text.split()

    features = [
        # Basic keyword presence
        sum(1 for k in PERSONS if k in text),  # Person mention count
        sum(1 for k in PARTIES if k in text),  # Party mention count
        sum(1 for k in ELECTION if k.lower() in text_lower),  # Election terms
        sum(1 for k in CORRUPTION if k in text),  # Corruption terms
        sum(1 for k in VIOLENCE if k in text),  # Violence terms
        sum(1 for k in GOVT if k.lower() in text_lower),  # Government terms
        sum(1 for k in BANGLISH if k in text_lower),  # Banglish terms
        sum(1 for k in SLANG if k in text),  # Slang count
        sum(1 for k in SYMBOLS if k in text),  # Symbol words

        # Sentiment & punctuation
        sum(1 for k in SENTIMENT_NEG if k in text),  # Negative sentiment
        text.count('!') + text.count('?'),  # Exclamation/question marks
        text.count('।') + text.count('.'),  # Sentence count

        # Text statistics
        len(words),  # Total word count
        len(text),  # Character count
        1 if len(words) > 10 else 0,  # Long text flag
        1 if len(words) < 3 else 0,  # Very short text flag

        # Advanced patterns
        len(re.findall(r'\b[A-Z]{3,}\b', text)),  # ALL CAPS words
        1 if re.search(r'\b(না|নেই|কোনো)[^।]*?(ভালো|উন্নয়ন|সুবিধা)', text) else 0,  # Negation pattern
        text.count('#'),  # Hashtag count
        1 if any(x in text for x in ['হাসিনা', 'খালেদা']) else 0,  # Major political figures

        # Combined features
        1 if (any(k in text for k in PARTIES) and any(k in text for k in ELECTION)) else 0,  # Party + Election
        1 if (any(k in text for k in VIOLENCE) and any(k in text for k in GOVT)) else 0,  # Violence + Govt
        1 if (any(k in text for k in CORRUPTION) and any(k in text for k in PERSONS)) else 0,  # Corruption + Person

        # Specific phrases
        1 if 'joy bangla' in text_lower or 'জয় বাংলা' in text else 0,
        1 if 'জিন্দাবাদ' in text else 0,
    ]

    return np.array(features, dtype=np.float32)

print("🔄 Extracting comprehensive features...")
train_features = np.stack(train_df['clean_text'].apply(extract_features).values)
test_features  = np.stack(test_df['clean_text'].apply(extract_features).values)
NUM_EXTRA_FEATURES = train_features.shape[1]
print(f"✅ Extracted {NUM_EXTRA_FEATURES} features per sample")

# Feature names
feature_names = [
    'Person_Count', 'Party_Count', 'Election_Terms', 'Corruption_Terms',
    'Violence_Terms', 'Govt_Terms', 'Banglish_Terms', 'Slang_Count',
    'Symbol_Words', 'Negative_Sentiment', 'Punctuation', 'Sentence_Count',
    'Word_Count', 'Char_Count', 'Is_Long_Text', 'Is_Very_Short',
    'AllCaps_Count', 'Negation_Pattern', 'Hashtag_Count', 'Major_Figure',
    'Party_Election_Combo', 'Violence_Govt_Combo', 'Corruption_Person_Combo',
    'Joy_Bangla', 'Jindabad'
]

# 5. Feature Correlation Heatmap
print("\n📊 Computing Feature Correlations...")
viz_df = pd.DataFrame(train_features, columns=feature_names)
viz_df['Target'] = train_df['Label'].map({'NonPolitical': 0, 'Political': 1})

# Correlation with target
target_corr = viz_df.corr()[['Target']].sort_values(by='Target', ascending=False)
print("\n🎯 Feature Correlations with Political Class:")
print(target_corr.to_string())

plt.figure(figsize=(10, 12))
sns.heatmap(target_corr, annot=True, cmap='RdYlGn', center=0, vmin=-1, vmax=1,
            fmt='.3f', linewidths=0.5)
plt.title('Feature Correlation with Political Class', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

# Full correlation matrix
plt.figure(figsize=(16, 14))
corr_matrix = viz_df.corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='coolwarm', center=0,
            linewidths=0.5, cbar_kws={"shrink": 0.8})
plt.title('Feature Correlation Matrix', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

# 6. Feature Distribution by Class
print("\n📊 Feature Distribution Analysis...")
fig, axes = plt.subplots(5, 5, figsize=(20, 20))
axes = axes.flatten()

for i, feature in enumerate(feature_names):
    if i < len(axes):
        for label in [0, 1]:
            data = viz_df[viz_df['Target'] == label][feature]
            axes[i].hist(data, bins=20, alpha=0.6,
                        label=['NonPolitical', 'Political'][label])
        axes[i].set_title(feature, fontsize=10, fontweight='bold')
        axes[i].legend(fontsize=8)
        axes[i].tick_params(labelsize=8)

plt.tight_layout()
plt.show()

# --------------------- STEP 5: ADVANCED BERT MODEL ---------------------
print("\n" + "="*70)
print("🤖 STEP 5: BUILDING ADVANCED BERT MODEL")
print("="*70)

MODEL_NAME = "csebuetnlp/banglabert"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

class EnhancedBanglaBERT(nn.Module):
    def __init__(self, num_extra_features):
        super().__init__()
        self.bert = AutoModel.from_pretrained(MODEL_NAME)
        self.hidden_size = self.bert.config.hidden_size

        # Feature projection
        self.feature_projection = nn.Sequential(
            nn.Linear(num_extra_features, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # Combined classifier with residual connections
        input_dim = self.hidden_size + 128

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 768),
            nn.LayerNorm(768),
            nn.GELU(),
            nn.Dropout(0.3),

            nn.Linear(768, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.2),

            nn.Linear(256, 2)
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for module in self.classifier:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, input_ids, attention_mask, extra_features, labels=None):
        # BERT encoding
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        # Use CLS token + mean pooling
        cls_token = outputs.last_hidden_state[:, 0, :]
        mean_pool = torch.mean(outputs.last_hidden_state, dim=1)
        bert_output = (cls_token + mean_pool) / 2

        # Project features
        projected_features = self.feature_projection(extra_features)

        # Combine
        combined = torch.cat([bert_output, projected_features], dim=1)
        logits = self.classifier(combined)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, 2), labels.view(-1))

        return {'loss': loss, 'logits': logits} if loss is not None else {'logits': logits}

# Dataset Class with validation split
class MemeDataset(Dataset):
    def __init__(self, texts, features, labels=None):
        self.texts = texts
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = tokenizer(
            str(self.texts[idx]),
            truncation=True,
            padding='max_length',
            max_length=128,
            return_tensors='pt'
        )
        item = {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'extra_features': torch.tensor(self.features[idx], dtype=torch.float32)
        }
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

# Prepare datasets with validation split
train_texts = train_df['clean_text'].tolist()
train_labels = train_df['Label'].map({'NonPolitical': 0, 'Political': 1}).tolist()

# Split for validation
train_texts_split, val_texts, train_features_split, val_features, train_labels_split, val_labels = train_test_split(
    train_texts, train_features, train_labels, test_size=0.15, random_state=42, stratify=train_labels
)

train_dataset = MemeDataset(train_texts_split, train_features_split, train_labels_split)
val_dataset = MemeDataset(val_texts, val_features, val_labels)
test_dataset = MemeDataset(test_df['clean_text'].tolist(), test_features)

print(f"📊 Training samples: {len(train_dataset)}")
print(f"📊 Validation samples: {len(val_dataset)}")
print(f"📊 Test samples: {len(test_dataset)}")

# Initialize model
model = EnhancedBanglaBERT(num_extra_features=NUM_EXTRA_FEATURES)
print(f"\n✅ Model initialized with {NUM_EXTRA_FEATURES} extra features")

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=8,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=1e-5,
    weight_decay=0.01,
    warmup_steps=100,
    logging_steps=50,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    fp16=True,
    report_to=[],
    gradient_accumulation_steps=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

# --------------------- STEP 6: TRAINING ---------------------
print("\n" + "="*70)
print("🚀 STEP 6: MODEL TRAINING")
print("="*70)

trainer.train()

# --------------------- STEP 7: VALIDATION & METRICS ---------------------
print("\n" + "="*70)
print("📈 STEP 7: VALIDATION PERFORMANCE")
print("="*70)

val_preds = trainer.predict(val_dataset)
val_pred_labels = np.argmax(val_preds.predictions, axis=1)

print("\n🎯 Validation Classification Report:")
print(classification_report(val_labels, val_pred_labels,
                          target_names=['NonPolitical', 'Political'],
                          digits=4))

# Confusion Matrix
cm = confusion_matrix(val_labels, val_pred_labels)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['NonPolitical', 'Political'],
            yticklabels=['NonPolitical', 'Political'])
plt.title('Validation Confusion Matrix', fontsize=14, fontweight='bold')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# --------------------- STEP 8: TEST PREDICTIONS ---------------------
print("\n" + "="*70)
print("🔮 STEP 8: GENERATING TEST PREDICTIONS")
print("="*70)

preds = trainer.predict(test_dataset)
final_preds = np.argmax(preds.predictions, axis=1)
labels_map = {0: 'NonPolitical', 1: 'Political'}
final_labels = [labels_map[x] for x in final_preds]

# Prediction distribution
pred_dist = pd.Series(final_labels).value_counts()
print("\n📊 Test Prediction Distribution:")
print(pred_dist)

plt.figure(figsize=(8, 5))
pred_dist.plot(kind='bar', color=['#FF6B6B', '#4ECDC4'])
plt.title('Test Set Prediction Distribution', fontsize=14, fontweight='bold')
plt.xlabel('Predicted Label')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# --------------------- STEP 9: SUBMISSION ---------------------
submission = pd.DataFrame({
    'Image_name': test_df['Image_name'],
    'Label': final_labels
})

submission.to_csv('submission_enhanced.csv', index=False)
print("\n" + "="*70)
print("✅ SUBMISSION FILE CREATED: submission_enhanced.csv")
print("="*70)

# Final summary
print("\n📋 PIPELINE SUMMARY:")
print(f"  • OCR Extraction: ✅")
print(f"  • Feature Engineering: {NUM_EXTRA_FEATURES} features ✅")
print(f"  • Model Training: 8 epochs ✅")
print(f"  • Validation F1-Score: {f1_score(val_labels, val_pred_labels, average='macro'):.4f}")
print(f"  • Test Predictions: {len(final_labels)} samples ✅")
print("\n🎉 All done! Good luck with your submission!")


