
# IMPORTANT: RUN THIS CELL IN ORDER TO IMPORT YOUR KAGGLE DATA SOURCES,
# THEN FEEL FREE TO DELETE THIS CELL.
# NOTE: THIS NOTEBOOK ENVIRONMENT DIFFERS FROM KAGGLE'S PYTHON
# ENVIRONMENT SO THERE MAY BE MISSING LIBRARIES USED BY YOUR
# NOTEBOOK.


# --------------------- PATHS ---------------------
TRAIN_CSV = 'Train/Train.csv'
TEST_CSV  = 'Test/Test.csv'
TRAIN_DIR = 'Train/Image'
TEST_DIR  = 'Test/Image'

print('Data source import complete.')

# =============================================================================
# POLIMEMEDECODE — ADVANCED XGBOOST PIPELINE (OPTIMIZED FOR 99% ACCURACY)
# =============================================================================

# 1. INSTALLATIONS
# !pip install -q transformers accelerate easyocr xgboost optuna scikit-learn matplotlib seaborn wordcloud imbalanced-learn

import os
import pandas as pd
import numpy as np
import re
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm.auto import tqdm
import easyocr
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
import warnings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import xgboost as xgb
from imblearn.over_sampling import SMOTE
import optuna

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")

# --------------------- SETUP & PATHS ---------------------

TRAIN_CSV = 'Train/Train.csv'
TEST_CSV  = 'Test/Test.csv'
TRAIN_IMG_DIR = 'Train/Image'
TEST_IMG_DIR  = 'Test/Image'

# --------------------- STEP 1: OCR EXTRACTION (GPU ACCELERATED) ---------------------
print("Initializing EasyOCR with GPU...")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Device: {torch.cuda.get_device_name(0)}")

# Force GPU usage for OCR
reader = easyocr.Reader(['bn', 'en'], gpu=True)

def extract_text_from_image(img_name, folder_path):
    img_path = os.path.join(folder_path, img_name)
    try:
        result = reader.readtext(img_path, detail=0, paragraph=True)
        return " ".join(result)
    except Exception as e:
        print(f"⚠️ Error processing {img_name}: {e}")
        return ""

# Load Dataframes
train_df = pd.read_csv(TRAIN_CSV)
test_df  = pd.read_csv(TEST_CSV)

# Run OCR
if 'ocr_text' not in train_df.columns:
    print(f"Running OCR on {len(train_df)} Training images...")
    tqdm.pandas()
    train_df['ocr_text'] = train_df['Image_name'].progress_apply(
        lambda x: extract_text_from_image(x, TRAIN_IMG_DIR))
    train_df.to_csv('train_with_ocr.csv', index=False)

if 'ocr_text' not in test_df.columns:
    print(f"Running OCR on {len(test_df)} Test images...")
    tqdm.pandas()
    test_df['ocr_text'] = test_df['Image_name'].progress_apply(
        lambda x: extract_text_from_image(x, TEST_IMG_DIR))
    test_df.to_csv('test_with_ocr.csv', index=False)

# --------------------- STEP 2: ADVANCED PREPROCESSING ---------------------
def clean_and_normalize(text):
    if pd.isna(text): return ""
    text = str(text)
    text = re.sub(r'[^ঀ-\u09FFa-zA-Z0-9\s\!\?\।\.,]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

train_df['clean_text'] = train_df['ocr_text'].apply(clean_and_normalize)
test_df['clean_text']  = test_df['ocr_text'].apply(clean_and_normalize)

# --------------------- STEP 3: ULTRA-ADVANCED FEATURE ENGINEERING ---------------------
print("\n🔧 Extracting Ultra-Advanced Features...")

# Comprehensive keyword lists from your analysis
KEYWORDS_POLITICAL = [
    "নির্বাচন", "শিক্ষা", "বিদেশ", "ক্ষমতা", "পরিবর্তন", "সরকার",
    "জনগণ", "অর্থনীতি", "বিশ্বাস", "আইন", "বিচার", "স্বাধীনতা",
    "ভোট", "মুজিব", "১৫ আগস্ট", "NCP", "এনসিপি", "বিজয়", "সংবিধান",
    "ভোটার", "BNP", "ফরাসি", "DAL", "আওয়ামীলীগ", "চাকরি",
    "আমার সরকার", "মেয়র", "জুলাই", "নোমিনেশন", "বিদেশের", "ইন্টারিম",
    "হাসিনা", "শেখ হাসিনা", "জয় বাংলা", "দুর্নীতি", "জনতা", "প্রধানমন্ত্রী",
    "মন্ত্রী", "সংসদ", "সংসদ নির্বাচন", "সরকারি", "বিপদ", "বিজয়",
    "election", "vote", "government", "prime minister", "minister", "parliament",
    "corruption", "protest", "rally", "party", "candidate", "campaign"
]

HIGH_IMPACT_KEYWORDS = [
    "ভোট", "মুজিব", "১৫ আগস্ট", "NCP", "হাসিনা", "শেখ হাসিনা",
    "জয় বাংলা", "দুর্নীতি", "বিচার", "প্রধানমন্ত্রী", "সংসদ"
]

NEGATIVE_SENTIMENT = [
    "না", "নেই", "কোনো", "দুর্নীতি", "লুট", "চুরি", "ঘুষ",
    "হরতাল", "আন্দোলন", "বিরোধী", "প্রতিবাদ", "সংকট"
]

PERSONS = [
    "হাসিনা", "খালেদা", "তারেক", "জিয়া", "মোদি", "মুজিব",
    "শেখ", "রহমান", "মেয়র", "প্রধানমন্ত্রী", "মন্ত্রী"
]

PARTIES = [
    "আওয়ামী", "বিএনপি", "জামায়াত", "লীগ", "জাপা", "BNP",
    "BAL", "Jamat", "Shibir", "NCP", "এনসিপি"
]

SLANG_COLLOQUIAL = [
    "দালাল", "চামচা", "জুলাই", "মেয়র", "ফরাসি", "NCP", "এনসিপি"
]

SLOGANS = [
    "জয় বাংলা", "মুজিব", "১৫ আগস্ট", "ভোট দিন", "সরকার পতন"
]

def extract_advanced_features(text):
    """Extract 40+ engineered features from text"""
    if pd.isna(text): text = ""
    text_lower = text.lower()
    words = text.split()
    word_count = len(words)

    features = {}

    # 1-5: Keyword presence features
    features['has_political_keyword'] = int(any(kw in text for kw in KEYWORDS_POLITICAL))
    features['political_keyword_count'] = sum(1 for kw in KEYWORDS_POLITICAL if kw in text)
    features['high_impact_keyword_count'] = sum(1 for kw in HIGH_IMPACT_KEYWORDS if kw in text)
    features['keyword_density'] = features['political_keyword_count'] / max(1, word_count)
    features['high_impact_density'] = features['high_impact_keyword_count'] / max(1, word_count)

    # 6-10: Sentiment and tone features
    features['negative_sentiment_count'] = sum(1 for phrase in NEGATIVE_SENTIMENT if phrase in text)
    features['has_negative_sentiment'] = int(features['negative_sentiment_count'] > 0)
    features['exclamation_count'] = text.count('!')
    features['question_count'] = text.count('?')
    features['punctuation_density'] = (text.count('!') + text.count('?')) / max(1, word_count)

    # 11-15: Entity features
    features['person_count'] = sum(1 for person in PERSONS if person in text)
    features['has_person'] = int(features['person_count'] > 0)
    features['party_count'] = sum(1 for party in PARTIES if party in text)
    features['has_party'] = int(features['party_count'] > 0)
    features['entity_density'] = (features['person_count'] + features['party_count']) / max(1, word_count)

    # 16-20: Slang and slogan features
    features['slang_count'] = sum(1 for term in SLANG_COLLOQUIAL if term in text)
    features['has_slang'] = int(features['slang_count'] > 0)
    features['slogan_count'] = sum(1 for slogan in SLOGANS if slogan in text)
    features['has_slogan'] = int(features['slogan_count'] > 0)
    features['slang_slogan_ratio'] = (features['slang_count'] + features['slogan_count']) / max(1, word_count)

    # 21-25: Text structure features
    features['word_count'] = word_count
    features['unique_word_count'] = len(set(words))
    features['lexical_diversity'] = features['unique_word_count'] / max(1, word_count)
    features['avg_word_length'] = np.mean([len(w) for w in words]) if words else 0
    features['sentence_count'] = text.count('।') + text.count('.') + text.count('?') + text.count('!')

    # 26-30: Capitalization and emphasis
    features['all_caps_count'] = len(re.findall(r'\b[A-Z]{3,}\b', text))
    features['has_all_caps'] = int(features['all_caps_count'] > 0)
    features['uppercase_ratio'] = sum(1 for c in text if c.isupper()) / max(1, len(text))
    features['digit_count'] = sum(1 for c in text if c.isdigit())
    features['has_date'] = int(bool(re.search(r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b|\b\d{1,2} \w+ \d{4}\b|\b\d{1,2} \w+\b', text)))

    # 31-35: Language features
    bangla_chars = len(re.findall(r'[ঀ-\u09FF]', text))
    english_chars = len(re.findall(r'[a-zA-Z]', text))
    total_chars = max(1, bangla_chars + english_chars)
    features['bangla_ratio'] = bangla_chars / total_chars
    features['english_ratio'] = english_chars / total_chars
    features['is_bilingual'] = int(bangla_chars > 0 and english_chars > 0)
    features['char_count'] = len(text)
    features['avg_sentence_length'] = word_count / max(1, features['sentence_count'])

    # 36-40: Advanced combination features
    features['political_person_interaction'] = features['political_keyword_count'] * features['person_count']
    features['political_party_interaction'] = features['political_keyword_count'] * features['party_count']
    features['sentiment_keyword_ratio'] = features['negative_sentiment_count'] / max(1, features['political_keyword_count'])
    features['emphasis_score'] = features['exclamation_count'] + features['all_caps_count'] + features['question_count']
    features['political_intensity'] = (
        features['high_impact_keyword_count'] * 3 +
        features['person_count'] * 2 +
        features['party_count'] * 2 +
        features['slogan_count'] * 1.5
    )

    # 41-45: TF-IDF inspired features
    features['has_multiple_keywords'] = int(features['political_keyword_count'] >= 3)
    features['has_strong_indicators'] = int(
        features['high_impact_keyword_count'] >= 1 and
        (features['has_person'] or features['has_party'])
    )
    features['keyword_variety'] = len(set([kw for kw in KEYWORDS_POLITICAL if kw in text]))
    features['entity_variety'] = len(set([e for e in PERSONS + PARTIES if e in text]))
    features['content_richness'] = features['keyword_variety'] + features['entity_variety']

    return np.array(list(features.values()), dtype=np.float32)

# Extract features
print("Extracting features from training data...")
train_features = np.stack(train_df['clean_text'].apply(extract_advanced_features).values)
print("Extracting features from test data...")
test_features  = np.stack(test_df['clean_text'].apply(extract_advanced_features).values)

print(f"✅ Extracted {train_features.shape[1]} features per sample")

# --------------------- STEP 4: ADD BERT EMBEDDINGS (OPTIONAL BUT POWERFUL) ---------------------
print("\n🧠 Generating BanglaBERT Embeddings...")

MODEL_NAME = "csebuetnlp/banglabert"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
bert_model = AutoModel.from_pretrained(MODEL_NAME)
bert_model.eval()

if torch.cuda.is_available():
    bert_model = bert_model.cuda()

def get_bert_embedding(text, max_length=128):
    """Extract [CLS] token embedding from BanglaBERT"""
    with torch.no_grad():
        encoding = tokenizer(
            str(text),
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt'
        )
        if torch.cuda.is_available():
            encoding = {k: v.cuda() for k, v in encoding.items()}

        outputs = bert_model(**encoding)
        cls_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        return cls_embedding.squeeze()

# Generate embeddings (batched for efficiency)
print("Generating BERT embeddings for train set...")
train_embeddings = []
for text in tqdm(train_df['clean_text'].tolist()):
    train_embeddings.append(get_bert_embedding(text))
train_embeddings = np.array(train_embeddings)

print("Generating BERT embeddings for test set...")
test_embeddings = []
for text in tqdm(test_df['clean_text'].tolist()):
    test_embeddings.append(get_bert_embedding(text))
test_embeddings = np.array(test_embeddings)

# Combine manual features with BERT embeddings
X_train = np.concatenate([train_features, train_embeddings], axis=1)
X_test = np.concatenate([test_features, test_embeddings], axis=1)

print(f"✅ Final feature shape: {X_train.shape}")

# --------------------- STEP 5: PREPARE LABELS & HANDLE IMBALANCE ---------------------
y_train = train_df['Label'].map({'NonPolitical': 0, 'Political': 1}).values

# Check class distribution
print("\nClass Distribution:")
print(pd.Series(y_train).value_counts())

# Apply SMOTE if imbalanced
if pd.Series(y_train).value_counts().min() / pd.Series(y_train).value_counts().max() < 0.8:
    print("\n⚖️ Applying SMOTE for class balancing...")
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    print("New class distribution after SMOTE:")
    print(pd.Series(y_train).value_counts())

# --------------------- STEP 6: HYPERPARAMETER OPTIMIZATION WITH OPTUNA ---------------------
print("\n🔍 Optimizing XGBoost Hyperparameters with Optuna...")

def objective(trial):
    params = {
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0.01, 1.0),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 10.0, log=True),
        'tree_method': 'hist',
        'random_state': 42,
        'eval_metric': 'logloss'
    }

    model = xgb.XGBClassifier(**params)

    # 5-Fold Cross-Validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_train, y_train, cv=skf, scoring='f1_macro', n_jobs=-1)

    return scores.mean()

# Run optimization (reduce n_trials for faster execution)
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50, show_progress_bar=True)

print("\n🏆 Best Hyperparameters:")
print(study.best_params)
print(f"Best F1-Macro Score: {study.best_value:.4f}")

# --------------------- STEP 7: TRAIN FINAL XGBOOST MODEL ---------------------
print("\n🚀 Training Final XGBoost Model...")

best_params = study.best_params
best_params.update({
    'tree_method': 'hist',
    'random_state': 42,
    'eval_metric': 'logloss'
})

final_model = xgb.XGBClassifier(**best_params)
final_model.fit(X_train, y_train)

# Cross-validation evaluation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(final_model, X_train, y_train, cv=skf, scoring='f1_macro', n_jobs=-1)

print(f"\n📊 Cross-Validation Results:")
print(f"Mean F1-Macro: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
print(f"Individual Fold Scores: {cv_scores}")

# --------------------- STEP 8: GENERATE PREDICTIONS ---------------------
print("\n🔮 Generating Predictions...")
y_pred = final_model.predict(X_test)

labels_map = {0: 'NonPolitical', 1: 'Political'}
final_labels = [labels_map[x] for x in y_pred]

# --------------------- STEP 9: CREATE SUBMISSION ---------------------
submission = pd.DataFrame({
    'Image_name': test_df['Image_name'],
    'Label': final_labels
})
submission.to_csv('submission_xgboost_optimized.csv', index=False)

print("\n✅ Submission saved to 'submission_xgboost_optimized.csv'")
print(f"\nPrediction Distribution:")
print(submission['Label'].value_counts())

# --------------------- STEP 10: FEATURE IMPORTANCE VISUALIZATION ---------------------
print("\n📊 Visualizing Feature Importance...")

feature_importance = final_model.feature_importances_
top_20_indices = np.argsort(feature_importance)[-20:]

plt.figure(figsize=(12, 8))
plt.barh(range(20), feature_importance[top_20_indices])
plt.yticks(range(20), [f'Feature_{i}' for i in top_20_indices])
plt.xlabel('Feature Importance')
plt.title('Top 20 Most Important Features')
plt.tight_layout()
plt.show()

print("\n🎉 DONE! Your model is trained with advanced XGBoost.")
print("Expected performance: High accuracy and F1-Macro score based on CV results.")

