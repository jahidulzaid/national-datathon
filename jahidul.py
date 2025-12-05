#!/usr/bin/env python3
"""
Qwen3-VL-8B-Instruct-FP8 based classifier for PoliMemeDecode.

The script loads the bilingual vision-language model, runs zero-shot
classification on each meme (Bangla + English text supported), and writes a
Kaggle-ready submission CSV with labels Political / NonPolitical.

Expected environment: RTX 4090 (fits FP8 weights with device_map="auto").
Install deps once: pip install --upgrade "transformers>=4.45.0" accelerate pillow pandas tqdm
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
import torch
from PIL import Image
from tqdm.auto import tqdm
from transformers import AutoModelForVision2Seq, AutoProcessor


MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct-FP8"
TRAIN_CSV = Path("Train/Train.csv")
TEST_CSV = Path("Test/Test.csv")
TRAIN_IMG_DIR = Path("Train/Image")
TEST_IMG_DIR = Path("Test/Image")
DEFAULT_SUBMISSION = Path("submission_qwen3vl_fp8.csv")


KEYWORD_HINTS = """
Bangladeshi political cues (Bangla or Banglish):
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
"""


def get_device_and_dtype() -> Tuple[str, torch.dtype]:
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16
    return "cpu", torch.float32


def load_qwen(model_id: str) -> Tuple[AutoModelForVision2Seq, AutoProcessor, str]:
    device, dtype = get_device_and_dtype()
    print(f"Loading {model_id} on {device} with dtype {dtype}...")
    model = AutoModelForVision2Seq.from_pretrained(
        model_id,
        dtype=dtype,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model.eval()
    return model, processor, device


def build_prompt(processor: AutoProcessor) -> str:
    keyword_text = KEYWORD_HINTS.strip()
    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "You are a precise meme classifier for Bangladeshi politics. "
                        "Label a meme as Political if visuals or text mention parties, leaders, elections, "
                        "government, corruption, protests, or insults/slogans about them—even when humorous or satirical. "
                        "If no political cues appear, label NonPolitical. Read Bangla and English (Banglish) text. "
                        "Helpful cues to treat as Political:\n" + keyword_text + "\n"
                        "If any of these cues are present in text, logos, banners, or uniforms, prefer Political. "
                        "Return ONLY a JSON object like {\"label\": \"Political\", \"confidence\": 0.0-1.0}."
                    ),
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "Classify this meme as Political or NonPolitical based on visuals and any "
                        "Bangla or English text. Return only the JSON object."
                    ),
                },
                {"type": "image"},
            ],
        },
    ]
    return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def parse_label(output_text: str) -> Optional[str]:
    text = output_text.lower()
    if "nonpolitical" in text or "non political" in text:
        return "NonPolitical"
    if "political" in text:
        return "Political"
    return None


def parse_confidence(output_text: str) -> Optional[float]:
    match = re.search(r"confidence[^0-9]*([01](?:\.\d+)?)", output_text, re.IGNORECASE)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None
    return None


def classify_image(
    image_path: Path,
    model: AutoModelForVision2Seq,
    processor: AutoProcessor,
    prompt: str,
    device: str,
    max_new_tokens: int,
) -> Tuple[str, str, Optional[float]]:
    image = Image.open(image_path).convert("RGB")
    inputs = processor(
        text=[prompt],
        images=[image],
        return_tensors="pt",
    )
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            repetition_penalty=1.05,
        )
    decoded = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    label = parse_label(decoded) or "NonPolitical"
    confidence = parse_confidence(decoded)
    return label, decoded, confidence


def predict_dataframe(
    df: pd.DataFrame,
    image_dir: Path,
    model: AutoModelForVision2Seq,
    processor: AutoProcessor,
    prompt: str,
    device: str,
    max_new_tokens: int,
) -> Tuple[List[str], List[str]]:
    labels: List[str] = []
    raw_outputs: List[str] = []
    for img_name in tqdm(df["Image_name"].tolist(), desc="Classifying", ncols=100):
        image_path = image_dir / img_name
        label, decoded, _ = classify_image(
            image_path=image_path,
            model=model,
            processor=processor,
            prompt=prompt,
            device=device,
            max_new_tokens=max_new_tokens,
        )
        labels.append(label)
        raw_outputs.append(decoded)
    return labels, raw_outputs


def evaluate_holdout(
    train_df: pd.DataFrame,
    image_dir: Path,
    model: AutoModelForVision2Seq,
    processor: AutoProcessor,
    prompt: str,
    device: str,
    max_new_tokens: int,
    fraction: float,
    seed: int,
) -> None:
    holdout = train_df.sample(frac=fraction, random_state=seed)
    preds, _ = predict_dataframe(
        holdout,
        image_dir,
        model,
        processor,
        prompt,
        device,
        max_new_tokens,
    )
    accuracy = (holdout["Label"].values == preds).mean()
    print(f"Holdout accuracy on {len(holdout)} samples: {accuracy:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen3-VL FP8 political meme classifier")
    parser.add_argument("--model-id", default=MODEL_ID, help="Hugging Face model id")
    parser.add_argument("--train-csv", type=Path, default=TRAIN_CSV, help="Path to train CSV")
    parser.add_argument("--test-csv", type=Path, default=TEST_CSV, help="Path to test CSV")
    parser.add_argument("--train-dir", type=Path, default=TRAIN_IMG_DIR, help="Directory with train images")
    parser.add_argument("--test-dir", type=Path, default=TEST_IMG_DIR, help="Directory with test images")
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_SUBMISSION, help="Submission file path")
    parser.add_argument("--max-new-tokens", type=int, default=32, help="Generation length (keep small for speed)")
    parser.add_argument("--eval-fraction", type=float, default=0.05, help="Fraction of train used for quick holdout eval")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for holdout split")
    parser.add_argument("--save-raw", type=Path, default=None, help="Optional path to save raw model text outputs")
    args = parser.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    train_df = pd.read_csv(args.train_csv)
    test_df = pd.read_csv(args.test_csv)
    model, processor, device = load_qwen(args.model_id)
    prompt = build_prompt(processor)

    if args.eval_fraction and 0 < args.eval_fraction < 1.0:
        evaluate_holdout(
            train_df=train_df,
            image_dir=args.train_dir,
            model=model,
            processor=processor,
            prompt=prompt,
            device=device,
            max_new_tokens=args.max_new_tokens,
            fraction=args.eval_fraction,
            seed=args.seed,
        )

    print(f"Running inference on {len(test_df)} test images...")
    test_labels, raw_outputs = predict_dataframe(
        test_df,
        args.test_dir,
        model,
        processor,
        prompt,
        device,
        args.max_new_tokens,
    )

    submission = pd.DataFrame({"Image_name": test_df["Image_name"], "Label": test_labels})
    submission.to_csv(args.output_csv, index=False)
    print(f"Saved predictions to {args.output_csv}")

    if args.save_raw:
        pd.DataFrame({"Image_name": test_df["Image_name"], "raw_output": raw_outputs}).to_csv(
            args.save_raw, index=False
        )
        print(f"Saved raw model outputs to {args.save_raw}")


if __name__ == "__main__":
    main()
