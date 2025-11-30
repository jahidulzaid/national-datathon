"""
Lightweight baseline for PoliMemeDecode: extract simple color/edge features
from memes, train a class-balanced logistic regression classifier, and
generate submission.csv predictions for the test set.

Usage:
    python3 solution.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def extract_features(
    image_path: Path,
    resize_to: int = 256,
    hist_bins: int = 16,
) -> np.ndarray:
    """
    Turn an image into a compact feature vector:
    - Resize to a fixed square to normalize scale.
    - Per-channel color histograms (normalized).
    - Per-channel mean/std.
    - Sobel edge magnitude histogram + mean/std to capture text/edges.
    """
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as exc:  # pragma: no cover - defensive for corrupt files
        raise RuntimeError(f"Failed to load {image_path}: {exc}") from exc

    img = img.resize((resize_to, resize_to), Image.BICUBIC)
    arr = np.asarray(img, dtype=np.float32) / 255.0  # scale to [0,1]

    feats: list[float] = []

    # Color histogram + moments
    for c in range(3):
        channel = arr[:, :, c]
        hist, _ = np.histogram(channel, bins=hist_bins, range=(0.0, 1.0), density=True)
        feats.extend(hist.tolist())
        feats.append(float(channel.mean()))
        feats.append(float(channel.std()))

    # Simple Sobel edges on grayscale without extra deps
    gray = arr.mean(axis=2)
    padded = np.pad(gray, 1, mode="reflect")
    gx = (
        -1 * padded[:-2, :-2]
        + 1 * padded[:-2, 2:]
        - 2 * padded[1:-1, :-2]
        + 2 * padded[1:-1, 2:]
        - 1 * padded[2:, :-2]
        + 1 * padded[2:, 2:]
    )
    gy = (
        -1 * padded[:-2, :-2]
        - 2 * padded[:-2, 1:-1]
        - 1 * padded[:-2, 2:]
        + 1 * padded[2:, :-2]
        + 2 * padded[2:, 1:-1]
        + 1 * padded[2:, 2:]
    )
    mag = np.hypot(gx, gy)
    norm = mag.max() + 1e-6
    mag = mag / norm  # keep values in [0,1]
    edge_hist, _ = np.histogram(mag, bins=8, range=(0.0, 1.0), density=True)
    feats.extend(edge_hist.tolist())
    feats.append(float(mag.mean()))
    feats.append(float(mag.std()))

    return np.asarray(feats, dtype=np.float32)


def build_feature_matrix(
    image_names: Iterable[str], image_dir: Path
) -> Tuple[np.ndarray, list[str]]:
    features = []
    failed: list[str] = []
    for name in image_names:
        path = image_dir / name
        try:
            feat = extract_features(path)
        except Exception as exc:  # pragma: no cover - defensive
            failed.append(f"{name}: {exc}")
            feat = np.zeros_like(extract_features(image_dir / image_names[0]))
        features.append(feat)
    return np.vstack(features), failed


def train_and_predict(
    train_df: pd.DataFrame, test_df: pd.DataFrame, base_dir: Path
) -> pd.DataFrame:
    train_images = base_dir / "Train" / "Image"
    test_images = base_dir / "Test" / "Image"

    X, failed_train = build_feature_matrix(train_df["Image_name"], train_images)
    if failed_train:
        print("Warnings while loading training images:", *failed_train, sep="\n- ", file=sys.stderr)
    y = train_df["Label"].to_numpy()

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=400, class_weight="balanced")),
        ]
    )
    pipeline.fit(X_train, y_train)

    # Tune probability threshold for the minority "Political" class on the hold-out split.
    proba = pipeline.predict_proba(X_val)
    classes = list(pipeline.classes_)
    pos_label = "Political"
    neg_label = [c for c in classes if c != pos_label][0]
    pos_idx = classes.index(pos_label)

    best_thresh = 0.5
    best_f1 = -1.0
    for thresh in np.linspace(0.3, 0.7, 17):
        preds = np.where(proba[:, pos_idx] >= thresh, pos_label, neg_label)
        f1 = f1_score(y_val, preds, average="macro")
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = float(thresh)

    print(f"Validation macro F1: {best_f1:.4f} at threshold {best_thresh:.2f}")

    # Refit on full training data
    pipeline.fit(X, y)

    X_test, failed_test = build_feature_matrix(test_df["Image_name"], test_images)
    if failed_test:
        print("Warnings while loading test images:", *failed_test, sep="\n- ", file=sys.stderr)

    test_proba = pipeline.predict_proba(X_test)[:, pos_idx]
    test_preds = np.where(test_proba >= best_thresh, pos_label, neg_label)

    submission = pd.DataFrame(
        {
            "Image_name": test_df["Image_name"],
            "Label": test_preds,
        }
    )
    return submission


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Baseline classifier for PoliMemeDecode.")
    parser.add_argument(
        "--output",
        default="submission.csv",
        help="Path to write predictions CSV (default: submission.csv)",
    )
    parser.add_argument(
        "--base-dir",
        default=".",
        help="Dataset root containing Train/ and Test/ folders (default: current directory)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_dir = Path(args.base_dir)

    train_csv = base_dir / "Train" / "Train.csv"
    test_csv = base_dir / "Test" / "Test.csv"
    if not train_csv.exists() or not test_csv.exists():
        raise SystemExit("Could not find Train/Train.csv or Test/Test.csv; check --base-dir path.")

    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    submission = train_and_predict(train_df, test_df, base_dir)
    submission.to_csv(args.output, index=False)
    print(f"Wrote predictions to {args.output}")


if __name__ == "__main__":
    main()
