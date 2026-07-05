from __future__ import annotations

import json
import os
import random
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup


CODEBERT_DIR = "/path/software_defects/CWEMAP_data/CodeBERT"


TRAIN_INPUT_JSON = "/path/software_defects/CWEMAP_data/dataset_name/dataset_name_train_set.json"
VALID_INPUT_JSON = "/path/software_defects/CWEMAP_data/dataset_name/dataset_name_vali_set.json"
TEST_INPUT_JSON = "/path/software_defects/CWEMAP_data/dataset_name/dataset_name_test_set.json"

RAW_INPUT_JSON = ""

OUTPUT_DIR = "/path/software_defects/CWEMAP_data/dataset_name/treevul_baseline_results"

RANDOM_SEED = 42
TRAIN_RATIO = 0.80
VALID_RATIO = 0.10
TEST_RATIO = 0.10

MAX_SEQ_LEN = 512
BATCH_SIZE = 8
EVAL_BATCH_SIZE = 16
EPOCHS = 10
PATIENCE = 3
DROPOUT = 0.20
CODEBERT_LR = 2e-5
HEAD_LR = 1e-4
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.10
GRAD_CLIP_NORM = 1.0

USE_CUDA = True
NUM_WORKERS = 0
SAVE_NORMALIZED_SPLITS = True

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_json_any(path: str | Path) -> List[Dict[str, Any]]:
    """Read either a JSON list or JSONL file."""
    p = Path(path)
    if not p.exists():
        return []
    text = p.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text.startswith("["):
        obj = json.loads(text)
        if not isinstance(obj, list):
            raise ValueError(f"Top-level JSON must be a list: {p}")
        return [x for x in obj if isinstance(x, dict)]

    rows: List[Dict[str, Any]] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at {p}:{line_no}: {exc}") from exc
        if isinstance(obj, dict):
            rows.append(obj)
    return rows


def write_json(path: str | Path, obj: Any) -> None:
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: str | Path, rows: Iterable[Dict[str, Any]]) -> None:
    with Path(path).open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def safe_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return "\n".join(str(x) for x in value if str(x).strip())
    return str(value)


def normalize_cwe_label(value: Any) -> Optional[str]:
    """Normalize a CWE label to a canonical string such as CWE-79."""
    if value is None:
        return None
    if isinstance(value, (list, tuple, set)):
        for item in value:
            normalized = normalize_cwe_label(item)
            if normalized:
                return normalized
        return None
    if isinstance(value, (bool, np.bool_)):
        return None
    if isinstance(value, (int, np.integer)):
        if int(value) > 0:
            return f"CWE-{int(value)}"
        return None
    if isinstance(value, float):
        return None

    text = str(value).strip()
    if not text:
        return None

    match = re.search(r"cwe[-_ ]?0*(\d+)", text, flags=re.IGNORECASE)
    if match:
        return f"CWE-{int(match.group(1))}"
    if re.fullmatch(r"\d+", text):
        return f"CWE-{int(text)}"
    return text.strip().upper().replace(" ", "-")


def build_cwe_label_mapping(rows: Sequence[Dict[str, Any]]) -> Tuple[Dict[str, int], List[str]]:
    labels = sorted({row["cwe_label"] for row in rows if row.get("cwe_label")}, key=lambda x: x)
    mapping = {label: idx for idx, label in enumerate(labels)}
    return mapping, labels


def assign_cwe_targets(rows: Sequence[Dict[str, Any]], mapping: Dict[str, int]) -> None:
    for row in rows:
        label = row.get("cwe_label")
        if label is None:
            raise ValueError("Every example must have a normalized CWE label.")
        row["target"] = mapping[label]


def make_example_id(record: Dict[str, Any], row_index: int) -> str:
    for key in ["idx", "commit_id", "hash", "big_vul_idx"]:
        val = record.get(key)
        if val is not None and str(val).strip():
            return str(val)
    return f"row_{row_index}"

# Data preparation for the provided schema

def normalize_record(record: Dict[str, Any], row_index: int) -> Optional[Dict[str, Any]]:
    """
    Normalize one dataset row.

    Input feature is func only. The ground-truth target is a CWE label,
    but it is never concatenated into model_input.
    """
    func = safe_text(record.get("func", "")).strip()
    cwe_label = (
        normalize_cwe_label(record.get("cwe"))
        or normalize_cwe_label(record.get("cwe_id"))
        or normalize_cwe_label(record.get("label"))
        or normalize_cwe_label(record.get("target"))
    )

    if not func or not cwe_label:
        return None

    return {
        "example_id": make_example_id(record, row_index),
        "project": safe_text(record.get("project", "")),
        "commit_id": safe_text(record.get("commit_id", "")),
        "cwe": safe_text(record.get("cwe", "")),
        "cwe_label": cwe_label,
        "big_vul_idx": record.get("big_vul_idx", ""),
        "idx": record.get("idx", row_index),
        "hash": safe_text(record.get("hash", "")),
        "func": func,
        "target": None,
        "model_input": func,
    }


def normalize_records(rows: Sequence[Dict[str, Any]], split_name: str) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    skipped = 0
    for i, row in enumerate(rows):
        item = normalize_record(row, i)
        if item is None:
            skipped += 1
            continue
        normalized.append(item)
    print(f"{split_name}: {len(normalized)} kept, {skipped} skipped because func/target was missing or invalid.")
    return normalized


def stratified_80_10_10_split(rows: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    labels = [r.get("cwe_label") or r.get("target") for r in rows]
    test_size = TEST_RATIO
    valid_from_train_size = VALID_RATIO / (TRAIN_RATIO + VALID_RATIO)

    try:
        train_valid, test = train_test_split(
            rows,
            test_size=test_size,
            random_state=RANDOM_SEED,
            stratify=labels,
        )
        train_valid_labels = [r.get("cwe_label") or r.get("target") for r in train_valid]
        train, valid = train_test_split(
            train_valid,
            test_size=valid_from_train_size,
            random_state=RANDOM_SEED,
            stratify=train_valid_labels,
        )
    except ValueError:
        rng = random.Random(RANDOM_SEED)
        shuffled = list(rows)
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_train = int(n * TRAIN_RATIO)
        n_valid = int(n * VALID_RATIO)
        train = shuffled[:n_train]
        valid = shuffled[n_train:n_train + n_valid]
        test = shuffled[n_train + n_valid:]

    return list(train), list(valid), list(test)


def load_dataset_splits() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[str]]:
    train_path = Path(TRAIN_INPUT_JSON)
    valid_path = Path(VALID_INPUT_JSON)
    test_path = Path(TEST_INPUT_JSON)

    if train_path.exists() and valid_path.exists() and test_path.exists():
        print("Loading existing TreeVul train/validation/test files.")
        train = normalize_records(read_json_any(train_path), "training")
        valid = normalize_records(read_json_any(valid_path), "validation")
        test = normalize_records(read_json_any(test_path), "testing")
    elif RAW_INPUT_JSON and Path(RAW_INPUT_JSON).exists():
        print("Existing split files were not found. Loading RAW_INPUT_JSON and creating 80/10/10 split.")
        rows = normalize_records(read_json_any(RAW_INPUT_JSON), "raw")
        train, valid, test = stratified_80_10_10_split(rows)
    else:
        raise FileNotFoundError(
            "Dataset files were not found. Please check TRAIN_INPUT_JSON, VALID_INPUT_JSON, "
            "TEST_INPUT_JSON, or set RAW_INPUT_JSON in the configuration block."
        )

    if not train or not valid or not test:
        raise RuntimeError("Loaded splits are empty after preprocessing.")

    label_mapping, class_names = build_cwe_label_mapping(train + valid + test)
    assign_cwe_targets(train, label_mapping)
    assign_cwe_targets(valid, label_mapping)
    assign_cwe_targets(test, label_mapping)
    return train, valid, test, class_names


def label_distribution(rows: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for r in rows:
        label = str(r.get("cwe_label") or r.get("target"))
        counts[label] = counts.get(label, 0) + 1
    return counts

class TreeVulFunctionDataset(Dataset):
    def __init__(self, rows: Sequence[Dict[str, Any]], tokenizer: Any):
        self.rows = list(rows)
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = self.rows[index]
        encoded = self.tokenizer(
            row["model_input"],
            truncation=True,
            padding="max_length",
            max_length=MAX_SEQ_LEN,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "label": torch.tensor(int(row["target"]), dtype=torch.long),
            "meta": row,
        }


def collate_batch(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "input_ids": torch.stack([x["input_ids"] for x in batch], dim=0),
        "attention_mask": torch.stack([x["attention_mask"] for x in batch], dim=0),
        "labels": torch.stack([x["label"] for x in batch], dim=0),
        "metas": [x["meta"] for x in batch],
    }


class TreeVulBaselineModel(nn.Module):
    """CodeBERT multi-class CWE classifier over the func field."""

    def __init__(self, codebert_dir: str, num_classes: int, dropout: float = DROPOUT):
        super().__init__()
        self.codebert = AutoModel.from_pretrained(codebert_dir)
        hidden_size = int(self.codebert.config.hidden_size)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        outputs = self.codebert(input_ids=input_ids, attention_mask=attention_mask)
        if getattr(outputs, "pooler_output", None) is not None:
            pooled = outputs.pooler_output
        else:
            pooled = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(pooled)
        result: Dict[str, Any] = {"logits": logits}
        if labels is not None:
            result["loss"] = nn.functional.cross_entropy(logits, labels)
        return result

# Training and evaluation

def move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    moved = dict(batch)
    for key in ["input_ids", "attention_mask", "labels"]:
        moved[key] = moved[key].to(device)
    return moved


def build_dataloaders(train: List[Dict[str, Any]], valid: List[Dict[str, Any]], test: List[Dict[str, Any]], tokenizer: Any) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_loader = DataLoader(
        TreeVulFunctionDataset(train, tokenizer),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=collate_batch,
    )
    valid_loader = DataLoader(
        TreeVulFunctionDataset(valid, tokenizer),
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_batch,
    )
    test_loader = DataLoader(
        TreeVulFunctionDataset(test, tokenizer),
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_batch,
    )
    return train_loader, valid_loader, test_loader


def train_one_epoch(model: TreeVulBaselineModel, loader: DataLoader, optimizer: torch.optim.Optimizer, scheduler: Any, device: torch.device) -> float:
    model.train()
    losses: List[float] = []
    progress = tqdm(loader, desc="train", leave=False)
    for batch in progress:
        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        out = model(batch["input_ids"], batch["attention_mask"], labels=batch["labels"])
        loss = out["loss"]
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        losses.append(float(loss.item()))
        progress.set_postfix(loss=float(np.mean(losses)))
    return float(np.mean(losses)) if losses else 0.0


@torch.no_grad()
def predict(model: TreeVulBaselineModel, loader: DataLoader, device: torch.device, class_names: Sequence[str]) -> List[Dict[str, Any]]:
    model.eval()
    rows: List[Dict[str, Any]] = []
    for batch in tqdm(loader, desc="predict", leave=False):
        metas = batch["metas"]
        batch = move_batch_to_device(batch, device)
        out = model(batch["input_ids"], batch["attention_mask"])
        logits = out["logits"]
        probs = torch.softmax(logits, dim=-1)
        pred_labels = torch.argmax(probs, dim=-1)

        for i, meta in enumerate(metas):
            true_label = int(meta["target"])
            pred_label = int(pred_labels[i].detach().cpu().item())
            prob_vector = {name: float(probs[i, idx].detach().cpu().item()) for idx, name in enumerate(class_names)}
            pred_cwe = class_names[pred_label] if pred_label < len(class_names) else str(pred_label)
            rows.append({
                "example_id": meta.get("example_id", ""),
                "project": meta.get("project", ""),
                "commit_id": meta.get("commit_id", ""),
                "hash": meta.get("hash", ""),
                "idx": meta.get("idx", ""),
                "big_vul_idx": meta.get("big_vul_idx", ""),
                "cwe": meta.get("cwe", ""),
                "true_cwe_label": meta.get("cwe_label", ""),
                "true_target": true_label,
                "pred_target": pred_label,
                "pred_cwe_label": pred_cwe,
                "pred_confidence": prob_vector.get(pred_cwe, 0.0),
                "probabilities": prob_vector,
                "correct": int(true_label == pred_label),
                "func_preview": safe_text(meta.get("func", ""))[:300],
            })
    return rows


def compute_metrics(preds: Sequence[Dict[str, Any]], class_names: Sequence[str]) -> Dict[str, Any]:
    if not preds:
        return {
            "num_predictions": 0,
            "num_classes": len(class_names),
            "Accuracy": 0.0,
            "Precision_macro": 0.0,
            "Recall_macro": 0.0,
            "F1_macro": 0.0,
            "Precision_weighted": 0.0,
            "Recall_weighted": 0.0,
            "F1_weighted": 0.0,
            "F1_micro": 0.0,
            "MCC": 0.0,
            "confusion_matrix_labels": class_names,
            "confusion_matrix": [],
        }

    y_true = [int(p["true_target"]) for p in preds]
    y_pred = [int(p["pred_target"]) for p in preds]
    labels = list(range(len(class_names)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    precision_macro = float(precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))
    recall_macro = float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))
    f1_macro = float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))
    precision_weighted = float(precision_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0))
    recall_weighted = float(recall_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0))
    f1_weighted = float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0))
    f1_micro = float(f1_score(y_true, y_pred, labels=labels, average="micro", zero_division=0))
    mcc = float(matthews_corrcoef(y_true, y_pred)) if len(set(y_true) | set(y_pred)) > 1 else 0.0

    return {
        "num_predictions": len(preds),
        "num_classes": len(class_names),
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision": precision_macro,
        "Recall": recall_macro,
        "F1_score": f1_macro,
        "Weighted_F1": f1_weighted,
        "Macro_F1": f1_macro,
        "Precision_macro": precision_macro,
        "Recall_macro": recall_macro,
        "F1_macro": f1_macro,
        "Precision_weighted": precision_weighted,
        "Recall_weighted": recall_weighted,
        "F1_weighted": f1_weighted,
        "F1_micro": f1_micro,
        "MCC": mcc,
        "AUC": None,
        "confusion_matrix_labels": class_names,
        "confusion_matrix": cm.tolist(),
    }


def metrics_to_dataframe(metrics: Dict[str, Any]) -> pd.DataFrame:
    scalar_items = []
    for key, value in metrics.items():
        if key in {"confusion_matrix", "confusion_matrix_labels"}:
            continue
        scalar_items.append({"metric": key, "value": value})
    return pd.DataFrame(scalar_items)


def confusion_matrix_dataframe(metrics: Dict[str, Any]) -> pd.DataFrame:
    cm = metrics.get("confusion_matrix", [])
    labels = metrics.get("confusion_matrix_labels", [])
    if not cm:
        cm = [[0 for _ in labels] for _ in labels]
    return pd.DataFrame(cm, index=[f"true_{label}" for label in labels], columns=[f"pred_{label}" for label in labels])


def save_split_files(out_dir: Path, train: List[Dict[str, Any]], valid: List[Dict[str, Any]], test: List[Dict[str, Any]]) -> None:
    if not SAVE_NORMALIZED_SPLITS:
        return
    for name, rows in [("training", train), ("validation", valid), ("testing", test)]:
        write_json(out_dir / f"{name}.json", rows)
        write_jsonl(out_dir / f"{name}.jsonl", rows)
        pd.DataFrame(rows).to_csv(out_dir / f"{name}.csv", index=False)


def save_outputs(
    out_dir: Path,
    train: List[Dict[str, Any]],
    valid: List[Dict[str, Any]],
    test: List[Dict[str, Any]],
    history: List[Dict[str, Any]],
    valid_preds: List[Dict[str, Any]],
    test_preds: List[Dict[str, Any]],
    valid_metrics: Dict[str, Any],
    test_metrics: Dict[str, Any],
    config: Dict[str, Any],
    class_names: Sequence[str],
) -> None:
    ensure_dir(out_dir)
    save_split_files(out_dir, train, valid, test)

    write_json(out_dir / "run_config.json", config)
    write_json(out_dir / "training_history.json", history)
    pd.DataFrame(history).to_csv(out_dir / "training_history.csv", index=False)

    write_json(out_dir / "validation_predictions.json", valid_preds)
    write_jsonl(out_dir / "validation_predictions.jsonl", valid_preds)
    pd.DataFrame(valid_preds).to_csv(out_dir / "validation_predictions.csv", index=False)

    write_json(out_dir / "testing_predictions.json", test_preds)
    write_jsonl(out_dir / "testing_predictions.jsonl", test_preds)
    pd.DataFrame(test_preds).to_csv(out_dir / "testing_predictions.csv", index=False)

    write_json(out_dir / "validation_metrics.json", valid_metrics)
    write_json(out_dir / "testing_metrics.json", test_metrics)
    metrics_to_dataframe(valid_metrics).to_csv(out_dir / "validation_metrics.csv", index=False)
    metrics_to_dataframe(test_metrics).to_csv(out_dir / "testing_metrics.csv", index=False)
    confusion_matrix_dataframe(valid_metrics).to_csv(out_dir / "validation_confusion_matrix.csv")
    confusion_matrix_dataframe(test_metrics).to_csv(out_dir / "testing_confusion_matrix.csv")

    split_summary_rows = []
    for split_name, rows in [("training", train), ("validation", valid), ("testing", test)]:
        counts = label_distribution(rows)
        row = {"split": split_name, "num_examples": len(rows)}
        for cls_name in class_names:
            row[f"label_{cls_name}"] = counts.get(cls_name, 0)
        split_summary_rows.append(row)
    split_summary = pd.DataFrame(split_summary_rows)
    split_summary.to_csv(out_dir / "split_summary.csv", index=False)
    write_json(out_dir / "split_summary.json", split_summary.to_dict(orient="records"))

    xlsx_path = out_dir / "treevul_baseline_results.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        metrics_to_dataframe(test_metrics).to_excel(writer, sheet_name="test_metrics", index=False)
        metrics_to_dataframe(valid_metrics).to_excel(writer, sheet_name="valid_metrics", index=False)
        confusion_matrix_dataframe(test_metrics).to_excel(writer, sheet_name="test_confusion")
        confusion_matrix_dataframe(valid_metrics).to_excel(writer, sheet_name="valid_confusion")
        pd.DataFrame(test_preds).to_excel(writer, sheet_name="test_predictions", index=False)
        pd.DataFrame(valid_preds).to_excel(writer, sheet_name="valid_predictions", index=False)
        pd.DataFrame(history).to_excel(writer, sheet_name="training_history", index=False)
        split_summary.to_excel(writer, sheet_name="split_summary", index=False)
        pd.DataFrame([config]).to_excel(writer, sheet_name="run_config", index=False)

def build_optimizer_and_scheduler(model: TreeVulBaselineModel, train_loader: DataLoader) -> Tuple[torch.optim.Optimizer, Any]:
    codebert_params = []
    head_params = []
    for name, param in model.named_parameters():
        if name.startswith("codebert."):
            codebert_params.append(param)
        else:
            head_params.append(param)

    optimizer = torch.optim.AdamW(
        [
            {"params": codebert_params, "lr": CODEBERT_LR},
            {"params": head_params, "lr": HEAD_LR},
        ],
        weight_decay=WEIGHT_DECAY,
    )
    total_steps = max(1, len(train_loader) * EPOCHS)
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    return optimizer, scheduler


def main() -> None:
    set_seed(RANDOM_SEED)
    out_dir = ensure_dir(OUTPUT_DIR)

    config = {
        "CODEBERT_DIR": CODEBERT_DIR,
        "TRAIN_INPUT_JSON": TRAIN_INPUT_JSON,
        "VALID_INPUT_JSON": VALID_INPUT_JSON,
        "TEST_INPUT_JSON": TEST_INPUT_JSON,
        "RAW_INPUT_JSON": RAW_INPUT_JSON,
        "OUTPUT_DIR": OUTPUT_DIR,
        "RANDOM_SEED": RANDOM_SEED,
        "TRAIN_RATIO": TRAIN_RATIO,
        "VALID_RATIO": VALID_RATIO,
        "TEST_RATIO": TEST_RATIO,
        "MAX_SEQ_LEN": MAX_SEQ_LEN,
        "BATCH_SIZE": BATCH_SIZE,
        "EVAL_BATCH_SIZE": EVAL_BATCH_SIZE,
        "EPOCHS": EPOCHS,
        "PATIENCE": PATIENCE,
        "DROPOUT": DROPOUT,
        "CODEBERT_LR": CODEBERT_LR,
        "HEAD_LR": HEAD_LR,
        "WEIGHT_DECAY": WEIGHT_DECAY,
        "WARMUP_RATIO": WARMUP_RATIO,
        "input_feature": "func only",
        "label_field": "cwe_label",
        "leakage_prevention": "cwe_label is never tokenized or concatenated into model_input",
    }
    write_json(out_dir / "run_config.json", config)

    print("=" * 90)
    print("TREEVUL baseline: CodeBERT multi-class CWE classifier over func")
    print("=" * 90)
    print(f"CodeBERT directory: {CODEBERT_DIR}")
    print(f"Training file     : {TRAIN_INPUT_JSON}")
    print(f"Validation file   : {VALID_INPUT_JSON}")
    print(f"Testing file      : {TEST_INPUT_JSON}")
    print(f"Output directory  : {OUTPUT_DIR}")
    print("Input feature     : func")
    print("Label field       : cwe_label")
    print("=" * 90)

    train, valid, test, class_names = load_dataset_splits()
    if not train or not valid or not test:
        raise RuntimeError(f"Empty split after preprocessing: train={len(train)}, valid={len(valid)}, test={len(test)}")

    config["num_classes"] = len(class_names)
    config["class_names"] = class_names
    write_json(out_dir / "run_config.json", config)

    print(f"Train/Validation/Test: {len(train)} / {len(valid)} / {len(test)}")
    print(f"Classes ({len(class_names)}): {class_names}")
    print(f"Train label distribution     : {label_distribution(train)}")
    print(f"Validation label distribution: {label_distribution(valid)}")
    print(f"Test label distribution      : {label_distribution(test)}")

    device = torch.device("cuda" if USE_CUDA and torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(CODEBERT_DIR)
    train_loader, valid_loader, test_loader = build_dataloaders(train, valid, test, tokenizer)

    model = TreeVulBaselineModel(CODEBERT_DIR, num_classes=len(class_names), dropout=DROPOUT).to(device)
    optimizer, scheduler = build_optimizer_and_scheduler(model, train_loader)

    best_valid_macro_f1 = -1.0
    bad_epochs = 0
    history: List[Dict[str, Any]] = []
    best_model_path = out_dir / "best_treevul_baseline_model.pt"

    for epoch in range(1, EPOCHS + 1):
        start = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, device)
        valid_preds = predict(model, valid_loader, device, class_names)
        valid_metrics = compute_metrics(valid_preds, class_names)
        monitor = float(valid_metrics["Macro_F1"])

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "valid_accuracy": valid_metrics["Accuracy"],
            "valid_precision": valid_metrics["Precision"],
            "valid_recall": valid_metrics["Recall"],
            "valid_f1": valid_metrics["F1_score"],
            "valid_weighted_f1": valid_metrics["Weighted_F1"],
            "valid_macro_f1": valid_metrics["Macro_F1"],
            "valid_mcc": valid_metrics["MCC"],
            "valid_auc": valid_metrics["AUC"],
            "seconds": round(time.time() - start, 2),
        }
        history.append(row)
        print(json.dumps(row, indent=2))

        if monitor > best_valid_macro_f1:
            best_valid_macro_f1 = monitor
            bad_epochs = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": config,
                "best_epoch": epoch,
                "best_valid_macro_f1": best_valid_macro_f1,
            }, best_model_path)
            print(f"Saved new best model to {best_model_path}")
        else:
            bad_epochs += 1
            print(f"No improvement. bad_epochs={bad_epochs}/{PATIENCE}")
            if bad_epochs >= PATIENCE:
                print("Early stopping triggered.")
                break

    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    valid_preds = predict(model, valid_loader, device, class_names)
    test_preds = predict(model, test_loader, device, class_names)
    valid_metrics = compute_metrics(valid_preds, class_names)
    test_metrics = compute_metrics(test_preds, class_names)

    save_outputs(out_dir, train, valid, test, history, valid_preds, test_preds, valid_metrics, test_metrics, config, class_names)

    print("\nValidation metrics:")
    print(json.dumps(valid_metrics, indent=2))
    print("\nTesting metrics:")
    print(json.dumps(test_metrics, indent=2))
    print("\nSaved outputs to:")
    print(out_dir)
    print(f"Best model: {best_model_path}")
    print(f"Excel     : {out_dir / 'treevul_baseline_results.xlsx'}")
    print(f"JSON      : {out_dir / 'testing_metrics.json'}")
    print(f"CSV       : {out_dir / 'testing_metrics.csv'}")


if __name__ == "__main__":
    main()
