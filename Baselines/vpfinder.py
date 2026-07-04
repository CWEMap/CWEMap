#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
import random
import re
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    precision_recall_fscore_support,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup


# =============================================================================
# EDIT HERE ONLY
# =============================================================================

# Local pretrained model folders.
BERT_DIR = "/path/software_defects/CWEMAP_data/BERT"
CODEBERT_DIR = "/path/software_defects/CWEMAP_data/CodeBERT"


TRAIN_INPUT_JSON = "/path/dataset_name/training.json"
INPUT = TRAIN_INPUT_JSON
VALID_INPUT_JSON = "/path/dataset_name/validation.json"
TEST_INPUT_JSON = "/path/dataset_name/testing.json"

OUTPUT = "/path/dataset_name/dataset_name_results"
OUTPUT_DIR = OUTPUT

TRAIN_RATIO = 0.80
VALID_RATIO = 0.10
TEST_RATIO = 0.10
RANDOM_SEED = 42


NUM_EPOCHS = 10
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 1
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.10
MAX_GRAD_NORM = 1.0
EARLY_STOPPING_PATIENCE = 3
TYPE_LOSS_WEIGHT = 1.0
DROPOUT = 0.30
NUM_ATTENTION_HEADS = 8


MAX_TEXT_LENGTH = 512
MAX_CODE_LENGTH = 512


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_AMP = True  # mixed precision on CUDA
NUM_WORKERS = 2
FREEZE_ENCODERS = False 


BUILD_CWE_LABEL_MAP_FROM = "all" 
ADD_OTHER_CWE_LABEL = True
OTHER_CWE_LABEL = "CWE-OTHER"


MIN_CWE_SUPPORT = 1


MAX_ROWS_DEBUG: Optional[int] = None


# =============================================================================
# Utilities
# =============================================================================

CWE_RE = re.compile(r"CWE[-_ ]?(\d+)", re.IGNORECASE)
CVE_RE = re.compile(r"CVE-\d{4}-\d+", re.IGNORECASE)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def exists_nonempty(path: str | Path) -> bool:
    try:
        p = Path(path)
        return p.exists() and p.is_file() and p.stat().st_size > 0
    except Exception:
        return False


def read_json_or_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    p = Path(path)
    text = p.read_text(encoding="utf-8", errors="replace").strip()
    if not text:
        return []

    if text.startswith("["):
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError(f"Expected a JSON list in {p}")
        return [x for x in data if isinstance(x, dict)]

    if text.startswith("{"):
        obj = json.loads(text)
        if isinstance(obj, list):
            return [x for x in obj if isinstance(x, dict)]
        if isinstance(obj, dict):
            for key in ["data", "samples", "rows", "items", "records"]:
                if isinstance(obj.get(key), list):
                    return [x for x in obj[key] if isinstance(x, dict)]
            # If it is a single record, return it as one-row dataset.
            return [obj]

    rows = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return [x for x in rows if isinstance(x, dict)]


def write_json(path: str | Path, obj: Any) -> None:
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: str | Path, rows: Iterable[Dict[str, Any]]) -> None:
    with Path(path).open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def get_first(row: Dict[str, Any], keys: Sequence[str], default: Any = "") -> Any:
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
        # case-insensitive fallback
        for k, v in row.items():
            if k.lower() == key.lower() and v is not None:
                return v
    return default


def to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return "\n".join(str(x) for x in value if str(x).strip())
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def normalize_cwe(value: Any) -> str:
    text = to_text(value).strip()
    if not text:
        return "NO_CWE"
    lowered = text.lower()
    if lowered in {"none", "null", "nan", "no_cwe", "nocwe", "n/a", "na", "-"}:
        return "NO_CWE"
    if "noinfo" in lowered or "other" == lowered or "unknown" in lowered:
        return "NO_CWE"
    m = CWE_RE.search(text)
    if m:
        return f"CWE-{int(m.group(1))}"
    if text.isdigit():
        return f"CWE-{int(text)}"
    return "NO_CWE"


def normalize_cve(value: Any) -> str:
    text = to_text(value).strip()
    m = CVE_RE.search(text)
    return m.group(0).upper() if m else text


def infer_binary_label(row: Dict[str, Any], cwe: str, cve: str) -> int:
    explicit = get_first(
        row,
        [
            "binary_label", "vul_label", "vulnerability_label", "is_vulnerable",
            "is_vul", "flag", "target", "label", "y", "class",
        ],
        default=None,
    )
    if explicit is not None:
        if isinstance(explicit, bool):
            return int(explicit)
        if isinstance(explicit, (int, float)) and not isinstance(explicit, bool):
            return 1 if int(explicit) == 1 else 0
        s = str(explicit).strip().lower()
        if s in {"1", "true", "yes", "y", "vul", "vulnerable", "vr", "security", "positive"}:
            return 1
        if s in {"0", "false", "no", "n", "non-vul", "non_vul", "nvul", "nvr", "negative", "benign"}:
            return 0


    if cwe != "NO_CWE":
        return 1
    if CVE_RE.search(cve or ""):
        return 1
    return 0


def extract_added_deleted_from_diff(diff_patch: str) -> Tuple[str, str]:
    added: List[str] = []
    deleted: List[str] = []
    for line in (diff_patch or "").splitlines():
        if line.startswith("+++") or line.startswith("---") or line.startswith("@@"):
            continue
        if line.startswith("+"):
            added.append(line[1:])
        elif line.startswith("-"):
            deleted.append(line[1:])
    return "\n".join(added), "\n".join(deleted)


def normalize_row(row: Dict[str, Any]) -> Dict[str, Any]:
    commit_id = to_text(get_first(row, ["Commit_id", "commit_id", "commit", "sha", "hash", "commit_hash"], ""))
    commit_message = to_text(get_first(row, ["commit_message", "Commit_message", "message", "msg", "commit_msg"], ""))
    diff_patch = to_text(get_first(row, ["diff_patch", "patch", "diff", "Patch", "code_patch"], ""))
    added_code = to_text(get_first(row, ["added_code", "patch_add", "add_code", "ADD_DIFF", "added", "patch_added"], ""))
    deleted_code = to_text(get_first(row, ["deleted_code", "removed_code", "patch_del", "del_code", "REM_DIFF", "deleted", "patch_deleted"], ""))
    if (not added_code.strip() or not deleted_code.strip()) and diff_patch.strip():
        add_from_diff, del_from_diff = extract_added_deleted_from_diff(diff_patch)
        if not added_code.strip():
            added_code = add_from_diff
        if not deleted_code.strip():
            deleted_code = del_from_diff

    cve_id = normalize_cve(get_first(row, ["cve_id", "CVE_ID", "cve", "cve_list"], ""))
    cve_description = to_text(get_first(row, ["cve_description", "CVE_description", "description", "text", "bug_report", "issue_description", "issue_text"], ""))
    cwe_id = normalize_cwe(get_first(row, ["cwe_id", "CWE_ID", "cwe", "cwe_list"], ""))
    binary_label = infer_binary_label(row, cwe_id, cve_id)

    if binary_label == 0:
        cwe_id = "NO_CWE"

    return {
        "commit_id": commit_id,
        "commit_message": commit_message,
        "diff_patch": diff_patch,
        "added_code": added_code,
        "deleted_code": deleted_code,
        "cve_id": cve_id,
        "cve_description": cve_description,
        "cwe_id": cwe_id,
        "binary_label": int(binary_label),
        "original_row": row,
    }


def stratify_key(row: Dict[str, Any]) -> str:
    if int(row.get("binary_label", 0)) == 0:
        return "NON_VUL"
    return normalize_cwe(row.get("cwe_id"))


def can_stratify(labels: Sequence[str], min_count: int = 2) -> bool:
    counts = Counter(labels)
    return len(counts) > 1 and all(v >= min_count for v in counts.values())


def split_80_10_10(rows: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    if not math.isclose(TRAIN_RATIO + VALID_RATIO + TEST_RATIO, 1.0, abs_tol=1e-6):
        raise ValueError("TRAIN_RATIO + VALID_RATIO + TEST_RATIO must equal 1.0")

    labels = [stratify_key(r) for r in rows]
    stratify = labels if can_stratify(labels, min_count=2) else None

    train_rows, temp_rows = train_test_split(
        rows,
        test_size=(VALID_RATIO + TEST_RATIO),
        random_state=RANDOM_SEED,
        shuffle=True,
        stratify=stratify,
    )

    temp_labels = [stratify_key(r) for r in temp_rows]
    stratify_temp = temp_labels if can_stratify(temp_labels, min_count=2) else None
    valid_fraction_of_temp = VALID_RATIO / (VALID_RATIO + TEST_RATIO)

    valid_rows, test_rows = train_test_split(
        temp_rows,
        test_size=(1.0 - valid_fraction_of_temp),
        random_state=RANDOM_SEED,
        shuffle=True,
        stratify=stratify_temp,
    )
    return train_rows, valid_rows, test_rows


def save_split_files(out_dir: Path, split_name: str, rows: List[Dict[str, Any]]) -> None:
    write_json(out_dir / f"{split_name}.json", rows)
    write_jsonl(out_dir / f"{split_name}.jsonl", rows)
    pd.DataFrame([{k: v for k, v in r.items() if k != "original_row"} for r in rows]).to_csv(out_dir / f"{split_name}.csv", index=False)


def build_cwe_label_map(train: List[Dict[str, Any]], valid: List[Dict[str, Any]], test: List[Dict[str, Any]]) -> Tuple[Dict[str, int], Dict[int, str], Dict[str, str]]:
    source_rows = train if BUILD_CWE_LABEL_MAP_FROM.lower() == "train" else (train + valid + test)
    train_counts = Counter(normalize_cwe(r.get("cwe_id")) for r in train if int(r.get("binary_label", 0)) == 1 and normalize_cwe(r.get("cwe_id")) != "NO_CWE")

    all_cwes = sorted({
        normalize_cwe(r.get("cwe_id"))
        for r in source_rows
        if int(r.get("binary_label", 0)) == 1 and normalize_cwe(r.get("cwe_id")) != "NO_CWE"
    })

    cwe_remap: Dict[str, str] = {}
    kept = []
    for cwe in all_cwes:
        if train_counts.get(cwe, 0) >= MIN_CWE_SUPPORT:
            kept.append(cwe)
            cwe_remap[cwe] = cwe
        else:
            cwe_remap[cwe] = OTHER_CWE_LABEL

    if ADD_OTHER_CWE_LABEL and (OTHER_CWE_LABEL not in kept):
        kept.append(OTHER_CWE_LABEL)

    kept = sorted(set(kept), key=lambda x: (x == OTHER_CWE_LABEL, x))
    cwe_to_id = {cwe: i for i, cwe in enumerate(kept)}
    id_to_cwe = {i: cwe for cwe, i in cwe_to_id.items()}
    return cwe_to_id, id_to_cwe, cwe_remap


def map_cwe_for_training(cwe: str, cwe_to_id: Dict[str, int], cwe_remap: Dict[str, str]) -> str:
    cwe = normalize_cwe(cwe)
    if cwe == "NO_CWE":
        return "NO_CWE"
    mapped = cwe_remap.get(cwe, cwe)
    if mapped in cwe_to_id:
        return mapped
    if ADD_OTHER_CWE_LABEL and OTHER_CWE_LABEL in cwe_to_id:
        return OTHER_CWE_LABEL
    return "NO_CWE"


# =============================================================================
# Dataset
# =============================================================================

class VPFinderDataset(Dataset):
    def __init__(
        self,
        rows: List[Dict[str, Any]],
        text_tokenizer: AutoTokenizer,
        code_tokenizer: AutoTokenizer,
        cwe_to_id: Dict[str, int],
        cwe_remap: Dict[str, str],
    ) -> None:
        self.rows = rows
        self.text_tokenizer = text_tokenizer
        self.code_tokenizer = code_tokenizer
        self.cwe_to_id = cwe_to_id
        self.cwe_remap = cwe_remap

    def __len__(self) -> int:
        return len(self.rows)

    def _tok_text(self, text: str) -> Dict[str, torch.Tensor]:
        enc = self.text_tokenizer(
            text or "",
            max_length=MAX_TEXT_LENGTH,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {k: v.squeeze(0) for k, v in enc.items() if k in {"input_ids", "attention_mask", "token_type_ids"}}

    def _tok_code(self, text: str) -> Dict[str, torch.Tensor]:
        enc = self.code_tokenizer(
            text or "",
            max_length=MAX_CODE_LENGTH,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {k: v.squeeze(0) for k, v in enc.items() if k in {"input_ids", "attention_mask", "token_type_ids"}}

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.rows[idx]
        desc = self._tok_text(row.get("cve_description", ""))
        msg = self._tok_text(row.get("commit_message", ""))
        add = self._tok_code(row.get("added_code", ""))
        dele = self._tok_code(row.get("deleted_code", ""))

        binary_label = int(row.get("binary_label", 0))
        mapped_cwe = map_cwe_for_training(row.get("cwe_id", "NO_CWE"), self.cwe_to_id, self.cwe_remap)
        cwe_label = self.cwe_to_id.get(mapped_cwe, -100) if binary_label == 1 else -100

        return {
            "desc_input_ids": desc["input_ids"],
            "desc_attention_mask": desc["attention_mask"],
            "msg_input_ids": msg["input_ids"],
            "msg_attention_mask": msg["attention_mask"],
            "add_input_ids": add["input_ids"],
            "add_attention_mask": add["attention_mask"],
            "del_input_ids": dele["input_ids"],
            "del_attention_mask": dele["attention_mask"],
            "binary_label": torch.tensor(binary_label, dtype=torch.long),
            "cwe_label": torch.tensor(cwe_label, dtype=torch.long),
            "row_index": torch.tensor(idx, dtype=torch.long),
        }


# =============================================================================
# Model
# =============================================================================

class MLPHead(nn.Module):
    def __init__(self, dims: Sequence[int], dropout: float) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class VPFinderModel(nn.Module):
    def __init__(self, bert_dir: str, codebert_dir: str, num_cwe_labels: int, dropout: float = 0.3, num_heads: int = 8) -> None:
        super().__init__()
        self.text_encoder = AutoModel.from_pretrained(bert_dir)
        self.code_encoder = AutoModel.from_pretrained(codebert_dir)
        hidden = int(self.text_encoder.config.hidden_size)
        code_hidden = int(self.code_encoder.config.hidden_size)
        if hidden != code_hidden:
            raise ValueError(f"BERT hidden size ({hidden}) must equal CodeBERT hidden size ({code_hidden})")
        if hidden % num_heads != 0:
            raise ValueError(f"hidden size {hidden} must be divisible by NUM_ATTENTION_HEADS={num_heads}")

        self.hidden_size = hidden
        self.att_text_patch = nn.MultiheadAttention(embed_dim=hidden, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.att_msg_patch = nn.MultiheadAttention(embed_dim=hidden, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

        fused_dim = hidden * 5
        self.binary_head = MLPHead([fused_dim, hidden * 3, hidden, 256, 64, 2], dropout=dropout)
        self.type_head = MLPHead([fused_dim, hidden * 3, hidden, 256, num_cwe_labels], dropout=dropout)

    @staticmethod
    def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        m = mask.float().unsqueeze(-1)
        return (x * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)

    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = out.last_hidden_state
        cls = hidden[:, 0, :]
        return hidden, cls

    def encode_code(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.code_encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = out.last_hidden_state
        cls = hidden[:, 0, :]
        return hidden, cls

    def forward(
        self,
        desc_input_ids: torch.Tensor,
        desc_attention_mask: torch.Tensor,
        msg_input_ids: torch.Tensor,
        msg_attention_mask: torch.Tensor,
        add_input_ids: torch.Tensor,
        add_attention_mask: torch.Tensor,
        del_input_ids: torch.Tensor,
        del_attention_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        vt, vcls_t = self.encode_text(desc_input_ids, desc_attention_mask)
        vc, vcls_c = self.encode_text(msg_input_ids, msg_attention_mask)

        va, vcls_a = self.encode_code(add_input_ids, add_attention_mask)
        vd, vcls_d = self.encode_code(del_input_ids, del_attention_mask)

        vp = torch.cat([va, vd], dim=1)
        patch_mask = torch.cat([add_attention_mask, del_attention_mask], dim=1)
        vcls_p = 0.5 * (vcls_a + vcls_d)

        key_padding_mask = patch_mask.eq(0)
        mt, _ = self.att_text_patch(query=vt, key=vp, value=vp, key_padding_mask=key_padding_mask, need_weights=False)
        vt_att = self.masked_mean(mt, desc_attention_mask)

        mc, _ = self.att_msg_patch(query=vc, key=vp, value=vp, key_padding_mask=key_padding_mask, need_weights=False)
        vc_att = self.masked_mean(mc, msg_attention_mask)

        vt_res = vt_att + vcls_p
        vc_res = vc_att + vcls_p

        vx = torch.cat([vt_res, vc_res, vcls_t, vcls_c, vcls_p], dim=-1)
        vx = self.dropout(vx)

        binary_logits = self.binary_head(vx)
        type_logits = self.type_head(vx)
        return {"binary_logits": binary_logits, "type_logits": type_logits, "fused": vx}


def freeze_encoder_parameters(model: VPFinderModel) -> None:
    for p in model.text_encoder.parameters():
        p.requires_grad = False
    for p in model.code_encoder.parameters():
        p.requires_grad = False


# =============================================================================
# Training and evaluation
# =============================================================================

@dataclass
class EpochResult:
    epoch: int
    train_loss: float
    train_binary_loss: float
    train_type_loss: float
    valid_binary_f1_weighted: float
    valid_type_f1_weighted: float
    valid_objective: float
    elapsed_seconds: float


def move_batch_to_device(batch: Dict[str, Any], device: str) -> Dict[str, Any]:
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device) if torch.is_tensor(v) else v
    return out


def compute_losses(outputs: Dict[str, torch.Tensor], binary_labels: torch.Tensor, cwe_labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    binary_loss_fn = nn.CrossEntropyLoss()
    type_loss_fn = nn.CrossEntropyLoss()
    binary_loss = binary_loss_fn(outputs["binary_logits"], binary_labels)
    valid_type_mask = cwe_labels.ne(-100)
    if valid_type_mask.any():
        type_loss = type_loss_fn(outputs["type_logits"][valid_type_mask], cwe_labels[valid_type_mask])
    else:
        type_loss = outputs["binary_logits"].new_tensor(0.0)
    total_loss = binary_loss + TYPE_LOSS_WEIGHT * type_loss
    return total_loss, binary_loss, type_loss


def train_one_epoch(
    model: VPFinderModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: torch.cuda.amp.GradScaler,
    epoch: int,
) -> Tuple[float, float, float]:
    model.train()
    total_loss_sum = 0.0
    binary_loss_sum = 0.0
    type_loss_sum = 0.0
    num_batches = 0
    optimizer.zero_grad(set_to_none=True)

    pbar = tqdm(loader, desc=f"Training epoch {epoch}")
    for step, batch in enumerate(pbar, start=1):
        batch = move_batch_to_device(batch, DEVICE)
        with torch.cuda.amp.autocast(enabled=(USE_AMP and DEVICE == "cuda")):
            outputs = model(
                desc_input_ids=batch["desc_input_ids"],
                desc_attention_mask=batch["desc_attention_mask"],
                msg_input_ids=batch["msg_input_ids"],
                msg_attention_mask=batch["msg_attention_mask"],
                add_input_ids=batch["add_input_ids"],
                add_attention_mask=batch["add_attention_mask"],
                del_input_ids=batch["del_input_ids"],
                del_attention_mask=batch["del_attention_mask"],
            )
            loss, binary_loss, type_loss = compute_losses(outputs, batch["binary_label"], batch["cwe_label"])
            loss = loss / GRADIENT_ACCUMULATION_STEPS

        scaler.scale(loss).backward()

        if step % GRADIENT_ACCUMULATION_STEPS == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()

        total_loss_sum += float(loss.detach().cpu()) * GRADIENT_ACCUMULATION_STEPS
        binary_loss_sum += float(binary_loss.detach().cpu())
        type_loss_sum += float(type_loss.detach().cpu())
        num_batches += 1
        pbar.set_postfix(loss=total_loss_sum / max(num_batches, 1), bin=binary_loss_sum / max(num_batches, 1), typ=type_loss_sum / max(num_batches, 1))

    return total_loss_sum / max(num_batches, 1), binary_loss_sum / max(num_batches, 1), type_loss_sum / max(num_batches, 1)


@torch.no_grad()
def predict(
    model: VPFinderModel,
    loader: DataLoader,
    rows: List[Dict[str, Any]],
    id_to_cwe: Dict[int, str],
) -> List[Dict[str, Any]]:
    model.eval()
    predictions: List[Dict[str, Any]] = []

    for batch in tqdm(loader, desc="Predicting"):
        original_indices = batch["row_index"].cpu().numpy().tolist()
        batch = move_batch_to_device(batch, DEVICE)
        outputs = model(
            desc_input_ids=batch["desc_input_ids"],
            desc_attention_mask=batch["desc_attention_mask"],
            msg_input_ids=batch["msg_input_ids"],
            msg_attention_mask=batch["msg_attention_mask"],
            add_input_ids=batch["add_input_ids"],
            add_attention_mask=batch["add_attention_mask"],
            del_input_ids=batch["del_input_ids"],
            del_attention_mask=batch["del_attention_mask"],
        )
        binary_probs = torch.softmax(outputs["binary_logits"], dim=-1).detach().cpu().numpy()
        binary_preds = binary_probs.argmax(axis=1)
        type_probs = torch.softmax(outputs["type_logits"], dim=-1).detach().cpu().numpy()
        type_preds = type_probs.argmax(axis=1)

        true_binary = batch["binary_label"].detach().cpu().numpy().tolist()
        true_cwe_ids = batch["cwe_label"].detach().cpu().numpy().tolist()

        for local_i, row_idx in enumerate(original_indices):
            row = rows[row_idx]
            pred_type_id = int(type_preds[local_i])
            pred_cwe = id_to_cwe.get(pred_type_id, OTHER_CWE_LABEL)
            pred_binary = int(binary_preds[local_i])
            pipeline_pred_cwe = pred_cwe if pred_binary == 1 else "NO_CWE"
            true_cwe_id = int(true_cwe_ids[local_i])
            mapped_true_cwe = id_to_cwe.get(true_cwe_id, "NO_CWE") if true_cwe_id >= 0 else "NO_CWE"
            predictions.append({
                "commit_id": row.get("commit_id", ""),
                "cve_id": row.get("cve_id", ""),
                "true_binary_label": int(true_binary[local_i]),
                "pred_binary_label": pred_binary,
                "prob_non_vulnerable": float(binary_probs[local_i][0]),
                "prob_vulnerable": float(binary_probs[local_i][1]) if binary_probs.shape[1] > 1 else 0.0,
                "true_cwe_original": row.get("cwe_id", "NO_CWE"),
                "true_cwe_mapped": mapped_true_cwe,
                "pred_cwe_from_type_head": pred_cwe,
                "pred_cwe_pipeline": pipeline_pred_cwe,
                "pred_cwe_confidence": float(type_probs[local_i][pred_type_id]),
                "commit_message": row.get("commit_message", ""),
                "cve_description": row.get("cve_description", ""),
            })
    return predictions


def safe_auc(y_true: Sequence[int], y_score: Sequence[float]) -> Optional[float]:
    try:
        if len(set(y_true)) < 2:
            return None
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return None


def binary_metrics(preds: List[Dict[str, Any]]) -> Dict[str, Any]:
    y_true = [int(p["true_binary_label"]) for p in preds]
    y_pred = [int(p["pred_binary_label"]) for p in preds]
    y_prob = [float(p["prob_vulnerable"]) for p in preds]
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
    return {
        "num_examples": len(preds),
        "Accuracy": float(accuracy_score(y_true, y_pred)) if preds else 0.0,
        "Precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)) if preds else 0.0,
        "Recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)) if preds else 0.0,
        "F1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)) if preds else 0.0,
        "Precision_positive": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)) if preds else 0.0,
        "Recall_positive": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)) if preds else 0.0,
        "F1_positive": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)) if preds else 0.0,
        "MCC": float(matthews_corrcoef(y_true, y_pred)) if len(set(y_true) | set(y_pred)) > 1 else 0.0,
        "AUC": safe_auc(y_true, y_prob),
        "TP": int(tp),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "confusion_matrix_labels": [0, 1],
        "confusion_matrix": cm.tolist(),
    }


def type_metrics(preds: List[Dict[str, Any]]) -> Dict[str, Any]:
    rows = [p for p in preds if int(p["true_binary_label"]) == 1 and p.get("true_cwe_mapped", "NO_CWE") != "NO_CWE"]
    if not rows:
        return {"num_examples": 0}
    y_true = [p["true_cwe_mapped"] for p in rows]
    y_pred = [p["pred_cwe_from_type_head"] for p in rows]
    labels = sorted(set(y_true) | set(y_pred))
    return {
        "num_examples": len(rows),
        "num_labels": len(labels),
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision_weighted": float(precision_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "Recall_weighted": float(recall_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "F1_weighted": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "Precision_macro": float(precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "Recall_macro": float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "F1_macro": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "MCC": float(matthews_corrcoef(y_true, y_pred)) if len(labels) > 1 else 0.0,
        "labels": labels,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def evaluate_predictions(preds: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "binary_vulnerability_identification": binary_metrics(preds),
        "cwe_type_classification_true_vulnerable_only": type_metrics(preds),
    }


def save_predictions_and_metrics(out_dir: Path, split_name: str, preds: List[Dict[str, Any]], metrics: Dict[str, Any]) -> None:
    write_json(out_dir / f"{split_name}_predictions.json", preds)
    write_jsonl(out_dir / f"{split_name}_predictions.jsonl", preds)
    pd.DataFrame(preds).to_csv(out_dir / f"{split_name}_predictions.csv", index=False)
    write_json(out_dir / f"{split_name}_metrics.json", metrics)

    flat_rows = []
    for block_name, block in metrics.items():
        for k, v in block.items():
            if isinstance(v, (list, dict)):
                continue
            flat_rows.append({"metric_group": block_name, "metric": k, "value": v})
    pd.DataFrame(flat_rows).to_csv(out_dir / f"{split_name}_metrics.csv", index=False)

    bin_cm = metrics.get("binary_vulnerability_identification", {}).get("confusion_matrix")
    if bin_cm is not None:
        pd.DataFrame(bin_cm, index=["true_0", "true_1"], columns=["pred_0", "pred_1"]).to_csv(out_dir / f"{split_name}_binary_confusion_matrix.csv")

    type_block = metrics.get("cwe_type_classification_true_vulnerable_only", {})
    type_cm = type_block.get("confusion_matrix")
    labels = type_block.get("labels")
    if type_cm is not None and labels:
        pd.DataFrame(type_cm, index=[f"true_{x}" for x in labels], columns=[f"pred_{x}" for x in labels]).to_csv(out_dir / f"{split_name}_cwe_confusion_matrix.csv")


def save_excel_report(
    out_dir: Path,
    train: List[Dict[str, Any]],
    valid: List[Dict[str, Any]],
    test: List[Dict[str, Any]],
    valid_preds: List[Dict[str, Any]],
    test_preds: List[Dict[str, Any]],
    valid_metrics: Dict[str, Any],
    test_metrics: Dict[str, Any],
    history: List[Dict[str, Any]],
    cwe_to_id: Dict[str, int],
) -> None:
    xlsx_path = out_dir / "vpfinder_results.xlsx"
    try:
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            pd.DataFrame([{k: v for k, v in r.items() if k != "original_row"} for r in train]).to_excel(writer, sheet_name="training", index=False)
            pd.DataFrame([{k: v for k, v in r.items() if k != "original_row"} for r in valid]).to_excel(writer, sheet_name="validation", index=False)
            pd.DataFrame([{k: v for k, v in r.items() if k != "original_row"} for r in test]).to_excel(writer, sheet_name="testing", index=False)
            pd.DataFrame(history).to_excel(writer, sheet_name="training_history", index=False)
            pd.DataFrame(valid_preds).to_excel(writer, sheet_name="validation_predictions", index=False)
            pd.DataFrame(test_preds).to_excel(writer, sheet_name="testing_predictions", index=False)
            pd.DataFrame([{"cwe_label": k, "id": v} for k, v in cwe_to_id.items()]).to_excel(writer, sheet_name="cwe_label_map", index=False)

            metric_rows = []
            for split_name, metrics in [("validation", valid_metrics), ("testing", test_metrics)]:
                for block_name, block in metrics.items():
                    for k, v in block.items():
                        if isinstance(v, (list, dict)):
                            continue
                        metric_rows.append({"split": split_name, "metric_group": block_name, "metric": k, "value": v})
            pd.DataFrame(metric_rows).to_excel(writer, sheet_name="metrics", index=False)
    except Exception as e:
        print(f"[WARN] Could not save Excel report {xlsx_path}: {e}")
        print("       Install openpyxl if needed: pip install openpyxl")


def make_dataloaders(
    train: List[Dict[str, Any]],
    valid: List[Dict[str, Any]],
    test: List[Dict[str, Any]],
    text_tokenizer: AutoTokenizer,
    code_tokenizer: AutoTokenizer,
    cwe_to_id: Dict[str, int],
    cwe_remap: Dict[str, str],
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_ds = VPFinderDataset(train, text_tokenizer, code_tokenizer, cwe_to_id, cwe_remap)
    valid_ds = VPFinderDataset(valid, text_tokenizer, code_tokenizer, cwe_to_id, cwe_remap)
    test_ds = VPFinderDataset(test, text_tokenizer, code_tokenizer, cwe_to_id, cwe_remap)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=(DEVICE == "cuda"))
    valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=(DEVICE == "cuda"))
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=(DEVICE == "cuda"))
    return train_loader, valid_loader, test_loader


def save_checkpoint(out_dir: Path, model: VPFinderModel, cwe_to_id: Dict[str, int], cwe_remap: Dict[str, str], metrics: Dict[str, Any]) -> None:
    ckpt = {
        "model_state_dict": model.state_dict(),
        "cwe_to_id": cwe_to_id,
        "cwe_remap": cwe_remap,
        "config": {
            "BERT_DIR": BERT_DIR,
            "CODEBERT_DIR": CODEBERT_DIR,
            "MAX_TEXT_LENGTH": MAX_TEXT_LENGTH,
            "MAX_CODE_LENGTH": MAX_CODE_LENGTH,
            "NUM_ATTENTION_HEADS": NUM_ATTENTION_HEADS,
            "DROPOUT": DROPOUT,
        },
        "best_validation_metrics": metrics,
    }
    torch.save(ckpt, out_dir / "best_vpfinder_model.pt")


def load_best_checkpoint(out_dir: Path, model: VPFinderModel) -> None:
    ckpt_path = out_dir / "best_vpfinder_model.pt"
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        model.load_state_dict(ckpt["model_state_dict"])


# =============================================================================
# Main workflow
# =============================================================================


def load_or_create_splits(out_dir: Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    if exists_nonempty(TRAIN_INPUT_JSON) and exists_nonempty(VALID_INPUT_JSON) and exists_nonempty(TEST_INPUT_JSON):
        print("Using existing training/validation/testing input files.")
        train_raw = read_json_or_jsonl(TRAIN_INPUT_JSON)
        valid_raw = read_json_or_jsonl(VALID_INPUT_JSON)
        test_raw = read_json_or_jsonl(TEST_INPUT_JSON)
        train = [normalize_row(r) for r in train_raw]
        valid = [normalize_row(r) for r in valid_raw]
        test = [normalize_row(r) for r in test_raw]
    elif exists_nonempty(INPUT):
        print("Using INPUT (alias of TRAIN_INPUT_JSON) and creating 80/10/10 split.")
        raw = read_json_or_jsonl(INPUT)
        rows = [normalize_row(r) for r in raw]
        if MAX_ROWS_DEBUG is not None:
            rows = rows[:MAX_ROWS_DEBUG]
        train, valid, test = split_80_10_10(rows)
    else:
        raise FileNotFoundError(
            "No valid input found. Set INPUT to an existing JSON/JSONL file or set TRAIN_INPUT_JSON, VALID_INPUT_JSON, and TEST_INPUT_JSON to existing files."
        )

    if MAX_ROWS_DEBUG is not None and exists_nonempty(TRAIN_INPUT_JSON):
        train = train[:MAX_ROWS_DEBUG]
        valid = valid[: max(1, MAX_ROWS_DEBUG // 10)]
        test = test[: max(1, MAX_ROWS_DEBUG // 10)]

    save_split_files(out_dir, "training", train)
    save_split_files(out_dir, "validation", valid)
    save_split_files(out_dir, "testing", test)
    return train, valid, test


def dataset_summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    bin_counts = Counter(int(r.get("binary_label", 0)) for r in rows)
    cwe_counts = Counter(normalize_cwe(r.get("cwe_id")) for r in rows if int(r.get("binary_label", 0)) == 1)
    return {
        "num_rows": len(rows),
        "binary_counts": dict(bin_counts),
        "num_vulnerable_cwe_labels": len(cwe_counts),
        "top_cwe_counts": dict(cwe_counts.most_common(20)),
    }


def main() -> None:
    set_seed(RANDOM_SEED)
    out_dir = ensure_dir(OUTPUT_DIR)

    print("=" * 100)
    print("VPFinder reproduction: BERT + CodeBERT + multi-head attention fusion")
    print("=" * 100)
    print(f"BERT_DIR     : {BERT_DIR}")
    print(f"CODEBERT_DIR : {CODEBERT_DIR}")
    print(f"INPUT        : {INPUT}")
    print(f"OUTPUT       : {OUTPUT}")
    print(f"DEVICE       : {DEVICE}")
    print("=" * 100)

    train, valid, test = load_or_create_splits(out_dir)

    cwe_to_id, id_to_cwe, cwe_remap = build_cwe_label_map(train, valid, test)
    if not cwe_to_id:
        raise ValueError("No CWE labels found for vulnerability type classification. Check cwe_id and binary labels.")

    run_config = {
        "BERT_DIR": BERT_DIR,
        "CODEBERT_DIR": CODEBERT_DIR,
        "INPUT": INPUT,
        "TRAIN_INPUT_JSON": TRAIN_INPUT_JSON,
        "VALID_INPUT_JSON": VALID_INPUT_JSON,
        "TEST_INPUT_JSON": TEST_INPUT_JSON,
        "OUTPUT": OUTPUT,
        "OUTPUT_DIR": OUTPUT_DIR,
        "TRAIN_RATIO": TRAIN_RATIO,
        "VALID_RATIO": VALID_RATIO,
        "TEST_RATIO": TEST_RATIO,
        "RANDOM_SEED": RANDOM_SEED,
        "NUM_EPOCHS": NUM_EPOCHS,
        "BATCH_SIZE": BATCH_SIZE,
        "LEARNING_RATE": LEARNING_RATE,
        "MAX_TEXT_LENGTH": MAX_TEXT_LENGTH,
        "MAX_CODE_LENGTH": MAX_CODE_LENGTH,
        "NUM_ATTENTION_HEADS": NUM_ATTENTION_HEADS,
        "DROPOUT": DROPOUT,
        "FREEZE_ENCODERS": FREEZE_ENCODERS,
        "TYPE_LOSS_WEIGHT": TYPE_LOSS_WEIGHT,
        "BUILD_CWE_LABEL_MAP_FROM": BUILD_CWE_LABEL_MAP_FROM,
        "MIN_CWE_SUPPORT": MIN_CWE_SUPPORT,
        "num_cwe_labels": len(cwe_to_id),
        "dataset_summary": {
            "training": dataset_summary(train),
            "validation": dataset_summary(valid),
            "testing": dataset_summary(test),
        },
    }
    write_json(out_dir / "run_config.json", run_config)
    write_json(out_dir / "cwe_label_map.json", {"cwe_to_id": cwe_to_id, "id_to_cwe": id_to_cwe, "cwe_remap": cwe_remap})

    print("Dataset summary:")
    print(json.dumps(run_config["dataset_summary"], indent=2))
    print(f"CWE labels: {len(cwe_to_id)}")

    print("Loading tokenizers and models...")
    text_tokenizer = AutoTokenizer.from_pretrained(BERT_DIR)
    code_tokenizer = AutoTokenizer.from_pretrained(CODEBERT_DIR)
    train_loader, valid_loader, test_loader = make_dataloaders(train, valid, test, text_tokenizer, code_tokenizer, cwe_to_id, cwe_remap)

    model = VPFinderModel(BERT_DIR, CODEBERT_DIR, num_cwe_labels=len(cwe_to_id), dropout=DROPOUT, num_heads=NUM_ATTENTION_HEADS).to(DEVICE)
    if FREEZE_ENCODERS:
        print("Freezing BERT and CodeBERT encoder parameters.")
        freeze_encoder_parameters(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    total_update_steps = math.ceil(len(train_loader) / GRADIENT_ACCUMULATION_STEPS) * NUM_EPOCHS
    warmup_steps = int(total_update_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_update_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=(USE_AMP and DEVICE == "cuda"))

    history: List[Dict[str, Any]] = []
    best_objective = -1.0
    bad_epochs = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        start = time.time()
        train_loss, train_binary_loss, train_type_loss = train_one_epoch(model, train_loader, optimizer, scheduler, scaler, epoch)

        valid_preds = predict(model, valid_loader, valid, id_to_cwe)
        valid_metrics = evaluate_predictions(valid_preds)
        valid_bin_f1 = float(valid_metrics["binary_vulnerability_identification"].get("F1_weighted", 0.0))
        valid_type_f1 = float(valid_metrics["cwe_type_classification_true_vulnerable_only"].get("F1_weighted", 0.0))
        valid_objective = valid_bin_f1 + valid_type_f1

        result = EpochResult(
            epoch=epoch,
            train_loss=train_loss,
            train_binary_loss=train_binary_loss,
            train_type_loss=train_type_loss,
            valid_binary_f1_weighted=valid_bin_f1,
            valid_type_f1_weighted=valid_type_f1,
            valid_objective=valid_objective,
            elapsed_seconds=time.time() - start,
        )
        history.append(asdict(result))
        write_json(out_dir / "training_history.json", history)
        pd.DataFrame(history).to_csv(out_dir / "training_history.csv", index=False)

        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_bin_f1={valid_bin_f1:.4f}, val_type_f1={valid_type_f1:.4f}, objective={valid_objective:.4f}")

        if valid_objective > best_objective:
            best_objective = valid_objective
            bad_epochs = 0
            save_checkpoint(out_dir, model, cwe_to_id, cwe_remap, valid_metrics)
            save_predictions_and_metrics(out_dir, "validation", valid_preds, valid_metrics)
            print(f"Saved new best model with validation objective {best_objective:.4f}")
        else:
            bad_epochs += 1
            if bad_epochs >= EARLY_STOPPING_PATIENCE:
                print(f"Early stopping after {epoch} epochs.")
                break

    print("Loading best validation checkpoint for final testing...")
    load_best_checkpoint(out_dir, model)

    valid_preds = predict(model, valid_loader, valid, id_to_cwe)
    valid_metrics = evaluate_predictions(valid_preds)
    save_predictions_and_metrics(out_dir, "validation", valid_preds, valid_metrics)

    test_preds = predict(model, test_loader, test, id_to_cwe)
    test_metrics = evaluate_predictions(test_preds)
    save_predictions_and_metrics(out_dir, "testing", test_preds, test_metrics)

    save_excel_report(out_dir, train, valid, test, valid_preds, test_preds, valid_metrics, test_metrics, history, cwe_to_id)

    print("\nFinal validation metrics:")
    print(json.dumps(valid_metrics, indent=2))
    print("\nFinal testing metrics:")
    print(json.dumps(test_metrics, indent=2))
    print("\nDone. Results saved to:")
    print(out_dir)


if __name__ == "__main__":
    main()
