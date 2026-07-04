#!/usr/bin/env python3
from __future__ import annotations

import difflib
import json
import math
import os
import random
import re
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, matthews_corrcoef
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup


# =============================================================================
# EDIT HERE ONLY
# =============================================================================

CODEBERT_DIR = "/path/software_defects/CWEMAP_data/CodeBERT"

# Main input dataset path. Use the actual training file directly.
TRAIN_INPUT_JSON = "/path/dataset_name/training.json"
INPUT = TRAIN_INPUT_JSON
VALID_INPUT_JSON = "/path/dataset_name/validation.json"
TEST_INPUT_JSON = "/path/dataset_name/testing.json"

OUTPUT_DIR = "/path/dataset_name/dataset_name_results"

RANDOM_SEED = 42
TARGET_DEPTH = 3
TRAIN_RATIO = 0.80
VALID_RATIO = 0.10
TEST_RATIO = 0.10


MAX_HUNKS = 8
MAX_REM_TOKENS = 128
MAX_ADD_TOKENS = 128
MAX_SEQ_LEN = 260
BATCH_SIZE = 4
EVAL_BATCH_SIZE = 4
EPOCHS = 20
PATIENCE = 5
DROPOUT = 0.10
BEAM_SIZE = 5

CODEBERT_LR = 5e-5
HEAD_LR = 1e-3
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 3000
GRAD_CLIP_NORM = 1.0

USE_CUDA = True
NUM_WORKERS = 0
SAVE_NORMALIZED_SPLITS = True


# =============================================================================
# Utilities
# =============================================================================

CWE_RE = re.compile(r"CWE-\d+", re.IGNORECASE)
EDIT_TO_ID = {"special": 0, "equal": 1, "replace": 2, "insert": 3, "delete": 4}
ID_TO_EDIT = {v: k for k, v in EDIT_TO_ID.items()}
ROOT = "CWE-1000"


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
        return obj
    rows: List[Dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def write_json(path: str | Path, obj: Any) -> None:
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: str | Path, rows: Iterable[Dict[str, Any]]) -> None:
    with Path(path).open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_cwe(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        value = " ".join(str(x) for x in value)
    match = CWE_RE.search(str(value))
    return match.group(0).upper() if match else ""


def cwe_sort_key(cwe: str) -> Tuple[int, str]:
    m = re.search(r"\d+", cwe or "")
    return (int(m.group(0)) if m else 10**9, cwe)


def list_to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        if value and all(isinstance(x, list) for x in value):
            return "\n".join("\n".join(str(y) for y in x) for x in value)
        return "\n".join(str(x) for x in value if str(x).strip())
    return str(value)


def truncate_text(text: str, max_chars: int = 500) -> str:
    text = text or ""
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 30] + " ...[TRUNCATED]... " + text[-15:]


# =============================================================================
# Data preparation
# =============================================================================


def parse_unified_diff(diff_text: str) -> List[Dict[str, str]]:
    """Very lightweight unified diff parser for fallback input records."""
    hunks: List[Dict[str, str]] = []
    rem_lines: List[str] = []
    add_lines: List[str] = []

    def flush() -> None:
        nonlocal rem_lines, add_lines
        if rem_lines or add_lines:
            hunks.append({"rem": "\n".join(rem_lines), "add": "\n".join(add_lines)})
        rem_lines, add_lines = [], []

    for raw in (diff_text or "").splitlines():
        if raw.startswith("@@"):
            flush()
            continue
        if raw.startswith("---") or raw.startswith("+++"):
            continue
        if raw.startswith("-"):
            rem_lines.append(raw[1:])
        elif raw.startswith("+"):
            add_lines.append(raw[1:])
    flush()
    return hunks


def extract_hunks(record: Dict[str, Any]) -> List[Dict[str, str]]:
    """Extract rem/add hunks from several common TreeVul-like schemas."""

    hunks_obj = record.get("hunks") or record.get("HUNKS")
    if isinstance(hunks_obj, list):
        out: List[Dict[str, str]] = []
        for h in hunks_obj:
            if not isinstance(h, dict):
                continue
            rem = h.get("rem") or h.get("removed") or h.get("removed_code") or h.get("REM_DIFF") or ""
            add = h.get("add") or h.get("added") or h.get("added_code") or h.get("ADD_DIFF") or ""
            rem_t, add_t = list_to_text(rem), list_to_text(add)
            if rem_t.strip() or add_t.strip():
                out.append({"rem": rem_t, "add": add_t})
        if out:
            return out


    for diff_key in ["diff", "patch", "commit_diff", "raw_diff"]:
        if isinstance(record.get(diff_key), str) and record.get(diff_key, "").strip():
            parsed = parse_unified_diff(record[diff_key])
            if parsed:
                return parsed

    rem_obj = record.get("REM_DIFF", record.get("removed_code", record.get("rem_code", "")))
    add_obj = record.get("ADD_DIFF", record.get("added_code", record.get("add_code", "")))

    if isinstance(rem_obj, list) and isinstance(add_obj, list):
        if rem_obj and add_obj and all(isinstance(x, list) for x in rem_obj) and all(isinstance(x, list) for x in add_obj):
            out = []
            for rem_h, add_h in zip(rem_obj, add_obj):
                rem_t, add_t = list_to_text(rem_h), list_to_text(add_h)
                if rem_t.strip() or add_t.strip():
                    out.append({"rem": rem_t, "add": add_t})
            if out:
                return out

    rem_t, add_t = list_to_text(rem_obj), list_to_text(add_obj)
    if rem_t.strip() or add_t.strip():
        return [{"rem": rem_t, "add": add_t}]
    return []


def extract_path_candidates(value: Any) -> List[List[str]]:
    paths: List[List[str]] = []
    if isinstance(value, list):
        # List of paths or one path.
        if value and all(not isinstance(x, list) for x in value):
            cwes = [normalize_cwe(x) for x in value]
            cwes = [x for x in cwes if x]
            if cwes:
                paths.append(cwes)
        else:
            for p in value:
                if isinstance(p, list):
                    cwes = [normalize_cwe(x) for x in p]
                    cwes = [x for x in cwes if x]
                    if cwes:
                        paths.append(cwes)
    elif isinstance(value, str):
        cwes = [m.group(0).upper() for m in CWE_RE.finditer(value)]
        if cwes:
            paths.append(cwes)
    return paths


def get_depth_path(record: Dict[str, Any], target_depth: int = TARGET_DEPTH) -> Optional[List[str]]:
    """
    Return y1..yd, excluding root CWE-1000. If original path is deeper than d, use the depth-d ancestor.
    """
    candidates: List[List[str]] = []
    for key in ["path_list", "cwe_path", "path", "paths", "true_path", "true_path_list"]:
        candidates.extend(extract_path_candidates(record.get(key)))

    if not candidates:
        cwe = normalize_cwe(record.get("cwe_list") or record.get("cwe") or record.get("CWE") or record.get("cwe_id"))
        if cwe:
            candidates = [[cwe]]

    for path in candidates:
        path = [x for x in path if x]
        if path and path[0] == ROOT:
            path = path[1:]
        path = [x for x in path if x != ROOT]
        if len(path) >= target_depth:
            return path[:target_depth]
    return None


def normalize_record(record: Dict[str, Any], idx: int, target_depth: int = TARGET_DEPTH) -> Optional[Dict[str, Any]]:
    path = get_depth_path(record, target_depth)
    if not path or len(path) < target_depth:
        return None
    hunks = extract_hunks(record)
    if not hunks:
        return None
    return {
        "example_id": str(record.get("example_id") or record.get("id") or f"commit_{idx}"),
        "source_index": int(record.get("source_index", idx)) if str(record.get("source_index", idx)).isdigit() else idx,
        "repo": record.get("repo", record.get("repository", "")),
        "commit_id": record.get("commit_id", record.get("commit", record.get("hash", ""))),
        "file_name": record.get("file_name", record.get("filename", record.get("file", ""))),
        "PL": record.get("PL", record.get("language", record.get("programming_language", ""))),
        "cve": record.get("cve", record.get("cve_list", record.get("CVE", ""))),
        "true_path": path,
        "true_cwe_depth3": path[target_depth - 1],
        "hunks": hunks[:MAX_HUNKS],
        "num_hunks_original": len(hunks),
        "raw_cwe": record.get("cwe_list", record.get("cwe", record.get("cwe_id", ""))),
    }


def normalize_records(rows: List[Dict[str, Any]], target_depth: int = TARGET_DEPTH) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    skipped = 0
    for i, row in enumerate(rows):
        norm = normalize_record(row, i, target_depth)
        if norm is None:
            skipped += 1
            continue
        out.append(norm)
    print(f"Normalized records: {len(out)} kept, {skipped} skipped because path/hunks were missing.")
    return out


def manual_stratified_split(
    rows: List[Dict[str, Any]],
    train_ratio: float,
    valid_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    assert abs(train_ratio + valid_ratio + test_ratio - 1.0) < 1e-6
    rng = random.Random(seed)
    by_label: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_label[row["true_cwe_depth3"]].append(row)

    train: List[Dict[str, Any]] = []
    valid: List[Dict[str, Any]] = []
    test: List[Dict[str, Any]] = []

    for label, group in by_label.items():
        rng.shuffle(group)
        n = len(group)
        n_test = int(round(n * test_ratio))
        n_valid = int(round(n * valid_ratio))
        if n >= 10:
            n_test = max(1, n_test)
            n_valid = max(1, n_valid)
        elif n >= 3:
            n_test = max(1, n_test)
            n_valid = max(0, n_valid)
        else:
            n_test = 0
            n_valid = 0
        if n_test + n_valid >= n:
            n_test = max(0, min(n_test, n - 1))
            n_valid = max(0, min(n_valid, n - n_test - 1))
        n_train = n - n_valid - n_test
        train.extend(group[:n_train])
        valid.extend(group[n_train:n_train + n_valid])
        test.extend(group[n_train + n_valid:])

    rng.shuffle(train)
    rng.shuffle(valid)
    rng.shuffle(test)
    return train, valid, test


def load_or_split_dataset() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    train_path = Path(TRAIN_INPUT_JSON)
    valid_path = Path(VALID_INPUT_JSON)
    test_path = Path(TEST_INPUT_JSON)

    if train_path.exists() and valid_path.exists() and test_path.exists():
        print("Loading existing training/validation/testing files.")
        train = normalize_records(read_json_any(train_path))
        valid = normalize_records(read_json_any(valid_path))
        test = normalize_records(read_json_any(test_path))
        return train, valid, test

    raw_path = Path(INPUT)
    if not raw_path.exists():
        raise FileNotFoundError(
            "No existing split files were found and INPUT does not exist. "
            "Edit TRAIN_INPUT_JSON / VALID_INPUT_JSON / TEST_INPUT_JSON or INPUT in the CONFIG block."
        )

    print("Loading raw input and creating 80/10/10 stratified split.")
    rows = normalize_records(read_json_any(raw_path))
    train, valid, test = manual_stratified_split(rows, TRAIN_RATIO, VALID_RATIO, TEST_RATIO, RANDOM_SEED)
    return train, valid, test


# =============================================================================
# CWE schema
# =============================================================================


@dataclass
class CWESchema:
    depth_labels: List[List[str]]
    label_to_idx: List[Dict[str, int]]
    idx_to_label: List[Dict[int, str]]
    children: Dict[str, List[str]]
    parent: Dict[str, str]
    target_depth: int = TARGET_DEPTH

    def to_dict(self) -> Dict[str, Any]:
        return {
            "depth_labels": self.depth_labels,
            "label_to_idx": self.label_to_idx,
            "idx_to_label": [{str(k): v for k, v in m.items()} for m in self.idx_to_label],
            "children": self.children,
            "parent": self.parent,
            "target_depth": self.target_depth,
        }


def build_cwe_schema(all_rows: List[Dict[str, Any]], target_depth: int = TARGET_DEPTH) -> CWESchema:
    labels_by_depth: List[set[str]] = [set() for _ in range(target_depth)]
    children: Dict[str, set[str]] = defaultdict(set)
    parent: Dict[str, str] = {}

    for row in all_rows:
        path = row["true_path"][:target_depth]
        for d, cwe in enumerate(path):
            labels_by_depth[d].add(cwe)
            par = ROOT if d == 0 else path[d - 1]
            children[par].add(cwe)
            parent[cwe] = par

    depth_labels = [sorted(list(s), key=cwe_sort_key) for s in labels_by_depth]
    label_to_idx = [{lab: i for i, lab in enumerate(labs)} for labs in depth_labels]
    idx_to_label = [{i: lab for lab, i in m.items()} for m in label_to_idx]
    children_plain = {k: sorted(v, key=cwe_sort_key) for k, v in children.items()}
    return CWESchema(depth_labels, label_to_idx, idx_to_label, children_plain, parent, target_depth)


def attach_label_indices(rows: List[Dict[str, Any]], schema: CWESchema) -> List[Dict[str, Any]]:
    out = []
    for row in rows:
        labels = []
        ok = True
        for d, cwe in enumerate(row["true_path"][:schema.target_depth]):
            if cwe not in schema.label_to_idx[d]:
                ok = False
                break
            labels.append(schema.label_to_idx[d][cwe])
        if ok and len(labels) == schema.target_depth:
            r = dict(row)
            r["label_indices"] = labels
            out.append(r)
    return out


# =============================================================================
# Tokenization dataset
# =============================================================================


class TreeVulDataset(Dataset):
    def __init__(self, rows: List[Dict[str, Any]], tokenizer: Any, max_hunks: int = MAX_HUNKS):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_hunks = max_hunks
        self.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 1
        self.cls_token_id = tokenizer.cls_token_id if tokenizer.cls_token_id is not None else tokenizer.convert_tokens_to_ids("<s>")
        self.sep_token_id = tokenizer.sep_token_id if tokenizer.sep_token_id is not None else tokenizer.convert_tokens_to_ids("</s>")
        self.eos_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else self.sep_token_id

    def __len__(self) -> int:
        return len(self.rows)

    def _align_edits(self, rem_tokens: List[str], add_tokens: List[str]) -> Tuple[List[str], List[str]]:
        rem_edit = ["delete"] * len(rem_tokens)
        add_edit = ["insert"] * len(add_tokens)
        matcher = difflib.SequenceMatcher(a=rem_tokens, b=add_tokens, autojunk=False)
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                for i in range(i1, i2):
                    rem_edit[i] = "equal"
                for j in range(j1, j2):
                    add_edit[j] = "equal"
            elif tag == "replace":
                for i in range(i1, i2):
                    rem_edit[i] = "replace"
                for j in range(j1, j2):
                    add_edit[j] = "replace"
            elif tag == "delete":
                for i in range(i1, i2):
                    rem_edit[i] = "delete"
            elif tag == "insert":
                for j in range(j1, j2):
                    add_edit[j] = "insert"
        return rem_edit, add_edit

    def _encode_hunk(self, rem: str, add: str) -> Dict[str, List[int]]:
        rem_tokens = self.tokenizer.tokenize(rem or "")[:MAX_REM_TOKENS]
        add_tokens = self.tokenizer.tokenize(add or "")[:MAX_ADD_TOKENS]
        rem_edit, add_edit = self._align_edits(rem_tokens, add_tokens)

        tokens = [self.tokenizer.cls_token] + rem_tokens + [self.tokenizer.sep_token] + add_tokens + [self.tokenizer.eos_token]
        edit_names = ["special"] + rem_edit + ["special"] + add_edit + ["special"]
        tokens = tokens[:MAX_SEQ_LEN]
        edit_names = edit_names[:MAX_SEQ_LEN]
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        edit_ids = [EDIT_TO_ID.get(e, 0) for e in edit_names]
        attention_mask = [1] * len(input_ids)
        return {"input_ids": input_ids, "edit_ids": edit_ids, "attention_mask": attention_mask}

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.rows[idx]
        hunk_features = []
        for h in row["hunks"][:self.max_hunks]:
            hunk_features.append(self._encode_hunk(h.get("rem", ""), h.get("add", "")))
        if not hunk_features:
            hunk_features.append(self._encode_hunk("", ""))
        return {
            "features": hunk_features,
            "labels": torch.tensor(row["label_indices"], dtype=torch.long),
            "example_id": row["example_id"],
            "true_path": row["true_path"],
            "meta": row,
        }


def collate_treevul(batch: List[Dict[str, Any]], pad_token_id: int) -> Dict[str, Any]:
    batch_size = len(batch)
    max_hunks = max(len(x["features"]) for x in batch)
    max_len = max(len(h["input_ids"]) for x in batch for h in x["features"])
    max_len = min(max_len, MAX_SEQ_LEN)

    input_ids = torch.full((batch_size, max_hunks, max_len), pad_token_id, dtype=torch.long)
    edit_ids = torch.zeros((batch_size, max_hunks, max_len), dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_hunks, max_len), dtype=torch.long)
    hunk_mask = torch.zeros((batch_size, max_hunks), dtype=torch.float)
    labels = torch.stack([x["labels"] for x in batch], dim=0)

    for b, item in enumerate(batch):
        for h_idx, h in enumerate(item["features"]):
            seq_len = min(len(h["input_ids"]), max_len)
            input_ids[b, h_idx, :seq_len] = torch.tensor(h["input_ids"][:seq_len], dtype=torch.long)
            edit_ids[b, h_idx, :seq_len] = torch.tensor(h["edit_ids"][:seq_len], dtype=torch.long)
            attention_mask[b, h_idx, :seq_len] = torch.tensor(h["attention_mask"][:seq_len], dtype=torch.long)
            hunk_mask[b, h_idx] = 1.0

    return {
        "input_ids": input_ids,
        "edit_ids": edit_ids,
        "attention_mask": attention_mask,
        "hunk_mask": hunk_mask,
        "labels": labels,
        "example_ids": [x["example_id"] for x in batch],
        "true_paths": [x["true_path"] for x in batch],
        "metas": [x["meta"] for x in batch],
    }


# =============================================================================
# TREEVUL model
# =============================================================================


class TreeVulModel(nn.Module):
    def __init__(
        self,
        codebert_dir: str,
        num_labels_by_depth: Sequence[int],
        dropout: float = DROPOUT,
        label_init_tensors: Optional[List[torch.Tensor]] = None,
    ):
        super().__init__()
        self.codebert = AutoModel.from_pretrained(codebert_dir)
        hidden = int(self.codebert.config.hidden_size)
        self.hidden_size = hidden
        self.edit_embeddings = nn.Embedding(len(EDIT_TO_ID), hidden)


        self.depth2_encoder = nn.LSTM(hidden, hidden // 2, num_layers=1, batch_first=True, bidirectional=True)
        self.depth3_encoder = nn.LSTM(hidden, hidden // 2, num_layers=1, batch_first=True, bidirectional=True)

        self.dropout = nn.Dropout(dropout)
        self.label_emb_depth1 = nn.Embedding(num_labels_by_depth[0], hidden)
        self.label_emb_depth2 = nn.Embedding(num_labels_by_depth[1], hidden)

        if label_init_tensors is not None:
            with torch.no_grad():
                if label_init_tensors[0].shape == self.label_emb_depth1.weight.shape:
                    self.label_emb_depth1.weight.copy_(label_init_tensors[0])
                if label_init_tensors[1].shape == self.label_emb_depth2.weight.shape:
                    self.label_emb_depth2.weight.copy_(label_init_tensors[1])

        self.classifier1 = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, 512), nn.ReLU(), nn.Dropout(dropout), nn.Linear(512, num_labels_by_depth[0]))
        self.classifier2 = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, 512), nn.ReLU(), nn.Dropout(dropout), nn.Linear(512, num_labels_by_depth[1]))
        self.classifier3 = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, 512), nn.ReLU(), nn.Dropout(dropout), nn.Linear(512, num_labels_by_depth[2]))

    def _masked_mean(self, seq: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask_f = mask.float().unsqueeze(-1)
        summed = (seq * mask_f).sum(dim=1)
        denom = mask_f.sum(dim=1).clamp(min=1.0)
        return summed / denom

    def encode_representations(
        self,
        input_ids: torch.Tensor,
        edit_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        hunk_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return commit representations for depth 1, depth 2, and depth 3."""
        bsz, nhunks, seq_len = input_ids.shape
        flat_input = input_ids.view(bsz * nhunks, seq_len)
        flat_edit = edit_ids.view(bsz * nhunks, seq_len)
        flat_mask = attention_mask.view(bsz * nhunks, seq_len)

        word_embeds = self.codebert.embeddings.word_embeddings(flat_input)
        edit_embeds = self.edit_embeddings(flat_edit)
        inputs_embeds = word_embeds + edit_embeds

        out = self.codebert(inputs_embeds=inputs_embeds, attention_mask=flat_mask)
        seq1 = out.last_hidden_state
        rep1_hunk = self._masked_mean(seq1, flat_mask)

        seq2, _ = self.depth2_encoder(seq1)
        rep2_hunk = self._masked_mean(seq2, flat_mask)

        seq3, _ = self.depth3_encoder(seq2)
        rep3_hunk = self._masked_mean(seq3, flat_mask)

        rep1_hunk = rep1_hunk.view(bsz, nhunks, -1)
        rep2_hunk = rep2_hunk.view(bsz, nhunks, -1)
        rep3_hunk = rep3_hunk.view(bsz, nhunks, -1)

        hmask = hunk_mask.float().unsqueeze(-1)
        denom = hmask.sum(dim=1).clamp(min=1.0)
        rep1 = (rep1_hunk * hmask).sum(dim=1) / denom
        rep2 = (rep2_hunk * hmask).sum(dim=1) / denom
        rep3 = (rep3_hunk * hmask).sum(dim=1) / denom
        return rep1, rep2, rep3

    def forward(
        self,
        input_ids: torch.Tensor,
        edit_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        hunk_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        rep1, rep2, rep3 = self.encode_representations(input_ids, edit_ids, attention_mask, hunk_mask)
        logits1 = self.classifier1(self.dropout(rep1))

        if labels is not None:
            p1 = labels[:, 0]
            p2 = labels[:, 1]
        else:
            p1 = torch.argmax(logits1, dim=-1)
            p2 = None

        logits2 = self.classifier2(self.dropout(rep2 + self.label_emb_depth1(p1)))
        if p2 is None:
            p2 = torch.argmax(logits2, dim=-1)
        logits3 = self.classifier3(self.dropout(rep3 + self.label_emb_depth2(p2)))

        result = {"logits": [logits1, logits2, logits3], "representations": [rep1, rep2, rep3]}
        if labels is not None:
            losses = []
            for d, logits in enumerate(result["logits"]):
                losses.append(nn.functional.cross_entropy(logits, labels[:, d]))
            result["loss"] = sum(losses) / len(losses)
        return result

    def logits_for_depth(self, rep: torch.Tensor, depth_idx: int, parent_idx: Optional[int] = None) -> torch.Tensor:
        """Single-sample logits for beam search. depth_idx: 0, 1, 2."""
        if rep.dim() == 1:
            rep = rep.unsqueeze(0)
        if depth_idx == 0:
            return self.classifier1(rep).squeeze(0)
        if depth_idx == 1:
            if parent_idx is None:
                raise ValueError("parent_idx is required for depth 2")
            p = torch.tensor([parent_idx], device=rep.device, dtype=torch.long)
            return self.classifier2(rep + self.label_emb_depth1(p)).squeeze(0)
        if depth_idx == 2:
            if parent_idx is None:
                raise ValueError("parent_idx is required for depth 3")
            p = torch.tensor([parent_idx], device=rep.device, dtype=torch.long)
            return self.classifier3(rep + self.label_emb_depth2(p)).squeeze(0)
        raise ValueError(f"Unsupported depth_idx={depth_idx}")


@torch.no_grad()
def build_label_init_tensors(codebert_dir: str, tokenizer: Any, schema: CWESchema, device: torch.device) -> List[torch.Tensor]:
    """Initialize parent label embeddings using CodeBERT embeddings of CWE names/IDs."""
    encoder = AutoModel.from_pretrained(codebert_dir).to(device)
    encoder.eval()
    tensors: List[torch.Tensor] = []
    for depth in [0, 1]:
        reps = []
        for cwe in schema.depth_labels[depth]:
            text = cwe
            enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=32).to(device)
            out = encoder(**enc).last_hidden_state[:, 0, :].squeeze(0)
            reps.append(out.detach().cpu())
        tensors.append(torch.stack(reps, dim=0))
    del encoder
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return tensors


# =============================================================================
# Beam search inference
# =============================================================================


def allowed_indices(schema: CWESchema, parent_cwe: str, depth_idx: int) -> List[int]:
    labels = schema.children.get(parent_cwe, [])
    idxs = [schema.label_to_idx[depth_idx][lab] for lab in labels if lab in schema.label_to_idx[depth_idx]]
    if not idxs:
        idxs = list(range(len(schema.depth_labels[depth_idx])))
    return idxs


def topk_from_allowed(log_probs: torch.Tensor, allowed: List[int], k: int) -> List[Tuple[int, float]]:
    allowed_tensor = torch.tensor(allowed, device=log_probs.device, dtype=torch.long)
    values = log_probs.index_select(0, allowed_tensor)
    kk = min(k, values.numel())
    top_vals, top_pos = torch.topk(values, k=kk)
    return [(int(allowed[int(pos.item())]), float(val.item())) for val, pos in zip(top_vals, top_pos)]


@torch.no_grad()
def beam_search_one(model: TreeVulModel, reps: Sequence[torch.Tensor], schema: CWESchema, beam_size: int = BEAM_SIZE) -> Tuple[List[str], float]:
    """Tree-aware beam search for one sample. Returns predicted CWE path y1..y3 and log score."""
    model.eval()
    # Depth 1 from root.
    logits1 = model.logits_for_depth(reps[0], 0)
    lp1 = torch.log_softmax(logits1, dim=-1)
    beams: List[Tuple[List[int], List[str], float]] = []
    for idx1, score1 in topk_from_allowed(lp1, allowed_indices(schema, ROOT, 0), beam_size):
        cwe1 = schema.idx_to_label[0][idx1]
        beams.append(([idx1], [cwe1], score1))

    # Depth 2 and 3.
    for depth_idx in [1, 2]:
        new_beams: List[Tuple[List[int], List[str], float]] = []
        for idx_path, cwe_path, score in beams:
            parent_idx = idx_path[-1]
            parent_cwe = cwe_path[-1]
            logits = model.logits_for_depth(reps[depth_idx], depth_idx, parent_idx=parent_idx)
            lp = torch.log_softmax(logits, dim=-1)
            allowed = allowed_indices(schema, parent_cwe, depth_idx)
            for next_idx, next_score in topk_from_allowed(lp, allowed, beam_size):
                next_cwe = schema.idx_to_label[depth_idx][next_idx]
                new_beams.append((idx_path + [next_idx], cwe_path + [next_cwe], score + next_score))
        new_beams.sort(key=lambda x: x[2], reverse=True)
        beams = new_beams[:beam_size]

    best = max(beams, key=lambda x: x[2])
    return best[1], best[2]


# =============================================================================
# Training / evaluation
# =============================================================================


def move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out = dict(batch)
    for key in ["input_ids", "edit_ids", "attention_mask", "hunk_mask", "labels"]:
        out[key] = out[key].to(device)
    return out


def train_one_epoch(model: TreeVulModel, loader: DataLoader, optimizer: torch.optim.Optimizer, scheduler: Any, device: torch.device) -> float:
    model.train()
    losses = []
    progress = tqdm(loader, desc="train", leave=False)
    for batch in progress:
        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        out = model(batch["input_ids"], batch["edit_ids"], batch["attention_mask"], batch["hunk_mask"], labels=batch["labels"])
        loss = out["loss"]
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        losses.append(float(loss.item()))
        progress.set_postfix(loss=np.mean(losses))
    return float(np.mean(losses)) if losses else 0.0


@torch.no_grad()
def predict(model: TreeVulModel, loader: DataLoader, schema: CWESchema, device: torch.device) -> List[Dict[str, Any]]:
    model.eval()
    preds: List[Dict[str, Any]] = []
    for batch in tqdm(loader, desc="predict", leave=False):
        batch_dev = move_batch_to_device(batch, device)
        reps = model.encode_representations(
            batch_dev["input_ids"], batch_dev["edit_ids"], batch_dev["attention_mask"], batch_dev["hunk_mask"]
        )
        bsz = batch_dev["input_ids"].shape[0]
        for i in range(bsz):
            sample_reps = [r[i].detach() for r in reps]
            pred_path, score = beam_search_one(model, sample_reps, schema, BEAM_SIZE)
            true_path = batch["true_paths"][i]
            meta = batch["metas"][i]
            preds.append({
                "example_id": batch["example_ids"][i],
                "repo": meta.get("repo", ""),
                "commit_id": meta.get("commit_id", ""),
                "file_name": meta.get("file_name", ""),
                "PL": meta.get("PL", ""),
                "cve": meta.get("cve", ""),
                "true_depth1": true_path[0],
                "true_depth2": true_path[1],
                "true_depth3": true_path[2],
                "pred_depth1": pred_path[0],
                "pred_depth2": pred_path[1],
                "pred_depth3": pred_path[2],
                "true_path": " > ".join(true_path),
                "pred_path": " > ".join(pred_path),
                "beam_log_score": score,
                "path_fraction": path_fraction(true_path, pred_path),
                "correct_depth1": int(true_path[0] == pred_path[0]),
                "correct_depth2": int(true_path[1] == pred_path[1]),
                "correct_depth3": int(true_path[2] == pred_path[2]),
            })
    return preds


def path_fraction(true_path: Sequence[str], pred_path: Sequence[str]) -> float:
    if not true_path:
        return 0.0
    return float(len(set(true_path) & set(pred_path)) / len(true_path))


def depth_metrics(preds: List[Dict[str, Any]], depth: int) -> Dict[str, Any]:
    y_true = [p[f"true_depth{depth}"] for p in preds]
    y_pred = [p[f"pred_depth{depth}"] for p in preds]
    labels = sorted(set(y_true) | set(y_pred), key=cwe_sort_key)
    return {
        "depth": depth,
        "num_examples": len(preds),
        "num_labels": len(labels),
        "Accuracy": float(accuracy_score(y_true, y_pred)) if preds else 0.0,
        "Weighted_F1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)) if preds else 0.0,
        "Macro_F1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)) if preds else 0.0,
        "MCC": float(matthews_corrcoef(y_true, y_pred)) if preds and len(labels) > 1 else 0.0,
    }


def compute_metrics(preds: List[Dict[str, Any]]) -> Dict[str, Any]:
    d1 = depth_metrics(preds, 1)
    d2 = depth_metrics(preds, 2)
    d3 = depth_metrics(preds, 3)
    return {
        "num_predictions": len(preds),
        "Depth_1": d1,
        "Depth_2": d2,
        "Depth_3": d3,
        "Path_Fraction_PF": float(np.mean([p["path_fraction"] for p in preds])) if preds else 0.0,
    }


def make_confusion_df(preds: List[Dict[str, Any]], depth: int) -> pd.DataFrame:
    y_true = [p[f"true_depth{depth}"] for p in preds]
    y_pred = [p[f"pred_depth{depth}"] for p in preds]
    labels = sorted(set(y_true) | set(y_pred), key=cwe_sort_key)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return pd.DataFrame(cm, index=[f"true_{x}" for x in labels], columns=[f"pred_{x}" for x in labels])


def flatten_metrics(metrics: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    for key in ["Depth_1", "Depth_2", "Depth_3"]:
        block = dict(metrics[key])
        block["metric_group"] = key
        rows.append(block)
    rows.append({
        "metric_group": "Path",
        "depth": "all",
        "num_examples": metrics.get("num_predictions", 0),
        "num_labels": "",
        "Accuracy": "",
        "Weighted_F1": "",
        "Macro_F1": "",
        "MCC": "",
        "Path_Fraction_PF": metrics.get("Path_Fraction_PF", 0.0),
    })
    cols = ["metric_group", "depth", "num_examples", "num_labels", "Accuracy", "Weighted_F1", "Macro_F1", "MCC", "Path_Fraction_PF"]
    return pd.DataFrame(rows)[cols]


def save_outputs(
    out_dir: Path,
    train: List[Dict[str, Any]],
    valid: List[Dict[str, Any]],
    test: List[Dict[str, Any]],
    schema: CWESchema,
    history: List[Dict[str, Any]],
    valid_preds: List[Dict[str, Any]],
    test_preds: List[Dict[str, Any]],
    valid_metrics: Dict[str, Any],
    test_metrics: Dict[str, Any],
) -> None:
    ensure_dir(out_dir)

    if SAVE_NORMALIZED_SPLITS:
        for name, rows in [("training", train), ("validation", valid), ("testing", test)]:
            write_json(out_dir / f"{name}.json", rows)
            write_jsonl(out_dir / f"{name}.jsonl", rows)
            pd.DataFrame(rows).to_csv(out_dir / f"{name}.csv", index=False)

    write_json(out_dir / "cwe_schema.json", schema.to_dict())
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
    flatten_metrics(valid_metrics).to_csv(out_dir / "validation_metrics.csv", index=False)
    flatten_metrics(test_metrics).to_csv(out_dir / "testing_metrics.csv", index=False)

    for split_name, preds in [("validation", valid_preds), ("testing", test_preds)]:
        for depth in [1, 2, 3]:
            make_confusion_df(preds, depth).to_csv(out_dir / f"{split_name}_confusion_depth{depth}.csv")

    # Excel workbook with all important results.
    xlsx_path = out_dir / "treevul_results.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        flatten_metrics(test_metrics).to_excel(writer, sheet_name="test_metrics", index=False)
        flatten_metrics(valid_metrics).to_excel(writer, sheet_name="valid_metrics", index=False)
        pd.DataFrame(test_preds).to_excel(writer, sheet_name="test_predictions", index=False)
        pd.DataFrame(valid_preds).to_excel(writer, sheet_name="valid_predictions", index=False)
        pd.DataFrame(history).to_excel(writer, sheet_name="training_history", index=False)
        pd.DataFrame({
            "split": ["training", "validation", "testing"],
            "num_examples": [len(train), len(valid), len(test)],
        }).to_excel(writer, sheet_name="split_summary", index=False)
        pd.DataFrame([
            {"depth": d + 1, "label_index": i, "cwe": cwe}
            for d, labels in enumerate(schema.depth_labels)
            for i, cwe in enumerate(labels)
        ]).to_excel(writer, sheet_name="label_mapping", index=False)
        for depth in [1, 2, 3]:
            make_confusion_df(test_preds, depth).to_excel(writer, sheet_name=f"test_cm_d{depth}")


def build_dataloaders(train: List[Dict[str, Any]], valid: List[Dict[str, Any]], test: List[Dict[str, Any]], tokenizer: Any) -> Tuple[DataLoader, DataLoader, DataLoader]:
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 1
    collate_fn = lambda batch: collate_treevul(batch, pad_id)
    train_loader = DataLoader(TreeVulDataset(train, tokenizer), batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, collate_fn=collate_fn)
    valid_loader = DataLoader(TreeVulDataset(valid, tokenizer), batch_size=EVAL_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, collate_fn=collate_fn)
    test_loader = DataLoader(TreeVulDataset(test, tokenizer), batch_size=EVAL_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, collate_fn=collate_fn)
    return train_loader, valid_loader, test_loader


def main() -> None:
    set_seed(RANDOM_SEED)
    out_dir = ensure_dir(OUTPUT_DIR)
    config = {
        "CODEBERT_DIR": CODEBERT_DIR,
        "INPUT": INPUT,
        "TRAIN_INPUT_JSON": TRAIN_INPUT_JSON,
        "VALID_INPUT_JSON": VALID_INPUT_JSON,
        "TEST_INPUT_JSON": TEST_INPUT_JSON,
        "OUTPUT_DIR": OUTPUT_DIR,
        "RANDOM_SEED": RANDOM_SEED,
        "TARGET_DEPTH": TARGET_DEPTH,
        "TRAIN_RATIO": TRAIN_RATIO,
        "VALID_RATIO": VALID_RATIO,
        "TEST_RATIO": TEST_RATIO,
        "MAX_HUNKS": MAX_HUNKS,
        "MAX_REM_TOKENS": MAX_REM_TOKENS,
        "MAX_ADD_TOKENS": MAX_ADD_TOKENS,
        "BATCH_SIZE": BATCH_SIZE,
        "EPOCHS": EPOCHS,
        "PATIENCE": PATIENCE,
        "BEAM_SIZE": BEAM_SIZE,
    }
    write_json(out_dir / "run_config.json", config)

    print("=" * 100)
    print("TREEVUL reproduction: CodeBERT + edit embeddings + hierarchical chained heads + beam search")
    print("=" * 100)
    print(f"CodeBERT directory: {CODEBERT_DIR}")
    print(f"Output directory  : {OUTPUT_DIR}")

    train, valid, test = load_or_split_dataset()
    all_rows = train + valid + test
    if not train or not valid or not test:
        raise RuntimeError(f"Empty split after preprocessing: train={len(train)}, valid={len(valid)}, test={len(test)}")

    schema = build_cwe_schema(all_rows, TARGET_DEPTH)
    train = attach_label_indices(train, schema)
    valid = attach_label_indices(valid, schema)
    test = attach_label_indices(test, schema)

    print(f"Train/Validation/Test: {len(train)} / {len(valid)} / {len(test)}")
    print(f"Depth labels: {[len(x) for x in schema.depth_labels]}")

    device = torch.device("cuda" if USE_CUDA and torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(CODEBERT_DIR)
    train_loader, valid_loader, test_loader = build_dataloaders(train, valid, test, tokenizer)

    print("Building CodeBERT-initialized CWE parent label embeddings...")
    label_init_tensors = build_label_init_tensors(CODEBERT_DIR, tokenizer, schema, device)

    model = TreeVulModel(
        CODEBERT_DIR,
        num_labels_by_depth=[len(x) for x in schema.depth_labels],
        dropout=DROPOUT,
        label_init_tensors=label_init_tensors,
    ).to(device)

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
    warmup_steps = min(WARMUP_STEPS, max(1, total_steps // 3))
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    best_metric = -1.0
    bad_epochs = 0
    history: List[Dict[str, Any]] = []
    best_path = out_dir / "best_treevul_model.pt"

    for epoch in range(1, EPOCHS + 1):
        start = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, device)
        valid_preds = predict(model, valid_loader, schema, device)
        valid_metrics = compute_metrics(valid_preds)
        monitor = valid_metrics["Depth_3"]["Macro_F1"]
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "valid_depth3_macro_f1": valid_metrics["Depth_3"]["Macro_F1"],
            "valid_depth3_weighted_f1": valid_metrics["Depth_3"]["Weighted_F1"],
            "valid_depth3_mcc": valid_metrics["Depth_3"]["MCC"],
            "valid_pf": valid_metrics["Path_Fraction_PF"],
            "seconds": round(time.time() - start, 2),
        }
        history.append(row)
        print(json.dumps(row, indent=2))

        if monitor > best_metric:
            best_metric = monitor
            bad_epochs = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "schema": schema.to_dict(),
                "config": config,
                "best_epoch": epoch,
                "best_valid_depth3_macro_f1": best_metric,
            }, best_path)
            print(f"Saved new best model to {best_path}")
        else:
            bad_epochs += 1
            print(f"No improvement. bad_epochs={bad_epochs}/{PATIENCE}")
            if bad_epochs >= PATIENCE:
                print("Early stopping triggered.")
                break

    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    valid_preds = predict(model, valid_loader, schema, device)
    test_preds = predict(model, test_loader, schema, device)
    valid_metrics = compute_metrics(valid_preds)
    test_metrics = compute_metrics(test_preds)

    save_outputs(out_dir, train, valid, test, schema, history, valid_preds, test_preds, valid_metrics, test_metrics)

    print("\nValidation metrics:")
    print(json.dumps(valid_metrics, indent=2))
    print("\nTesting metrics:")
    print(json.dumps(test_metrics, indent=2))
    print("\nSaved outputs to:")
    print(out_dir)
    print(f"Excel: {out_dir / 'treevul_results.xlsx'}")
    print(f"JSON:  {out_dir / 'testing_metrics.json'}")
    print(f"CSV:   {out_dir / 'testing_metrics.csv'}")


if __name__ == "__main__":
    main()
