from __future__ import annotations

import json
import os
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from tqdm import tqdm

try:
    from openai import OpenAI
except Exception: 
    OpenAI = None


TRAIN_INPUT_JSON = "/path/software_defects/CWEMAP_data/dataset_name/dataset_name_train_set.json"
VALID_INPUT_JSON = "/path/software_defects/CWEMAP_data/dataset_name/dataset_name_vali_set.json"
TEST_INPUT_JSON = "/path/software_defects/CWEMAP_data/dataset_name/dataset_name_test_set.json"

RAW_INPUT_JSON = ""
OUTPUT_DIR = "/path/software_defects/CWEMAP_data/dataset_name/vulread_baseline_results"

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-v4-flash"
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "PASTE_YOUR_DEEPSEEK_API_KEY_HERE")

RANDOM_SEED = 42
TRAIN_RATIO = 0.80
VALID_RATIO = 0.10
TEST_RATIO = 0.10

RUN_DEEPSEEK_INFERENCE = True
RUN_EVALUATION = True
MAX_VALID_API_SAMPLES: Optional[int] = None
MAX_TEST_API_SAMPLES: Optional[int] = None
API_TEMPERATURE = 0.0
API_MAX_TOKENS = 700
API_SLEEP_SECONDS = 0.10
MAX_CODE_CHARS = 8500
MAX_KG_CONTEXT_CHARS = 3500

CANDIDATE_CWE_TOP_K = 12
FEWSHOT_K = 3
FORCE_CWE_FROM_CANDIDATES = True
FALLBACK_TO_RETRIEVAL_TOP1_CWE = True
SAVE_NORMALIZED_SPLITS = True
OTHER_LABEL = "CWE-OTHER"


ABSTRACTION_CLASSES: Dict[str, Dict[str, Any]] = {
    "Input Validation": {
        "description": "Improper validation, sanitization, neutralization, or checking of external input.",
        "keywords": ["input", "validate", "validation", "sanitize", "sanitization", "check", "parse", "filter", "escape", "length", "range"],
    },
    "File and Path Handling": {
        "description": "Unsafe file, directory, archive, symbolic-link, pathname, or resource path handling.",
        "keywords": ["path", "file", "directory", "dir", "archive", "zip", "tar", "traversal", "link", "symlink", "filename", "origin"],
    },
    "Memory Management": {
        "description": "Memory corruption, bounds, allocation, deallocation, leaks, use-after-free, buffer errors.",
        "keywords": ["memory", "buffer", "overflow", "underflow", "free", "malloc", "alloc", "calloc", "realloc", "leak", "bounds", "pointer", "null"],
    },
    "Access Control": {
        "description": "Missing or incorrect authorization, authentication, privilege, permission, or access checks.",
        "keywords": ["access", "authorization", "authentication", "permission", "privilege", "secure", "suid", "guid", "credentials", "role"],
    },
    "Cryptography": {
        "description": "Weak, missing, or incorrect cryptographic use, key management, randomness, or certificate validation.",
        "keywords": ["crypto", "cryptographic", "encrypt", "decrypt", "cipher", "certificate", "random", "key", "hash", "tls", "ssl"],
    },
    "Error and Exception Handling": {
        "description": "Unchecked return values, improper exception handling, ignored errors, or incomplete error propagation.",
        "keywords": ["error", "exception", "return value", "unchecked", "failure", "status", "errno", "warn", "assert"],
    },
    "Resource Management": {
        "description": "Improper handling of resources, lifetime, locks, sessions, handles, sockets, or cleanup.",
        "keywords": ["resource", "handle", "socket", "lock", "unlock", "release", "cleanup", "close", "lifetime", "destroy"],
    },
    "Concurrency": {
        "description": "Race conditions, synchronization bugs, deadlocks, time-of-check time-of-use errors.",
        "keywords": ["race", "concurrent", "thread", "mutex", "synchronization", "deadlock", "toctou"],
    },
    "Information Exposure": {
        "description": "Unintended exposure of sensitive information, credentials, memory contents, or system details.",
        "keywords": ["information", "exposure", "leak", "disclosure", "sensitive", "secret", "credential", "privacy"],
    },
    "Code Injection and Command Execution": {
        "description": "Unsafe command execution, code injection, eval, deserialization, script execution, or interpreter abuse.",
        "keywords": ["command", "execute", "execution", "eval", "injection", "deserialize", "script", "shell", "interpreter"],
    },
    "Configuration and Environment": {
        "description": "Insecure defaults, deployment misconfiguration, environment assumptions, or policy errors.",
        "keywords": ["configuration", "config", "default", "environment", "policy", "setting", "option"],
    },
    "Numeric and Type Handling": {
        "description": "Integer overflows, truncation, signedness, casting, type confusion, numeric boundary errors.",
        "keywords": ["integer", "overflow", "underflow", "signed", "unsigned", "type", "cast", "conversion", "numeric", "size", "len", "capacity"],
    },
    "Protocol and State Logic": {
        "description": "Incorrect protocol state, workflow, ordering, session state, or business/security logic.",
        "keywords": ["state", "protocol", "logic", "workflow", "sequence", "session", "order", "transition"],
    },
}

CWE_RE = re.compile(r"CWE[-_ ]?(\d+)", re.IGNORECASE)
JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)
TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}|CWE-\d+|\$ORIGIN|\$PLATFORM", re.IGNORECASE)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_json_any(path: str | Path) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    text = p.read_text(encoding="utf-8", errors="replace").strip()
    if not text:
        return []
    if text.startswith("["):
        obj = json.loads(text)
        if not isinstance(obj, list):
            raise ValueError(f"Top-level JSON must be a list: {p}")
        return [x for x in obj if isinstance(x, dict)]
    if text.startswith("{"):
        obj = json.loads(text)
        if isinstance(obj, dict):
            for key in ["data", "samples", "rows", "items", "records"]:
                if isinstance(obj.get(key), list):
                    return [x for x in obj[key] if isinstance(x, dict)]
            return [obj]
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
        if value and all(isinstance(x, list) for x in value):
            return "\n".join("\n".join(str(y) for y in x) for x in value)
        return "\n".join(str(x) for x in value if str(x).strip())
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def normalize_cwe(value: Any) -> str:
    if value is None:
        return "NO_CWE"
    if isinstance(value, (list, tuple)):
        value = " ".join(str(x) for x in value)
    text = str(value).strip()
    if not text or text.lower() in {"none", "null", "nan", "no_cwe", "nocwe", "n/a", "na", "-"}:
        return "NO_CWE"
    if "noinfo" in text.lower() or "unknown" in text.lower():
        return "NO_CWE"
    m = CWE_RE.search(text)
    if m:
        return f"CWE-{int(m.group(1))}"
    if text.isdigit():
        return f"CWE-{int(text)}"
    return "NO_CWE"


def cwe_sort_key(label: str) -> Tuple[int, str]:
    if label == OTHER_LABEL:
        return (10**12, label)
    m = re.search(r"\d+", label or "")
    return (int(m.group(0)) if m else 10**9, label)


def extract_all_cwes(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        value = " ".join(str(x) for x in value)
    return [f"CWE-{int(m.group(1))}" for m in CWE_RE.finditer(str(value))]


def extract_path_list(value: Any) -> List[List[str]]:
    paths: List[List[str]] = []
    if isinstance(value, str):
        cwes = extract_all_cwes(value)
        if cwes:
            paths.append(cwes)
    elif isinstance(value, list):
        if value and all(not isinstance(x, list) for x in value):
            cwes = [normalize_cwe(x) for x in value]
            cwes = [x for x in cwes if x != "NO_CWE"]
            if cwes:
                paths.append(cwes)
        else:
            for item in value:
                if isinstance(item, list):
                    cwes = [normalize_cwe(x) for x in item]
                    cwes = [x for x in cwes if x != "NO_CWE"]
                    if cwes:
                        paths.append(cwes)
                else:
                    cwe = normalize_cwe(item)
                    if cwe != "NO_CWE":
                        paths.append([cwe])
    return paths


def label_from_record(record: Dict[str, Any]) -> Tuple[str, List[List[str]]]:
    cwe = normalize_cwe(record.get("cwe_list"))
    if cwe == "NO_CWE":
        cwe = normalize_cwe(record.get("cwe") or record.get("cwe_id"))
    paths = extract_path_list(record.get("path_list") or record.get("cwe_path") or record.get("path"))
    if cwe == "NO_CWE" and paths:
        cwe = paths[0][-1]
    return cwe, paths


def truncate(text: str, n: int) -> str:
    text = text or ""
    if len(text) <= n:
        return text
    return text[: n - 80] + "\n...[TRUNCATED]...\n" + text[-40:]


def tokens(text: str, max_tokens: int = 2500) -> List[str]:
    toks = [t.lower() for t in TOKEN_RE.findall(text or "")]
    stop = {"return", "const", "char", "int", "void", "static", "while", "for", "else", "if", "ifdef", "endif", "struct", "class"}
    return [t for t in toks if t not in stop and len(t) > 2][:max_tokens]


def simple_code_entities(text: str, max_entities: int = 80) -> List[str]:
    ents: List[str] = []
    for m in re.finditer(r"['\"]([^'\"]{1,120})['\"]", text or ""):
        s = m.group(1)
        if any(ch in s for ch in ["/", "\\", ".", "$", "%"]):
            ents.append(s)
    for m in re.finditer(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(", text or ""):
        ents.append(m.group(1))
    ents.extend(tokens(text, max_tokens=1200))
    seen, out = set(), []
    for e in ents:
        if e not in seen:
            seen.add(e)
            out.append(e)
        if len(out) >= max_entities:
            break
    return out


def choose_abstraction_classes(text: str, cwe: str = "NO_CWE") -> List[str]:
    hay = (text or "").lower() + " " + (cwe or "").lower()
    scored: List[Tuple[int, str]] = []
    for cls, meta in ABSTRACTION_CLASSES.items():
        score = sum(1 for kw in meta["keywords"] if kw.lower() in hay)
        if score:
            scored.append((score, cls))
    scored.sort(reverse=True)
    if scored:
        return [c for _, c in scored[:3]]
    return []


def make_model_input(record: Dict[str, Any]) -> str:
    msg = safe_text(record.get("msg") or record.get("commit_message") or record.get("message") or "")
    rem = safe_text(record.get("REM_DIFF") or record.get("removed_code") or record.get("deleted_code") or "")
    add = safe_text(record.get("ADD_DIFF") or record.get("added_code") or "")
    # Ground-truth CWE/path fields are intentionally excluded from this input.
    return (
        "Commit message:\n" + msg.strip() +
        "\n\nRemoved code (REM_DIFF):\n" + rem.strip() +
        "\n\nAdded code (ADD_DIFF):\n" + add.strip()
    ).strip()


def make_example_id(record: Dict[str, Any], row_index: int) -> str:
    for key in ["idx", "commit_id", "hash", "cve_list", "cve"]:
        val = record.get(key)
        if val is not None and str(val).strip():
            return str(val)
    return f"row_{row_index}"


def normalize_record(record: Dict[str, Any], row_index: int) -> Optional[Dict[str, Any]]:
    model_input = make_model_input(record)
    label, paths = label_from_record(record)
    if not model_input.strip():
        return None
    if label == "NO_CWE":
        return None
    return {
        "example_id": make_example_id(record, row_index),
        "project": safe_text(record.get("project") or record.get("repo") or ""),
        "commit_id": safe_text(record.get("commit_id") or record.get("commit") or record.get("hash") or ""),
        "hash": safe_text(record.get("hash") or ""),
        "msg": safe_text(record.get("msg") or record.get("commit_message") or record.get("message") or ""),
        "REM_DIFF": safe_text(record.get("REM_DIFF") or record.get("removed_code") or record.get("deleted_code") or ""),
        "ADD_DIFF": safe_text(record.get("ADD_DIFF") or record.get("added_code") or ""),
        "cwe_list": record.get("cwe_list", record.get("cwe", record.get("cwe_id", ""))),
        "path_list": paths,
        "label_cwe": label,
        "code": model_input,
        "entities": simple_code_entities(model_input),
        "abstraction_classes": choose_abstraction_classes(model_input),
    }


def normalize_records(rows: Sequence[Dict[str, Any]], split_name: str) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    skipped = 0
    for i, row in enumerate(rows):
        item = normalize_record(row, i)
        if item is None:
            skipped += 1
        else:
            normalized.append(item)
    print(f"{split_name}: kept {len(normalized)} records, skipped {skipped} records.")
    return normalized


def stratify_labels(rows: Sequence[Dict[str, Any]]) -> Optional[List[str]]:
    labels = [r["label_cwe"] for r in rows]
    counts = Counter(labels)
    if len(counts) > 1 and min(counts.values()) >= 2:
        return labels
    return None


def split_raw(rows: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    if not np.isclose(TRAIN_RATIO + VALID_RATIO + TEST_RATIO, 1.0):
        raise ValueError("TRAIN_RATIO + VALID_RATIO + TEST_RATIO must equal 1.0")
    stratify = stratify_labels(rows)
    train, temp = train_test_split(rows, test_size=VALID_RATIO + TEST_RATIO, random_state=RANDOM_SEED, shuffle=True, stratify=stratify)
    temp_stratify = stratify_labels(temp)
    valid_fraction = VALID_RATIO / (VALID_RATIO + TEST_RATIO)
    valid, test = train_test_split(temp, test_size=1.0 - valid_fraction, random_state=RANDOM_SEED, shuffle=True, stratify=temp_stratify)
    return train, valid, test


def load_dataset() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    train_path = Path(TRAIN_INPUT_JSON)
    valid_path = Path(VALID_INPUT_JSON)
    test_path = Path(TEST_INPUT_JSON)
    if train_path.exists() and valid_path.exists() and test_path.exists():
        print("Loading existing TreeVulDataset train/validation/test split files.")
        train = normalize_records(read_json_any(train_path), "training")
        valid = normalize_records(read_json_any(valid_path), "validation")
        test = normalize_records(read_json_any(test_path), "testing")
        return train, valid, test
    if RAW_INPUT_JSON and Path(RAW_INPUT_JSON).exists():
        rows = normalize_records(read_json_any(RAW_INPUT_JSON), "raw")
        return split_raw(rows)
    raise FileNotFoundError("Set TRAIN_INPUT_JSON/VALID_INPUT_JSON/TEST_INPUT_JSON to existing files, or set RAW_INPUT_JSON.")

# KG and retrieval

def build_kg(train_examples: List[Dict[str, Any]]) -> Dict[str, Any]:
    kg: Dict[str, Any] = {
        "abstraction_classes": ABSTRACTION_CLASSES,
        "cwe_to_classes": defaultdict(Counter),
        "class_to_cwes": defaultdict(Counter),
        "cwe_to_entities": defaultdict(Counter),
        "entity_to_cwes": defaultdict(Counter),
        "cwe_to_tokens": defaultdict(Counter),
        "cwe_paths": defaultdict(list),
        "cwe_frequency": Counter(),
    }
    for ex in train_examples:
        cwe = normalize_cwe(ex.get("label_cwe"))
        if cwe == "NO_CWE":
            continue
        kg["cwe_frequency"][cwe] += 1
        for path in ex.get("path_list") or []:
            if path and path not in kg["cwe_paths"][cwe]:
                kg["cwe_paths"][cwe].append(path)
        for cls in ex.get("abstraction_classes", []):
            kg["cwe_to_classes"][cwe][cls] += 1
            kg["class_to_cwes"][cls][cwe] += 1
        for ent in ex.get("entities", [])[:80]:
            kg["cwe_to_entities"][cwe][ent] += 1
            kg["entity_to_cwes"][ent][cwe] += 1
        for tok in tokens(ex.get("code", ""), max_tokens=1500):
            kg["cwe_to_tokens"][cwe][tok] += 1

    out: Dict[str, Any] = {"abstraction_classes": ABSTRACTION_CLASSES}
    for k in ["cwe_to_classes", "class_to_cwes", "cwe_to_entities", "entity_to_cwes", "cwe_to_tokens"]:
        out[k] = {kk: dict(vv) for kk, vv in kg[k].items()}
    out["cwe_paths"] = dict(kg["cwe_paths"])
    out["cwe_frequency"] = dict(kg["cwe_frequency"])
    return out


def build_retrieval_index(train_examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    index = []
    for ex in train_examples:
        cwe = normalize_cwe(ex.get("label_cwe"))
        if cwe == "NO_CWE":
            continue
        index.append({
            "example_id": ex["example_id"],
            "cwe": cwe,
            "path_list": ex.get("path_list", []),
            "tokens": set(tokens(ex.get("code", ""), max_tokens=1600)),
            "summary": truncate(ex.get("code", ""), 900),
        })
    return index


def retrieve_similar(ex: Dict[str, Any], retrieval_index: List[Dict[str, Any]], k: int = 5) -> List[Dict[str, Any]]:
    q = set(tokens(ex.get("code", ""), max_tokens=1600))
    scored = []
    for row in retrieval_index:
        inter = len(q & row["tokens"])
        union = len(q | row["tokens"]) or 1
        score = inter / union
        if score > 0:
            scored.append((score, row))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [r for _, r in scored[:k]]


def retrieve_kg_context(ex: Dict[str, Any], kg: Dict[str, Any], retrieval_index: List[Dict[str, Any]]) -> Dict[str, Any]:
    entities = ex.get("entities") or simple_code_entities(ex.get("code", ""))
    classes = ex.get("abstraction_classes") or choose_abstraction_classes(ex.get("code", ""))
    similar = retrieve_similar(ex, retrieval_index, k=max(FEWSHOT_K, CANDIDATE_CWE_TOP_K))

    scores = Counter()
    for rank, row in enumerate(similar):
        scores[row["cwe"]] += 100.0 / (rank + 1)
    for ent in entities:
        for cwe, freq in kg.get("entity_to_cwes", {}).get(ent, {}).items():
            scores[cwe] += min(freq, 15)
    for cls in classes:
        for cwe, freq in kg.get("class_to_cwes", {}).get(cls, {}).items():
            scores[cwe] += 2 * min(freq, 20)
    q_toks = Counter(tokens(ex.get("code", ""), max_tokens=1600))
    for cwe, tok_counts in kg.get("cwe_to_tokens", {}).items():
        overlap = 0
        for tok, cnt in q_toks.items():
            if tok in tok_counts:
                overlap += min(cnt, tok_counts[tok], 3)
        if overlap:
            scores[cwe] += min(overlap, 50)
    for cwe, freq in kg.get("cwe_frequency", {}).items():
        scores[cwe] += 0.01 * freq

    candidates = [c for c, _ in scores.most_common(CANDIDATE_CWE_TOP_K) if c != "NO_CWE"]
    if not candidates:
        candidates = [c for c, _ in Counter(kg.get("cwe_frequency", {})).most_common(CANDIDATE_CWE_TOP_K)]

    return {
        "entities": entities[:40],
        "abstraction_classes": classes[:3],
        "candidate_cwes_ranked": candidates,
        "candidate_paths": {cwe: kg.get("cwe_paths", {}).get(cwe, [])[:3] for cwe in candidates},
        "class_descriptions": {cls: ABSTRACTION_CLASSES.get(cls, {}).get("description", "") for cls in classes[:3]},
        "few_shot_examples": [
            {"cwe": row["cwe"], "path_list": row.get("path_list", [])[:2], "evidence_excerpt": row["summary"]}
            for row in similar[:FEWSHOT_K]
        ],
    }


def kg_context_to_text(ctx: Dict[str, Any]) -> str:
    return truncate(json.dumps(ctx, ensure_ascii=False, indent=2), MAX_KG_CONTEXT_CHARS)

# DeepSeek API and prompt

def get_deepseek_client() -> Optional[Any]:
    if not RUN_DEEPSEEK_INFERENCE:
        return None
    if OpenAI is None:
        print("[WARN] OpenAI SDK is not installed. Using retrieval-only fallback.")
        return None
    key = (DEEPSEEK_API_KEY or "").strip()
    if not key or key == "PASTE_YOUR_DEEPSEEK_API_KEY_HERE":
        print("[WARN] DEEPSEEK_API_KEY is missing. Using retrieval-only fallback.")
        return None
    return OpenAI(api_key=key, base_url=DEEPSEEK_BASE_URL)


def call_deepseek(client: Any, system_prompt: str, user_prompt: str, cache_key: str, cache_dir: Path) -> str:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{cache_key}.json"
    if cache_file.exists():
        try:
            return json.loads(cache_file.read_text(encoding="utf-8"))["response"]
        except Exception:
            pass
    resp = client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
        temperature=API_TEMPERATURE,
        max_tokens=API_MAX_TOKENS,
    )
    text = resp.choices[0].message.content or ""
    cache_file.write_text(json.dumps({"response": text}, ensure_ascii=False, indent=2), encoding="utf-8")
    time.sleep(API_SLEEP_SECONDS)
    return text


def parse_json_from_response(text: str) -> Dict[str, Any]:
    cleaned = (text or "").strip().replace("```json", "```").replace("```JSON", "```").strip()
    if "```" in cleaned:
        parts = cleaned.split("```")
        cleaned = max(parts, key=len).strip()
    m = JSON_OBJECT_RE.search(cleaned)
    if m:
        cleaned = m.group(0)
    try:
        return json.loads(cleaned)
    except Exception:
        cwes = extract_all_cwes(text)
        return {
            "cwe_id": cwes[0] if cwes else "NO_CWE",
            "confidence": 0.3,
            "reasoning": (text or "")[:1000],
            "parse_error": True,
        }


def make_inference_prompt(ex: Dict[str, Any], kg_ctx: Dict[str, Any]) -> Tuple[str, str]:
    candidates = kg_ctx.get("candidate_cwes_ranked", [])
    system = (
        "You are a senior software security analyst. Predict the CWE type of a security patch. "
        "Return only valid JSON, no markdown."
    )
    user = f"""
Analyze the TreeVul commit evidence using KG-guided vulnerability reasoning.

Input evidence contains only:
- commit message
- removed code from REM_DIFF
- added code from ADD_DIFF

Important leakage rule:
- The ground-truth CWE label and CWE path are NOT provided as input evidence.
- Predict the CWE from the code/message evidence only.

Candidate CWE rule:
- Choose cwe_id from this candidate list only: {candidates}
- Do not invent a CWE outside candidate_cwes_ranked.
- Prefer the CWE that best matches the vulnerability cause and repair transformation.

Output JSON fields exactly:
{{
  "cwe_id": "CWE-xxx",
  "confidence": float from 0 to 1,
  "entities": ["..."],
  "abstraction_classes": ["..."],
  "reasoning": "concise entity -> abstraction class -> CWE explanation"
}}

KG context and retrieved examples:
{kg_context_to_text(kg_ctx)}

Commit evidence:
{truncate(ex.get('code', ''), MAX_CODE_CHARS)}
""".strip()
    return system, user


def postprocess_prediction(parsed: Dict[str, Any], kg_ctx: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    candidates = [normalize_cwe(c) for c in kg_ctx.get("candidate_cwes_ranked", []) if normalize_cwe(c) != "NO_CWE"]
    raw_cwe = normalize_cwe(parsed.get("cwe_id", "NO_CWE"))
    if FORCE_CWE_FROM_CANDIDATES:
        mentioned = [c for c in extract_all_cwes(json.dumps(parsed, ensure_ascii=False)) if c in candidates]
        if raw_cwe in candidates:
            pred_cwe = raw_cwe
        elif mentioned:
            pred_cwe = mentioned[0]
        elif FALLBACK_TO_RETRIEVAL_TOP1_CWE and candidates:
            pred_cwe = candidates[0]
        else:
            pred_cwe = raw_cwe if raw_cwe != "NO_CWE" else (candidates[0] if candidates else "NO_CWE")
    else:
        pred_cwe = raw_cwe if raw_cwe != "NO_CWE" else (candidates[0] if candidates else "NO_CWE")
    parsed["postprocess_note"] = {
        "force_cwe_from_candidates": FORCE_CWE_FROM_CANDIDATES,
        "candidate_cwes_ranked": candidates,
        "raw_cwe": raw_cwe,
    }
    return pred_cwe, parsed

# Inference and evaluation

def run_inference(split_rows: List[Dict[str, Any]], split_name: str, kg: Dict[str, Any], retrieval_index: List[Dict[str, Any]], out_dir: Path, max_samples: Optional[int]) -> List[Dict[str, Any]]:
    client = get_deepseek_client()
    rows = split_rows[:max_samples] if max_samples is not None else split_rows
    cache_dir = out_dir / "api_cache" / split_name
    preds: List[Dict[str, Any]] = []

    for ex in tqdm(rows, desc=f"VulRead inference {split_name}"):
        ctx = retrieve_kg_context(ex, kg, retrieval_index)
        raw = ""
        if client is not None:
            system, user = make_inference_prompt(ex, ctx)
            cache_key = re.sub(r"[^A-Za-z0-9_.-]", "_", ex["example_id"])
            try:
                raw = call_deepseek(client, system, user, cache_key, cache_dir)
                parsed = parse_json_from_response(raw)
            except Exception as exc:
                parsed = {"cwe_id": "NO_CWE", "reasoning": f"API_ERROR: {exc}"}
        else:
            parsed = {
                "cwe_id": (ctx.get("candidate_cwes_ranked") or ["NO_CWE"])[0],
                "confidence": 0.0,
                "reasoning": "Retrieval-only fallback because API client is unavailable.",
            }

        pred_cwe, parsed = postprocess_prediction(parsed, ctx)
        true_cwe = normalize_cwe(ex.get("label_cwe"))
        preds.append({
            "example_id": ex.get("example_id", ""),
            "project": ex.get("project", ""),
            "commit_id": ex.get("commit_id", ""),
            "hash": ex.get("hash", ""),
            "true_cwe": true_cwe,
            "pred_cwe": pred_cwe,
            "correct": int(true_cwe == pred_cwe),
            "path_list": ex.get("path_list", []),
            "candidate_cwes_ranked": ctx.get("candidate_cwes_ranked", []),
            "raw_response": raw,
            "parsed_response": parsed,
            "msg": ex.get("msg", ""),
        })

    write_json(out_dir / f"{split_name}_predictions.json", preds)
    write_jsonl(out_dir / f"{split_name}_predictions.jsonl", preds)
    pd.DataFrame(preds).to_csv(out_dir / f"{split_name}_predictions.csv", index=False)
    return preds


def per_label_counts(y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]) -> List[Dict[str, Any]]:
    rows = []
    for lab in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == lab and p == lab)
        tn = sum(1 for t, p in zip(y_true, y_pred) if t != lab and p != lab)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != lab and p == lab)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == lab and p != lab)
        rows.append({"label": lab, "TP": tp, "TN": tn, "FP": fp, "FN": fn})
    return rows


def evaluate_predictions(preds: List[Dict[str, Any]], name: str, out_dir: Path) -> Dict[str, Any]:
    y_true = [normalize_cwe(r.get("true_cwe")) for r in preds]
    y_pred = [normalize_cwe(r.get("pred_cwe")) for r in preds]
    labels = sorted(set(y_true) | set(y_pred), key=cwe_sort_key)
    if not preds:
        return {"num_predictions": 0}
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    pd.DataFrame(cm, index=[f"true_{x}" for x in labels], columns=[f"pred_{x}" for x in labels]).to_csv(out_dir / f"{name}_confusion_matrix.csv")
    pd.DataFrame(per_label_counts(y_true, y_pred, labels)).to_csv(out_dir / f"{name}_per_cwe_tp_tn_fp_fn.csv", index=False)
    metrics = {
        "num_predictions": len(preds),
        "num_labels": len(labels),
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "Recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "F1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "Precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "Recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "F1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "F1_micro": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "MCC": float(matthews_corrcoef(y_true, y_pred)) if len(labels) > 1 else 0.0,
        "labels": labels,
        "confusion_matrix": cm.tolist(),
    }
    write_json(out_dir / f"{name}_metrics.json", metrics)
    pd.DataFrame([{ "metric": k, "value": v } for k, v in metrics.items() if not isinstance(v, (list, dict))]).to_csv(out_dir / f"{name}_metrics.csv", index=False)
    return metrics


def save_split(out_dir: Path, name: str, rows: List[Dict[str, Any]]) -> None:
    if not SAVE_NORMALIZED_SPLITS:
        return
    clean_rows = [{k: v for k, v in r.items() if k not in {"code", "entities", "abstraction_classes"}} for r in rows]
    write_json(out_dir / f"{name}.json", clean_rows)
    write_jsonl(out_dir / f"{name}.jsonl", clean_rows)
    pd.DataFrame(clean_rows).to_csv(out_dir / f"{name}.csv", index=False)


def save_excel(out_dir: Path, valid_preds: List[Dict[str, Any]], test_preds: List[Dict[str, Any]], valid_metrics: Dict[str, Any], test_metrics: Dict[str, Any], split_summary: Dict[str, Any]) -> None:
    with pd.ExcelWriter(out_dir / "vulread_baseline_results.xlsx", engine="openpyxl") as writer:
        pd.DataFrame(valid_preds).to_excel(writer, sheet_name="validation_predictions", index=False)
        pd.DataFrame(test_preds).to_excel(writer, sheet_name="testing_predictions", index=False)
        pd.DataFrame([{ "metric": k, "value": v } for k, v in valid_metrics.items() if not isinstance(v, (list, dict))]).to_excel(writer, sheet_name="validation_metrics", index=False)
        pd.DataFrame([{ "metric": k, "value": v } for k, v in test_metrics.items() if not isinstance(v, (list, dict))]).to_excel(writer, sheet_name="testing_metrics", index=False)
        pd.DataFrame([split_summary]).to_excel(writer, sheet_name="split_summary", index=False)

# Main

def main() -> None:
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    out_dir = ensure_dir(OUTPUT_DIR)

    print("=" * 100)
    print("VulRead baseline: KG-guided LLM/retrieval CWE prediction from msg + REM_DIFF + ADD_DIFF")
    print("=" * 100)
    print(f"TRAIN: {TRAIN_INPUT_JSON}")
    print(f"VALID: {VALID_INPUT_JSON}")
    print(f"TEST : {TEST_INPUT_JSON}")
    print(f"OUTPUT: {OUTPUT_DIR}")
    print("Leakage rule: cwe_list/path_list are labels only, never prompt input evidence.")

    train, valid, test = load_dataset()
    if not train or not valid or not test:
        raise RuntimeError(f"Empty split after preprocessing: train={len(train)}, valid={len(valid)}, test={len(test)}")

    for name, rows in [("training", train), ("validation", valid), ("testing", test)]:
        save_split(out_dir, name, rows)

    kg = build_kg(train)
    retrieval_index = build_retrieval_index(train)
    write_json(out_dir / "kg.json", kg)

    split_summary = {
        "train": len(train),
        "validation": len(valid),
        "testing": len(test),
        "train_unique_cwes": len(set(r["label_cwe"] for r in train)),
        "valid_unique_cwes": len(set(r["label_cwe"] for r in valid)),
        "test_unique_cwes": len(set(r["label_cwe"] for r in test)),
        "input_features": "msg + REM_DIFF + ADD_DIFF",
        "label_fields": "cwe_list or path_list",
        "leakage_rule": "CWE labels are not included in prompt evidence.",
        "retrieval_index_size": len(retrieval_index),
        "deepseek_model": DEEPSEEK_MODEL,
    }
    write_json(out_dir / "run_config.json", {
        "TRAIN_INPUT_JSON": TRAIN_INPUT_JSON,
        "VALID_INPUT_JSON": VALID_INPUT_JSON,
        "TEST_INPUT_JSON": TEST_INPUT_JSON,
        "OUTPUT_DIR": OUTPUT_DIR,
        "DEEPSEEK_BASE_URL": DEEPSEEK_BASE_URL,
        "DEEPSEEK_MODEL": DEEPSEEK_MODEL,
        "CANDIDATE_CWE_TOP_K": CANDIDATE_CWE_TOP_K,
        "FEWSHOT_K": FEWSHOT_K,
        "MAX_VALID_API_SAMPLES": MAX_VALID_API_SAMPLES,
        "MAX_TEST_API_SAMPLES": MAX_TEST_API_SAMPLES,
        "split_summary": split_summary,
    })
    write_json(out_dir / "split_summary.json", split_summary)

    valid_preds = run_inference(valid, "validation", kg, retrieval_index, out_dir, MAX_VALID_API_SAMPLES)
    test_preds = run_inference(test, "testing", kg, retrieval_index, out_dir, MAX_TEST_API_SAMPLES)

    valid_metrics = evaluate_predictions(valid_preds, "validation", out_dir) if RUN_EVALUATION else {}
    test_metrics = evaluate_predictions(test_preds, "testing", out_dir) if RUN_EVALUATION else {}
    save_excel(out_dir, valid_preds, test_preds, valid_metrics, test_metrics, split_summary)

    print("\nValidation metrics:")
    print(json.dumps(valid_metrics, indent=2))
    print("\nTesting metrics:")
    print(json.dumps(test_metrics, indent=2))
    print(f"\nDone. Results saved to: {out_dir}")


if __name__ == "__main__":
    main()
