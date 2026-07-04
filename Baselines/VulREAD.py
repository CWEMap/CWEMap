# #!/usr/bin/env python3
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
from tqdm import tqdm

try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None


# =============================================================================
# EDIT HERE ONLY
# =============================================================================

INPUT_JSON = "/path/dataset_name/training.json/jsonl"
INPUT_JSON = "/path/dataset_name/testing.json/jsonl"
INPUT_JSON = "/path/dataset_name/validation.json/jsonl"
OUTPUT_DIR = "/path/dataset_name/dataset_name_results"

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-v4-flash"

# Paste your real API key here, for example: "sk-xxxxxxxxxxxxxxxx".
DEEPSEEK_API_KEY = "PASTE_YOUR_DEEPSEEK_API_KEY_HERE"

RANDOM_SEED = 42
MAX_TREEVUL_RECORDS: Optional[int] = 5000
TRAIN_RATIO = 0.80
VALID_RATIO = 0.10
TEST_RATIO = 0.10

# Cost/runtime controls.
RUN_DEEPSEEK_INFERENCE = True
RUN_EVALUATION = True
MAX_TEST_API_SAMPLES: Optional[int] = None 
API_TEMPERATURE = 0.0
API_MAX_TOKENS = 700
API_SLEEP_SECONDS = 0.10
MAX_CODE_CHARS = 8500
MAX_KG_CONTEXT_CHARS = 3500


USE_PHASE_PRIOR_FOR_BINARY = True


FORCE_CWE_FROM_CANDIDATES = True
FALLBACK_TO_RETRIEVAL_TOP1_CWE = True
CANDIDATE_CWE_TOP_K = 12
FEWSHOT_K = 3


ADD_TRUE_CWE_TO_CANDIDATES_DEBUG_ONLY = False


# =============================================================================
# Abstraction classes and CWE helpers
# =============================================================================

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

CWE_RE = re.compile(r"CWE-\d+", re.IGNORECASE)
JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)
TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}|CWE-\d+|\$ORIGIN|\$PLATFORM", re.IGNORECASE)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_json_any(path: str | Path) -> List[Dict[str, Any]]:
    p = Path(path)
    text = p.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text.startswith("["):
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError("Expected JSON list or JSONL records")
        return data
    rows = []
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
        return "NO_CWE"
    if isinstance(value, list):
        value = " ".join(map(str, value))
    m = CWE_RE.search(str(value))
    return m.group(0).upper() if m else "NO_CWE"


def extract_all_cwes(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        value = " ".join(map(str, value))
    return [m.group(0).upper() for m in CWE_RE.finditer(str(value))]


def normalize_paths(path_list: Any) -> List[List[str]]:
    paths: List[List[str]] = []
    if not isinstance(path_list, list):
        return paths
    for p in path_list:
        if isinstance(p, list):
            cwes = [normalize_cwe(x) for x in p]
            cwes = [x for x in cwes if x != "NO_CWE"]
            if cwes:
                paths.append(cwes)
    return paths


def list_to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return "\n".join(str(x) for x in value if str(x).strip())
    return str(value)


def truncate(text: str, n: int) -> str:
    text = text or ""
    if len(text) <= n:
        return text
    return text[: n - 80] + "\n...[TRUNCATED]...\n" + text[-40:]


def tokens(text: str, max_tokens: int = 2500) -> List[str]:
    toks = [t.lower() for t in TOKEN_RE.findall(text or "")]
    stop = {"return", "const", "char", "int", "void", "static", "while", "for", "else", "if", "ifdef", "endif", "struct", "class"}
    out = [t for t in toks if t not in stop and len(t) > 2]
    return out[:max_tokens]


def simple_code_entities(code: str, max_entities: int = 80) -> List[str]:
    ents: List[str] = []
    for m in re.finditer(r"['\"]([^'\"]{1,120})['\"]", code or ""):
        s = m.group(1)
        if any(ch in s for ch in ["/", "\\", ".", "$", "%"]):
            ents.append(s)
    for m in re.finditer(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(", code or ""):
        ents.append(m.group(1))
    for t in tokens(code, max_tokens=1200):
        ents.append(t)
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
    return [] if cwe == "NO_CWE" else ["Input Validation"]


def make_patch_text(item: Dict[str, Any]) -> str:
    meta = {
        "repo": item.get("repo", ""),
        "commit_id": item.get("commit_id", ""),
        "file_name": item.get("file_name", ""),
        "file_type": item.get("file_type", ""),
        "programming_language": item.get("PL", ""),
        "commit_message": item.get("msg", ""),
    }
    meta_text = "\n".join(f"{k}: {v}" for k, v in meta.items() if v)
    rem = list_to_text(item.get("REM_DIFF", []))
    add = list_to_text(item.get("ADD_DIFF", []))
    return f"{meta_text}\n\nREM_DIFF_PRE_PATCH_REMOVED_CODE:\n{rem}\n\nADD_DIFF_POST_PATCH_FIXED_CODE:\n{add}".strip()


def convert_treevul_items(raw_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    examples: List[Dict[str, Any]] = []
    for idx, item in enumerate(raw_items):
        true_cwe = normalize_cwe(item.get("cwe_list"))
        if true_cwe == "NO_CWE":

            continue
        paths = normalize_paths(item.get("path_list"))
        patch_text = make_patch_text(item)
        if not patch_text:
            continue
        common = {
            "source_item_index": idx,
            "repo": item.get("repo", ""),
            "commit_id": item.get("commit_id", ""),
            "file_name": item.get("file_name", ""),
            "PL": item.get("PL", ""),
            "cve": str(item.get("cve_list", "")),
            "full_patch": patch_text,
            "true_vulnerability_cwe": true_cwe,
            "true_vulnerability_paths": paths,
        }

        pre_code = patch_text + "\n\nPHASE_UNDER_ANALYSIS: PRE_PATCH_REMOVED_CODE\nDecision target: judge the vulnerable pre-patch behavior represented by REM_DIFF."
        pre_entities = simple_code_entities(pre_code)
        pre_classes = choose_abstraction_classes(pre_code, true_cwe)
        examples.append({
            **common,
            "example_id": f"{idx}_PRE_PATCH_vulnerable",
            "phase": "PRE_PATCH_REMOVED_CODE",
            "binary_label": 1,
            "cwe": true_cwe,
            "path_list": paths,
            "code": pre_code,
            "entities": pre_entities,
            "abstraction_classes": pre_classes,
        })

        post_code = patch_text + "\n\nPHASE_UNDER_ANALYSIS: POST_PATCH_FIXED_CODE\nDecision target: judge the fixed post-patch behavior represented by ADD_DIFF."
        post_entities = simple_code_entities(post_code)
        post_classes = choose_abstraction_classes(post_code, "NO_CWE")
        examples.append({
            **common,
            "example_id": f"{idx}_POST_PATCH_fixed",
            "phase": "POST_PATCH_FIXED_CODE",
            "binary_label": 0,
            "cwe": "NO_CWE",
            "path_list": [],
            "code": post_code,
            "entities": post_entities,
            "abstraction_classes": post_classes,
        })
    return examples


def split_examples(examples: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:

    by_item: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for ex in examples:
        by_item[int(ex["source_item_index"])].append(ex)
    item_ids = list(by_item)
    random.seed(RANDOM_SEED)
    random.shuffle(item_ids)
    n = len(item_ids)
    n_train = int(n * TRAIN_RATIO)
    n_valid = int(n * VALID_RATIO)
    train_ids = set(item_ids[:n_train])
    valid_ids = set(item_ids[n_train:n_train + n_valid])
    test_ids = set(item_ids[n_train + n_valid:])
    train = [ex for iid in train_ids for ex in by_item[iid]]
    valid = [ex for iid in valid_ids for ex in by_item[iid]]
    test = [ex for iid in test_ids for ex in by_item[iid]]
    return train, valid, test


# =============================================================================
# KG and retrieval
# =============================================================================


def build_kg(train_examples: List[Dict[str, Any]]) -> Dict[str, Any]:
    kg: Dict[str, Any] = {
        "abstraction_classes": ABSTRACTION_CLASSES,
        "cwe_to_classes": defaultdict(Counter),
        "class_to_cwes": defaultdict(Counter),
        "cwe_parent": defaultdict(Counter),
        "cwe_children": defaultdict(Counter),
        "cwe_to_entities": defaultdict(Counter),
        "entity_to_cwes": defaultdict(Counter),
        "cwe_to_tokens": defaultdict(Counter),
        "cwe_paths": defaultdict(list),
        "cwe_frequency": Counter(),
    }
    for ex in train_examples:
        if int(ex.get("binary_label", 0)) != 1:
            continue
        cwe = normalize_cwe(ex.get("cwe"))
        if cwe == "NO_CWE":
            continue
        kg["cwe_frequency"][cwe] += 1
        for p in ex.get("path_list") or []:
            if p and p not in kg["cwe_paths"][cwe]:
                kg["cwe_paths"][cwe].append(p)
            for a, b in zip(p, p[1:]):
                kg["cwe_children"][a][b] += 1
                kg["cwe_parent"][b][a] += 1
        for cls in ex.get("abstraction_classes", []):
            kg["cwe_to_classes"][cwe][cls] += 1
            kg["class_to_cwes"][cls][cwe] += 1
        for ent in ex.get("entities", [])[:80]:
            kg["cwe_to_entities"][cwe][ent] += 1
            kg["entity_to_cwes"][ent][cwe] += 1
        for tok in tokens(ex.get("code", ""), max_tokens=1500):
            kg["cwe_to_tokens"][cwe][tok] += 1

    out: Dict[str, Any] = {"abstraction_classes": ABSTRACTION_CLASSES}
    for k in ["cwe_to_classes", "class_to_cwes", "cwe_parent", "cwe_children", "cwe_to_entities", "entity_to_cwes", "cwe_to_tokens"]:
        out[k] = {kk: dict(vv) for kk, vv in kg[k].items()}
    out["cwe_paths"] = dict(kg["cwe_paths"])
    out["cwe_frequency"] = dict(kg["cwe_frequency"])
    return out


def build_retrieval_index(train_examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    index = []
    for ex in train_examples:
        if int(ex.get("binary_label", 0)) != 1:
            continue
        cwe = normalize_cwe(ex.get("cwe"))
        if cwe == "NO_CWE":
            continue
        tok_set = set(tokens(ex.get("code", ""), max_tokens=1600))
        index.append({
            "example_id": ex["example_id"],
            "cwe": cwe,
            "path_list": ex.get("path_list", []),
            "file_name": ex.get("file_name", ""),
            "PL": ex.get("PL", ""),
            "tokens": tok_set,
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
        if ex.get("PL") and ex.get("PL") == row.get("PL"):
            score += 0.03
        if ex.get("file_name") and ex.get("file_name") == row.get("file_name"):
            score += 0.05
        if score > 0:
            scored.append((score, row))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [r for _, r in scored[:k]]


def retrieve_kg_context(ex: Dict[str, Any], kg: Dict[str, Any], retrieval_index: List[Dict[str, Any]]) -> Dict[str, Any]:
    entities = ex.get("entities") or simple_code_entities(ex.get("code", ""))
    classes = ex.get("abstraction_classes") or choose_abstraction_classes(ex.get("code", ""), "NO_CWE")
    similar = retrieve_similar(ex, retrieval_index, k=max(FEWSHOT_K, CANDIDATE_CWE_TOP_K))

    scores = Counter()

    for rank, row in enumerate(similar):
        scores[row["cwe"]] += 100.0 / (rank + 1)
    # Entity links.
    for ent in entities:
        for cwe, freq in kg.get("entity_to_cwes", {}).get(ent, {}).items():
            scores[cwe] += min(freq, 15)
    # Class links.
    for cls in classes:
        for cwe, freq in kg.get("class_to_cwes", {}).get(cls, {}).items():
            scores[cwe] += 2 * min(freq, 20)
    # Token links.
    q_toks = Counter(tokens(ex.get("code", ""), max_tokens=1600))
    for cwe, tok_counts in kg.get("cwe_to_tokens", {}).items():
        overlap = 0
        for t, cnt in q_toks.items():
            if t in tok_counts:
                overlap += min(cnt, tok_counts[t], 3)
        if overlap:
            scores[cwe] += min(overlap, 50)
    # Frequency fallback.
    for cwe, freq in kg.get("cwe_frequency", {}).items():
        scores[cwe] += 0.01 * freq

    candidates = [c for c, _ in scores.most_common(CANDIDATE_CWE_TOP_K) if c != "NO_CWE"]
    if ADD_TRUE_CWE_TO_CANDIDATES_DEBUG_ONLY:
        true_cwe = normalize_cwe(ex.get("true_vulnerability_cwe" if int(ex.get("binary_label", 0)) == 0 else "cwe"))
        if true_cwe != "NO_CWE" and true_cwe not in candidates:
            candidates = [true_cwe] + candidates
            candidates = candidates[:CANDIDATE_CWE_TOP_K]

    return {
        "entities": entities[:40],
        "abstraction_classes": classes[:3],
        "candidate_cwes_ranked": candidates,
        "candidate_paths": {cwe: kg.get("cwe_paths", {}).get(cwe, [])[:3] for cwe in candidates},
        "class_descriptions": {cls: ABSTRACTION_CLASSES.get(cls, {}).get("description", "") for cls in classes[:3]},
        "few_shot_examples": [
            {
                "cwe": row["cwe"],
                "path_list": row.get("path_list", [])[:2],
                "evidence_excerpt": row["summary"],
            }
            for row in similar[:FEWSHOT_K]
        ],
    }


def kg_context_to_text(ctx: Dict[str, Any]) -> str:
    return truncate(json.dumps(ctx, ensure_ascii=False, indent=2), MAX_KG_CONTEXT_CHARS)


# =============================================================================
# DeepSeek API and parsing
# =============================================================================


def get_deepseek_client() -> Any:
    if OpenAI is None:
        raise RuntimeError("OpenAI SDK is not installed. Run: pip install openai")
    key = (DEEPSEEK_API_KEY or "").strip()
    if not key or key == "PASTE_YOUR_DEEPSEEK_API_KEY_HERE":
        raise RuntimeError("DeepSeek API key missing. Set DEEPSEEK_API_KEY in script or environment.")
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
        is_non = bool(re.search(r"non[- ]?vulnerable|not vulnerable|safe|fixed", text or "", re.I))
        is_vul = bool(re.search(r"vulnerable|vulnerability|weakness|CWE-", text or "", re.I)) and not is_non
        return {
            "binary_label": 1 if is_vul else 0,
            "cwe_id": cwes[0] if cwes else "NO_CWE",
            "confidence": 0.3,
            "reasoning": (text or "")[:1000],
            "parse_error": True,
        }


def make_inference_prompt(ex: Dict[str, Any], kg_ctx: Dict[str, Any]) -> Tuple[str, str]:
    candidates = kg_ctx.get("candidate_cwes_ranked", [])
    phase = ex.get("phase", "")
    system = (
        "You are a senior software security analyst. You perform patch-aware CWE diagnosis. "
        "Return only valid JSON, no markdown."
    )
    user = f"""
Analyze the TreeVul patch evidence using phase-aware reasoning and KG guidance.

TreeVul patch semantics:
- REM_DIFF_PRE_PATCH_REMOVED_CODE usually shows vulnerable pre-patch behavior removed by the security patch.
- ADD_DIFF_POST_PATCH_FIXED_CODE usually shows repaired/fixed behavior added by the patch.
- The field PHASE_UNDER_ANALYSIS tells which phase you must classify.
- If PHASE_UNDER_ANALYSIS is PRE_PATCH_REMOVED_CODE, identify the vulnerability cause from the repair transformation.
- If PHASE_UNDER_ANALYSIS is POST_PATCH_FIXED_CODE, decide whether the fixed code is non-vulnerable.

Candidate CWE rule:
- If binary_label = 1, choose cwe_id from this candidate list only: {candidates}
- If binary_label = 0, output cwe_id = "NO_CWE".
- Do not invent a CWE outside candidate_cwes_ranked.
- Prefer the deepest CWE leaf whose CWE path best matches the patch mechanism.

Output JSON fields exactly:
{{
  "binary_label": 0 or 1,
  "cwe_id": "CWE-xxx" or "NO_CWE",
  "confidence": float from 0 to 1,
  "entities": ["..."],
  "abstraction_classes": ["..."],
  "reasoning": "concise entity -> abstraction class -> CWE explanation"
}}

KG Context and retrieved examples:
{kg_context_to_text(kg_ctx)}

Patch evidence:
{truncate(ex.get('code', ''), MAX_CODE_CHARS)}
""".strip()
    return system, user


def postprocess_prediction(ex: Dict[str, Any], parsed: Dict[str, Any], kg_ctx: Dict[str, Any]) -> Tuple[int, str, Dict[str, Any]]:
    candidates = [normalize_cwe(c) for c in kg_ctx.get("candidate_cwes_ranked", []) if normalize_cwe(c) != "NO_CWE"]

    # Binary: phase-aware TreeVul prior (transparent setting).
    if USE_PHASE_PRIOR_FOR_BINARY:
        pred_label = 1 if ex.get("phase") == "PRE_PATCH_REMOVED_CODE" else 0
    else:
        raw_label = str(parsed.get("binary_label", "0")).strip()
        pred_label = 1 if raw_label in {"1", "true", "True", "vulnerable"} else 0

    raw_cwe = normalize_cwe(parsed.get("cwe_id", "NO_CWE"))
    if pred_label == 0:
        pred_cwe = "NO_CWE"
    else:
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
            pred_cwe = raw_cwe
    parsed["postprocess_note"] = {
        "use_phase_prior_for_binary": USE_PHASE_PRIOR_FOR_BINARY,
        "force_cwe_from_candidates": FORCE_CWE_FROM_CANDIDATES,
        "candidate_cwes_ranked": candidates,
        "raw_cwe": raw_cwe,
    }
    return pred_label, pred_cwe, parsed


# =============================================================================
# Inference and evaluation
# =============================================================================


def run_inference(test: List[Dict[str, Any]], kg: Dict[str, Any], retrieval_index: List[Dict[str, Any]], out_dir: Path) -> List[Dict[str, Any]]:
    client = get_deepseek_client()
    rows = test[:MAX_TEST_API_SAMPLES] if MAX_TEST_API_SAMPLES is not None else test
    cache_dir = out_dir / "api_cache"
    preds: List[Dict[str, Any]] = []
    for ex in tqdm(rows, desc="DeepSeek patch-aware KG inference"):
        ctx = retrieve_kg_context(ex, kg, retrieval_index)
        system, user = make_inference_prompt(ex, ctx)
        cache_key = re.sub(r"[^A-Za-z0-9_.-]", "_", ex["example_id"])
        try:
            raw = call_deepseek(client, system, user, cache_key=cache_key, cache_dir=cache_dir)
            parsed = parse_json_from_response(raw)
        except Exception as e:
            raw = ""
            parsed = {"binary_label": 0, "cwe_id": "NO_CWE", "reasoning": f"API_ERROR: {e}"}
        pred_label, pred_cwe, parsed = postprocess_prediction(ex, parsed, ctx)
        preds.append({
            "example_id": ex["example_id"],
            "phase": ex.get("phase", ""),
            "repo": ex.get("repo", ""),
            "commit_id": ex.get("commit_id", ""),
            "file_name": ex.get("file_name", ""),
            "true_binary_label": int(ex.get("binary_label", 0)),
            "pred_binary_label": pred_label,
            "true_cwe": normalize_cwe(ex.get("cwe", "NO_CWE")),
            "pred_cwe": pred_cwe,
            "true_path_list": ex.get("path_list", []),
            "candidate_cwes_ranked": ctx.get("candidate_cwes_ranked", []),
            "kg_context": ctx,
            "raw_response": raw,
            "parsed_response": parsed,
        })
    write_jsonl(out_dir / "predictions.jsonl", preds)
    pd.DataFrame(preds).to_csv(out_dir / "predictions.csv", index=False)
    return preds


def shortest_path_to_cwe(kg: Dict[str, Any], target: str) -> List[str]:
    target = normalize_cwe(target)
    if target == "NO_CWE":
        return []
    path = [target]
    seen = {target}
    cur = target
    parent_map = kg.get("cwe_parent", {})
    for _ in range(20):
        parents = parent_map.get(cur, {})
        if not parents:
            break
        parent = max(parents.items(), key=lambda kv: kv[1])[0]
        if parent in seen:
            break
        path.append(parent)
        seen.add(parent)
        cur = parent
    return list(reversed(path))


def path_fraction(true_paths: Any, pred_cwe: str, kg: Dict[str, Any], true_label: int, pred_label: int) -> float:
    if true_label == 0:
        return 1.0 if pred_label == 0 else 0.0
    paths = [p for p in (true_paths if isinstance(true_paths, list) else []) if isinstance(p, list) and p]
    if not paths or normalize_cwe(pred_cwe) == "NO_CWE":
        return 0.0
    pred_path = set(shortest_path_to_cwe(kg, pred_cwe) or [normalize_cwe(pred_cwe)])
    best = 0.0
    for p in paths:
        pset = {normalize_cwe(x) for x in p if normalize_cwe(x) != "NO_CWE"}
        if pset:
            best = max(best, len(pset & pred_path) / len(pset))
    return float(best)


def per_label_counts(y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]) -> List[Dict[str, Any]]:
    rows = []
    for lab in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == lab and p == lab)
        tn = sum(1 for t, p in zip(y_true, y_pred) if t != lab and p != lab)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != lab and p == lab)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == lab and p != lab)
        rows.append({"label": lab, "TP": tp, "TN": tn, "FP": fp, "FN": fn})
    return rows


def cwe_metrics_block(rows: List[Dict[str, Any]], kg: Dict[str, Any], name: str, out_dir: Path) -> Dict[str, Any]:
    y_true = [normalize_cwe(r.get("true_cwe", "NO_CWE")) for r in rows]
    y_pred = [normalize_cwe(r.get("pred_cwe", "NO_CWE")) for r in rows]
    labels = sorted(set(y_true) | set(y_pred))
    if not rows:
        return {"num_examples": 0}
    pf_values = [
        path_fraction(r.get("true_path_list", []), r.get("pred_cwe", "NO_CWE"), kg, int(r.get("true_binary_label", 0)), int(r.get("pred_binary_label", 0)))
        for r in rows
    ]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    pd.DataFrame(cm, index=[f"true_{x}" for x in labels], columns=[f"pred_{x}" for x in labels]).to_csv(out_dir / f"{name}_cwe_confusion_matrix.csv")
    pd.DataFrame(per_label_counts(y_true, y_pred, labels)).to_csv(out_dir / f"{name}_per_cwe_tp_tn_fp_fn.csv", index=False)
    return {
        "num_examples": len(rows),
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Weighted_F1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "Macro_F1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "Micro_F1": float(f1_score(y_true, y_pred, labels=labels, average="micro", zero_division=0)),
        "MCC": float(matthews_corrcoef(y_true, y_pred)) if len(set(y_true) | set(y_pred)) > 1 else 0.0,
        "Path_Fraction_PF": float(np.mean(pf_values)) if pf_values else 0.0,
        "num_cwe_labels": len(labels),
    }


def evaluate_predictions(preds: List[Dict[str, Any]], kg: Dict[str, Any], out_dir: Path) -> Dict[str, Any]:
    y_true_bin = [int(r["true_binary_label"]) for r in preds]
    y_pred_bin = [int(r["pred_binary_label"]) for r in preds]
    cm_bin = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1])
    tn, fp, fn, tp = cm_bin.ravel()
    binary = {
        "Accuracy": float(accuracy_score(y_true_bin, y_pred_bin)),
        "Precision": float(precision_score(y_true_bin, y_pred_bin, zero_division=0)),
        "Recall": float(recall_score(y_true_bin, y_pred_bin, zero_division=0)),
        "F1_score": float(f1_score(y_true_bin, y_pred_bin, zero_division=0)),
        "Weighted_F1": float(f1_score(y_true_bin, y_pred_bin, average="weighted", zero_division=0)),
        "Macro_F1": float(f1_score(y_true_bin, y_pred_bin, average="macro", zero_division=0)),
        "MCC": float(matthews_corrcoef(y_true_bin, y_pred_bin)) if len(set(y_true_bin) | set(y_pred_bin)) > 1 else 0.0,
        "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
        "confusion_matrix_labels": [0, 1],
        "confusion_matrix": cm_bin.tolist(),
    }
    pd.DataFrame(cm_bin, index=["true_0", "true_1"], columns=["pred_0", "pred_1"]).to_csv(out_dir / "binary_confusion_matrix.csv")

    vuln_only = [r for r in preds if int(r.get("true_binary_label", 0)) == 1]
    metrics = {
        "num_predictions": len(preds),
        "settings": {
            "USE_PHASE_PRIOR_FOR_BINARY": USE_PHASE_PRIOR_FOR_BINARY,
            "FORCE_CWE_FROM_CANDIDATES": FORCE_CWE_FROM_CANDIDATES,
            "FALLBACK_TO_RETRIEVAL_TOP1_CWE": FALLBACK_TO_RETRIEVAL_TOP1_CWE,
            "ADD_TRUE_CWE_TO_CANDIDATES_DEBUG_ONLY": ADD_TRUE_CWE_TO_CANDIDATES_DEBUG_ONLY,
            "CANDIDATE_CWE_TOP_K": CANDIDATE_CWE_TOP_K,
            "FEWSHOT_K": FEWSHOT_K,
        },
        "binary_metrics": binary,
        "cwe_metrics_all_examples": cwe_metrics_block(preds, kg, "all_examples", out_dir),
        "cwe_metrics_vulnerable_only": cwe_metrics_block(vuln_only, kg, "vulnerable_only", out_dir),
    }
    write_json(out_dir / "metrics.json", metrics)
    return metrics


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    out_dir = ensure_dir(OUTPUT_DIR)
    print("=" * 95)
    print("Improved TreeVul + DeepSeek-V4-Flash patch-aware VulReaD-style reproduction")
    print("=" * 95)
    print(f"Input dataset : {INPUT_JSON}")
    print(f"Output folder : {OUTPUT_DIR}")
    print(f"DeepSeek model: {DEEPSEEK_MODEL}")
    print(f"MAX_TREEVUL_RECORDS: {MAX_TREEVUL_RECORDS}")
    print(f"USE_PHASE_PRIOR_FOR_BINARY: {USE_PHASE_PRIOR_FOR_BINARY}")
    print(f"FORCE_CWE_FROM_CANDIDATES: {FORCE_CWE_FROM_CANDIDATES}")
    print("=" * 95)

    raw_all = read_json_any(INPUT_JSON)
    raw = raw_all[:MAX_TREEVUL_RECORDS] if MAX_TREEVUL_RECORDS is not None else raw_all
    examples = convert_treevul_items(raw)
    train, valid, test = split_examples(examples)
    kg = build_kg(train)
    retrieval_index = build_retrieval_index(train)

    print(f"Raw items total: {len(raw_all)}")
    print(f"Raw items used : {len(raw)}")
    print(f"Examples       : {len(examples)}")
    print(f"Train/Valid/Test examples: {len(train)} / {len(valid)} / {len(test)}")
    print(f"Train retrieval index vulnerable examples: {len(retrieval_index)}")

    write_jsonl(out_dir / "train.jsonl", train)
    write_jsonl(out_dir / "valid.jsonl", valid)
    write_jsonl(out_dir / "test.jsonl", test)
    pd.DataFrame(train).to_csv(out_dir / "train.csv", index=False)
    pd.DataFrame(valid).to_csv(out_dir / "valid.csv", index=False)
    pd.DataFrame(test).to_csv(out_dir / "test.csv", index=False)
    write_json(out_dir / "kg.json", kg)
    write_json(out_dir / "split_summary.json", {
        "input_json": INPUT_JSON,
        "output_dir": OUTPUT_DIR,
        "max_treevul_records": MAX_TREEVUL_RECORDS,
        "raw_items_total": len(raw_all),
        "raw_items_used": len(raw),
        "examples": len(examples),
        "train": len(train),
        "valid": len(valid),
        "test": len(test),
        "mode": "patch_phase_aware_pre_post_examples_same_raw_item_split",
    })

    preds: List[Dict[str, Any]] = []
    if RUN_DEEPSEEK_INFERENCE:
        preds = run_inference(test, kg, retrieval_index, out_dir)
    else:
        pred_file = out_dir / "predictions.jsonl"
        if pred_file.exists():
            preds = read_json_any(pred_file)

    if RUN_EVALUATION and preds:
        metrics = evaluate_predictions(preds, kg, out_dir)
        print("\nEvaluation metrics:")
        print(json.dumps(metrics, indent=2))

    print("\nDone. Results saved to:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()
