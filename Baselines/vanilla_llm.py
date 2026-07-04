#!/usr/bin/env python3
"""
Baseline: Vanilla LLM (Few-Shot) CWE Path Prediction for CWEMap.
export DEEPSEEK_API_KEY="your-api-key-here"

"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import networkx as nx
import pandas as pd
from sklearn.metrics import f1_score, matthews_corrcoef
from tqdm import tqdm

NO_PREDICTION_LABEL = "NO_PREDICTION"
DEFAULT_MODEL = "DeepSeek-V4-Flash"
DEFAULT_BASE_URL = "https://api.deepseek.com"


def to_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def stable_id(record: Dict[str, Any]) -> str:
    if record.get("hash") not in (None, ""):
        return str(record["hash"])
    raw = "||".join(
        [str(record.get("project", "")), str(record.get("commit_id", "")), str(record.get("idx", "")), str(record.get("func", ""))]
    )
    return hashlib.sha256(raw.encode("utf-8", errors="ignore")).hexdigest()


def normalize_cwe(value: Any) -> Optional[str]:
    if not value:
        return None
    if isinstance(value, (list, tuple)):
        value = value[0] if value else None
    if not value:
        return None
    match = re.search(r"CWE-?(\d+)", str(value), flags=re.IGNORECASE)
    return f"CWE-{match.group(1)}" if match else None


def iter_jsonl(path: str) -> Iterable[Tuple[int, Dict[str, Any]]]:
    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield line_no, json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {line_no} in {path}: {exc}") from exc


def load_primevul_jsonl(
    path: str,
    max_items: Optional[int] = None,
    vulnerable_only: bool = True,
    require_cwe: bool = True,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for line_no, item in iter_jsonl(path):
        func = item.get("func")
        if not isinstance(func, str) or not func.strip():
            continue
        if vulnerable_only and to_int(item.get("target"), 0) != 1:
            continue
        if require_cwe and not item.get("cwe"):
            continue

        item = dict(item)
        item["_line_no"] = line_no
        item["_stable_id"] = stable_id(item)
        item["_cwe_norm"] = normalize_cwe(item.get("cwe"))
        records.append(item)

        if max_items is not None and len(records) >= max_items:
            break
    return records


def sanitize_code(text: str, max_chars: int) -> str:
    text = re.sub(r"CVE-\d{4}-\d+", " CVE_ID ", text, flags=re.IGNORECASE)
    text = re.sub(r"CWE-\d+", " CWE_ID ", text, flags=re.IGNORECASE)  # never leak the label into the prompt
    text = text.strip()
    return text[:max_chars]


def load_cwe_taxonomy_digraph(path: str) -> nx.DiGraph:
    graph = nx.DiGraph()
    suffix = Path(path).suffix.lower()
    if suffix in (".jsonl", ".ndjson"):
        with open(path, "r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                parent = normalize_cwe(item.get("parent_cwe") or item.get("parent"))
                child = normalize_cwe(item.get("child_cwe") or item.get("child"))
                if parent and child:
                    graph.add_edge(parent, child)
    else:
        with open(path, "r", encoding="utf-8", errors="ignore") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                parent = normalize_cwe(row.get("parent_cwe") or row.get("parent"))
                child = normalize_cwe(row.get("child_cwe") or row.get("child"))
                if parent and child:
                    graph.add_edge(parent, child)
    if graph.number_of_nodes() == 0:
        raise ValueError(f"No usable parent/child CWE edges found in {path}.")
    return graph


def true_cwe_path(taxonomy: nx.DiGraph, true_cwe: Optional[str]) -> Optional[List[str]]:
    """Shortest root -> true_cwe path in GCWE (deterministic tie-break by
    lexicographically smallest path). None if unreachable/unknown."""
    if not true_cwe or true_cwe not in taxonomy:
        return None
    roots = [n for n in taxonomy.nodes if taxonomy.in_degree(n) == 0]
    candidates: List[List[str]] = []
    for root in roots:
        if root == true_cwe:
            candidates.append([root])
            continue
        try:
            candidates.append(nx.shortest_path(taxonomy, root, true_cwe))
        except nx.NetworkXNoPath:
            continue
    if not candidates:
        return None
    min_len = min(len(p) for p in candidates)
    shortest = sorted(p for p in candidates if len(p) == min_len)
    return shortest[0]


def path_fraction(predicted_path: Optional[List[str]], truth_path: Optional[List[str]]) -> Optional[float]:
    """PF = |predicted ∩ true| / |true|. None only when truth_path itself is
    undefined; 0.0 when the model gave no usable path but ground truth exists."""
    if truth_path is None or len(truth_path) == 0:
        return None
    if not predicted_path:
        return 0.0
    overlap = len(set(predicted_path) & set(truth_path))
    return round(overlap / len(truth_path), 6)


def compute_performance_metrics(y_true: List[str], y_pred: List[str], pf_values: List[float]) -> Dict[str, Any]:
    """Weighted F1 / Macro F1 / MCC (terminal-CWE classification) + mean
    Path Fraction - identical definitions to phases/phase4_hierarchical_reasoning.py."""
    if not y_true:
        return {
            "num_evaluated_queries": 0,
            "weighted_f1": None,
            "macro_f1": None,
            "mcc": None,
            "path_fraction_mean": None,
            "exact_match_accuracy": None,
            "note": "No queries had a known ground-truth CWE; no classification metrics could be computed.",
        }

    labels = sorted(set(y_true) | set(y_pred))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0))
    mcc = float(matthews_corrcoef(y_true, y_pred))
    exact_match = sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)
    pf_mean = float(sum(pf_values) / len(pf_values)) if pf_values else None

    return {
        "num_evaluated_queries": len(y_true),
        "num_path_fraction_evaluated": len(pf_values),
        "weighted_f1": round(weighted_f1, 6),
        "macro_f1": round(macro_f1, 6),
        "mcc": round(mcc, 6),
        "path_fraction_mean": round(pf_mean, 6) if pf_mean is not None else None,
        "exact_match_accuracy": round(exact_match, 6),
        "note": None,
    }


# ---------------------------------------------------------------------------
# Few-shot example selection
# ---------------------------------------------------------------------------
def select_fewshot_examples(
    pool: List[Dict[str, Any]],
    taxonomy: nx.DiGraph,
    num_shots: int,
    max_code_chars: int,
    exclude_stable_ids: set,
    seed: int,
) -> List[Dict[str, Any]]:
    """Pick up to `num_shots` pool examples, preferring one per distinct CWE
    (for taxonomic diversity) and skipping anything that (a) overlaps with
    the test set by stable_id (leakage guard) or (b) has no valid root->leaf
    path in the taxonomy (each shown example must be a *correct*, gold-path
    demonstration for few-shot to be meaningful)."""
    rng = random.Random(seed)
    candidates = [r for r in pool if r["_stable_id"] not in exclude_stable_ids and r.get("_cwe_norm")]
    rng.shuffle(candidates)

    by_cwe: Dict[str, Dict[str, Any]] = {}
    for rec in candidates:
        cwe = rec["_cwe_norm"]
        if cwe in by_cwe:
            continue
        path = true_cwe_path(taxonomy, cwe)
        if not path:
            continue
        by_cwe[cwe] = {"record": rec, "path": path}
        if len(by_cwe) >= num_shots:
            break

    if len(by_cwe) < num_shots:
        used_ids = {v["record"]["_stable_id"] for v in by_cwe.values()}
        for rec in candidates:
            if len(by_cwe) >= num_shots:
                break
            if rec["_stable_id"] in used_ids:
                continue
            path = true_cwe_path(taxonomy, rec["_cwe_norm"])
            if not path:
                continue
            by_cwe[f"{rec['_cwe_norm']}::{rec['_stable_id']}"] = {"record": rec, "path": path}
            used_ids.add(rec["_stable_id"])

    shots = []
    for entry in by_cwe.values():
        shots.append(
            {
                "code": sanitize_code(entry["record"].get("func", ""), max_code_chars),
                "cwe_path": entry["path"],
            }
        )
    return shots


# ---------------------------------------------------------------------------
# Prompting
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a security analyst classifying source-code vulnerabilities into \
the CWE (Common Weakness Enumeration) hierarchy.

Given a single java/C/C++ function, output the full CWE path from a top-level \
weakness category down to the single most specific CWE that applies, as a \
JSON array of CWE IDs ordered from most general (root) to most specific (leaf), \
e.g. ["CWE-664", "CWE-118", "CWE-119"].

Rules:
- Respond with ONLY a JSON object of the form {"cwe_path": ["CWE-XXX", ...]}.
- No prose, no markdown fences, no explanation - JSON only.
- The path must end in the single CWE you believe best classifies the function.
- If you are not confident any CWE applies, respond with {"cwe_path": []}.
"""


def build_fewshot_block(shots: List[Dict[str, Any]]) -> str:
    blocks = []
    for i, shot in enumerate(shots, start=1):
        path_json = json.dumps(shot["cwe_path"])
        blocks.append(f"Example {i}:\nCode:\n{shot['code']}\n\nAnswer: {{\"cwe_path\": {path_json}}}")
    return "\n\n".join(blocks)


def build_user_prompt(fewshot_block: str, query_code: str) -> str:
    parts = []
    if fewshot_block:
        parts.append(fewshot_block)
    parts.append(f"Now classify this function:\nCode:\n{query_code}\n\nAnswer:")
    return "\n\n---\n\n".join(parts)


# ---------------------------------------------------------------------------
# DeepSeek-V4-Flash 
# ---------------------------------------------------------------------------
def make_deepseek_client(api_key: Optional[str], base_url: str) -> Any:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError("Missing dependency: install with `pip install openai`.") from exc

    key = api_key or os.getenv("DEEPSEEK_API_KEY")
    if not key:
        raise EnvironmentError("DEEPSEEK_API_KEY is not set. Export it or pass --api-key.")
    return OpenAI(api_key=key, base_url=base_url)


def call_llm_for_path(
    client: Any,
    model: str,
    system_prompt: str,
    user_prompt: str,
    retries: int,
    sleep_seconds: float,
    temperature: float,
) -> str:
    last_error: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            return response.choices[0].message.content or ""
        except Exception as exc: 
            last_error = exc
            time.sleep(sleep_seconds * attempt)
    raise RuntimeError(f"LLM call failed after {retries} attempt(s): {last_error}") from last_error


def parse_predicted_path(raw_text: str) -> Optional[List[str]]:
    """Extract a validated list of CWE-ID strings from the model's raw
    response. Returns None if nothing usable could be parsed."""
    if not raw_text:
        return None
    cleaned = raw_text.strip()
    cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()

    parsed: Any = None
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                parsed = json.loads(cleaned[start : end + 1])
            except json.JSONDecodeError:
                parsed = None

    raw_path: Any = None
    if isinstance(parsed, dict):
        raw_path = parsed.get("cwe_path")
    elif isinstance(parsed, list):
        raw_path = parsed

    if raw_path is None:
        # last-resort fallback: pull every CWE-like token in order of appearance
        found = re.findall(r"CWE-?\d+", cleaned, flags=re.IGNORECASE)
        raw_path = found

    if not isinstance(raw_path, list):
        return None

    path = [normalize_cwe(x) for x in raw_path]
    path = [p for p in path if p]
    # de-duplicate consecutive repeats while preserving order
    deduped: List[str] = []
    for cwe in path:
        if not deduped or deduped[-1] != cwe:
            deduped.append(cwe)
    return deduped or None


# ---------------------------------------------------------------------------
# Response cache
# ---------------------------------------------------------------------------
CACHE_VERSION = "vanilla_llm_fewshot_v1"


def read_cache(path: Optional[str]) -> Dict[str, Any]:
    cache: Dict[str, Any] = {}
    if not path or not os.path.exists(path):
        return cache
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = entry.get("_cache_key")
            if key:
                cache[key] = entry
    return cache


def append_cache_jsonl(path: Optional[str], entry: Dict[str, Any]) -> None:
    if not path:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry) + "\n")


def cache_key(stable_id_: str, model: str, num_shots: int) -> str:
    raw = f"{CACHE_VERSION}::{model}::{num_shots}::{stable_id_}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def run_vanilla_llm_baseline(args: argparse.Namespace) -> Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print(f"Loading test queries:   {args.test_input}")
    queries = load_primevul_jsonl(args.test_input, max_items=args.max_queries)
    print(f"Loading few-shot pool:  {args.fewshot_pool}")
    pool = load_primevul_jsonl(args.fewshot_pool)
    if not queries:
        raise ValueError("No query records loaded. Check --test-input and filtering flags.")
    if not pool:
        raise ValueError("No few-shot pool records loaded. Check --fewshot-pool.")

    print(f"Loading CWE taxonomy:   {args.cwe_taxonomy}")
    taxonomy = load_cwe_taxonomy_digraph(args.cwe_taxonomy)

    test_ids = {q["_stable_id"] for q in queries}
    shots = select_fewshot_examples(
        pool, taxonomy, args.num_shots, args.max_code_chars, exclude_stable_ids=test_ids, seed=args.seed
    )
    print(f"Selected {len(shots)} few-shot example(s) (requested {args.num_shots}).")
    fewshot_block = build_fewshot_block(shots)

    client = make_deepseek_client(args.api_key, args.base_url)
    cache = read_cache(args.cache)

    output_records: List[Dict[str, Any]] = []
    path_rows: List[Dict[str, Any]] = []
    prediction_rows: List[Dict[str, Any]] = []

    y_true: List[str] = []
    y_pred: List[str] = []
    pf_values: List[float] = []

    for query in tqdm(queries, desc="VanillaLLM-FewShot"):
        stable = query["_stable_id"]
        true_cwe = query.get("_cwe_norm")
        truth_path = true_cwe_path(taxonomy, true_cwe)

        key = cache_key(stable, args.model, args.num_shots)
        if key in cache:
            raw_response = cache[key]["raw_response"]
        else:
            code = sanitize_code(query.get("func", ""), args.max_code_chars)
            user_prompt = build_user_prompt(fewshot_block, code)
            raw_response = call_llm_for_path(
                client, args.model, SYSTEM_PROMPT, user_prompt, args.retries, args.sleep, args.temperature
            )
            append_cache_jsonl(
                args.cache,
                {"_cache_key": key, "stable_id": stable, "model": args.model, "raw_response": raw_response},
            )
            cache[key] = {"raw_response": raw_response}

        predicted_path = parse_predicted_path(raw_response)
        predicted_cwe = predicted_path[-1] if predicted_path else None
        pf = path_fraction(predicted_path, truth_path)

        output_records.append(
            {
                "target_graph_id": stable,
                "idx": query.get("idx"),
                "project": query.get("project"),
                "target_true_cwe_eval_only": true_cwe,
                "raw_llm_response": raw_response,
                "predicted_cwe_path": predicted_path,
                "predicted_cwe": predicted_cwe,
                "path_fraction": pf,
            }
        )
        path_rows.append(
            {
                "target_graph_id": stable,
                "target_true_cwe": true_cwe,
                "predicted_cwe_path": " -> ".join(predicted_path) if predicted_path else None,
                "terminal_cwe": predicted_cwe,
                "path_depth": len(predicted_path) if predicted_path else 0,
                "path_fraction": pf,
            }
        )

        if true_cwe:
            y_true.append(true_cwe)
            y_pred.append(predicted_cwe or NO_PREDICTION_LABEL)
            if pf is not None:
                pf_values.append(pf)
            prediction_rows.append(
                {
                    "target_graph_id": stable,
                    "target_true_cwe": true_cwe,
                    "predicted_cwe": predicted_cwe or NO_PREDICTION_LABEL,
                    "exact_match": true_cwe == predicted_cwe,
                    "path_fraction": pf,
                }
            )

    performance_metrics = compute_performance_metrics(y_true, y_pred, pf_values)

    run_config = {
        "baseline": "Vanilla LLM (Few-Shot) CWE Path Prediction",
        "model": args.model,
        "base_url": args.base_url,
        "num_shots_requested": args.num_shots,
        "num_shots_used": len(shots),
        "test_input": args.test_input,
        "fewshot_pool": args.fewshot_pool,
        "cwe_taxonomy": args.cwe_taxonomy,
        "max_code_chars": args.max_code_chars,
        "temperature": args.temperature,
        "seed": args.seed,
        "num_queries_evaluated": len(queries),
    }

    output_json = {"run_config": run_config, "performance_metrics": performance_metrics, "records": output_records}
    path_df = pd.DataFrame(path_rows)
    metrics_df = pd.DataFrame([performance_metrics])
    predictions_df = pd.DataFrame(prediction_rows)
    return output_json, path_df, metrics_df, predictions_df


def save_outputs(
    output_json: Dict[str, Any],
    path_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    output_json_path: str,
    output_excel_path: str,
    output_csv_prefix: Optional[str],
) -> Dict[str, str]:
    Path(output_json_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as handle:
        json.dump(output_json, handle, indent=2, default=str)

    Path(output_excel_path).parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_excel_path, engine="openpyxl") as writer:
        path_df.to_excel(writer, sheet_name="predicted_paths", index=False)
        metrics_df.to_excel(writer, sheet_name="performance_metrics", index=False)
        predictions_df.to_excel(writer, sheet_name="predictions_vs_groundtruth", index=False)
        pd.DataFrame([output_json["run_config"]]).to_excel(writer, sheet_name="run_config", index=False)

    csv_paths: Dict[str, str] = {}
    if output_csv_prefix:
        Path(output_csv_prefix).parent.mkdir(parents=True, exist_ok=True)
        csv_paths["predicted_paths"] = f"{output_csv_prefix}_predicted_paths.csv"
        csv_paths["performance_metrics"] = f"{output_csv_prefix}_performance_metrics.csv"
        csv_paths["predictions_vs_groundtruth"] = f"{output_csv_prefix}_predictions_vs_groundtruth.csv"
        path_df.to_csv(csv_paths["predicted_paths"], index=False)
        metrics_df.to_csv(csv_paths["performance_metrics"], index=False)
        predictions_df.to_csv(csv_paths["predictions_vs_groundtruth"], index=False)
    return csv_paths


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Vanilla LLM (few-shot) CWE path prediction baseline.")
    parser.add_argument("--test-input", required=True, help="PrimeVul-style JSONL of queries to evaluate.")
    parser.add_argument("--fewshot-pool", required=True, help="PrimeVul-style JSONL to draw few-shot examples from (should not overlap --test-input).")
    parser.add_argument("--cwe-taxonomy", required=True, help="CWE taxonomy edge list CSV/JSONL (parent_cwe,child_cwe) - used only for scoring.")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-excel", required=True)
    parser.add_argument("--output-csv-prefix", default=None)
    parser.add_argument("--cache", default=None, help="JSONL response cache path (skips repeat LLM calls on re-run).")

    parser.add_argument("--model", default=DEFAULT_MODEL, help="Underlying LLM (default: DeepSeek-V4-Flash).")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--api-key", default=None, help="Defaults to $DEEPSEEK_API_KEY.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--sleep", type=float, default=0.5)

    parser.add_argument("--num-shots", type=int, default=5)
    parser.add_argument("--max-code-chars", type=int, default=4000)
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    output_json, path_df, metrics_df, predictions_df = run_vanilla_llm_baseline(args)
    csv_paths = save_outputs(
        output_json, path_df, metrics_df, predictions_df,
        args.output_json, args.output_excel, args.output_csv_prefix,
    )

    print(f"Saved JSON output:  {args.output_json}")
    print(f"Saved Excel output: {args.output_excel}")
    for name, path in csv_paths.items():
        print(f"Saved CSV output ({name}): {path}")

    metrics = output_json["performance_metrics"]
    if metrics.get("note"):
        print(f"Performance metrics: {metrics['note']}")
    else:
        print(
            "Performance metrics — "
            f"Weighted F1: {metrics.get('weighted_f1')}, "
            f"Macro F1: {metrics.get('macro_f1')}, "
            f"MCC: {metrics.get('mcc')}, "
            f"Path Fraction (mean): {metrics.get('path_fraction_mean')}, "
            f"evaluated on {metrics.get('num_evaluated_queries')} queries."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
