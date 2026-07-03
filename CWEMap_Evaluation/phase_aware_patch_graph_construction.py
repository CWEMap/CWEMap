#!/usr/bin/env python3
"""
Phase 2: Patch Graph Construction / Triple-Aware Knowledge Materialisation for CWEMap.

Consumes the JSON produced by Phase 1 (Patch-Aware Vulnerability Retrieval)
and, for the target query patch plus its top-k retrieved historical patches,
while preserving the PrimeVul split discipline established in Phase 1:
primevul/primevul_training.jsonl is the historical retrieval corpus,
primevul/primevul_validation.jsonl is the validation query split, and
primevul/primevul_testing.jsonl is the held-out final test query split.
calls the OpenAI ChatGPT-4o API with a fixed, non-negotiable system prompt to
extract security-relevant knowledge triples and materialise them as
triple-aware patch knowledge graphs anchored to three temporal phases of
each patch:

  T_before  - vulnerable program state before the fix
  T_delta   - security-relevant modification introduced by the patch
  T_after   - secure program state after the patch

The extractor is intentionally conservative: it must not infer facts that
are not directly supported by the patch text, and it must return machine
JSON only (no prose). This script enforces that contract at the pipeline
level by validating every triple returned by the model and dropping/logging
anything malformed rather than silently accepting it.

Example:
  export OPENAI_API_KEY="your-api-key-here"

  python scripts/phase2_triple_extraction.py \
    --phase1-json outputs/pvr/primevul_pvr_results.json \
    --patch-source Dataset_Preprocessing/dataset/primevul/primevul_validation.jsonl \
    --output-json outputs/triples/primevul_triples.json \
    --output-excel outputs/triples/primevul_triples.xlsx
"""

import argparse
import difflib
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm


# Allow both package import via main.py and direct standalone execution from the project root.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.namespace_adapter import describe_source, namespace_from_config

# Training must be used as the retrieval corpus; validation/testing are unseen
# query splits used to validate/test the CWEMap approach.
PRIMEVUL_DATASET_SPLITS = {
    "training": "primevul/primevul_training.jsonl",
    "validation": "primevul/primevul_validation.jsonl",
    "testing": "primevul/primevul_testing.jsonl",
}

PRIMEVUL_DEFAULT_PATHS = {
    split: f"Dataset_Preprocessing/dataset/{relative_path}"
    for split, relative_path in PRIMEVUL_DATASET_SPLITS.items()
}

# ---------------------------------------------------------------------------
# System prompt (verbatim contract for PhaseAwareTripleExtractor).
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are PhaseAwareTripleExtractor, a cybersecurity knowledge extraction agent.
Your task is to transform vulnerability-fixing patches into structured security knowledge suitable for graph construction.
You will receive:
1. The target vulnerability-fixing patch.
2. Top-k historical retrieved patches from Phase 1.
our  objective is to identify only security-relevant information.
Represent every patch using three temporal phases:
T_before
    Vulnerable program state before the fix.
T_delta
    Security-relevant modification introduced by the patch.
T_after
    Secure program state after the patch.
Extract only facts that are directly supported by the patch.
Return JSON only."""


RESPONSE_SHAPE_NOTE = (
    'Wrap your output as a single JSON object of the exact shape '
    '{"triples": [{"subject": "...", "relation": "...", "object": "...", "phase": "..."}, ...]}. '
    "If a phase has no directly supported facts, simply omit triples for that phase. "
    "If nothing is directly supported at all, return {\"triples\": []}."
)

ALLOWED_PHASES = {"T_before", "T_delta", "T_after"}
REQUIRED_TRIPLE_KEYS = {"subject", "relation", "object", "phase"}


# ---------------------------------------------------------------------------
# Generic helpers (mirrors Phase 1 conventions for a self-contained script)
# ---------------------------------------------------------------------------
def ensure_parent(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def to_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


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


def sanitize_text(text: Optional[str]) -> Optional[str]:
    """Strip label-revealing strings before sending code to the model."""
    if not text:
        return text
    text = re.sub(r"CVE-\d{4}-\d+", " CVE_ID ", text, flags=re.IGNORECASE)
    text = re.sub(r"CWE-\d+", " CWE_ID ", text, flags=re.IGNORECASE)
    text = re.sub(r"https?://\S+", " URL ", text)
    return text


def truncate(text: Optional[str], max_chars: int) -> Optional[str]:
    if text is None:
        return None
    return text[:max_chars]


def patch_stable_id(meta: Dict[str, Any]) -> str:
    """Stable identifier for a patch, used for caching and dedup."""
    if meta.get("hash") not in (None, ""):
        return f"hash:{meta['hash']}"
    raw = "||".join(
        [
            str(meta.get("project", "")),
            str(meta.get("commit_id", "")),
            str(meta.get("idx", "")),
        ]
    )
    return "meta:" + hashlib.sha256(raw.encode("utf-8", errors="ignore")).hexdigest()


def parse_json_object(text: str) -> Dict[str, Any]:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start >= 0 and end > start:
            try:
                parsed = json.loads(cleaned[start : end + 1])
            except json.JSONDecodeError:
                return {}
        else:
            return {}

    if isinstance(parsed, dict):
        return parsed
    if isinstance(parsed, list):
        return {"triples": parsed}
    return {}


def validate_triples(raw_triples: Any) -> Tuple[List[Dict[str, str]], int]:
    """Keep only well-formed triples; return (valid_triples, num_dropped)."""
    valid: List[Dict[str, str]] = []
    dropped = 0
    if not isinstance(raw_triples, list):
        return valid, dropped

    for item in raw_triples:
        if not isinstance(item, dict):
            dropped += 1
            continue
        if not REQUIRED_TRIPLE_KEYS.issubset(item.keys()):
            dropped += 1
            continue
        subject, relation, obj, phase = (
            item.get("subject"),
            item.get("relation"),
            item.get("object"),
            item.get("phase"),
        )
        if not all(isinstance(v, str) and v.strip() for v in (subject, relation, obj, phase)):
            dropped += 1
            continue
        if phase not in ALLOWED_PHASES:
            dropped += 1
            continue
        valid.append(
            {
                "subject": subject.strip(),
                "relation": relation.strip(),
                "object": obj.strip(),
                "phase": phase,
            }
        )
    return valid, dropped


# ---------------------------------------------------------------------------
# Phase 1 output + optional before/after patch source loading
# ---------------------------------------------------------------------------
def load_phase1_output(source: Any) -> Dict[str, Any]:
    """Accepts either the Phase 1 output already in memory (a dict, as
    passed by main.py when chaining phases directly) or a path to Phase 1's
    saved JSON file (standalone CLI use)."""
    if isinstance(source, dict):
        return source
    with open(source, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_patch_source(path: Optional[str]) -> Dict[str, Dict[str, Any]]:
    """Optional JSONL with richer per-patch fields: func_before, func_after,
    diff/patch, keyed by hash and by (project, commit_id, idx). If not
    provided, Phase 2 falls back to whatever single `func` snapshot Phase 1
    already embedded in its output."""
    index: Dict[str, Dict[str, Any]] = {}
    if not path:
        return index

    for _, item in iter_jsonl(path):
        entry = {
            "func_before": item.get("func_before") or item.get("before"),
            "func_after": item.get("func_after") or item.get("after"),
            "diff": item.get("diff") or item.get("patch"),
        }
        if item.get("hash") not in (None, ""):
            index[f"hash:{item['hash']}"] = entry
        meta_key = "meta:" + hashlib.sha256(
            "||".join(
                [str(item.get("project", "")), str(item.get("commit_id", "")), str(item.get("idx", ""))]
            ).encode("utf-8", errors="ignore")
        ).hexdigest()
        index[meta_key] = entry
    return index


def build_patch_view(
    meta: Dict[str, Any],
    patch_source: Dict[str, Dict[str, Any]],
    max_chars: int,
) -> Dict[str, Optional[str]]:
    """Resolve T_before / T_delta / T_after raw text for one patch.

    Falls back gracefully when only a single code snapshot is available
    (Phase 1's default output only carries one `func` per record). The
    extractor system prompt already forbids inferring unsupported facts,
    so missing phases are simply sent as null rather than fabricated.
    """
    sid = patch_stable_id(meta)
    lookup = patch_source.get(sid)

    before = after = delta = None
    note = None

    if lookup:
        before = lookup.get("func_before")
        after = lookup.get("func_after")
        delta = lookup.get("diff")
        if not delta and before and after:
            diff_lines = difflib.unified_diff(
                before.splitlines(), after.splitlines(), lineterm="", n=2
            )
            delta = "\n".join(diff_lines) or None
        if not before and not after:
            lookup = None  # nothing usable; fall through to single-snapshot mode

    if not lookup:
        func = meta.get("func")
        target = to_int(meta.get("target"), -1)
        if func:
            if target == 1:
                before = func
                note = "only the vulnerable snapshot was available (no matched fix pair)"
            elif target == 0:
                after = func
                note = "only the patched snapshot was available (no matched vulnerable pair)"
            else:
                before = func
                note = "target label unknown; snapshot treated as T_before only"

    before = truncate(sanitize_text(before), max_chars)
    delta = truncate(sanitize_text(delta), max_chars)
    after = truncate(sanitize_text(after), max_chars)

    return {"T_before": before, "T_delta": delta, "T_after": after, "note": note}


# ---------------------------------------------------------------------------
# Offline heuristic fallback (--no-llm)
# ---------------------------------------------------------------------------
def _heuristic_calls(code: Optional[str], max_calls: int = 8) -> List[str]:
    if not code:
        return []
    calls = re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(", code)
    seen: List[str] = []
    for call in calls:
        if call not in seen:
            seen.append(call)
        if len(seen) >= max_calls:
            break
    return seen


def heuristic_extract_triples(patch_view: Dict[str, Optional[str]]) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
    """Deterministic, non-LLM approximation of PhaseAwareTripleExtractor.

    This is NOT a substitute for the ChatGPT-4o extractor and is only used
    when --no-llm is passed (offline development/testing/CI, no API key
    required). It mirrors the same T_before/T_delta/T_after contract and
    triple schema so downstream Phase 3/4 consumers see an identical shape,
    but the facts it extracts are shallow (function-call presence only).
    """
    triples: List[Dict[str, str]] = []
    before_calls = set(_heuristic_calls(patch_view.get("T_before")))
    after_calls = set(_heuristic_calls(patch_view.get("T_after")))

    for call in sorted(before_calls):
        triples.append({"subject": call, "relation": "contains_call", "object": "code_block", "phase": "T_before"})
    for call in sorted(after_calls):
        triples.append({"subject": call, "relation": "contains_call", "object": "code_block", "phase": "T_after"})

    added = sorted(after_calls - before_calls)
    removed = sorted(before_calls - after_calls)
    for call in added:
        triples.append({"subject": call, "relation": "adds_check", "object": "code_block", "phase": "T_delta"})
    for call in removed:
        triples.append({"subject": call, "relation": "removes_check", "object": "code_block", "phase": "T_delta"})

    valid, dropped = validate_triples(triples)
    return valid, {"dropped_triples": dropped, "error": None}


# ---------------------------------------------------------------------------
# ChatGPT-4o triple extraction and triple-aware knowledge materialisation
# ---------------------------------------------------------------------------
def make_openai_client(api_key: Optional[str], base_url: Optional[str]) -> Any:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError("Missing dependency: install with `pip install openai`.") from exc

    key = api_key or os.getenv("OPENAI_API_KEY")
    if not key:
        raise EnvironmentError("OPENAI_API_KEY is not set. Export it or pass --api-key.")

    kwargs: Dict[str, Any] = {"api_key": key}
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)


def extract_triples_for_patch(
    client: Any,
    patch_view: Dict[str, Optional[str]],
    model: str,
    retries: int,
    sleep_seconds: float,
) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
    """One ChatGPT-4o call for one patch's T_before/T_delta/T_after view.

    The returned triples are the materialised knowledge units used by Phase 3
    to construct phase-aware patch graphs.
    """
    user_payload = {
        "T_before": patch_view["T_before"],
        "T_delta": patch_view["T_delta"],
        "T_after": patch_view["T_after"],
    }
    user_content = RESPONSE_SHAPE_NOTE + "\n\n" + json.dumps(user_payload, ensure_ascii=False)

    diagnostics: Dict[str, Any] = {"dropped_triples": 0, "error": None}
    last_error: Optional[str] = None

    for attempt in range(1, retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content or ""
            parsed = parse_json_object(content)
            valid, dropped = validate_triples(parsed.get("triples", []))
            diagnostics["dropped_triples"] = dropped
            time.sleep(sleep_seconds)
            return valid, diagnostics
        except Exception as exc:  # noqa: BLE001 - external API call, broad catch is intentional
            last_error = str(exc)
            if attempt >= retries:
                break
            time.sleep(max(1.0, sleep_seconds) * attempt)

    diagnostics["error"] = last_error
    return [], diagnostics


# ---------------------------------------------------------------------------
# Cache (JSONL, keyed by patch stable id + a hash of the exact view sent)
# ---------------------------------------------------------------------------
def view_fingerprint(patch_view: Dict[str, Optional[str]]) -> str:
    raw = json.dumps(
        {"T_before": patch_view["T_before"], "T_delta": patch_view["T_delta"], "T_after": patch_view["T_after"]},
        sort_keys=True,
    )
    return hashlib.sha256(raw.encode("utf-8", errors="ignore")).hexdigest()


def read_cache(cache_path: str) -> Dict[str, Dict[str, Any]]:
    cache: Dict[str, Dict[str, Any]] = {}
    path = Path(cache_path)
    if not path.exists():
        return cache
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                key = item.get("cache_key")
                if key:
                    cache[key] = item
            except json.JSONDecodeError:
                continue
    return cache


def append_cache(cache_path: str, cache_key: str, triples: List[Dict[str, str]], diagnostics: Dict[str, Any]) -> None:
    ensure_parent(cache_path)
    with open(cache_path, "a", encoding="utf-8") as handle:
        handle.write(
            json.dumps({"cache_key": cache_key, "triples": triples, "diagnostics": diagnostics}, ensure_ascii=False)
            + "\n"
        )


def get_or_extract(
    stable_id: str,
    patch_view: Dict[str, Optional[str]],
    cache: Dict[str, Dict[str, Any]],
    cache_path: str,
    client: Any,
    model: str,
    retries: int,
    sleep_seconds: float,
    use_llm: bool = True,
) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
    cache_key = f"{stable_id}:{view_fingerprint(patch_view)}:{'llm' if use_llm else 'heuristic'}"
    if cache_key in cache:
        entry = cache[cache_key]
        return entry.get("triples", []), entry.get("diagnostics", {})

    if use_llm:
        triples, diagnostics = extract_triples_for_patch(client, patch_view, model, retries, sleep_seconds)
    else:
        triples, diagnostics = heuristic_extract_triples(patch_view)

    cache[cache_key] = {"triples": triples, "diagnostics": diagnostics}
    append_cache(cache_path, cache_key, triples, diagnostics)
    return triples, diagnostics


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def patch_summary_meta(meta: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "project": meta.get("project"),
        "commit_id": meta.get("commit_id"),
        "idx": meta.get("idx"),
        "hash": meta.get("hash"),
        "cwe_eval_only": meta.get("cwe"),
    }


def run_extraction(args: argparse.Namespace) -> Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    phase1_output = load_phase1_output(args.phase1_json)
    records = phase1_output.get("records", [])
    if args.max_queries is not None:
        records = records[: args.max_queries]
    if not records:
        raise ValueError("No records found in Phase 1 JSON output.")

    patch_source = load_patch_source(args.patch_source)
    cache = read_cache(args.cache)
    use_llm = not args.no_llm
    client = make_openai_client(args.api_key, args.base_url) if use_llm else None

    triple_rows: List[Dict[str, Any]] = []
    patch_rows: List[Dict[str, Any]] = []
    output_records: List[Dict[str, Any]] = []

    for record in tqdm(records, desc="TripleExtraction"):
        query_meta = record.get("query", {})
        query_view = build_patch_view(query_meta, patch_source, args.max_chars)
        query_sid = patch_stable_id(query_meta)
        query_triples, query_diag = get_or_extract(
            query_sid, query_view, cache, args.cache, client, args.model, args.retries, args.sleep, use_llm
        )

        target_entry = {
            "role": "target",
            "rank": None,
            "meta": patch_summary_meta(query_meta),
            "view": {k: v for k, v in query_view.items() if k != "note"},
            "note": query_view.get("note"),
            "num_triples": len(query_triples),
            "dropped_triples": query_diag.get("dropped_triples", 0),
            "error": query_diag.get("error"),
            "triples": query_triples,
        }
        patch_entries = [target_entry]

        for triple in query_triples:
            triple_rows.append(
                {
                    "query_idx": query_meta.get("idx"),
                    "query_hash": query_meta.get("hash"),
                    "patch_role": "target",
                    "rank": None,
                    "patch_project": query_meta.get("project"),
                    "patch_commit_id": query_meta.get("commit_id"),
                    "patch_idx": query_meta.get("idx"),
                    "patch_hash": query_meta.get("hash"),
                    **triple,
                }
            )
        patch_rows.append(
            {
                "query_idx": query_meta.get("idx"),
                "patch_role": "target",
                "rank": None,
                "patch_project": query_meta.get("project"),
                "patch_commit_id": query_meta.get("commit_id"),
                "patch_idx": query_meta.get("idx"),
                "num_triples": len(query_triples),
                "dropped_triples": query_diag.get("dropped_triples", 0),
                "error": query_diag.get("error"),
            }
        )

        if args.include_retrieved:
            for cand in record.get("retrieved_cases", []):
                cand_view = build_patch_view(cand, patch_source, args.max_chars)
                cand_sid = patch_stable_id(cand)
                cand_triples, cand_diag = get_or_extract(
                    cand_sid, cand_view, cache, args.cache, client, args.model, args.retries, args.sleep, use_llm
                )
                rank = cand.get("rank")

                patch_entries.append(
                    {
                        "role": "retrieved",
                        "rank": rank,
                        "meta": patch_summary_meta(cand),
                        "view": {k: v for k, v in cand_view.items() if k != "note"},
                        "note": cand_view.get("note"),
                        "num_triples": len(cand_triples),
                        "dropped_triples": cand_diag.get("dropped_triples", 0),
                        "error": cand_diag.get("error"),
                        "triples": cand_triples,
                    }
                )

                for triple in cand_triples:
                    triple_rows.append(
                        {
                            "query_idx": query_meta.get("idx"),
                            "query_hash": query_meta.get("hash"),
                            "patch_role": "retrieved",
                            "rank": rank,
                            "patch_project": cand.get("project"),
                            "patch_commit_id": cand.get("commit_id"),
                            "patch_idx": cand.get("idx"),
                            "patch_hash": cand.get("hash"),
                            **triple,
                        }
                    )
                patch_rows.append(
                    {
                        "query_idx": query_meta.get("idx"),
                        "patch_role": "retrieved",
                        "rank": rank,
                        "patch_project": cand.get("project"),
                        "patch_commit_id": cand.get("commit_id"),
                        "patch_idx": cand.get("idx"),
                        "num_triples": len(cand_triples),
                        "dropped_triples": cand_diag.get("dropped_triples", 0),
                        "error": cand_diag.get("error"),
                    }
                )

        output_records.append({"query_idx": query_meta.get("idx"), "query_hash": query_meta.get("hash"), "patches": patch_entries})

    run_config = {
        "phase": "Patch Graph Construction / Triple-Aware Knowledge Materialisation",
        "phase_id": "PGC_TKM",
        "extractor_agent": "PhaseAwareTripleExtractor",
        "underlying_model": args.model if use_llm else None,
        "production_underlying_model": "ChatGPT-4o",
        "phase1_input": describe_source(args.phase1_json),
        "phase1_input_role": "Phase 1 output consumed by Phase 2",
        "patch_source": args.patch_source,
        "primevul_dataset_splits": PRIMEVUL_DATASET_SPLITS,
        "dataset_split_usage": "Triples are extracted for the Phase 1 query split while retrieved examples must originate from primevul_training.jsonl.",
        "model": args.model,
        "use_llm": use_llm,
        "materialisation_unit": "validated security triple with phase label",
        "graph_materialisation_note": "Phase 2 outputs subject-relation-object-phase triples for each target/retrieved patch; Phase 3 materialises these triples into directed phase-aware graphs for alignment.",
        "include_retrieved": args.include_retrieved,
        "max_chars": args.max_chars,
        "allowed_phases": sorted(ALLOWED_PHASES),
        "num_records": len(records),
        "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
    }

    output_json = {"run_config": run_config, "records": output_records}
    triples_df = pd.DataFrame(triple_rows)
    patches_df = pd.DataFrame(patch_rows)
    config_df = pd.DataFrame([run_config])
    return output_json, triples_df, patches_df, config_df


def save_outputs(
    output_json: Dict[str, Any],
    triples_df: pd.DataFrame,
    patches_df: pd.DataFrame,
    config_df: pd.DataFrame,
    output_json_path: str,
    output_excel_path: str,
) -> None:
    ensure_parent(output_json_path)
    ensure_parent(output_excel_path)

    with open(output_json_path, "w", encoding="utf-8") as handle:
        json.dump(output_json, handle, indent=2, ensure_ascii=False)

    with pd.ExcelWriter(output_excel_path, engine="openpyxl") as writer:
        triples_df.to_excel(writer, sheet_name="Triples", index=False)
        patches_df.to_excel(writer, sheet_name="Patches_Summary", index=False)
        config_df.to_excel(writer, sheet_name="Run_Config", index=False)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CWEMap Phase 2: Patch Graph Construction / Triple-Aware Knowledge Materialisation with ChatGPT-4o over Phase 1 output."
    )
    parser.add_argument("--phase1-json", required=True, help="Path to Phase 1 output JSON.")
    parser.add_argument(
        "--patch-source",
        default=None,
        help="Optional JSONL providing func_before/func_after/diff per patch "
        "(matched by hash, else project+commit_id+idx). Without this, only "
        "the single code snapshot already in the Phase 1 output is used.",
    )
    parser.add_argument("--output-json", required=True, help="Output JSON path for downstream phases.")
    parser.add_argument("--output-excel", required=True, help="Output Excel path for inspection.")
    parser.add_argument(
        "--cache",
        default="outputs/cache/gpt4o_triples.jsonl",
        help="JSONL cache for extracted triples, keyed by patch id + input fingerprint.",
    )

    parser.add_argument("--model", default="gpt-4o", help="OpenAI chat model name.")
    parser.add_argument("--api-key", default=None, help="OpenAI API key. Defaults to env OPENAI_API_KEY.")
    parser.add_argument("--base-url", default=None, help="Optional OpenAI-compatible base URL override.")
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Disable ChatGPT-4o calls and use a deterministic local heuristic extractor instead "
        "(function-call presence only). For offline development/testing/CI without an API key; "
        "not a substitute for the LLM extractor in production runs.",
    )
    parser.add_argument("--retries", type=int, default=3, help="API retry count per patch.")
    parser.add_argument("--sleep", type=float, default=0.2, help="Sleep interval after successful API calls.")
    parser.add_argument("--max-chars", type=int, default=12000, help="Max characters per code field sent to the model.")
    parser.add_argument("--max-queries", type=int, default=None, help="Optional limit on number of query records.")
    parser.add_argument(
        "--no-retrieved-triples",
        dest="include_retrieved",
        action="store_false",
        help="Only extract triples for the target query patch, skip retrieved historical patches.",
    )
    parser.set_defaults(include_retrieved=True)
    return parser


# ---------------------------------------------------------------------------
# Pipeline entrypoint (used by main.py to chain Phase 1 -> Phase 2 -> ... )
# ---------------------------------------------------------------------------
def run_patch_graph_construction(phase1_output: Any, config: Dict[str, Any]) -> Dict[str, Any]:
    """Phase 2 entrypoint driven by a config dict instead of CLI flags.

    Args:
        phase1_output: the dict returned by
            `phase1_patch_aware_retrieval.run_patch_aware_retrieval(...)`
            (in-memory chaining), OR a path string to a saved Phase 1 JSON
            file (standalone use).
        config: the full pipeline config; only `config["phase2"]` is read.

    Returns the Phase 2 JSON output in-memory for Phase 3 to consume.
    """
    parser = build_arg_parser()
    required = {} if phase1_output is not None else {"phase1_json": "phase2.phase1_json"}
    args = namespace_from_config(parser, config.get("phase2", {}), required=required)
    # Phase 1's output feeds Phase 2 directly - this is the Phase1->Phase2 link.
    # main.py passes the in-memory dict; standalone/config-driven partial reruns can use phase2.phase1_json.
    if phase1_output is not None:
        args.phase1_json = phase1_output

    output_json, triples_df, patches_df, config_df = run_extraction(args)
    save_outputs(
        output_json=output_json,
        triples_df=triples_df,
        patches_df=patches_df,
        config_df=config_df,
        output_json_path=args.output_json,
        output_excel_path=args.output_excel,
    )

    print(f"[Phase 2] Saved JSON output:  {args.output_json}")
    print(f"[Phase 2] Saved Excel output: {args.output_excel}")
    return output_json

# ---------------------------------------------------------------------------
def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    output_json, triples_df, patches_df, config_df = run_extraction(args)
    save_outputs(
        output_json=output_json,
        triples_df=triples_df,
        patches_df=patches_df,
        config_df=config_df,
        output_json_path=args.output_json,
        output_excel_path=args.output_excel,
    )

    print(f"Saved JSON output:  {args.output_json}")
    print(f"Saved Excel output: {args.output_excel}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
