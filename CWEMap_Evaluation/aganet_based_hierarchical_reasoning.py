#!/usr/bin/env python3
"""
Phase 4: HierarchicalReasoningAgent for CWEMap.

Example:
  python scripts/phase4_hierarchical_reasoning.py \
    --phase3-json outputs/evidence/primevul_evidence.json \
    --cwe-taxonomy Dataset_Preprocessing/dataset/cwe/cwe_taxonomy_edges.csv \
    --output-json outputs/paths/primevul_cwe_paths.json \
    --output-excel outputs/paths/primevul_cwe_paths.xlsx
"""

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import pandas as pd
from sklearn.metrics import f1_score, matthews_corrcoef
from tqdm import tqdm


# Allow both package import via main.py and direct standalone execution from the project root.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.namespace_adapter import describe_source, namespace_from_config

PRIMEVUL_DATASET_SPLITS = {
    "training": "primevul/primevul_training.jsonl",
    "validation": "primevul/primevul_validation.jsonl",
    "testing": "primevul/primevul_testing.jsonl",
}

PRIMEVUL_DEFAULT_PATHS = {
    split: f"Dataset_Preprocessing/dataset/{relative_path}"
    for split, relative_path in PRIMEVUL_DATASET_SPLITS.items()
}


NO_PREDICTION_LABEL = "NO_PREDICTION"

METRICS_DEFINITION = (
    "Evaluated only on queries whose ground-truth CWE (target_true_cwe_eval_only, "
    "carried through from Phase 1/2 for evaluation purposes only - never used by "
    "Phase 3 alignment/HNERP scoring or by this script's reasoning/pruning logic) "
    "is known. predicted_cwe is the terminal node of predicted_cwe_path, or the "
    "sentinel 'NO_PREDICTION' when no path was returned. Weighted F1 and Macro F1 "
    "are sklearn.metrics.f1_score over the union of true/predicted CWE labels "
    "(NO_PREDICTION included as its own class). MCC is sklearn.metrics."
    "matthews_corrcoef over the same true/predicted label pairs. Path Fraction "
    "(PF) per query = |predicted_cwe_path nodes ∩ true_cwe_path nodes| / "
    "|true_cwe_path nodes|, where true_cwe_path is the shortest root-to-ground-"
    "truth path in GCWE (ties broken lexicographically for determinism); PF is "
    "0.0 when no path was predicted, and omitted from the average only when the "
    "ground-truth CWE itself is not reachable from any GCWE root."
)

REASONING_SCORE_DEFINITION = (
    "node_score(n, parent) = w_hnerp*hnerp_support(n) + w_cov*evidence_coverage(n) "
    "+ w_align*alignment_quality(n) + w_tax*taxonomy_consistency(n) "
    "+ w_pc*parent_child_compatibility(n, parent), where (all computed strictly "
    "from Z, never invented): "
    "hnerp_support(n) = mean HNERP score of evidence items whose reference_cwe is "
    "n or a taxonomy descendant of n; "
    "evidence_coverage(n) = |such evidence items| / |total placeable evidence items|; "
    "alignment_quality(n) = mean of (structural_consistency, relation_consistency, "
    "phase_consistency) over those evidence items; "
    "taxonomy_consistency(n) = 1.0, constant, since n is only ever reached via an "
    "actual GCWE parent-child edge (the gate, not a soft score); "
    "parent_child_compatibility(n, parent) = evidence-item count under n divided by "
    "evidence-item count under parent (branch purity relative to the parent; 1.0 "
    "for root nodes, which have no parent constraint). "
    "Path confidence = mean of node_score over all nodes on the path."
)

DEEPSEEK_REASONER_PROMPT = """You are the CWEMap HierarchicalReasoningAgent.
You receive graph-aligned vulnerability evidence Z, HNERP-ranked reference
CWEs, and a small set of taxonomy-valid candidate CWE paths produced by a
beam search over the frozen CWE taxonomy.

Choose the single best candidate path for the target vulnerability. You must
not invent CWE IDs, reorder path nodes, use the target ground-truth label, or
select a path outside the provided candidates. Base your decision only on the
HNERP evidence, alignment quality, phase/relation consistency, evidence
coverage, taxonomy consistency, and parent-child compatibility fields.

Return JSON only with this shape:
{
  "selected_candidate_index": 0,
  "selected_path": ["CWE-..."],
  "confidence": 0.0,
  "rationale": "brief evidence-based rationale without mentioning hidden or unavailable labels"
}
"""


@dataclass(frozen=True)
class ReasoningWeights:
    w_hnerp: float = 0.30
    w_cov: float = 0.25
    w_align: float = 0.20
    w_tax: float = 0.10
    w_pc: float = 0.15


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def ensure_parent(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def normalize_cwe(value: Any) -> Optional[str]:
    if not value:
        return None
    match = re.search(r"CWE-?(\d+)", str(value), flags=re.IGNORECASE)
    return f"CWE-{match.group(1)}" if match else None


def load_phase3_output(source: Any) -> Dict[str, Any]:
    """Accepts either Phase 3's output already in memory (a dict, as passed
    by main.py when chaining phases directly) or a path to Phase 3's saved
    JSON file (standalone CLI use)."""
    if isinstance(source, dict):
        return source
    with open(source, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_cwe_taxonomy_digraph(path: str) -> nx.DiGraph:
    """GCWE as a directed parent -> child taxonomy graph for top-down
    traversal (edge list CSV/JSONL with parent_cwe/child_cwe columns)."""
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


# ---------------------------------------------------------------------------
# Evidence indexing (strictly from Z - reference_cwe + quality scores)
# ---------------------------------------------------------------------------
def collect_evidence(record: Dict[str, Any], taxonomy: nx.DiGraph) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Returns (placeable_evidence, unplaceable_cwes). Evidence whose
    reference_cwe isn't a node in GCWE cannot be positioned by hierarchical
    reasoning and is excluded from scoring (but reported for transparency),
    never guessed at."""
    placeable: List[Dict[str, Any]] = []
    unplaceable: List[str] = []

    for ref in record.get("matched_reference_graphs", []):
        cwe = normalize_cwe(ref.get("reference_cwe"))
        item = {
            "reference_graph_id": ref.get("reference_graph_id"),
            "cwe": cwe,
            "hnerp_score": ref.get("hnerp_score", 0.0),
            "structural_consistency": ref.get("structural_consistency", 0.0),
            "relation_consistency": ref.get("relation_consistency", 0.0),
            "phase_consistency": ref.get("phase_consistency", 0.0),
        }
        if cwe and cwe in taxonomy:
            placeable.append(item)
        elif cwe:
            unplaceable.append(cwe)

    return placeable, unplaceable


def build_descendant_index(taxonomy: nx.DiGraph, nodes: List[str]) -> Dict[str, set]:
    """self + all descendants for each node actually needed, cached."""
    index: Dict[str, set] = {}
    for n in nodes:
        index[n] = {n} | nx.descendants(taxonomy, n)
    return index


# ---------------------------------------------------------------------------
# Ground-truth path derivation + performance metrics
# ---------------------------------------------------------------------------
def true_cwe_path(taxonomy: nx.DiGraph, true_cwe: Optional[str]) -> Optional[List[str]]:
    """Shortest root -> true_cwe path in GCWE, for Path Fraction scoring only.

    Ties (multiple shortest paths, e.g. multiple parents) are broken by
    picking the lexicographically smallest path so results are deterministic
    across runs. Returns None if true_cwe is missing or not reachable from
    any root (in-degree 0 node) in the taxonomy.
    """
    if not true_cwe or true_cwe not in taxonomy:
        return None

    roots = [n for n in taxonomy.nodes if taxonomy.in_degree(n) == 0]
    candidate_paths: List[List[str]] = []
    for root in roots:
        if root == true_cwe:
            candidate_paths.append([root])
            continue
        try:
            candidate_paths.append(nx.shortest_path(taxonomy, root, true_cwe))
        except nx.NetworkXNoPath:
            continue

    if not candidate_paths:
        return None

    min_len = min(len(p) for p in candidate_paths)
    shortest = [p for p in candidate_paths if len(p) == min_len]
    shortest.sort()
    return shortest[0]


def path_fraction(predicted_path: Optional[List[str]], truth_path: Optional[List[str]]) -> Optional[float]:
    """PF = |predicted ∩ true| / |true|. None only when truth_path itself is
    undefined (ground truth not reachable in GCWE); 0.0 when no prediction
    was made but ground truth is placeable."""
    if truth_path is None or len(truth_path) == 0:
        return None
    if not predicted_path:
        return 0.0
    overlap = len(set(predicted_path) & set(truth_path))
    return round(overlap / len(truth_path), 6)


def compute_performance_metrics(
    y_true: List[str],
    y_pred: List[str],
    pf_values: List[float],
) -> Dict[str, Any]:
    """Weighted F1, Macro F1, MCC (classification, terminal-CWE prediction)
    plus mean Path Fraction (hierarchical partial-credit). Returns a dict
    with None values and an explanatory note if no evaluable samples exist."""
    if not y_true:
        return {
            "num_evaluated_queries": 0,
            "weighted_f1": None,
            "macro_f1": None,
            "mcc": None,
            "path_fraction_mean": None,
            "exact_match_accuracy": None,
            "note": "No queries had a known ground-truth CWE (target_true_cwe_eval_only); "
            "no classification metrics could be computed.",
        }

    labels = sorted(set(y_true) | set(y_pred))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0))
    # this gracefully and emits a warning, so no extra guard is needed here.
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
# Scoring
# ---------------------------------------------------------------------------
def support_items(node: str, evidence: List[Dict[str, Any]], descendant_index: Dict[str, set]) -> List[Dict[str, Any]]:
    subtree = descendant_index.get(node, {node})
    return [ev for ev in evidence if ev["cwe"] in subtree]


def score_node(
    node: str,
    parent: Optional[str],
    evidence: List[Dict[str, Any]],
    descendant_index: Dict[str, set],
    total_placeable: int,
    weights: ReasoningWeights,
) -> Dict[str, Any]:
    items = support_items(node, evidence, descendant_index)
    n_support = len(items)

    hnerp_support = round(sum(i["hnerp_score"] for i in items) / n_support, 6) if n_support else 0.0
    evidence_coverage = round(n_support / total_placeable, 6) if total_placeable else 0.0
    alignment_quality = (
        round(
            sum((i["structural_consistency"] + i["relation_consistency"] + i["phase_consistency"]) / 3.0 for i in items)
            / n_support,
            6,
        )
        if n_support
        else 0.0
    )
    taxonomy_consistency = 1.0  # gated by construction: n is only visited via a real GCWE edge

    if parent is None:
        parent_child_compatibility = 1.0
    else:
        parent_support = len(support_items(parent, evidence, descendant_index))
        parent_child_compatibility = round(min(1.0, n_support / parent_support), 6) if parent_support else 0.0

    node_score = round(
        weights.w_hnerp * hnerp_support
        + weights.w_cov * evidence_coverage
        + weights.w_align * alignment_quality
        + weights.w_tax * taxonomy_consistency
        + weights.w_pc * parent_child_compatibility,
        6,
    )

    return {
        "cwe": node,
        "num_supporting_evidence": n_support,
        "hnerp_support": hnerp_support,
        "evidence_coverage": evidence_coverage,
        "alignment_quality": alignment_quality,
        "taxonomy_consistency": taxonomy_consistency,
        "parent_child_compatibility": parent_child_compatibility,
        "node_score": node_score,
    }


# ---------------------------------------------------------------------------
# Top-down beam search
# ---------------------------------------------------------------------------
def reason_path(
    taxonomy: nx.DiGraph,
    evidence: List[Dict[str, Any]],
    weights: ReasoningWeights,
    beam_width: int,
    min_branch_score: float,
    max_depth: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Returns (completed_paths_sorted_desc, pruned_branch_log).
    Each completed path is {"path": [...], "confidence": ..., "level_scores": [...]}.
    """
    if not evidence:
        return [], []

    all_cwe_nodes = list(taxonomy.nodes)
    descendant_index = build_descendant_index(taxonomy, all_cwe_nodes)
    total_placeable = len(evidence)

    roots = [n for n in taxonomy.nodes if taxonomy.in_degree(n) == 0]

    pruned_log: List[Dict[str, Any]] = []
    # active beam entries: (path_nodes, level_score_records)
    beam: List[Tuple[List[str], List[Dict[str, Any]]]] = []

    for r in roots:
        rec = score_node(r, None, evidence, descendant_index, total_placeable, weights)
        if rec["num_supporting_evidence"] == 0 or rec["node_score"] < min_branch_score:
            pruned_log.append({"cwe": r, "parent": None, "reason": "no_or_weak_support", **rec})
            continue
        beam.append(([r], [rec]))

    beam.sort(key=lambda pe: sum(rec["node_score"] for rec in pe[1]) / len(pe[1]), reverse=True)
    beam = beam[:beam_width]

    completed: List[Dict[str, Any]] = []
    depth = 0

    while beam and depth < max_depth:
        depth += 1
        next_beam: List[Tuple[List[str], List[Dict[str, Any]]]] = []

        for path_nodes, level_scores in beam:
            last = path_nodes[-1]
            children = list(taxonomy.successors(last))
            candidate_recs = []
            for c in children:
                if c in path_nodes:
                    continue  # guard against cycles in malformed taxonomy data
                rec = score_node(c, last, evidence, descendant_index, total_placeable, weights)
                if rec["num_supporting_evidence"] == 0 or rec["node_score"] < min_branch_score:
                    pruned_log.append({"cwe": c, "parent": last, "reason": "no_or_weak_support", **rec})
                    continue
                candidate_recs.append((c, rec))

            if not candidate_recs:
                confidence = round(sum(rec["node_score"] for rec in level_scores) / len(level_scores), 6)
                completed.append({"path": path_nodes, "confidence": confidence, "level_scores": level_scores})
                continue

            candidate_recs.sort(key=lambda cr: cr[1]["node_score"], reverse=True)
            for c, rec in candidate_recs[:beam_width]:
                next_beam.append((path_nodes + [c], level_scores + [rec]))

        next_beam.sort(
            key=lambda pe: sum(rec["node_score"] for rec in pe[1]) / len(pe[1]), reverse=True
        )
        beam = next_beam[:beam_width]

    # any beam entries still active when max_depth was hit are also completed
    for path_nodes, level_scores in beam:
        confidence = round(sum(rec["node_score"] for rec in level_scores) / len(level_scores), 6)
        completed.append({"path": path_nodes, "confidence": confidence, "level_scores": level_scores})

    completed.sort(key=lambda c: c["confidence"], reverse=True)
    return completed, pruned_log



# ---------------------------------------------------------------------------
# DeepSeek-V4-Flash reasoning / candidate selection
# ---------------------------------------------------------------------------
def make_deepseek_client(api_key: Optional[str], base_url: str) -> Any:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError("Missing dependency: install with `pip install openai`.") from exc

    key = api_key or os.getenv("DEEPSEEK_API_KEY")
    if not key:
        raise EnvironmentError("DEEPSEEK_API_KEY is not set. Export it or pass --api-key, or use --no-llm.")
    return OpenAI(api_key=key, base_url=base_url)


def parse_json_object(text: str) -> Dict[str, Any]:
    cleaned = (text or "").strip()
    cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()
    try:
        parsed = json.loads(cleaned)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start >= 0 and end > start:
            try:
                parsed = json.loads(cleaned[start : end + 1])
                return parsed if isinstance(parsed, dict) else {}
            except json.JSONDecodeError:
                return {}
    return {}


def read_reasoning_cache(cache_path: str) -> Dict[str, Dict[str, Any]]:
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


def append_reasoning_cache(cache_path: str, cache_key: str, selection: Dict[str, Any], diagnostics: Dict[str, Any]) -> None:
    ensure_parent(cache_path)
    with open(cache_path, "a", encoding="utf-8") as handle:
        handle.write(
            json.dumps({"cache_key": cache_key, "selection": selection, "diagnostics": diagnostics}, ensure_ascii=False)
            + "\n"
        )


def candidate_payload(completed: List[Dict[str, Any]], max_candidates: int) -> List[Dict[str, Any]]:
    payload: List[Dict[str, Any]] = []
    for idx, cand in enumerate(completed[:max_candidates]):
        payload.append(
            {
                "candidate_index": idx,
                "path": cand.get("path"),
                "deterministic_confidence": cand.get("confidence"),
                "level_scores": cand.get("level_scores", []),
            }
        )
    return payload


def reasoning_fingerprint(target_id: Any, evidence: List[Dict[str, Any]], candidates: List[Dict[str, Any]]) -> str:
    raw = json.dumps(
        {"target_graph_id": target_id, "evidence": evidence, "candidate_paths": candidates},
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(raw.encode("utf-8", errors="ignore")).hexdigest()


def select_path_with_deepseek(
    client: Any,
    model: str,
    target_id: Any,
    evidence: List[Dict[str, Any]],
    candidates: List[Dict[str, Any]],
    cache: Dict[str, Dict[str, Any]],
    cache_path: str,
    retries: int,
    sleep_seconds: float,
) -> Tuple[Optional[int], Dict[str, Any]]:
    """Ask DeepSeek-V4-Flash to select among already taxonomy-valid paths.

    The model is deliberately constrained to a candidate index; the returned
    selection is validated against the candidate list before it can affect the
    final prediction.
    """
    if not candidates:
        return None, {"used_llm": False, "error": "no_candidate_paths"}

    cache_key = f"deepseek_hra:{model}:{reasoning_fingerprint(target_id, evidence, candidates)}"
    if cache_key in cache:
        entry = cache[cache_key]
        selection = entry.get("selection", {})
        diagnostics = entry.get("diagnostics", {})
    else:
        user_payload = {
            "target_graph_id": target_id,
            "graph_aligned_evidence_Z": evidence,
            "taxonomy_valid_candidate_paths": candidates,
        }
        last_error: Optional[str] = None
        selection: Dict[str, Any] = {}
        diagnostics = {"used_llm": True, "error": None}
        for attempt in range(1, retries + 1):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": DEEPSEEK_REASONER_PROMPT},
                        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
                    ],
                    temperature=0.0,
                    max_tokens=700,
                    response_format={"type": "json_object"},
                )
                content = response.choices[0].message.content or ""
                selection = parse_json_object(content)
                time.sleep(sleep_seconds)
                break
            except Exception as exc:  # noqa: BLE001 - external API call, broad catch is intentional
                last_error = str(exc)
                if attempt >= retries:
                    diagnostics["error"] = last_error
                    break
                time.sleep(max(1.0, sleep_seconds) * attempt)
        append_reasoning_cache(cache_path, cache_key, selection, diagnostics)
        cache[cache_key] = {"selection": selection, "diagnostics": diagnostics}

    candidate_by_index = {item["candidate_index"]: item for item in candidates}
    selected_index = selection.get("selected_candidate_index")
    try:
        selected_index = int(selected_index)
    except (TypeError, ValueError):
        selected_index = None

    if selected_index not in candidate_by_index:
        return None, {
            **diagnostics,
            "used_llm": diagnostics.get("used_llm", True),
            "validated": False,
            "selection": selection,
            "error": diagnostics.get("error") or "DeepSeek selected an invalid candidate index",
        }

    selected_path = selection.get("selected_path")
    if selected_path and selected_path != candidate_by_index[selected_index].get("path"):
        return None, {
            **diagnostics,
            "used_llm": diagnostics.get("used_llm", True),
            "validated": False,
            "selection": selection,
            "error": diagnostics.get("error") or "DeepSeek selected_path did not match selected_candidate_index",
        }

    return selected_index, {
        **diagnostics,
        "used_llm": diagnostics.get("used_llm", True),
        "validated": True,
        "selection": selection,
        "rationale": selection.get("rationale"),
        "model_confidence": selection.get("confidence"),
    }


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def run_reasoning(args: argparse.Namespace) -> Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    phase3_output = load_phase3_output(args.phase3_json)
    records = phase3_output.get("records", [])
    if args.max_queries is not None:
        records = records[: args.max_queries]
    if not records:
        raise ValueError("No records found in Phase 3 JSON output.")

    taxonomy = load_cwe_taxonomy_digraph(args.cwe_taxonomy)
    weights = ReasoningWeights(
        w_hnerp=args.w_hnerp, w_cov=args.w_cov, w_align=args.w_align, w_tax=args.w_tax, w_pc=args.w_pc
    )
    use_llm = not args.no_llm
    reasoning_client = make_deepseek_client(args.api_key, args.base_url) if use_llm else None
    reasoning_cache = read_reasoning_cache(args.cache)

    output_records: List[Dict[str, Any]] = []
    path_rows: List[Dict[str, Any]] = []
    pruned_rows: List[Dict[str, Any]] = []
    prediction_rows: List[Dict[str, Any]] = []

    y_true: List[str] = []
    y_pred: List[str] = []
    pf_values: List[float] = []

    for record in tqdm(records, desc="HierarchicalReasoning"):
        target_id = record.get("target_graph_id")
        true_cwe = normalize_cwe(record.get("target_true_cwe_eval_only"))
        truth_path = true_cwe_path(taxonomy, true_cwe)
        evidence, unplaceable = collect_evidence(record, taxonomy)

        completed, pruned_log = reason_path(
            taxonomy, evidence, weights, args.beam_width, args.min_branch_score, args.max_depth
        )

        for p in pruned_log:
            pruned_rows.append({"target_graph_id": target_id, **p})

        if not completed:
            output_records.append(
                {
                    "target_graph_id": target_id,
                    "target_true_cwe_eval_only": true_cwe,
                    "predicted_cwe_path": None,
                    "predicted_cwe": None,
                    "path_confidence": 0.0,
                    "reasoning_mode": "deepseek_v4_flash" if use_llm else "deterministic_offline",
                    "reasoning_model": args.model if use_llm else None,
                    "llm_reasoning_used": False,
                    "refinement_applied": False,
                    "note": "insufficient taxonomy-placeable evidence in Z; no path could be supported"
                    if not evidence
                    else "no root node reached the support/score threshold",
                    "unplaceable_reference_cwes": sorted(set(unplaceable)),
                    "path_fraction": path_fraction(None, truth_path),
                }
            )
            if true_cwe:
                y_true.append(true_cwe)
                y_pred.append(NO_PREDICTION_LABEL)
                pf = path_fraction(None, truth_path)
                if pf is not None:
                    pf_values.append(pf)
                prediction_rows.append(
                    {
                        "target_graph_id": target_id,
                        "target_true_cwe": true_cwe,
                        "predicted_cwe": NO_PREDICTION_LABEL,
                        "exact_match": False,
                        "path_confidence": 0.0,
                        "path_fraction": pf,
                    }
                )
            continue

        best = completed[0]
        refinement_applied = False
        llm_diagnostics: Dict[str, Any] = {"used_llm": False, "validated": False}

        candidate_paths = candidate_payload(completed, args.max_alternatives + 1)
        if use_llm and reasoning_client is not None:
            selected_index, llm_diagnostics = select_path_with_deepseek(
                client=reasoning_client,
                model=args.model,
                target_id=target_id,
                evidence=evidence,
                candidates=candidate_paths,
                cache=reasoning_cache,
                cache_path=args.cache,
                retries=args.retries,
                sleep_seconds=args.sleep,
            )
            if selected_index is not None:
                best = completed[selected_index]
                refinement_applied = selected_index != 0

        if not use_llm and best["confidence"] < args.min_confidence and len(completed) > 1:
            # Offline deterministic mode: reconsider alternative taxonomy-valid
            # branches already carried in the beam - never invents new evidence.
            best = completed[0]
            refinement_applied = True

        alternatives = [
            {"path": c["path"], "confidence": c["confidence"]} for c in completed if c is not best
        ][: args.max_alternatives]

        predicted_cwe = best["path"][-1]
        pf = path_fraction(best["path"], truth_path)

        output_records.append(
            {
                "target_graph_id": target_id,
                "target_true_cwe_eval_only": true_cwe,
                "predicted_cwe_path": best["path"],
                "predicted_cwe": predicted_cwe,
                "path_confidence": best["confidence"],
                "reasoning_mode": "deepseek_v4_flash" if use_llm else "deterministic_offline",
                "reasoning_model": args.model if use_llm else None,
                "llm_reasoning_used": bool(llm_diagnostics.get("used_llm") and llm_diagnostics.get("validated")),
                "llm_reasoning_diagnostics": llm_diagnostics,
                "deepseek_rationale": llm_diagnostics.get("rationale"),
                "level_scores": best["level_scores"],
                "refinement_applied": refinement_applied,
                "below_min_confidence": best["confidence"] < args.min_confidence,
                "alternative_paths_considered": alternatives,
                "unplaceable_reference_cwes": sorted(set(unplaceable)),
                "path_fraction": pf,
            }
        )

        path_rows.append(
            {
                "target_graph_id": target_id,
                "target_true_cwe": true_cwe,
                "predicted_cwe_path": " -> ".join(best["path"]),
                "terminal_cwe": predicted_cwe,
                "path_confidence": best["confidence"],
                "reasoning_mode": "deepseek_v4_flash" if use_llm else "deterministic_offline",
                "reasoning_model": args.model if use_llm else None,
                "llm_reasoning_used": bool(llm_diagnostics.get("used_llm") and llm_diagnostics.get("validated")),
                "path_depth": len(best["path"]),
                "num_alternatives_considered": len(alternatives),
                "refinement_applied": refinement_applied,
                "path_fraction": pf,
            }
        )

        if true_cwe:
            y_true.append(true_cwe)
            y_pred.append(predicted_cwe)
            if pf is not None:
                pf_values.append(pf)
            prediction_rows.append(
                {
                    "target_graph_id": target_id,
                    "target_true_cwe": true_cwe,
                    "predicted_cwe": predicted_cwe,
                    "exact_match": true_cwe == predicted_cwe,
                    "path_confidence": best["confidence"],
                    "path_fraction": pf,
                }
            )

    performance_metrics = compute_performance_metrics(y_true, y_pred, pf_values)

    run_config = {
        "phase": "Hierarchical Reasoning with DeepSeek-V4-Flash",
        "phase_id": "HRA_DSV4F",
        "underlying_model": args.model if use_llm else None,
        "use_llm_reasoner": use_llm,
        "reasoning_mode": "deepseek_v4_flash" if use_llm else "deterministic_offline",
        "phase3_input": describe_source(args.phase3_json),
        "phase3_input_role": "Phase 3 output consumed by Phase 4",
        "cwe_taxonomy": args.cwe_taxonomy,
        "primevul_dataset_splits": PRIMEVUL_DATASET_SPLITS,
        "dataset_split_usage": "Final CWE path prediction is evaluated on validation/testing queries; supporting historical evidence comes from primevul_training.jsonl only.",
        "beam_width": args.beam_width,
        "min_branch_score": args.min_branch_score,
        "min_confidence": args.min_confidence,
        "max_depth": args.max_depth,
        "reasoning_weights": weights.__dict__,
        "reasoning_score_definition": REASONING_SCORE_DEFINITION,
        "deepseek_reasoner_prompt_sha256": hashlib.sha256(DEEPSEEK_REASONER_PROMPT.encode("utf-8")).hexdigest() if use_llm else None,
        "deepseek_validation_note": "DeepSeek may only select among taxonomy-valid beam-search candidate paths; invalid or unmatched selections are rejected and the deterministic top path is used.",
        "metrics_definition": METRICS_DEFINITION,
        "num_records": len(records),
    }

    output_json = {"run_config": run_config, "performance_metrics": performance_metrics, "records": output_records}
    path_df = pd.DataFrame(path_rows)
    pruned_df = pd.DataFrame(pruned_rows)
    config_df = pd.DataFrame([run_config])
    metrics_df = pd.DataFrame([performance_metrics])
    predictions_df = pd.DataFrame(prediction_rows)
    return output_json, path_df, pruned_df, config_df, metrics_df, predictions_df


def save_outputs(
    output_json: Dict[str, Any],
    path_df: pd.DataFrame,
    pruned_df: pd.DataFrame,
    config_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    output_json_path: str,
    output_excel_path: str,
    output_csv_prefix: str,
) -> None:
    ensure_parent(output_json_path)
    ensure_parent(output_excel_path)
    ensure_parent(output_csv_prefix)

    with open(output_json_path, "w", encoding="utf-8") as handle:
        json.dump(output_json, handle, indent=2, ensure_ascii=False)

    with pd.ExcelWriter(output_excel_path, engine="openpyxl") as writer:
        path_df.to_excel(writer, sheet_name="Predicted_Paths", index=False)
        predictions_df.to_excel(writer, sheet_name="Predictions_vs_GroundTruth", index=False)
        metrics_df.to_excel(writer, sheet_name="Performance_Metrics", index=False)
        pruned_df.to_excel(writer, sheet_name="Pruned_Branches", index=False)
        config_df.to_excel(writer, sheet_name="Run_Config", index=False)

    csv_paths = {
        "predicted_paths": f"{output_csv_prefix}_predicted_paths.csv",
        "predictions_vs_groundtruth": f"{output_csv_prefix}_predictions_vs_groundtruth.csv",
        "performance_metrics": f"{output_csv_prefix}_performance_metrics.csv",
        "pruned_branches": f"{output_csv_prefix}_pruned_branches.csv",
    }
    for path in csv_paths.values():
        ensure_parent(path)

    path_df.to_csv(csv_paths["predicted_paths"], index=False)
    predictions_df.to_csv(csv_paths["predictions_vs_groundtruth"], index=False)
    metrics_df.to_csv(csv_paths["performance_metrics"], index=False)
    pruned_df.to_csv(csv_paths["pruned_branches"], index=False)

    return csv_paths


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CWEMap Phase 4: HierarchicalReasoningAgent - top-down CWE path prediction from Phase 3's evidence package Z."
    )
    parser.add_argument("--phase3-json", required=True, help="Path to Phase 3 output JSON (evidence package Z per query).")
    parser.add_argument("--model", default="DeepSeek-V4-Flash", help="DeepSeek model name for the final hierarchical reasoning agent.")
    parser.add_argument("--base-url", default=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"), help="DeepSeek OpenAI-compatible base URL.")
    parser.add_argument("--api-key", default=None, help="DeepSeek API key. Defaults to env DEEPSEEK_API_KEY.")
    parser.add_argument("--cache", default="outputs/cache/deepseek_v4_flash_reasoning.jsonl", help="JSONL cache for DeepSeek reasoning selections.")
    parser.add_argument("--no-llm", action="store_true", help="Disable DeepSeek-V4-Flash and use deterministic offline beam reasoning only.")
    parser.add_argument("--retries", type=int, default=3, help="DeepSeek API retry count.")
    parser.add_argument("--sleep", type=float, default=0.2, help="Sleep interval after successful DeepSeek calls.")
    parser.add_argument(
        "--cwe-taxonomy",
        required=True,
        help="CSV/JSONL edge list for the frozen CWE taxonomy graph GCWE (columns/keys: parent_cwe, child_cwe).",
    )
    parser.add_argument("--output-json", required=True, help="Output JSON path with predicted CWE paths.")
    parser.add_argument("--output-excel", required=True, help="Output Excel path for inspection.")
    parser.add_argument(
        "--output-csv-prefix",
        required=True,
        help="Path prefix for CSV outputs (e.g. outputs/paths/primevul_cwe_paths). Writes "
        "<prefix>_predicted_paths.csv, <prefix>_predictions_vs_groundtruth.csv, "
        "<prefix>_performance_metrics.csv, and <prefix>_pruned_branches.csv.",
    )

    parser.add_argument("--beam-width", type=int, default=3, help="Number of alternative branches carried per level.")
    parser.add_argument(
        "--min-branch-score", type=float, default=0.05, help="Prune a candidate child if its node_score falls below this."
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.4,
        help="If the top path's confidence is below this, refinement re-ranks already-computed beam alternatives "
        "(flagged in output, no new evidence is invented).",
    )
    parser.add_argument("--max-depth", type=int, default=20, help="Safety cap on taxonomy traversal depth.")
    parser.add_argument("--max-alternatives", type=int, default=3, help="Number of alternative paths to report for audit.")
    parser.add_argument("--max-queries", type=int, default=None, help="Optional limit on number of query records.")

    weight_defaults = ReasoningWeights()
    parser.add_argument("--w-hnerp", type=float, default=weight_defaults.w_hnerp, help="Weight for hnerp_support.")
    parser.add_argument("--w-cov", type=float, default=weight_defaults.w_cov, help="Weight for evidence_coverage.")
    parser.add_argument("--w-align", type=float, default=weight_defaults.w_align, help="Weight for alignment_quality.")
    parser.add_argument("--w-tax", type=float, default=weight_defaults.w_tax, help="Weight for taxonomy_consistency.")
    parser.add_argument("--w-pc", type=float, default=weight_defaults.w_pc, help="Weight for parent_child_compatibility.")
    return parser


# ---------------------------------------------------------------------------
# Pipeline entrypoint (used by main.py to chain Phase 1 -> Phase 2 -> ... )
# ---------------------------------------------------------------------------
def run_hierarchical_reasoning(phase3_output: Any, config: Dict[str, Any]) -> Dict[str, Any]:
    """Phase 4 entrypoint driven by a config dict instead of CLI flags.

    Args:
        phase3_output: the dict returned by
            `phase3_evidence_alignment.run_evidence_alignment(...)`
            (in-memory chaining), OR a path string to a saved Phase 3 JSON
            file (standalone use).
        config: the full pipeline config; only `config["phase4"]` is read.

    Returns the Phase 4 JSON output (predicted CWE paths + performance
    metrics) - the final prediction artifact of the whole pipeline.
    """
    parser = build_arg_parser()
    required = {"cwe_taxonomy": "phase4.cwe_taxonomy"}
    if phase3_output is None:
        required["phase3_json"] = "phase4.phase3_json"
    args = namespace_from_config(
        parser,
        config.get("phase4", {}),
        required=required,
    )
    # Phase 3's output feeds Phase 4 directly - this is the Phase3->Phase4 link.
    # main.py passes the in-memory dict; standalone/config-driven partial reruns can use phase4.phase3_json.
    if phase3_output is not None:
        args.phase3_json = phase3_output

    output_json, path_df, pruned_df, config_df, metrics_df, predictions_df = run_reasoning(args)
    csv_paths = save_outputs(
        output_json=output_json,
        path_df=path_df,
        pruned_df=pruned_df,
        config_df=config_df,
        metrics_df=metrics_df,
        predictions_df=predictions_df,
        output_json_path=args.output_json,
        output_excel_path=args.output_excel,
        output_csv_prefix=args.output_csv_prefix,
    )

    print(f"[Phase 4] Saved JSON output:  {args.output_json}")
    print(f"[Phase 4] Saved Excel output: {args.output_excel}")
    for name, path in csv_paths.items():
        print(f"[Phase 4] Saved CSV output ({name}): {path}")

    metrics = output_json.get("performance_metrics", {})
    if metrics.get("note"):
        print(f"[Phase 4] Performance metrics: {metrics['note']}")
    else:
        print(
            "[Phase 4] Performance metrics — "
            f"Weighted F1: {metrics.get('weighted_f1')}, "
            f"Macro F1: {metrics.get('macro_f1')}, "
            f"MCC: {metrics.get('mcc')}, "
            f"Path Fraction (mean): {metrics.get('path_fraction_mean')}, "
            f"evaluated on {metrics.get('num_evaluated_queries')} queries."
        )
    return output_json


# ---------------------------------------------------------------------------
def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    output_json, path_df, pruned_df, config_df, metrics_df, predictions_df = run_reasoning(args)
    csv_paths = save_outputs(
        output_json=output_json,
        path_df=path_df,
        pruned_df=pruned_df,
        config_df=config_df,
        metrics_df=metrics_df,
        predictions_df=predictions_df,
        output_json_path=args.output_json,
        output_excel_path=args.output_excel,
        output_csv_prefix=args.output_csv_prefix,
    )

    print(f"Saved JSON output:  {args.output_json}")
    print(f"Saved Excel output: {args.output_excel}")
    for name, path in csv_paths.items():
        print(f"Saved CSV output ({name}): {path}")

    metrics = output_json.get("performance_metrics", {})
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
