#!/usr/bin/env python3
"""
Phase 3: EvidenceAlignmentAgent for CWEMap.

Consumes the JSON produced by Phase 2 (PhaseAwareTripleExtractor) and, for
each query from primevul/primevul_validation.jsonl or
primevul/primevul_testing.jsonl, aligns the target patch graph (Ginput)
against top-k historical graphs retrieved only from
primevul/primevul_training.jsonl (KGexamples), using the frozen
CWE taxonomy graph (GCWE) only as a *post-hoc coherence signal* among
already-accepted evidence (never using the target's own CWE label, which
is unknown at inference time and must not leak into alignment).

This script performs no LLM calls. It is a deterministic graph-reasoning
step:

VF2++
     filter the verified mappings by strict edge relation and temporal phase
     equality (T_before / T_delta / T_after). This implements the requested
     VF2++ matching/filtering stage while keeping phase-aware edge semantics
     explicit and auditable.

 Rank candidates by HNERP and assemble the evidence package Z per
     query, containing only structurally validated, ranked evidence.

Example:
  python scripts/phase3_evidence_alignment.py \
    --phase2-json outputs/triples/primevul_triples.json \
    --cwe-taxonomy Dataset_Preprocessing/dataset/cwe/cwe_taxonomy_edges.csv \
    --output-json outputs/evidence/primevul_evidence.json \
    --output-excel outputs/evidence/primevul_evidence.xlsx
"""

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import pandas as pd
from networkx.algorithms.isomorphism import MultiDiGraphMatcher
from tqdm import tqdm


# Allow both package import via main.py and direct standalone execution from the project root.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.namespace_adapter import describe_source, namespace_from_config

# Canonical PrimeVul split names used throughout the pipeline.
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

ALLOWED_PHASES = ("T_before", "T_delta", "T_after")

HNERP_DEFINITION = (
    "HNERP = w_node * node_resonance + w_edge * edge_resonance "
    "+ w_hier * hierarchy_resonance, where: "
    "node_resonance = |exact-identifier-matched target-core nodes| / |target-core nodes| "
    "(how much of the target's evidence graph is anchored by shared identifiers); "
    "edge_resonance = |edges of the target-core subgraph induced on the matched nodes| "
    "/ |edges of the full target-core subgraph| "
    "(how much of the target's evidence graph the verified isomorphic substructure covers); "
    "hierarchy_resonance = mean normalized CWE-taxonomy proximity (via GCWE shortest-path "
    "distance) between this candidate's CWE and the CWEs of the other candidates already "
    "accepted for the same query (peer coherence only - the target's own CWE is never used, "
    "since it is unknown at inference time). "
    "Weights are renormalized over whichever components have data available "
    "(hierarchy_resonance is dropped, not zero-filled, when GCWE or CWE labels are unavailable)."
)


@dataclass(frozen=True)
class HnerpWeights:
    w_node: float = 0.40
    w_edge: float = 0.40
    w_hier: float = 0.20


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def ensure_parent(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def normalize_label(text: Any) -> str:
    if not isinstance(text, str):
        return str(text).strip().lower()
    return re.sub(r"\s+", " ", text.strip().lower())


def normalize_cwe(value: Any) -> Optional[str]:
    """Extract a single canonical CWE-NNN id string from whatever shape the
    upstream `cwe` field takes (string, list, or list of dicts)."""
    if value is None:
        return None
    if isinstance(value, list):
        value = value[0] if value else None
    if isinstance(value, dict):
        value = value.get("cwe") or value.get("id")
    if not value:
        return None
    match = re.search(r"CWE-?(\d+)", str(value), flags=re.IGNORECASE)
    return f"CWE-{match.group(1)}" if match else None


# ---------------------------------------------------------------------------
# Loading Phase 2 output + optional CWE taxonomy
# ---------------------------------------------------------------------------
def load_phase2_output(source: Any) -> Dict[str, Any]:
    """Accepts either Phase 2's output already in memory (a dict, as passed
    by main.py when chaining phases directly) or a path to Phase 2's saved
    JSON file (standalone CLI use)."""
    if isinstance(source, dict):
        return source
    with open(source, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_cwe_taxonomy(path: Optional[str]) -> Optional[nx.Graph]:
    """GCWE: an undirected parent/child CWE taxonomy graph loaded from a
    CSV/JSONL edge list (columns/keys: parent_cwe, child_cwe). Returns None
    if not supplied - hierarchy_resonance is then dropped, never invented."""
    if not path:
        return None

    graph = nx.Graph()
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

    return graph if graph.number_of_nodes() > 0 else None


# ---------------------------------------------------------------------------
# Graph construction from Phase 2 triples
# ---------------------------------------------------------------------------
def build_patch_graph(patch_entry: Dict[str, Any]) -> nx.MultiDiGraph:
    """Ginput / one KGexample: nodes = exact identifiers from Phase 2
    triples, edges = (relation, phase). Node/edge content is taken
    verbatim from Phase 2 output - no new triples are extracted here."""
    graph = nx.MultiDiGraph()
    for triple in patch_entry.get("triples", []):
        subj, rel, obj, phase = (
            triple.get("subject"),
            triple.get("relation"),
            triple.get("object"),
            triple.get("phase"),
        )
        if not (subj and rel and obj and phase in ALLOWED_PHASES):
            continue
        graph.add_node(subj, label=normalize_label(subj))
        graph.add_node(obj, label=normalize_label(obj))
        graph.add_edge(subj, obj, key=f"{rel}::{phase}", relation=rel, phase=phase)
    return graph


def extract_core_subgraph(graph: nx.MultiDiGraph) -> nx.MultiDiGraph:
    """Security-relevant core subgraph: Phase 2 triples are already
    security-filtered by construction, so this step only drops isolated
    nodes (defensive - should rarely trigger given how edges are built)."""
    isolated = list(nx.isolates(graph))
    if not isolated:
        return graph
    core = graph.copy()
    core.remove_nodes_from(isolated)
    return core


# ---------------------------------------------------------------------------
# Candidate correspondence + VF2++ structural/semantic verification
# ---------------------------------------------------------------------------
def node_match(attrs1: Dict[str, Any], attrs2: Dict[str, Any]) -> bool:
    return attrs1.get("label") == attrs2.get("label")


def edge_match(attrs1: Dict[str, Any], attrs2: Dict[str, Any]) -> bool:
    return attrs1.get("relation") == attrs2.get("relation") and attrs1.get("phase") == attrs2.get("phase")


def verify_alignment(
    target_core: nx.MultiDiGraph,
    cand_core: nx.MultiDiGraph,
    min_shared_nodes: int,
) -> Optional[Dict[str, Any]]:
    """Derive the node correspondence from exact identifier overlap, then
    verify it with NetworkX VF2++ over normalized node labels. The accepted
    VF2++ mapping is subsequently filtered by relation and phase on every
    mapped edge; candidates that fail this semantic edge check are rejected.
    Returns None if rejected."""
    target_labels = {normalize_label(n): n for n in target_core.nodes}
    cand_labels = {normalize_label(n): n for n in cand_core.nodes}
    shared = sorted(set(target_labels) & set(cand_labels))

    if len(shared) < min_shared_nodes:
        return None

    target_nodes = [target_labels[lbl] for lbl in shared]
    cand_nodes = [cand_labels[lbl] for lbl in shared]

    target_sub = target_core.subgraph(target_nodes).copy()
    cand_sub = cand_core.subgraph(cand_nodes).copy()

    # VF2++ verifies the structurally compatible mapping under exact node
    # labels. Edge relation/phase semantics are checked immediately below and
    # must be fully consistent for a candidate to survive filtering.
    try:
        mapping_iter = nx.vf2pp_all_isomorphisms(target_sub, cand_sub, node_label="label", default_label="")
        node_mapping = next(mapping_iter, None)
    except Exception:  # pragma: no cover - defensive compatibility fallback
        matcher = MultiDiGraphMatcher(target_sub, cand_sub, node_match=node_match, edge_match=edge_match)
        node_mapping = dict(matcher.mapping) if matcher.is_isomorphic() else None

    if node_mapping is None:
        return None

    node_mapping = dict(node_mapping)  # target_node -> cand_node

    edge_mapping: List[Dict[str, Any]] = []
    relation_hits, phase_hits, total = 0, 0, 0
    for u, v, key, data in target_sub.edges(keys=True, data=True):
        total += 1
        mu, mv = node_mapping.get(u), node_mapping.get(v)
        cand_edges = cand_sub.get_edge_data(mu, mv) or {}
        matched = None
        for ckey, cdata in cand_edges.items():
            if cdata.get("relation") == data.get("relation") and cdata.get("phase") == data.get("phase"):
                matched = cdata
                break
        if matched is not None:
            relation_hits += 1
            phase_hits += 1
            edge_mapping.append(
                {
                    "target_edge": {"subject": u, "relation": data.get("relation"), "object": v, "phase": data.get("phase")},
                    "matched_edge": {"subject": mu, "relation": matched.get("relation"), "object": mv, "phase": matched.get("phase")},
                }
            )

    # VF2++ checked graph structure; this edge-semantic filter enforces the
    # phase-aware knowledge constraints. Without full relation/phase agreement,
    # the candidate is rejected rather than merely down-weighted.
    if total and len(edge_mapping) < total:
        return None

    relation_consistency = round(relation_hits / total, 6) if total else 0.0
    phase_consistency = round(phase_hits / total, 6) if total else 0.0

    node_resonance = round(len(shared) / target_core.number_of_nodes(), 6) if target_core.number_of_nodes() else 0.0
    edge_resonance = (
        round(target_sub.number_of_edges() / target_core.number_of_edges(), 6)
        if target_core.number_of_edges()
        else 0.0
    )

    return {
        "node_mapping": [{"target_node": u, "candidate_node": v} for u, v in node_mapping.items()],
        "edge_mapping": edge_mapping,
        "phase_consistency": phase_consistency,
        "relation_consistency": relation_consistency,
        "structural_consistency": 1.0,  # only reachable if matcher.is_isomorphic() was True
        "node_resonance": node_resonance,
        "edge_resonance": edge_resonance,
    }


# ---------------------------------------------------------------------------
# Hierarchy resonance (peer CWE-taxonomy coherence among accepted evidence)
# ---------------------------------------------------------------------------
def cwe_taxonomic_proximity(cwe_a: str, cwe_b: str, taxonomy: nx.Graph) -> Optional[float]:
    if cwe_a not in taxonomy or cwe_b not in taxonomy:
        return None
    try:
        distance = nx.shortest_path_length(taxonomy, cwe_a, cwe_b)
    except nx.NetworkXNoPath:
        return None
    return 1.0 / (1.0 + distance)


def compute_hierarchy_resonance(
    accepted: List[Dict[str, Any]],
    taxonomy: Optional[nx.Graph],
) -> None:
    """Mutates each accepted candidate in place, adding hierarchy_resonance
    (None if unavailable) based on peer CWE proximity only."""
    if taxonomy is None:
        for cand in accepted:
            cand["hierarchy_resonance"] = None
        return

    for cand in accepted:
        own_cwe = cand.get("_cwe")
        peers = [c for c in accepted if c is not cand]
        if not own_cwe or not peers:
            cand["hierarchy_resonance"] = None
            continue
        proximities = []
        for peer in peers:
            peer_cwe = peer.get("_cwe")
            if not peer_cwe:
                continue
            prox = cwe_taxonomic_proximity(own_cwe, peer_cwe, taxonomy)
            if prox is not None:
                proximities.append(prox)
        cand["hierarchy_resonance"] = round(sum(proximities) / len(proximities), 6) if proximities else None


def compute_hnerp_and_confidence(accepted: List[Dict[str, Any]], weights: HnerpWeights) -> None:
    for cand in accepted:
        components = [(weights.w_node, cand["node_resonance"]), (weights.w_edge, cand["edge_resonance"])]
        if cand.get("hierarchy_resonance") is not None:
            components.append((weights.w_hier, cand["hierarchy_resonance"]))

        weight_sum = sum(w for w, _ in components)
        hnerp = sum(w * s for w, s in components) / weight_sum if weight_sum else 0.0
        cand["hnerp_score"] = round(hnerp, 6)

        # confidence: precision/recall-style agreement between node and edge
        # resonance, discounted by relation/phase consistency of the verified
        # alignment. Distinct from HNERP (which also folds in taxonomy peer
        # coherence) - both are reported per the Z schema.
        nr, er = cand["node_resonance"], cand["edge_resonance"]
        base_conf = (nr * er) ** 0.5 if nr > 0 and er > 0 else 0.0
        consistency_factor = (cand["relation_consistency"] + cand["phase_consistency"]) / 2.0
        cand["confidence_score"] = round(base_conf * consistency_factor, 6)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def run_alignment(args: argparse.Namespace) -> Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    phase2_output = load_phase2_output(args.phase2_json)
    records = phase2_output.get("records", [])
    if args.max_queries is not None:
        records = records[: args.max_queries]
    if not records:
        raise ValueError("No records found in Phase 2 JSON output.")

    taxonomy = load_cwe_taxonomy(args.cwe_taxonomy)
    weights = HnerpWeights(w_node=args.w_node, w_edge=args.w_edge, w_hier=args.w_hier)

    z_records: List[Dict[str, Any]] = []
    evidence_rows: List[Dict[str, Any]] = []
    rejection_rows: List[Dict[str, Any]] = []

    for record in tqdm(records, desc="EvidenceAlignment"):
        patches = record.get("patches", [])
        target_entry = next((p for p in patches if p.get("role") == "target"), None)
        if target_entry is None:
            continue

        target_graph = extract_core_subgraph(build_patch_graph(target_entry))
        target_id = f"target::{record.get('query_idx')}::{record.get('query_hash')}"

        accepted: List[Dict[str, Any]] = []
        for cand_entry in patches:
            if cand_entry.get("role") != "retrieved":
                continue
            cand_meta = cand_entry.get("meta", {})
            cand_id = "cand::{project}::{commit_id}::{idx}".format(
                project=cand_meta.get("project"), commit_id=cand_meta.get("commit_id"), idx=cand_meta.get("idx")
            )
            cand_graph = extract_core_subgraph(build_patch_graph(cand_entry))

            if target_graph.number_of_nodes() == 0 or cand_graph.number_of_nodes() == 0:
                rejection_rows.append(
                    {"target_id": target_id, "candidate_id": cand_id, "reason": "empty_core_subgraph"}
                )
                continue

            verified = verify_alignment(target_graph, cand_graph, args.min_shared_nodes)
            if verified is None:
                rejection_rows.append(
                    {"target_id": target_id, "candidate_id": cand_id, "reason": "failed_structural_or_semantic_compatibility"}
                )
                continue

            verified["candidate_graph_id"] = cand_id
            verified["rank_hint"] = cand_entry.get("rank")
            verified["_cwe"] = normalize_cwe(cand_meta.get("cwe_eval_only"))
            accepted.append(verified)

        compute_hierarchy_resonance(accepted, taxonomy)
        compute_hnerp_and_confidence(accepted, weights)
        accepted.sort(key=lambda c: c["hnerp_score"], reverse=True)
        if args.max_evidence is not None:
            accepted = accepted[: args.max_evidence]

        matched_reference_graphs = []
        for cand in accepted:
            entry = {
                "reference_graph_id": cand["candidate_graph_id"],
                "reference_cwe": cand.get("_cwe"),
                "node_mapping": cand["node_mapping"],
                "edge_mapping": cand["edge_mapping"],
                "phase_consistency": cand["phase_consistency"],
                "relation_consistency": cand["relation_consistency"],
                "structural_consistency": cand["structural_consistency"],
                "hnerp_score": cand["hnerp_score"],
                "confidence_score": cand["confidence_score"],
            }
            matched_reference_graphs.append(entry)

            evidence_rows.append(
                {
                    "target_graph_id": target_id,
                    "target_true_cwe_eval_only": normalize_cwe((target_entry.get("meta") or {}).get("cwe_eval_only")),
                    "reference_graph_id": cand["candidate_graph_id"],
                    "reference_cwe": cand.get("_cwe"),
                    "node_resonance": cand["node_resonance"],
                    "edge_resonance": cand["edge_resonance"],
                    "hierarchy_resonance": cand["hierarchy_resonance"],
                    "hnerp_score": cand["hnerp_score"],
                    "confidence_score": cand["confidence_score"],
                    "phase_consistency": cand["phase_consistency"],
                    "relation_consistency": cand["relation_consistency"],
                    "num_matched_nodes": len(cand["node_mapping"]),
                    "num_matched_edges": len(cand["edge_mapping"]),
                }
            )

        z_records.append(
            {
                "target_graph_id": target_id,
                "target_true_cwe_eval_only": normalize_cwe(
                    (target_entry.get("meta") or {}).get("cwe_eval_only")
                ),
                "matched_reference_graphs": matched_reference_graphs,
            }
        )

    run_config = {
        "phase": "Evidence Alignment",
        "phase_id": "EAA",
        "phase2_input": describe_source(args.phase2_json),
        "phase2_input_role": "Phase 2 output consumed by Phase 3",
        "cwe_taxonomy": args.cwe_taxonomy,
        "primevul_dataset_splits": PRIMEVUL_DATASET_SPLITS,
        "dataset_split_usage": "Align validation/testing query graphs against training-derived retrieved graphs; do not use target CWE labels during matching/scoring.",
        "cwe_taxonomy_loaded": taxonomy is not None,
        "min_shared_nodes": args.min_shared_nodes,
        "max_evidence_per_query": args.max_evidence,
        "hnerp_weights": weights.__dict__,
        "hnerp_definition": HNERP_DEFINITION,
        "structural_verification_method": "NetworkX VF2++ (`vf2pp_all_isomorphisms`) over exact normalized node labels, followed by strict relation+phase edge filtering",
        "graph_matching_algorithm": "VF2++",
        "evidence_scoring_method": "Hierarchical Node-Edge Resonance Proximity (HNERP)",
        "target_true_cwe_eval_only_note": "Present in each output record for Phase 4 evaluation only. "
        "It is attached after alignment/HNERP scoring is fully computed and is never read by any "
        "candidate-matching or scoring function in this script, preserving the no-leakage guarantee.",
        "num_records": len(records),
    }

    output_json = {"run_config": run_config, "records": z_records}
    evidence_df = pd.DataFrame(evidence_rows)
    rejection_df = pd.DataFrame(rejection_rows)
    config_df = pd.DataFrame([run_config])
    return output_json, evidence_df, rejection_df, config_df


def save_outputs(
    output_json: Dict[str, Any],
    evidence_df: pd.DataFrame,
    rejection_df: pd.DataFrame,
    config_df: pd.DataFrame,
    output_json_path: str,
    output_excel_path: str,
) -> None:
    ensure_parent(output_json_path)
    ensure_parent(output_excel_path)

    with open(output_json_path, "w", encoding="utf-8") as handle:
        json.dump(output_json, handle, indent=2, ensure_ascii=False)

    with pd.ExcelWriter(output_excel_path, engine="openpyxl") as writer:
        evidence_df.to_excel(writer, sheet_name="Evidence_Z", index=False)
        rejection_df.to_excel(writer, sheet_name="Rejected_Candidates", index=False)
        config_df.to_excel(writer, sheet_name="Run_Config", index=False)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CWEMap Phase 3: EvidenceAlignmentAgent - VF2++ graph alignment/filtering + HNERP scoring over Phase 2 output."
    )
    parser.add_argument("--phase2-json", required=True, help="Path to Phase 2 output JSON.")
    parser.add_argument(
        "--cwe-taxonomy",
        default=None,
        help="Optional CSV/JSONL edge list for the frozen CWE taxonomy graph GCWE "
        "(columns/keys: parent_cwe, child_cwe). Without this, hierarchy_resonance "
        "is omitted rather than invented.",
    )
    parser.add_argument("--output-json", required=True, help="Output JSON path (the evidence package Z, per query).")
    parser.add_argument("--output-excel", required=True, help="Output Excel path for inspection.")

    parser.add_argument(
        "--min-shared-nodes",
        type=int,
        default=2,
        help="Minimum exact-identifier node overlap required before attempting isomorphism verification.",
    )
    parser.add_argument(
        "--max-evidence",
        type=int,
        default=None,
        help="Optional cap on number of ranked evidence entries kept per query in Z.",
    )
    parser.add_argument("--max-queries", type=int, default=None, help="Optional limit on number of query records.")

    parser.add_argument("--w-node", type=float, default=HnerpWeights().w_node, help="HNERP weight for node_resonance.")
    parser.add_argument("--w-edge", type=float, default=HnerpWeights().w_edge, help="HNERP weight for edge_resonance.")
    parser.add_argument("--w-hier", type=float, default=HnerpWeights().w_hier, help="HNERP weight for hierarchy_resonance.")
    return parser


# ---------------------------------------------------------------------------
# Pipeline entrypoint (used by main.py to chain Phase 1 -> Phase 2 -> ... )
# ---------------------------------------------------------------------------
def run_evidence_alignment(phase2_output: Any, config: Dict[str, Any]) -> Dict[str, Any]:
    """Phase 3 entrypoint driven by a config dict instead of CLI flags.

    Args:
        phase2_output: the dict returned by
            `phase2_patch_graph_construction.run_patch_graph_construction(...)`
            (in-memory chaining), OR a path string to a saved Phase 2 JSON
            file (standalone use).
        config: the full pipeline config; only `config["phase3"]` is read.

    Returns the Phase 3 JSON output (evidence package Z per query) in-memory
    for Phase 4 to consume.
    """
    parser = build_arg_parser()
    required = {} if phase2_output is not None else {"phase2_json": "phase3.phase2_json"}
    args = namespace_from_config(
        parser,
        config.get("phase3", {}),
        required=required,
    )
    # Phase 2's output feeds Phase 3 directly - this is the Phase2->Phase3 link.
    # main.py passes the in-memory dict; standalone/config-driven partial reruns can use phase3.phase2_json.
    if phase2_output is not None:
        args.phase2_json = phase2_output

    output_json, evidence_df, rejection_df, config_df = run_alignment(args)
    save_outputs(
        output_json=output_json,
        evidence_df=evidence_df,
        rejection_df=rejection_df,
        config_df=config_df,
        output_json_path=args.output_json,
        output_excel_path=args.output_excel,
    )

    print(f"[Phase 3] Saved JSON output:  {args.output_json}")
    print(f"[Phase 3] Saved Excel output: {args.output_excel}")
    return output_json


# ---------------------------------------------------------------------------
# Standalone CLI entrypoint (optional - main.py is the primary way to run
# the full pipeline; this remains for running/debugging Phase 3 in isolation)
# ---------------------------------------------------------------------------
def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    output_json, evidence_df, rejection_df, config_df = run_alignment(args)
    save_outputs(
        output_json=output_json,
        evidence_df=evidence_df,
        rejection_df=rejection_df,
        config_df=config_df,
        output_json_path=args.output_json,
        output_excel_path=args.output_excel,
    )

    print(f"Saved JSON output:  {args.output_json}")
    print(f"Saved Excel output: {args.output_excel}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
