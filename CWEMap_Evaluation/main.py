#!/usr/bin/env python3
"""
CWEMap - single entrypoint for the full 4-phase pipeline.

    Phase 1  Patch-Aware Vulnerability Retrieval  -->
    Phase 2  Patch Graph Construction (triple extraction)  -->
    Phase 3  Evidence Alignment (VF2++ + HNERP)  -->
    Phase 4  DeepSeek-V4-Flash Hierarchical Reasoning (CWE path prediction)
Usage:
    python main.py --config configs/default.yaml
    python main.py --config configs/primevul_validation.yaml
    python main.py --config configs/primevul_testing.yaml
    python main.py --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from phases.phase1_patch_aware_retrieval import run_patch_aware_retrieval
from phases.phase2_patch_graph_construction import run_patch_graph_construction
from phases.phase3_evidence_alignment import run_evidence_alignment
from phases.phase4_hierarchical_reasoning import run_hierarchical_reasoning
from utils.config import load_config, save_config


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the full CWEMap pipeline (Phase 1 -> Phase 2 -> Phase 3 -> Phase 4)."
    )
    parser.add_argument(
        "--config", default="configs/default.yaml", help="Path to a YAML config file (see configs/default.yaml)."
    )
    parser.add_argument("--input", default=None, help="Override phase1.input (PrimeVul query JSONL: primevul_training.jsonl, primevul_validation.jsonl, or primevul_testing.jsonl).")
    parser.add_argument("--corpus", default=None, help="Override phase1.corpus (normally primevul_training.jsonl as the training-only retrieval corpus).")
    parser.add_argument("--cwe-taxonomy", default=None, help="Override phase3/phase4 cwe_taxonomy path.")
    parser.add_argument(
        "--no-deepseek", action="store_true", help="Override phase1.no_deepseek (skip DeepSeek feature calls)."
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Override phase2.no_llm and phase4.no_llm (skip GPT-4o/DeepSeek calls for offline validation).",
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=None,
        help="Override max_queries for every phase at once (useful for a quick smoke-test run).",
    )
    return parser


def apply_cli_overrides(config: dict, args: argparse.Namespace) -> dict:
    if args.input is not None:
        config["phase1"]["input"] = args.input
    if args.corpus is not None:
        config["phase1"]["corpus"] = args.corpus
    if args.cwe_taxonomy is not None:
        config["phase3"]["cwe_taxonomy"] = args.cwe_taxonomy
        config["phase4"]["cwe_taxonomy"] = args.cwe_taxonomy
    if args.no_deepseek:
        config["phase1"]["no_deepseek"] = True
    if args.no_llm:
        config["phase2"]["no_llm"] = True
        config["phase4"]["no_llm"] = True
    if args.max_queries is not None:
        for phase in ("phase1", "phase2", "phase3", "phase4"):
            config[phase]["max_queries"] = args.max_queries
    return config


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    config = load_config(args.config)
    config = apply_cli_overrides(config, args)

    effective_config_path = str(Path(config["phase1"]["output_json"]).parent / f"{config['run_name']}.effective_config.yaml")
    save_config(config, effective_config_path)
    print(f"[main] Effective config saved to: {effective_config_path}")

    print("\n=== Phase 1: Patch-Aware Vulnerability Retrieval ===")
    retrieval = run_patch_aware_retrieval(config)

    print("\n=== Phase 2: Patch Graph Construction (Triple Extraction) ===")
    graphs = run_patch_graph_construction(retrieval, config)

    print("\n=== Phase 3: Evidence Alignment (VF2++ + HNERP) ===")
    evidence = run_evidence_alignment(graphs, config)

    print("\n=== Phase 4: DeepSeek-V4-Flash Hierarchical Reasoning (CWE Path Prediction) ===")
    predictions = run_hierarchical_reasoning(evidence, config)

    print("\nPipeline Finished.")
    metrics = predictions.get("performance_metrics", {})
    print(metrics if metrics else predictions)
    return 0


if __name__ == "__main__":
    sys.exit(main())
