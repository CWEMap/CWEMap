# 🛠️ CWEMap: Graph-Guided Reasoning for Fine-Grained CWE Classification of Security Patches

This repository contains the replication package for **CWEMap**:

> **CWEMap: Graph-Guided Reasoning for Fine-Grained CWE Classification of Security Patches**

![CWEMap Approach Overview](https://github.com/CWEMap/CWEMap/blob/d5f697b026bbc702f28bb528c013e9ed04648087/CWEMap.png?raw=true)

The replication package supports the experiments reported in the manuscript, including effectiveness comparison, ablation analysis, cross-backbone generalizability, and efficiency analysis.

## 🧭 1. Framework Overview

CWEMap is a graph-guided hierarchical reasoning framework for fine-grained commit-level Common Weakness Enumeration (CWE) classification of security patches. Given a vulnerability-fixing commit, CWEMap predicts a valid CWE path by integrating three complementary sources of evidence: patch-evolution semantics, structurally similar historical vulnerability cases, and the CWE taxonomy. The framework is organized into four core stages: Patch-Aware Vulnerability Retrieval, Phase-Aware Patch Graph Construction, Agent-Based Evidence Alignment, and Agent-Based Hierarchical Reasoning.

Rather than directly classifying raw patch text, CWEMap first retrieves patch-relevant historical cases, then converts both the target patch and retrieved cases into phase-aware patch graphs. These graphs explicitly model the vulnerable state, repair transformation, and mitigated state of a code change. CWEMap then verifies structurally compatible historical evidence through graph alignment and performs constrained top-down reasoning over the frozen CWE taxonomy graph to produce a taxonomy-consistent terminal CWE path.

### 🔎 Phase 1: Patch-Aware Vulnerability Retrieval (PVR)

CWEMap first retrieves the top-k patch-relevant historical vulnerability cases from a leakage-free training corpus. Each candidate case is represented using multiple evidence channels, including vulnerable code fragments, patched code fragments, diff hunks, localized method boundaries, and sanitized metadata. To prevent shortcut learning and train-test leakage, validation and test commits are excluded from the retrieval corpus, and label-revealing information such as CVE identifiers, issue links, and explicit CWE mentions is removed before retrieval.

### 🧱 Phase 2: Phase-Aware Patch Graph Construction (PGC)

CWEMap converts the target patch and retrieved cases into phase-aware security triples that represent the security-relevant transition introduced by the patch. Each patch is modeled using three temporal evidence phases:

- `T_before`: Vulnerable pre-patch state
- `T_delta`: Repair transformation
- `T_after`: Mitigated post-patch state

These triples are materialized into directed labeled graphs. The target patch becomes `G_input`, while the retrieved historical cases form the reference graph set `KG_examples`. Together with the frozen CWE taxonomy graph `G_CWE`, these graphs form the structured evidence workspace used by downstream reasoning.

### 🔗 Phase 3: Agent-Based Evidence Alignment (AEA)

CWEMap aligns the target patch graph with retrieved reference graphs to verify whether the retrieved historical cases are structurally compatible with the target security patch. This stage applies constrained subgraph matching, including topology, edge direction, relation compatibility, and phase consistency. It then ranks aligned subgraphs using resonance evidence scoring and produces a graph-aligned evidence package `Z` for hierarchical CWE reasoning.

### 🌳 Phase 4: Agent-Based Hierarchical Reasoning (AHR)

CWEMap performs constrained top-down decoding over the CWE taxonomy graph `G_CWE`. Rather than predicting an isolated flat CWE label, it expands only valid child nodes at each hierarchy level and scores candidate CWE paths using graph-aligned evidence, evidence coverage, mapping consistency, and taxonomy validity. If the confidence score is insufficient, CWEMap applies confidence-guided refinement before committing the final prediction.

> 💡 **Key Innovation:**  
> CWEMap’s key innovation is its graph-guided, taxonomy-aware reasoning mechanism that converts security patches into phase-aware evidence graphs, aligns them with structurally similar historical vulnerability cases, and predicts valid fine-grained CWE paths through constrained hierarchical decoding.


🎯 [Target Vulnerability-Fixing Commit]
        │
        ▼
🔎 1. Patch-Aware Vulnerability Retrieval (PVR)
   └── Retrieves top-k patch-relevant historical vulnerability cases
       from a leakage-free training corpus
        │
        ▼
🧱 2. Phase-Aware Patch Graph Construction (PGC)
   └── Extracts security triples and materializes phase-aware graphs
       using T_before, T_delta, and T_after
        │
        ▼
🔗 3. Agent-Based Evidence Alignment (AEA)
   └── Verifies structurally compatible historical cases through
       constrained subgraph matching and resonance evidence scoring
        │
        ▼
🌳 4. Agent-Based Hierarchical Reasoning (AHR)
   └── Performs top-down CWE path prediction over the frozen
       CWE taxonomy graph G_CWE with confidence-guided refinement
        │
        ▼
🏁 [Predicted Terminal CWE Path]


## 📊 Datasets

We evaluate CWEMap on two public vulnerability benchmarks: **TREEVUL** [1] and **PRIMEVUL** [2].

**TREEVUL** contains real-world security patches annotated with hierarchical CWE paths, making it suitable for evaluating fine-grained commit-level vulnerability type classification under the CWE tree structure.

**PRIMEVUL** is a large-scale vulnerability benchmark collected from multiple open-source ecosystems. It includes both vulnerable and benign code instances; in CWEMap, CWE-path evaluation is conducted on vulnerability-labeled instances with valid CWE annotations.


## 📂 2. Repository Structure

CWEMap/
├── README.md
│   └── Setup instructions, artifact description, and reproduction workflow
│
├── requirements.txt
│   └── Python dependencies for pip-based installation
│
├── environment.yml
│   └── Conda environment specification for artifact evaluation
│
├── configs/
│   ├── default.yaml
│   │   └── Default configuration for CWEMap experiments
│   ├── treevul.yaml
│   │   └── Dataset-specific configuration for TREEVUL
│   ├── primevul.yaml
│   │   └── Dataset-specific configuration for PRIMEVUL
│   └── llm_backbones.yaml
│       └── Configuration for cross-backbone LLM experiments
│
├── data/
│   ├── raw/
│   │   ├── treevul/
│   │   │   └── Raw TREEVUL benchmark files
│   │   └── primevul/
│   │       └── Raw PRIMEVUL benchmark files
│   │
│   ├── processed/
│   │   ├── treevul/
│   │   │   └── Preprocessed TREEVUL splits and graph-ready inputs
│   │   └── primevul/
│   │       └── Preprocessed PRIMEVUL splits and graph-ready inputs
│   │
│   └── cwe/
│       └── cwe_taxonomy.json
│           └── Frozen CWE taxonomy graph used for hierarchical decoding
│
├── src/
│   ├── retrieval/
│   │   └── Patch-Aware Vulnerability Retrieval implementation
│   ├── graph_construction/
│   │   └── Phase-aware triple extraction and patch graph materialization
│   ├── evidence_alignment/
│   │   └── Subgraph matching, structural compatibility checking, and evidence scoring
│   ├── reasoning/
│   │   └── Taxonomy-constrained CWE path decoding and confidence-guided refinement
│   ├── evaluation/
│   │   └── Metric computation, significance testing, and result aggregation utilities
│   └── utils/
│       └── Shared utilities for logging, configuration, caching, and data handling
│
├── scripts/
│   ├── preprocess_data.py
│   │   └── Preprocess raw datasets into train/validation/test splits
│   ├── build_retrieval_index.py
│   │   └── Build the training-only historical retrieval index
│   ├── run_cwemap.py
│   │   └── Execute the full CWEMap pipeline
│   ├── run_baselines.py
│   │   └── Run baseline methods used in the manuscript
│   ├── run_ablation.py
│   │   └── Run leave-one-component-out ablation experiments
│   ├── run_backbone_study.py
│   │   └── Evaluate CWEMap across different LLM backbones
│   ├── run_efficiency.py
│   │   └── Measure runtime, token usage, and inference cost
│   └── aggregate_results.py
│       └── Generate manuscript-ready tables and metrics
│
├── outputs/
│   ├── logs/
│   │   └── Runtime logs for each experiment
│   ├── predictions/
│   │   └── Predicted CWE paths and intermediate model outputs
│   ├── metrics/
│   │   └── Weighted F1, Macro F1, MCC, Path Fraction, and confidence intervals
│   ├── tables/
│   │   └── CSV/LaTeX tables corresponding to manuscript results
│   └── figures/
│       └── Generated figures and pipeline diagrams
│
└── replication/
    ├── run_all.sh
    │   └── End-to-end reproduction script for the full artifact
    ├── reproduce_rq1.sh
    │   └── Reproduce main effectiveness comparison
    ├── reproduce_rq2.sh
    │   └── Reproduce component ablation study
    ├── reproduce_rq3.sh
    │   └── Reproduce cross-backbone generalizability study
    └── reproduce_rq4.sh
        └── Reproduce computational efficiency analysis




### References

[1] S. Pan, L. Bao, X. Xia, D. Lo, and S. Li, “Fine-grained commit-level vulnerability type prediction by CWE tree structure,” in *Proceedings of the IEEE/ACM 45th International Conference on Software Engineering (ICSE)*, 2023, pp. 957–969.

[2] Y. Ding, Y. Fu, O. Ibrahim, C. Sitawarin, X. Chen, B. Alomair, D. Wagner, B. Ray, and Y. Chen, “Vulnerability detection with code language models: How far are we?” in *Proceedings of the IEEE/ACM 47th International Conference on Software Engineering (ICSE)*, 2025, pp. 1729–1741.
