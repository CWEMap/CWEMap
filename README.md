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

```text
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

---

## 📂 2. Repository Structure

```text
CWEMap/
├── README.md               # Setup and replication instructions
├── requirements.txt       # Python dependencies (pip)
├── environment.yml        # Anaconda environment specification
├── configs/               # Hyperparameter and configuration files
│   ├── default.yaml
│   ├── treevul.yaml
│   ├── primevul.yaml
│   └── llm_backbones.yaml
├── data/                  # Datasets and taxonomies
│   ├── raw/               # Raw benchmarks (TreeVul, PrimeVul)
│   │   ├── treevul/
│   │   └── primevul/
│   ├── processed/         # Tokenized, parsed, and graph-ready inputs
│   │   ├── treevul/
│   │   └── primevul/
│   └── cwe/
│       └── cwe_taxonomy.json
├── src/                   # Core implementation codebase
│   ├── retrieval/         # Code for PVR phase
│   ├── graph_construction/# Code for PGC phase
│   ├── evidence_alignment/# Code for AEA phase
│   ├── reasoning/         # Code for AHR phase
│   ├── evaluation/        # Evaluation metric calculation
│   └── utils/             # Helper utilities and loggers
├── scripts/               # Single-task execution scripts
│   ├── preprocess_data.py
│   ├── build_retrieval_index.py
│   ├── run_cwemap.py
│   ├── run_baselines.py
│   ├── run_ablation.py
│   ├── run_backbone_study.py
│   ├── run_efficiency.py
│   └── aggregate_results.py
├── outputs/               # Evaluation artifacts
│   ├── logs/              # Runtime execution logs
│   ├── predictions/       # Model output inference JSONs
│   ├── metrics/           # Calculated precision, recall, F1 scores
│   ├── tables/            # LaTeX/CSV tables generated for the paper
│   └── figures/           # Plot diagrams (PDF/PNG format)
└── replication/           # One-click reproduction workflows
    ├── run_all.sh         # Complete pipeline execution
    ├── reproduce_rq1.sh   # Main effectiveness comparison
    ├── reproduce_rq2.sh   # Ablation study variants
    ├── reproduce_rq3.sh   # Cross-backbone evaluation
    └── reproduce_rq4.sh   # Execution time & resource footprint analysis
