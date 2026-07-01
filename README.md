# 🛠️ CWEMap: Graph-Guided Reasoning for Fine-Grained CWE Classification of Security Patches

This repository contains the replication package for **CWEMap**:

> **CWEMap: Graph-Guided Reasoning for Fine-Grained CWE Classification of Security Patches**

![CWEMap Approach Overview](https://github.com/CWEMap/CWEMap/blob/d5f697b026bbc702f28bb528c013e9ed04648087/CWEMap.png?raw=true)

## Overview of CWEMap

CWEMap is a graph-guided hierarchical reasoning framework for fine-grained commit-level Common Weakness Enumeration (CWE) classification of security patches. Given a vulnerability-fixing commit, CWEMap predicts a taxonomy-consistent CWE path by integrating three complementary sources of evidence: patch-evolution semantics, structurally similar historical vulnerability cases, and the CWE taxonomy. Rather than directly classifying raw patch text, CWEMap first retrieves patch-relevant historical cases, then converts the target patch and retrieved cases into phase-aware patch graphs that explicitly model the vulnerable state, repair transformation, and mitigated state of the code change.

The framework then performs graph-based evidence alignment to verify whether retrieved historical cases are structurally compatible with the target patch. Finally, it conducts constrained top-down reasoning over the CWE taxonomy graph to produce a valid terminal CWE path. This design makes CWEMap especially suitable for sparse, noisy, and long-tailed security-patch classification settings, where fine-grained CWE types may be semantically close and difficult to distinguish using surface-level patch tokens alone.

```text
[Target Vulnerability-Fixing Patch]
        │
        ▼
1. Patch-Aware Vulnerability Retrieval
   └── Retrieves top-k patch-relevant historical vulnerability cases
        from a leakage-free training corpus
        │
        ▼
2. Phase-Aware Patch Graph Construction
   └── Extracts security triples and materializes phase-aware graphs:
        T_before  : vulnerable pre-patch state
        T_delta   : repair transformation
        T_after   : mitigated post-patch state
        │
        ▼
3. Agent-Based Evidence Alignment
   └── Verifies structurally compatible historical cases using
        VF2++-based subgraph matching and resonance evidence scoring
        │
        ▼
4. Agent-Based Hierarchical Reasoning
   └── Performs constrained top-down decoding over the frozen CWE
        taxonomy graph G_CWE, with confidence-guided refinement
        │
        ▼
[Predicted Terminal CWE Path]

The replication package supports the experiments reported in the ICSE manuscript, including effectiveness comparison, ablation analysis, cross-backbone generalizability, and efficiency analysis.

---

## 🧭 1. Framework Overview

Given a vulnerability-fixing commit, CWEMap operates across four core stages:

### 🔍 Phase 1: Patch-Aware Vulnerability Retrieval (PVR)
Retrieves patch-relevant historical vulnerability cases from the training corpus while strictly preventing train-test data leakage.

### 🌿 Phase 2: Phase-Aware Patch Graph Construction (PGC)
Converts each raw code patch into phase-aware security triples:
* `T_before`: Vulnerable pre-patch state
* `T_delta`: Repair transformation
* `T_after`: Mitigated post-patch state

### 🤝 Phase 3: Agent-Based Evidence Alignment (AEA)
Aligns target patch graphs with retrieved reference graphs using constrained subgraph matching and multi-dimensional evidence scoring.

### 🧠 Phase 4: Agent-Based Hierarchical Reasoning (AHR)
Performs taxonomy-constrained CWE path prediction and confidence-guided refinement over the hierarchical CWE structure.

> **Key Innovation:** Unlike traditional approaches that rely solely on lexical patch similarity or unconstrained direct LLM prompting, CWEMap grounds its CWE predictions in explicit, graph-mapped patch semantics and valid structural taxonomy paths.

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
