# 🛠️ CWEMap Replication Package 


> **CWEMap: Graph-Guided Reasoning for Fine-Grained CWE Classification of Security Patches**

![CWEMap Approach Overview](https://github.com/CWEMap/CWEMap/blob/d5f697b026bbc702f28bb528c013e9ed04648087/CWEMap.png?raw=true)

The replication package supports the experiments reported in the manuscript, including dataset preprocessing, execution scripts, cross-backbone generalizability evaluation, and efficiency analysis.

## 🧭 1. Framework Overview

CWEMap follows a four-stage graph-guided workflow for fine-grained commit-level CWE classification: it retrieves relevant historical cases, constructs phase-aware patch graphs, aligns structurally compatible evidence, and decodes a valid CWE path over the frozen CWE taxonomy graph.

### 🔎 Phase 1: Patch-Aware Vulnerability Retrieval (PVR)
Retrieves top-k patch-relevant historical cases from a leakage-free training corpus after removing label-revealing metadata.

### 🧱 Phase 2: Phase-Aware Patch Graph Construction (PGC)
Extracts security triples from the target patch and retrieved cases, then builds phase-aware graphs using `T_before`, `T_delta`, and `T_after`. The target graph is `G_input`, and retrieved reference graphs form `KG_examples`.

### 🔗 Phase 3: Agent-Based Evidence Alignment (AEA)
Aligns `G_input` with `KG_examples` through constrained subgraph matching and evidence scoring to produce the graph-aligned package `Z`.

### 🌳 Phase 4: Agent-Based Hierarchical Reasoning (AHR)
Performs top-down decoding over `G_CWE` to predict a taxonomy-consistent terminal CWE path, with confidence-guided refinement when needed.

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
```


## 📊 Datasets

We evaluate **CWEMap** on two public vulnerability benchmarks: **TREEVUL** [1] and **PRIMEVUL** [2].

The dataset and preprocessing scripts are available in [`Dataset_Preprocessing/`](./Dataset_Preprocessing).

**TREEVUL** contains real-world security patches annotated with hierarchical CWE paths, making it suitable for evaluating fine-grained commit-level vulnerability type classification.

**PRIMEVUL** is a large-scale vulnerability benchmark collected from multiple open-source ecosystems. It covers 140 Common Weakness Enumeration (CWE) classes among vulnerability-labeled training instances; in CWEMap, CWE-path evaluation is conducted only on vulnerability-labeled samples with valid CWE annotations.

The datasets are publicly available through Google Drive below.

Download Dataset: [Click here to access the dataset](https://drive.google.com/drive/folders/1ZNNrLlSb7GIvuvNFKMNDHEvGxci6WppK?usp=sharing)

## 📂 Repository Structure

```text
CWEMap/
├── README.md
│   └── Setup instructions, dataset access, and reproduction workflow
│
├── requirements.txt
│   └── Python dependencies for pip-based installation
│
├── environment.yml
│   └── Conda environment specification for evaluation
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
├── dataset/
│   ├── treevul/
│   │   ├── train_set.json
│   │   ├── validation_set.json
│   │   └── test_set.json
│   │
│   └── primevul/
│       ├── primevul_train.jsonl
│       ├── primevul_valid.jsonl
│       └── primevul_test.jsonl
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
└── outputs/
    ├── logs/
    │   └── Runtime logs for each experiment
    ├── predictions/
    │   └── Predicted CWE paths and intermediate model outputs
    ├── metrics/
    │   └── Weighted F1, Macro F1, MCC, and Path Fraction results
    └── tables/
        └── CSV/JSON output
```

## 📚 References

[1] S. Pan, L. Bao, X. Xia, D. Lo, and S. Li,  
“Fine-grained commit-level vulnerability type prediction by CWE tree structure,”  
in *Proceedings of the IEEE/ACM 45th International Conference on Software Engineering (ICSE)*,  
2023, pp. 957–969.

[2] Y. Ding, Y. Fu, O. Ibrahim, C. Sitawarin, X. Chen, B. Alomair, D. Wagner, B. Ray, and Y. Chen,  
“Vulnerability detection with code language models: How far are we?”  
in *Proceedings of the IEEE/ACM 47th International Conference on Software Engineering (ICSE)*,  
2025, pp. 1729–1741.
