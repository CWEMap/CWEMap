# 🛠️ CWEMap Replication Package 


> **CWEMap: Graph-Guided Reasoning for Fine-Grained CWE Classification of Security Patches**

![CWEMap Approach Overview](https://github.com/CWEMap/CWEMap/blob/d5f697b026bbc702f28bb528c013e9ed04648087/CWEMap.png?raw=true)

The replication package supports the experiments reported in the manuscript, including dataset preprocessing, execution scripts, cross-backbone generalizability evaluation, and efficiency analysis.

## 🧭 1. Framework Overview

CWEMap follows a four-stage graph-guided workflow for fine-grained commit-level CWE classification: it retrieves relevant historical cases, constructs phase-aware patch graphs, aligns structurally compatible evidence, and decodes a valid CWE path over the frozen CWE taxonomy graph.

#####  Phase 1: Patch-Aware Vulnerability Retrieval (PVR)
#####  Phase 2: Phase-Aware Patch Graph Construction (PGC)
#####  Phase 3: Agent-Based Evidence Alignment (AEA)
#####  Phase 4: Agent-Based Hierarchical Reasoning (AHR)
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

## 📂 Datasets Structure

```text
Datasets/
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
```
## 📂 Evaluation

CWEMap follows a four-phase evaluation pipeline. Each phase takes the output of the previous phase as input and produces structured evidence for the next stage.

##### Phase 1: Patch-Aware Vulnerability Retrieval (PVR)

PVR retrieves the top-k structurally and semantically relevant historical vulnerability cases from the training-only corpus.

```bash
python scripts/run_cwemap.py \
  --phase pvr \
  --input dataset/treevul/test_set.json \
  --corpus dataset/treevul/train_set.json \
  --output outputs/pvr/treevul_retrieved_cases.json

#####  Phase 2: Phase-Aware Patch Graph Construction (PGC)
Write  how we Do ( one line)
Command to execute with --input --output path

#####  Phase 3: Agent-Based Evidence Alignment (AEA)
Write  how we Do ( one line)
Command to execute with --input --output path

#####  Phase 4: Agent-Based Hierarchical Reasoning (AHR)
Write  how we Do ( one line)
Command to execute with --input --output path


## 📚 References

[1] S. Pan, L. Bao, X. Xia, D. Lo, and S. Li,  
“Fine-grained commit-level vulnerability type prediction by CWE tree structure,”  
in *Proceedings of the IEEE/ACM 45th International Conference on Software Engineering (ICSE)*,  
2023, pp. 957–969.

[2] Y. Ding, Y. Fu, O. Ibrahim, C. Sitawarin, X. Chen, B. Alomair, D. Wagner, B. Ray, and Y. Chen,  
“Vulnerability detection with code language models: How far are we?”  
in *Proceedings of the IEEE/ACM 47th International Conference on Software Engineering (ICSE)*,  
2025, pp. 1729–1741.
