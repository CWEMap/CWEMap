## CWEMap layout

```
CWEMap/
├── main.py                                # single pipeline entrypoint
├── configs/
├── phases/
│   ├── phase1_patch_aware_retrieval.py    # Phase 1
│   ├── phase2_patch_graph_construction.py # Phase 2
│   ├── phase3_evidence_alignment.py       # Phase 3
│   └── phase4_hierarchical_reasoning.py   # Phase 4
├── utils/
│   ├── config.py                          # load_config()/save_config()
│   └── namespace_adapter.py               # config-dict -> argparse.Namespace bridge
├── fixtures/                              # example dataset
└── outputs/                               # JSON/Excel/CSV 
```

## Running the pipeline

```bash
pip install numpy pandas scikit-learn networkx openpyxl tqdm openai pyyaml

export DEEPSEEK_API_KEY="..."   # Replace your API Key
export OPENAI_API_KEY="..."     # Replace your API Key
```

