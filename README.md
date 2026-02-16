# Topological Signatures of Resource-Constrained Language Model Training

This project investigates whether persistent homology can detect meaningful topological structures in language model embedding spaces during training, and whether these structures differ across model sizes (a proxy for computational resource constraints).

## Key Findings

- **Topological features strongly predict model quality**: H1 persistence entropy correlates with perplexity at |r| > 0.84 (p < 10^-5), outperforming spectral baselines (effective rank, participation ratio) by ~1.5x
- **Training follows a three-phase topological trajectory**: collapse (steps 0-32), reorganization (32-1000), and refinement (1000+), paralleling spectral phases from Li et al. (2025)
- **Larger models develop distinct topology**: Pythia-410M converges to fewer, more concentrated topological features (73 H1 features, entropy 3.88) compared to Pythia-70M (81 features, entropy 4.01)
- **H1 (1-cycles/loops) carries the signal**: H0 (connected components) is constant, while H1 features systematically evolve and correlate with performance
- **Spectral baselines fail for the largest model**: Effective rank is not significant (p=0.10) for Pythia-410M, while topological features remain highly significant

## Experimental Setup

- **Models**: Pythia-70M, 160M, 410M (EleutherAI) — 19 training checkpoints each
- **Data**: WikiText-2 validation set (200 texts)
- **Pipeline**: Embedding extraction → PCA (50D) → Vietoris-Rips persistent homology (Ripser) → Feature extraction
- **Baselines**: Effective rank (RankMe), participation ratio, eigenspectrum decay
- **Total runtime**: 19.6 minutes on NVIDIA RTX 3090

## Repository Structure

```
.
├── REPORT.md               # Full research report with results
├── README.md               # This file
├── planning.md             # Research plan and motivation
├── literature_review.md    # Literature synthesis
├── resources.md            # Resource catalog
├── src/
│   ├── extract_embeddings.py   # Core pipeline: embeddings, PH, spectral features
│   ├── run_experiments.py      # Main experiment runner
│   └── analyze_results.py      # Analysis and visualization
├── results/
│   ├── data/               # Raw results (JSON)
│   │   ├── all_results.json
│   │   ├── pythia-{70m,160m,410m}_results.json
│   │   ├── statistics.json
│   │   └── config.json
│   └── plots/              # Visualizations (PNG)
│       ├── summary_figure.png
│       ├── topological_evolution.png
│       ├── topology_vs_perplexity.png
│       ├── topo_vs_spectral_correlations.png
│       └── ...
├── datasets/               # Downloaded datasets (WikiText-2, TinyStories, GLUE MRPC)
├── papers/                 # Downloaded research papers
└── code/                   # Cloned repositories (Pythia, Ripser, Giotto-TDA)
```

## Reproducing Results

```bash
# 1. Create and activate virtual environment
uv venv && source .venv/bin/activate

# 2. Install dependencies
uv pip install torch numpy scipy matplotlib scikit-learn transformers datasets ripser persim tqdm

# 3. Run experiments (requires GPU, ~20 min)
export USER="researcher"  # needed for some environments
python src/run_experiments.py

# 4. Generate analysis and visualizations
python src/analyze_results.py
```

## Full Report

See [REPORT.md](REPORT.md) for the complete research report including methodology, all results, statistical analysis, and discussion.
