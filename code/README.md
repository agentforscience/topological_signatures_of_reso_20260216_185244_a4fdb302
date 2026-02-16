# Cloned Repositories

## Repo 1: Pythia
- **URL**: https://github.com/EleutherAI/pythia
- **Purpose**: Suite of 16 language models (70M-12B) with 154 intermediate training checkpoints each. Enables studying how topological features evolve throughout training.
- **Location**: `code/pythia/`
- **Key files**: Training configs, analysis scripts
- **Notes**: Models are loaded from HuggingFace Hub. The repo contains training configs and evaluation scripts. All models trained on identical data (The Pile) in identical order.

## Repo 2: Ripser.py
- **URL**: https://github.com/scikit-tda/ripser.py
- **Purpose**: Fast persistent homology computation library. Core tool for computing persistence diagrams from embedding point clouds.
- **Location**: `code/ripser/`
- **Key files**: `ripser/ripser.py` - main interface
- **Installation**: `pip install ripser`
- **Notes**: Lean Python wrapper around C++ Ripser engine. Supports sparse and dense distance matrices. Essential for computing Vietoris-Rips persistence.

## Repo 3: Giotto-TDA
- **URL**: https://github.com/giotto-ai/giotto-tda
- **Purpose**: High-performance TDA toolbox with scikit-learn API. Provides complete pipeline from point cloud to topological features.
- **Location**: `code/giotto-tda/`
- **Key files**: `gtda/` - main library package
- **Installation**: `pip install giotto-tda`
- **Notes**: Provides VietorisRipsPersistence, BettiCurve, PersistenceEntropy, and other topological feature extractors. Best for building end-to-end ML pipelines with TDA.

## Additional Libraries (Not Cloned)

### GUDHI
- **URL**: https://github.com/GUDHI/gudhi-devel
- **Installation**: `pip install gudhi`
- **Purpose**: Comprehensive TDA library with Alpha complexes, Cubical complexes, and more
- **Notes**: More complex types than Ripser but heavier dependency

### scikit-tda ecosystem
- **URL**: https://scikit-tda.org/
- **Purpose**: Collection of TDA libraries (persim, ripser, tadasets, etc.)
- **Notes**: persim provides tools for comparing persistence diagrams (bottleneck distance, Wasserstein distance)
