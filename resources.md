# Resources Catalog

## Summary

This document catalogs all resources gathered for the research project "Topological Signatures of Resource-Constrained Language Model Training." The project investigates whether persistent homology can detect distinct topological structures in LM embedding spaces that correlate with computational budget and model performance.

## Papers

Total papers downloaded: 25

| # | Title | Authors | Year | File | Category |
|---|-------|---------|------|------|----------|
| 1 | TDA of Neural Network Layer Representations | Shahidullah | 2022 | `papers/2208.06438_tda_nn_layer_representations.pdf` | TDA+NN |
| 2 | Can Neural Networks Learn Persistent Homology? | Montúfar et al. | 2020 | `papers/2011.14688_nn_learn_persistent_homology.pdf` | TDA+NN |
| 3 | TDA Beyond Persistent Homology (Review) | Various | 2025 | `papers/2507.19504_tda_beyond_persistent_homology_review.pdf` | TDA+NN |
| 4 | Expressivity of Persistent Homology in Graph Learning | Various | 2023 | `papers/2302.09826_expressivity_persistent_homology.pdf` | TDA+NN |
| 5 | Regularization of PH Gradient Computation | Various | 2020 | `papers/2011.05804_persistent_homology_gradient.pdf` | TDA+NN |
| 6 | Intrinsic Persistent Homology via Density-Based Metric | Various | 2020 | `papers/2012.07621_intrinsic_persistent_homology.pdf` | TDA+NN |
| 7 | Riemannian Geometry of NN Representations | Various | 2023 | `papers/2309.00254_riemannian_geometry_nn_representations.pdf` | Geometry |
| 8 | Intrinsic Dimensions of Language Fractal Structures | Various | 2023 | `papers/2311.10217_intrinsic_dimensions_language_fractal.pdf` | Geometry |
| 9 | Intrinsic Dimensionality Explains LM Fine-Tuning | Aghajanyan et al. | 2020 | `papers/2012.13255_intrinsic_dimensionality_finetuning.pdf` | Geometry |
| 10 | Representation Geometry: Pretraining to Post-training | Li et al. | 2025 | `papers/2509.23024_representation_geometry_lm.pdf` | Geometry |
| 11 | Linear Representation Hypothesis & LLM Geometry | Various | 2023 | `papers/2311.03658_linear_representation_hypothesis.pdf` | Geometry |
| 12 | Geometry of Truth in LLM Representations | Various | 2023 | `papers/2310.06824_geometry_truth_llm.pdf` | Geometry |
| 13 | Truthfulness via Local Intrinsic Dimension | Various | 2024 | `papers/2402.18048_truthfulness_intrinsic_dimension.pdf` | Geometry |
| 14 | Memorization via Intrinsic Dimension | Various | 2025 | `papers/2506.09591_memorization_intrinsic_dimension.pdf` | Geometry |
| 15 | Local Intrinsic Dimensions of Contextual LMs | Various | 2025 | `papers/2506.01034_local_intrinsic_dimensions_lm.pdf` | Geometry |
| 16 | Learning Paradigms in LLMs via Intrinsic Dimension | Various | 2024 | `papers/2412.06245_learning_paradigms_intrinsic_dimension.pdf` | Geometry |
| 17 | Vocabulary Embeddings & Linguistic Structure | Various | 2025 | `papers/2505.00773_vocab_embeddings_linguistic_structure.pdf` | Geometry |
| 18 | Training Compute-Optimal LLMs (Chinchilla) | Hoffmann et al. | 2022 | `papers/2203.15556_chinchilla_compute_optimal.pdf` | Scaling |
| 19 | Beyond Chinchilla-Optimal | Various | 2023 | `papers/2401.00448_beyond_chinchilla_optimal.pdf` | Scaling |
| 20 | Robustness of Chinchilla Scaling | Various | 2025 | `papers/2509.23963_robustness_chinchilla_scaling.pdf` | Scaling |
| 21 | Reconciling Kaplan and Chinchilla | Various | 2024 | `papers/2406.12907_reconciling_scaling_laws.pdf` | Scaling |
| 22 | Scaling Law with LR Annealing | Various | 2024 | `papers/2408.11029_scaling_law_lr_annealing.pdf` | Scaling |
| 23 | Scaling Laws for Downstream Performance | Various | 2024 | `papers/2402.04177_scaling_laws_downstream.pdf` | Scaling |
| 24 | Neural Scaling Laws from Data Distribution | Various | 2024 | `papers/2404.10102_scaling_laws_data_distribution.pdf` | Scaling |
| 25 | Visualizing the Loss Landscape of Neural Nets | Li et al. | 2017 | `papers/1712.09913_loss_landscape_visualization.pdf` | Supporting |

See `papers/README.md` for detailed descriptions.

## Datasets

Total datasets downloaded: 3 (+ model checkpoints accessed on-demand)

| Name | Source | Size | Task | Location | Notes |
|------|--------|------|------|----------|-------|
| WikiText-2 | HuggingFace `wikitext` | 36K/3.7K/4.3K examples | Perplexity eval | `datasets/wikitext-2/` | Standard LM benchmark |
| TinyStories | HuggingFace `roneneldan/TinyStories` | ~2.1M stories | LM training | `datasets/tinystories/` | Samples only; full dataset via streaming |
| GLUE MRPC | HuggingFace `nyu-mll/glue` | 3.7K/408/1.7K examples | Downstream eval | `datasets/glue-mrpc/` | Paraphrase detection |

### Model Checkpoints (On-Demand via HuggingFace)

| Model | Source | Checkpoints | Use |
|-------|--------|-------------|-----|
| Pythia-70M | `EleutherAI/pythia-70m` | 154 | Smallest model for rapid iteration |
| Pythia-160M | `EleutherAI/pythia-160m` | 154 | Small model for cross-scale comparison |
| Pythia-410M | `EleutherAI/pythia-410m` | 154 | Medium model for cross-scale comparison |
| OLMo-1B | `allenai/OLMo-1B-hf` | Multiple | Larger model validation |

See `datasets/README.md` for download instructions.

## Code Repositories

Total repositories cloned: 3

| Name | URL | Purpose | Location | Notes |
|------|-----|---------|----------|-------|
| Pythia | github.com/EleutherAI/pythia | LM checkpoints & analysis | `code/pythia/` | 154 intermediate checkpoints per model |
| Ripser.py | github.com/scikit-tda/ripser.py | Fast persistent homology | `code/ripser/` | Core TDA computation |
| Giotto-TDA | github.com/giotto-ai/giotto-tda | TDA ML toolkit | `code/giotto-tda/` | Scikit-learn compatible TDA pipelines |

### Additional Libraries (install via pip)

| Library | Install | Purpose |
|---------|---------|---------|
| `ripser` | `pip install ripser` | Vietoris-Rips persistence |
| `giotto-tda` | `pip install giotto-tda` | End-to-end TDA pipelines |
| `gudhi` | `pip install gudhi` | Comprehensive TDA library |
| `persim` | `pip install persim` | Persistence diagram comparison |
| `transformers` | `pip install transformers` | Load Pythia/OLMo models |
| `datasets` | `pip install datasets` | Load HuggingFace datasets |
| `torch` | `pip install torch` | PyTorch for model inference |

See `code/README.md` for detailed descriptions.

## Resource Gathering Notes

### Search Strategy
1. **Paper search**: Used arXiv API with 8+ targeted queries covering TDA+NN, representation geometry, intrinsic dimension, and scaling laws. Also searched Semantic Scholar.
2. **Dataset search**: Checked HuggingFace for datasets used in key papers (Aghajanyan, Li et al.) and models with intermediate checkpoints (Pythia, OLMo).
3. **Code search**: Identified TDA libraries (Ripser, Giotto-TDA, GUDHI) and model suites (Pythia, OLMo) with full checkpoint history.

### Selection Criteria
- **Papers**: Prioritized work at the intersection of TDA/persistent homology and neural networks; representation geometry of LMs during training; and scaling laws that define "resource-constrained."
- **Datasets**: Selected small-to-medium datasets feasible for experiments without massive compute, plus established benchmarks from the literature.
- **Code**: Selected well-maintained, actively developed libraries with good documentation.

### Challenges Encountered
- Paper-finder service was unavailable, requiring manual arXiv API and Semantic Scholar searches.
- Semantic Scholar API was rate-limited (429 errors), limiting results from that source.
- Most TDA+LM work is very recent (2023-2025), so the field is nascent with limited baselines.

### Gaps and Workarounds
- **No existing codebase directly computes persistent homology on LM embeddings during training**: Will need to build this pipeline from components (transformers + ripser/giotto-tda).
- **TDA computation on high-dimensional embeddings is expensive**: Literature suggests subsampling + PCA before computing persistence (Shahidullah 2022).
- **Limited prior work on "topological signatures" specifically**: This is genuinely novel territory. The closest work is spectral analysis by Li et al. (2025).

## Recommendations for Experiment Design

### 1. Primary Dataset(s)
- **Pythia checkpoints** (70M, 160M, 410M): Use these as the primary data source. Extract embeddings from multiple intermediate checkpoints and compute persistent homology features.
- **WikiText-2**: Use for evaluating perplexity at each checkpoint.
- **TinyStories**: Use for training small models from scratch under controlled compute budgets.

### 2. Baseline Methods
- **Spectral baselines** (RankMe, α_ReQ): Reproduce Li et al. (2025) spectral analysis as comparison
- **Intrinsic dimension** (d90): Reproduce Aghajanyan et al. (2020) as geometric baseline
- **Loss/perplexity curves**: Standard training dynamics baseline

### 3. Evaluation Metrics
- Betti numbers (b0, b1) at each training checkpoint
- Total persistence and persistence entropy
- Bottleneck/Wasserstein distance between consecutive checkpoints
- Correlation between topological features and downstream performance
- Phase detection: identify topological analogues of the three spectral phases

### 4. Code to Adapt/Reuse
- **Ripser.py** for persistent homology computation
- **Giotto-TDA** for Betti curve extraction and persistence diagram vectorization
- **HuggingFace transformers** for loading Pythia/OLMo checkpoints
- **Custom pipeline** needed to: extract embeddings → subsample → PCA → compute PH → extract features → correlate with metrics

### 5. Proposed Experiment Workflow
1. Load Pythia-70M at multiple training checkpoints (e.g., steps 0, 1000, 5000, 10000, 50000, 100000, 143000)
2. For each checkpoint: run a batch of text through the model, extract last-layer embeddings
3. Subsample embeddings (N=1000-5000 points), reduce dimension with PCA (to d=50-100)
4. Compute Vietoris-Rips persistent homology using Ripser
5. Extract Betti curves, persistence diagrams, total persistence
6. Also compute RankMe and α_ReQ as spectral baselines
7. Measure perplexity on WikiText-2 and accuracy on MRPC
8. Repeat for Pythia-160M and Pythia-410M
9. Compare topological trajectories across model sizes
10. Analyze whether under-trained (early stopped) models have distinct topological signatures vs. fully trained models
