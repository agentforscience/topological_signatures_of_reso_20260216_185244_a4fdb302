# Research Plan: Topological Signatures of Resource-Constrained Language Model Training

## Motivation & Novelty Assessment

### Why This Research Matters
Language models trained under computational constraints must learn to compress linguistic information into efficient representations. Understanding the geometric and topological structure of these representations can reveal *how* models organize information differently under budget pressure, informing model selection, training strategy optimization, and our fundamental understanding of representation learning. Persistent homology provides a richer descriptor of shape than spectral methods alone, capturing holes, voids, and connectivity patterns.

### Gap in Existing Work
Based on the literature review:
- Li et al. (2025) characterized three geometric phases during LM pretraining using **spectral** analysis (effective rank, eigenspectrum decay), but did not examine topological features.
- Shahidullah (2022) applied persistent homology to simple feedforward NNs, but never to language model embeddings across training.
- Aghajanyan et al. (2020) showed intrinsic dimension correlates with generalization, but intrinsic dimension is a scalar—persistent homology captures richer multi-scale structure.
- **No existing work applies persistent homology to track how topological features of LM embedding spaces evolve during training, or how resource constraints affect this evolution.**

### Our Novel Contribution
We provide the first systematic study of persistent homology applied to language model embedding spaces across training, comparing multiple model sizes (as a proxy for resource constraint). We test whether:
1. Topological features (Betti numbers, persistence) change systematically during training
2. Different model sizes exhibit distinct topological signatures
3. Topological features correlate with (and potentially predict) model performance
4. Topological analysis captures information beyond spectral baselines

### Experiment Justification
- **Experiment 1 (Topological evolution during training)**: Needed to establish that persistent homology detects meaningful changes during LM training—the foundational claim.
- **Experiment 2 (Cross-model comparison)**: Needed to test whether resource constraints (model size) create distinct topological signatures.
- **Experiment 3 (Topology-performance correlation)**: Needed to test whether topological features have predictive value for model quality.
- **Experiment 4 (Comparison with spectral baselines)**: Needed to demonstrate that persistent homology captures information beyond existing methods.

## Research Question
Do resource-constrained language models develop distinct topological structures in their embedding spaces, detectable via persistent homology, that correlate with model performance and differ systematically across computational budgets?

## Hypothesis Decomposition

**H1**: Persistent homology features (Betti numbers, total persistence) of LM embedding spaces change systematically during training, reflecting representation reorganization.

**H2**: Models of different sizes (70M, 160M, 410M parameters) exhibit quantitatively different topological signatures at equivalent training stages.

**H3**: Topological features correlate with model performance (perplexity), and this correlation is at least as strong as spectral baselines (effective rank).

**H4**: Topological evolution during training exhibits phase-like transitions analogous to the spectral phases identified by Li et al. (2025).

## Proposed Methodology

### Approach
Use Pythia model family (70M, 160M, 410M) with their 154 publicly available training checkpoints. Extract embeddings at selected checkpoints, compute persistent homology, and analyze topological features across training and model sizes.

### Experimental Steps

1. **Embedding extraction**: For each model size and selected checkpoint, run a fixed batch of WikiText-2 text through the model. Extract last-layer hidden states. Subsample to N=1000 points for computational feasibility.

2. **Dimensionality reduction**: Apply PCA to reduce embedding dimension to d=50 (following Shahidullah 2022 recommendation; balances preserving structure with computational cost of PH).

3. **Persistent homology computation**: Compute Vietoris-Rips persistence diagrams (H0, H1) using Ripser. Extract:
   - Betti numbers (b0, b1) at multiple filtration scales
   - Total persistence (sum of lifetimes)
   - Persistence entropy
   - Number of significant features (persistence > threshold)

4. **Spectral baselines**: Compute effective rank (RankMe) and participation ratio on the covariance matrix of embeddings at each checkpoint.

5. **Performance measurement**: Compute perplexity on WikiText-2 validation set at each checkpoint.

6. **Correlation analysis**: Spearman correlation between topological/spectral features and perplexity across checkpoints.

7. **Cross-model comparison**: Compare topological trajectories across the three model sizes.

### Checkpoints Selected
To balance thoroughness with compute time (~12-15 checkpoints per model):
- Steps: 0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000, 2000, 5000, 10000, 33000, 66000, 100000, 143000

This logarithmic spacing captures early rapid changes and later refinement.

### Baselines
1. **Effective rank (RankMe)**: Spectral baseline from Li et al. (2025)
2. **Participation ratio**: Alternative spectral metric
3. **Raw perplexity trajectory**: Null baseline for correlation

### Evaluation Metrics
- Betti numbers b0, b1 across filtration values
- Total persistence: Σ(death - birth) for all features
- Persistence entropy: -Σ p_i log(p_i) where p_i = persistence_i / total_persistence
- Wasserstein distance between consecutive persistence diagrams (rate of topological change)
- Spearman correlation between topological features and perplexity
- Effective rank and participation ratio (spectral baselines)

### Statistical Analysis Plan
- Spearman rank correlations (non-parametric, appropriate for potentially non-linear relationships)
- Bootstrap confidence intervals for correlations (1000 resamples)
- Compare correlation strengths between topological and spectral features using Fisher z-transformation
- Significance level: α = 0.05

## Expected Outcomes
- **Support for H1**: Betti numbers and persistence should change monotonically or with identifiable phases during training
- **Support for H2**: Smaller models should show different topological complexity (potentially simpler topology) than larger models
- **Support for H3**: Topological features should correlate with perplexity (r > 0.5)
- **Support for H4**: Rate of topological change (Wasserstein distance) should show peaks at phase boundaries

**Results that would refute**: If topological features are noisy/random across checkpoints, show no correlation with performance, or are indistinguishable across model sizes.

## Timeline and Milestones
- Planning + Setup: 10 min
- Embedding extraction + PH computation: 30-40 min (GPU accelerated)
- Analysis + Visualization: 15-20 min
- Documentation: 15 min

## Potential Challenges
1. **PH computation on 1000 points**: Ripser should handle this in seconds per diagram. If slow, reduce to 500 points.
2. **Downloading Pythia checkpoints**: Each checkpoint requires downloading model weights. Start with 70M (smallest). If slow, reduce checkpoint count.
3. **Noise in PH features**: Subsample multiple times and average to reduce variance.
4. **Memory**: Pythia-410M needs ~1.6GB VRAM per checkpoint. RTX 3090 (24GB) is ample.

## Success Criteria
1. Clear visualization showing topological features evolve during training (not random noise)
2. Statistically significant correlation between at least one topological feature and perplexity
3. Visible difference in topological signatures across model sizes
4. Complete REPORT.md with actual quantitative results
