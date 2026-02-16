# Literature Review: Topological Signatures of Resource-Constrained Language Model Training

## Research Area Overview

This research sits at the intersection of three fields: (1) topological data analysis (TDA) applied to neural network representations, (2) the geometry of learned representations in language models, and (3) scaling laws and resource-constrained training. The hypothesis is that cost-constrained language models develop distinct topological structures in their embedding spaces that can be characterized using persistent homology, with computational budget limitations creating measurable geometric signatures that correlate with model performance and generalization.

## Key Papers

### Pillar 1: TDA Applied to Neural Network Representations

#### Shahidullah (2022) - "Topological Data Analysis of Neural Network Layer Representations"
- **Source**: arXiv:2208.06438, Caltech
- **Key Contribution**: First direct study of how topological features (Betti numbers) are preserved in NN layer representations using persistent homology
- **Methodology**: Trained a feedforward NN on binary classification of a modified torus with Klein bottle-like twist. Computed persistence diagrams on layer representations using Ripser, after HDBSCAN clustering and PCA projection.
- **Key Results**:
  - Early layers approximate homeomorphisms (preserve input topology)
  - Deeper layers significantly change topology as representations consolidate
  - Tanh (bijective) activation preserves topology longer than ReLU (surjective)
  - Noise in representations hampers persistent homology computation
- **Datasets**: Synthetic modified torus (9,800 points in 4D)
- **Tools**: TensorFlow, Ripser, HDBSCAN, PCA
- **Relevance**: Directly demonstrates that persistent homology can track how NNs transform topological structure—foundation for our approach applied to LMs

#### Montúfar, Otter & Wang (2020) - "Can Neural Networks Learn Persistent Homology Features?"
- **Source**: arXiv:2011.14688
- **Key Contribution**: Shows NNs can approximate persistent homology computations
- **Relevance**: Validates that topological features are learnable, supporting the idea that LMs implicitly learn topological structure

#### Review: "TDA and Topological Deep Learning Beyond Persistent Homology" (2025)
- **Source**: arXiv:2507.19504
- **Key Contribution**: Comprehensive review of TDA methods beyond standard persistent homology
- **Relevance**: Survey of available tools and methods for our experiments

### Pillar 2: Geometry and Intrinsic Dimension of LM Representations

#### Aghajanyan, Zettlemoyer & Gupta (2020) - "Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning"
- **Source**: arXiv:2012.13255, Facebook AI
- **Key Contribution**: Shows pre-trained LMs have remarkably low intrinsic dimension for fine-tuning (d90 ≈ 200 for RoBERTa-Large on MRPC)
- **Methodology**:
  - Reparameterize fine-tuning: θ^D = θ^D_0 + θ^d M (Fastfood transform)
  - Binary search over d to find d90 (dimension achieving 90% of full performance)
  - Structure-Aware Intrinsic Dimension (SAID): adds per-layer scaling λ_i
- **Key Results**:
  - Pre-training monotonically decreases intrinsic dimension over training steps
  - Larger models have lower intrinsic dimension (inverse correlation)
  - Lower intrinsic dimension correlates with better generalization
  - Generalization bound: L(f) ≤ L̂(f) + O(√(d/m)), independent of full parameter count D
- **Datasets**: GLUE (MRPC, QQP), Yelp, SST-2, MNLI, ANLI
- **Relevance**: **Critical paper** — demonstrates that representation geometry evolves during pre-training and correlates with generalization. Our work extends this by using persistent homology instead of intrinsic dimension to capture richer topological structure.

#### Li et al. (2025) - "Tracing the Representation Geometry of Language Models from Pretraining to Post-training"
- **Source**: arXiv:2509.23024, McGill/Mila
- **Key Contribution**: Discovers three universal geometric phases during LM pretraining using spectral analysis
- **Methodology**:
  - Compute effective rank (RankMe) and eigenspectrum decay rate (α_ReQ) on covariance of last-token representations
  - Track across OLMo (1B-7B) and Pythia (160M-12B) checkpoints
- **Key Results — Three Phases**:
  1. **"Warmup" phase**: Rapid representational collapse during LR ramp-up
  2. **"Entropy-seeking" phase**: Manifold dimensionality expands 2-3×, coincides with n-gram memorization
  3. **"Compression-seeking" phase**: Anisotropic consolidation along dominant eigendirections, correlates with downstream task performance gains
- **Post-training**: SFT/DPO → entropy-seeking; RLVR → compression-seeking
- **Datasets**: FineWeb, OLMo/Pythia checkpoints
- **Tools**: Spectral analysis, RankMe, α_ReQ metrics
- **Relevance**: **Most directly relevant paper** — demonstrates non-monotonic geometric evolution during training. Our work would extend this spectral approach with topological (persistent homology) analysis, capturing richer structural information (holes, connected components) beyond eigenspectrum decay.

#### Park et al. (2023) - "The Linear Representation Hypothesis and the Geometry of Large Language Models"
- **Source**: arXiv:2311.03658
- **Key Contribution**: Investigates linear representations of concepts in LLMs
- **Relevance**: Provides geometric understanding of what LM embeddings encode

#### Marks & Tegmark (2023) - "The Geometry of Truth: Emergent Linear Structure in LLM Representations"
- **Source**: arXiv:2310.06824
- **Key Contribution**: Shows truth/falsehood is linearly encoded in LLM representations
- **Relevance**: Demonstrates that geometric properties of embeddings carry semantic meaning

#### Aghajanyan et al. (2024) - "Characterizing Truthfulness in LLM Generations with Local Intrinsic Dimension"
- **Source**: arXiv:2402.18048
- **Relevance**: Uses local intrinsic dimension (LID) to characterize LLM behavior—methodologically relevant

#### Multiple papers on intrinsic dimension of LMs (2025)
- arXiv:2506.09591 - Memorization through intrinsic dimension lens
- arXiv:2506.01034 - Local intrinsic dimensions of contextual LMs
- arXiv:2412.06245 - Comparative study of learning paradigms via intrinsic dimension
- **Relevance**: Active research area showing intrinsic dimension captures important properties

#### Vocabulary Embeddings paper (2025)
- **Source**: arXiv:2505.00773
- **Key Contribution**: Shows linguistic structure emerges early in embedding space during training
- **Relevance**: Early-training geometric organization relevant to our phase analysis

### Pillar 3: Scaling Laws and Resource-Constrained Training

#### Hoffmann et al. (2022) - "Training Compute-Optimal Large Language Models" (Chinchilla)
- **Source**: arXiv:2203.15556, DeepMind
- **Key Contribution**: Establishes compute-optimal scaling — model size and training tokens should scale equally with compute budget
- **Methodology**: Trained 400+ models (70M-16B params, 5B-500B tokens). Three approaches: (1) fix model sizes, vary tokens; (2) IsoFLOP profiles; (3) parametric loss function L(N,D) = E + A/N^α + B/D^β
- **Key Results**:
  - N_opt ∝ C^0.50, D_opt ∝ C^0.50 (equal scaling)
  - Chinchilla (70B, 1.4T tokens) outperforms Gopher (280B, 300B tokens) with same compute
  - Current LLMs are significantly under-trained
- **Relevance**: **Foundational** — defines the compute-optimal frontier. Resource-constrained models deviate from this frontier, and our hypothesis is that this deviation creates distinct topological signatures.

#### Beyond Chinchilla-Optimal (2023) - arXiv:2401.00448
- **Key Contribution**: Extends Chinchilla by accounting for inference cost, suggesting over-training is optimal when inference is considered
- **Relevance**: Defines a broader notion of "resource-constrained" that includes inference budget

#### Evaluating Robustness of Chinchilla Scaling (2025) - arXiv:2509.23963
- **Key Contribution**: Tests whether Chinchilla scaling laws hold under different conditions
- **Relevance**: Validates/challenges the baseline scaling framework

#### Reconciling Kaplan and Chinchilla (2024) - arXiv:2406.12907
- **Key Contribution**: Explains discrepancies between two major scaling law formulations
- **Relevance**: Understanding which scaling law to use for defining "resource-constrained"

#### Li et al. (2017) - "Visualizing the Loss Landscape of Neural Nets"
- **Source**: arXiv:1712.09913
- **Key Contribution**: Methods for visualizing and understanding NN loss landscapes
- **Relevance**: Complementary geometric perspective on training dynamics

## Common Methodologies

1. **Persistent Homology on Point Clouds**: Used by Shahidullah (2022). Compute Vietoris-Rips persistence diagrams on embeddings using Ripser/GUDHI/giotto-tda.
2. **Intrinsic Dimension Estimation**: Used by Aghajanyan et al. (2020). Fastfood transform reparameterization with binary search for d90.
3. **Spectral Analysis of Representations**: Used by Li et al. (2025). Effective rank (RankMe) and eigenspectrum decay (α_ReQ) on covariance matrices.
4. **Scaling Law Fitting**: Used by Hoffmann et al. (2022). Parametric loss models L(N,D) fit to training curves.

## Standard Baselines

For our experiments, relevant baselines include:
- **Spectral metrics** (RankMe, α_ReQ) as in Li et al. (2025)
- **Intrinsic dimension** (d90) as in Aghajanyan et al. (2020)
- **Standard loss/perplexity** tracking
- **Downstream task accuracy** (GLUE tasks)

## Evaluation Metrics

- **Topological**: Betti numbers (b0, b1, b2), persistence diagrams, Betti curves, bottleneck distance, Wasserstein distance between persistence diagrams
- **Geometric**: Intrinsic dimension (d90), effective rank (RankMe), eigenspectrum decay rate (α_ReQ)
- **Performance**: Perplexity (WikiText), downstream accuracy (GLUE), generalization gap
- **Correlation**: Spearman/Pearson correlations between topological features and performance metrics

## Datasets in the Literature

| Dataset | Used By | Task |
|---------|---------|------|
| Synthetic manifolds (torus, etc.) | Shahidullah (2022) | TDA validation |
| GLUE (MRPC, QQP, SST-2, MNLI) | Aghajanyan et al. (2020) | Intrinsic dimension, downstream eval |
| The Pile | Pythia models | LM pretraining |
| FineWeb | Li et al. (2025) | Representation geometry analysis |
| WikiText-103 | Various | Perplexity evaluation |
| TinyStories | Small LM research | Feasible LM training |

## Gaps and Opportunities

1. **No existing work applies persistent homology to track LM training dynamics**: Li et al. (2025) uses spectral methods; Shahidullah (2022) uses PH on simple feedforward NNs. Combining PH with LM checkpoint analysis is novel.

2. **Resource constraint effects on topology are unexplored**: Scaling law papers focus on loss; geometry papers focus on spectral properties. No one has examined how varying the compute budget affects the *topological* structure of embeddings.

3. **Connection between topology and generalization is underdeveloped**: Aghajanyan shows intrinsic dimension correlates with generalization. Persistent homology captures richer structure (holes, higher-order features) that may correlate even more strongly.

4. **Phase transitions in topological features during training**: Li et al. (2025) found three spectral phases. Do analogous topological phases exist? Do they differ for under-resourced models?

5. **Scalability of TDA to high-dimensional LM embeddings**: Key practical challenge — persistent homology is O(n^3) in the worst case. Subsampling, dimensionality reduction, and approximate methods are needed.

## Recommendations for Our Experiment

### Recommended Datasets
1. **Primary**: Pythia model checkpoints (70M, 160M, 410M) — 154 checkpoints per model, trained on identical data
2. **Evaluation**: WikiText-2 for perplexity, GLUE MRPC for downstream
3. **Small-scale training**: TinyStories for training models from scratch under different compute budgets

### Recommended Baselines
1. Spectral metrics (RankMe, α_ReQ) from Li et al. (2025)
2. Intrinsic dimension estimation following Aghajanyan et al. (2020)
3. Standard perplexity/loss curves

### Recommended Metrics
1. **Persistent homology features**: Betti curves, total persistence, persistence entropy across training checkpoints
2. **Correlation analysis**: Correlate topological features with performance metrics
3. **Cross-model comparison**: Compare topological signatures across model sizes (70M vs 160M vs 410M)

### Methodological Considerations
- Use Ripser.py for fast persistence computation on embedding subsamples
- Subsample embeddings (1000-5000 points) for computational feasibility
- Apply PCA to reduce embedding dimension before computing PH (as done by Shahidullah)
- Track Betti numbers b0 (connected components) and b1 (loops) across training checkpoints
- Compare "resource-constrained" (early stopping, small model) vs "compute-optimal" training regimes
- Use bottleneck/Wasserstein distance between persistence diagrams to quantify topological change rate
