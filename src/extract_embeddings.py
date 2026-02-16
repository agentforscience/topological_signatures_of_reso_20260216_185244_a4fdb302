"""
Extract embeddings from Pythia model checkpoints and compute topological + spectral features.

This module handles:
1. Loading Pythia models at specific training checkpoints
2. Running text through the model to extract last-layer embeddings
3. Computing persistent homology (H0, H1) via Ripser
4. Computing spectral baselines (effective rank, participation ratio)
5. Computing perplexity on WikiText-2
"""

import os
import json
import time
import random
import logging
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from sklearn.decomposition import PCA
from ripser import ripser
from persim import wasserstein as wasserstein_distance
from scipy.stats import entropy as scipy_entropy

# Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
PROJECT_ROOT = Path(__file__).parent.parent


def load_wikitext2_texts(n_texts: int = 200, min_length: int = 100) -> List[str]:
    """Load texts from WikiText-2 for embedding extraction."""
    from datasets import load_from_disk
    ds = load_from_disk(str(PROJECT_ROOT / "datasets" / "wikitext-2"))
    texts = []
    for item in ds["validation"]:
        text = item["text"].strip()
        if len(text) >= min_length:
            texts.append(text)
        if len(texts) >= n_texts:
            break
    logger.info(f"Loaded {len(texts)} texts from WikiText-2 validation set")
    return texts


def extract_embeddings(
    model_name: str,
    revision: str,
    texts: List[str],
    max_length: int = 128,
    batch_size: int = 32,
) -> np.ndarray:
    """
    Extract last-layer hidden states from a Pythia checkpoint.
    Returns array of shape (n_tokens, hidden_dim) by pooling over non-padding tokens.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info(f"Loading {model_name} revision={revision}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name, revision=revision, torch_dtype=torch.float16
    ).to(DEVICE)
    model.eval()

    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(
            batch_texts, return_tensors="pt", padding=True,
            truncation=True, max_length=max_length
        ).to(DEVICE)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        # Last hidden layer, mean-pool over sequence (excluding padding)
        hidden = outputs.hidden_states[-1]  # (batch, seq_len, hidden_dim)
        mask = inputs["attention_mask"].unsqueeze(-1).float()  # (batch, seq_len, 1)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1)  # (batch, hidden_dim)
        all_embeddings.append(pooled.cpu().float().numpy())

    # Clean up GPU memory
    del model
    torch.cuda.empty_cache()

    embeddings = np.concatenate(all_embeddings, axis=0)
    logger.info(f"Extracted embeddings shape: {embeddings.shape}")
    return embeddings


def compute_perplexity(
    model_name: str,
    revision: str,
    texts: List[str],
    max_length: int = 256,
) -> float:
    """Compute perplexity on a set of texts."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name, revision=revision, torch_dtype=torch.float16
    ).to(DEVICE)
    model.eval()

    total_loss = 0.0
    total_tokens = 0

    for text in texts[:100]:  # Use subset for speed
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length).to(DEVICE)
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
        n_tokens = inputs["input_ids"].shape[1]
        total_loss += outputs.loss.item() * n_tokens
        total_tokens += n_tokens

    del model
    torch.cuda.empty_cache()

    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)
    logger.info(f"Perplexity: {perplexity:.2f}")
    return perplexity


def compute_persistent_homology(
    embeddings: np.ndarray,
    n_subsample: int = 1000,
    pca_dim: int = 50,
    max_homology_dim: int = 1,
) -> Dict:
    """
    Compute persistent homology on embeddings.

    Steps:
    1. Subsample to n_subsample points
    2. PCA reduce to pca_dim dimensions
    3. Compute Vietoris-Rips persistence via Ripser

    Returns dict with Betti numbers, persistence stats, and raw diagrams.
    """
    n_points = embeddings.shape[0]

    # Subsample if needed
    if n_points > n_subsample:
        idx = np.random.choice(n_points, n_subsample, replace=False)
        data = embeddings[idx]
    else:
        data = embeddings.copy()

    # PCA reduction
    actual_pca_dim = min(pca_dim, data.shape[0] - 1, data.shape[1])
    pca = PCA(n_components=actual_pca_dim)
    data_pca = pca.fit_transform(data)
    variance_explained = pca.explained_variance_ratio_.sum()
    logger.info(f"PCA: {data.shape[1]}D -> {actual_pca_dim}D, variance explained: {variance_explained:.3f}")

    # Compute persistent homology
    t0 = time.time()
    result = ripser(data_pca, maxdim=max_homology_dim, thresh=np.inf)
    ph_time = time.time() - t0
    logger.info(f"Ripser computed in {ph_time:.1f}s")

    diagrams = result["dgms"]

    # Extract features for each homology dimension
    features = {
        "pca_variance_explained": float(variance_explained),
        "pca_dim": actual_pca_dim,
        "n_points": int(data.shape[0]),
        "ph_compute_time_s": float(ph_time),
    }

    for dim in range(max_homology_dim + 1):
        dgm = diagrams[dim]
        # Remove infinite death times
        finite_mask = np.isfinite(dgm[:, 1])
        dgm_finite = dgm[finite_mask]

        if len(dgm_finite) == 0:
            features[f"h{dim}_n_features"] = 0
            features[f"h{dim}_total_persistence"] = 0.0
            features[f"h{dim}_mean_persistence"] = 0.0
            features[f"h{dim}_max_persistence"] = 0.0
            features[f"h{dim}_persistence_entropy"] = 0.0
            features[f"h{dim}_n_significant"] = 0
            continue

        persistences = dgm_finite[:, 1] - dgm_finite[:, 0]
        persistences = persistences[persistences > 0]

        if len(persistences) == 0:
            features[f"h{dim}_n_features"] = 0
            features[f"h{dim}_total_persistence"] = 0.0
            features[f"h{dim}_mean_persistence"] = 0.0
            features[f"h{dim}_max_persistence"] = 0.0
            features[f"h{dim}_persistence_entropy"] = 0.0
            features[f"h{dim}_n_significant"] = 0
            continue

        total_pers = float(persistences.sum())
        # Persistence entropy
        probs = persistences / total_pers
        pers_entropy = float(-np.sum(probs * np.log(probs + 1e-12)))

        # Significant features: persistence > median
        median_pers = np.median(persistences)
        n_significant = int((persistences > median_pers).sum())

        features[f"h{dim}_n_features"] = int(len(persistences))
        features[f"h{dim}_total_persistence"] = total_pers
        features[f"h{dim}_mean_persistence"] = float(persistences.mean())
        features[f"h{dim}_max_persistence"] = float(persistences.max())
        features[f"h{dim}_persistence_entropy"] = pers_entropy
        features[f"h{dim}_n_significant"] = n_significant
        features[f"h{dim}_std_persistence"] = float(persistences.std())

    # Store raw diagrams for later Wasserstein distance computation
    features["_diagrams"] = [dgm.tolist() for dgm in diagrams]

    return features


def compute_spectral_features(embeddings: np.ndarray) -> Dict:
    """
    Compute spectral features of the embedding covariance matrix.

    - Effective rank (RankMe): exp(entropy of normalized singular values)
    - Participation ratio: (sum of eigenvalues)^2 / sum(eigenvalues^2)
    - Eigenspectrum decay rate (alpha): slope of log-log eigenvalue plot
    """
    # Center the data
    centered = embeddings - embeddings.mean(axis=0)

    # Compute covariance and eigenvalues
    cov = np.cov(centered.T)
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = np.sort(eigenvalues)[::-1]  # Descending
    eigenvalues = np.maximum(eigenvalues, 0)  # Ensure non-negative

    # Effective rank (RankMe)
    total = eigenvalues.sum()
    if total > 0:
        probs = eigenvalues / total
        probs = probs[probs > 1e-12]
        effective_rank = float(np.exp(-np.sum(probs * np.log(probs))))
    else:
        effective_rank = 1.0

    # Participation ratio
    if eigenvalues.sum() > 0:
        participation_ratio = float(eigenvalues.sum() ** 2 / (eigenvalues ** 2).sum())
    else:
        participation_ratio = 1.0

    # Eigenspectrum decay rate (power law fit on top eigenvalues)
    n_top = min(50, len(eigenvalues))
    top_eigs = eigenvalues[:n_top]
    top_eigs = top_eigs[top_eigs > 1e-12]
    if len(top_eigs) > 5:
        log_idx = np.log(np.arange(1, len(top_eigs) + 1))
        log_eig = np.log(top_eigs)
        # Linear fit in log-log space
        alpha = -float(np.polyfit(log_idx, log_eig, 1)[0])
    else:
        alpha = 0.0

    # Stable rank: ||A||_F^2 / ||A||_2^2
    if eigenvalues[0] > 0:
        stable_rank = float(eigenvalues.sum() / eigenvalues[0])
    else:
        stable_rank = 1.0

    return {
        "effective_rank": effective_rank,
        "participation_ratio": participation_ratio,
        "eigenspectrum_decay_alpha": alpha,
        "stable_rank": stable_rank,
        "top_eigenvalue": float(eigenvalues[0]) if len(eigenvalues) > 0 else 0.0,
        "eigenvalue_ratio_1_10": float(eigenvalues[0] / eigenvalues[9]) if len(eigenvalues) > 9 and eigenvalues[9] > 0 else float('inf'),
    }


def compute_wasserstein_distances(features_list: List[Dict]) -> List[float]:
    """Compute Wasserstein distances between consecutive persistence diagrams."""
    distances = []
    for i in range(1, len(features_list)):
        prev_dgms = features_list[i - 1].get("_diagrams", None)
        curr_dgms = features_list[i].get("_diagrams", None)
        if prev_dgms is None or curr_dgms is None:
            distances.append(float('nan'))
            continue

        # Use H1 diagrams for Wasserstein distance
        prev_h1 = np.array(prev_dgms[1]) if len(prev_dgms) > 1 else np.array([]).reshape(0, 2)
        curr_h1 = np.array(curr_dgms[1]) if len(curr_dgms) > 1 else np.array([]).reshape(0, 2)

        # Filter to finite entries
        if len(prev_h1) > 0:
            prev_h1 = prev_h1[np.isfinite(prev_h1).all(axis=1)]
        if len(curr_h1) > 0:
            curr_h1 = curr_h1[np.isfinite(curr_h1).all(axis=1)]

        if len(prev_h1) == 0:
            prev_h1 = np.array([[0, 0]])
        if len(curr_h1) == 0:
            curr_h1 = np.array([[0, 0]])

        try:
            d = wasserstein_distance(prev_h1, curr_h1)
            distances.append(float(d))
        except Exception as e:
            logger.warning(f"Wasserstein distance failed: {e}")
            distances.append(float('nan'))

    return distances


def run_experiment_for_model(
    model_name: str,
    checkpoints: List[str],
    texts: List[str],
    n_subsample: int = 1000,
    pca_dim: int = 50,
) -> List[Dict]:
    """Run the full experiment pipeline for a single model across checkpoints."""
    results = []
    for ckpt in checkpoints:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing {model_name} checkpoint: {ckpt}")
        logger.info(f"{'='*60}")

        try:
            t0 = time.time()

            # Extract embeddings
            embeddings = extract_embeddings(model_name, ckpt, texts)

            # Compute persistent homology
            ph_features = compute_persistent_homology(
                embeddings, n_subsample=n_subsample, pca_dim=pca_dim
            )

            # Compute spectral features
            spectral_features = compute_spectral_features(embeddings)

            # Compute perplexity
            perplexity = compute_perplexity(model_name, ckpt, texts)

            total_time = time.time() - t0

            result = {
                "model": model_name,
                "checkpoint": ckpt,
                "step": int(ckpt.replace("step", "")),
                "perplexity": perplexity,
                "embedding_dim": int(embeddings.shape[1]),
                "n_embeddings": int(embeddings.shape[0]),
                "total_time_s": float(total_time),
                **{k: v for k, v in ph_features.items() if not k.startswith("_")},
                **spectral_features,
            }
            # Keep raw diagrams for Wasserstein computation but don't serialize
            result["_diagrams"] = ph_features.get("_diagrams")
            results.append(result)

            logger.info(f"Checkpoint {ckpt} done in {total_time:.1f}s | "
                       f"PPL={perplexity:.1f} | b0={ph_features.get('h0_n_features', 0)} | "
                       f"b1={ph_features.get('h1_n_features', 0)} | "
                       f"eff_rank={spectral_features['effective_rank']:.1f}")

        except Exception as e:
            logger.error(f"Failed on {model_name} {ckpt}: {e}")
            import traceback
            traceback.print_exc()
            continue

    return results


if __name__ == "__main__":
    # This script is imported by run_experiments.py
    pass
