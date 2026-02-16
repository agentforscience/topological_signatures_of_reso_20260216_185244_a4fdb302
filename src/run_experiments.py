"""
Main experiment runner: Topological Signatures of Resource-Constrained LM Training.

Runs the full experimental pipeline:
1. Load WikiText-2 texts
2. For each Pythia model size (70M, 160M, 410M), extract embeddings at selected checkpoints
3. Compute persistent homology and spectral features
4. Save all results to JSON
"""

import json
import sys
import time
import logging
import numpy as np
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from extract_embeddings import (
    load_wikitext2_texts,
    run_experiment_for_model,
    compute_wasserstein_distances,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

# Experiment configuration
MODELS = {
    "EleutherAI/pythia-70m": "pythia-70m",
    "EleutherAI/pythia-160m": "pythia-160m",
    "EleutherAI/pythia-410m": "pythia-410m",
}

# Selected checkpoints: logarithmic spacing to capture early rapid changes
# Pythia has checkpoints at steps: 0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
# then every 1000 steps up to 143000
CHECKPOINTS = [
    "step0", "step1", "step2", "step4", "step8", "step16", "step32",
    "step64", "step128", "step256", "step512",
    "step1000", "step2000", "step5000", "step10000",
    "step33000", "step66000", "step100000", "step143000",
]

N_SUBSAMPLE = 1000
PCA_DIM = 50
N_TEXTS = 200  # Number of WikiText-2 texts to use


def main():
    start_time = time.time()
    results_dir = PROJECT_ROOT / "results" / "data"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load texts once
    logger.info("Loading WikiText-2 texts...")
    texts = load_wikitext2_texts(n_texts=N_TEXTS)

    all_results = {}

    for model_name, model_short in MODELS.items():
        logger.info(f"\n{'#'*70}")
        logger.info(f"# Running experiments for {model_name}")
        logger.info(f"{'#'*70}")

        model_results = run_experiment_for_model(
            model_name=model_name,
            checkpoints=CHECKPOINTS,
            texts=texts,
            n_subsample=N_SUBSAMPLE,
            pca_dim=PCA_DIM,
        )

        # Compute Wasserstein distances between consecutive checkpoints
        if len(model_results) > 1:
            wass_distances = compute_wasserstein_distances(model_results)
            for i, d in enumerate(wass_distances):
                model_results[i + 1]["wasserstein_h1_from_prev"] = d

        # Remove raw diagrams before saving (not JSON-serializable easily)
        for r in model_results:
            r.pop("_diagrams", None)

        all_results[model_short] = model_results

        # Save incrementally
        output_path = results_dir / f"{model_short}_results.json"
        with open(output_path, "w") as f:
            json.dump(model_results, f, indent=2, default=str)
        logger.info(f"Saved {model_short} results to {output_path}")

    # Save combined results
    combined_path = results_dir / "all_results.json"
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    total_time = time.time() - start_time
    logger.info(f"\nAll experiments completed in {total_time:.1f}s ({total_time/60:.1f} min)")

    # Save experiment config
    config = {
        "seed": 42,
        "models": list(MODELS.keys()),
        "checkpoints": CHECKPOINTS,
        "n_subsample": N_SUBSAMPLE,
        "pca_dim": PCA_DIM,
        "n_texts": N_TEXTS,
        "total_time_s": total_time,
    }
    with open(results_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)


if __name__ == "__main__":
    main()
