"""
Analysis and visualization of topological signatures experiment results.

Generates:
1. Topological feature evolution plots (Betti numbers, persistence across training)
2. Cross-model comparison plots
3. Correlation analysis (topology vs. perplexity)
4. Spectral vs. topological comparison
5. Phase detection analysis
6. Statistical summary tables
"""

import json
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import spearmanr, pearsonr
from scipy.ndimage import gaussian_filter1d

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "data"
PLOTS_DIR = PROJECT_ROOT / "results" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Plot style
plt.rcParams.update({
    'figure.figsize': (10, 6),
    'font.size': 12,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'legend.fontsize': 11,
    'figure.dpi': 150,
})

MODEL_COLORS = {
    "pythia-70m": "#1f77b4",
    "pythia-160m": "#ff7f0e",
    "pythia-410m": "#2ca02c",
}

MODEL_LABELS = {
    "pythia-70m": "Pythia-70M",
    "pythia-160m": "Pythia-160M",
    "pythia-410m": "Pythia-410M",
}


def load_results():
    """Load all experiment results."""
    combined_path = RESULTS_DIR / "all_results.json"
    with open(combined_path) as f:
        return json.load(f)


def plot_topological_evolution(all_results):
    """Plot how topological features evolve during training for each model."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    features = [
        ("h0_n_features", "H0 Feature Count (Connected Components)"),
        ("h1_n_features", "H1 Feature Count (Loops)"),
        ("h0_total_persistence", "H0 Total Persistence"),
        ("h1_total_persistence", "H1 Total Persistence"),
        ("h0_persistence_entropy", "H0 Persistence Entropy"),
        ("h1_persistence_entropy", "H1 Persistence Entropy"),
    ]

    for idx, (feature, title) in enumerate(features):
        ax = axes[idx // 3, idx % 3]
        for model_name, results in all_results.items():
            steps = [r["step"] for r in results if feature in r]
            values = [r[feature] for r in results if feature in r]
            if steps:
                ax.plot(steps, values, 'o-', color=MODEL_COLORS.get(model_name, 'gray'),
                       label=MODEL_LABELS.get(model_name, model_name), markersize=4)
        ax.set_xlabel("Training Step")
        ax.set_ylabel(title.split("(")[0].strip())
        ax.set_title(title)
        ax.set_xscale("symlog", linthresh=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "topological_evolution.png", bbox_inches='tight')
    plt.close()
    print(f"Saved topological_evolution.png")


def plot_spectral_evolution(all_results):
    """Plot spectral features across training."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    spectral_features = [
        ("effective_rank", "Effective Rank (RankMe)"),
        ("participation_ratio", "Participation Ratio"),
        ("eigenspectrum_decay_alpha", "Eigenspectrum Decay Rate (alpha)"),
    ]

    for idx, (feature, title) in enumerate(spectral_features):
        ax = axes[idx]
        for model_name, results in all_results.items():
            steps = [r["step"] for r in results if feature in r]
            values = [r[feature] for r in results if feature in r]
            if steps:
                ax.plot(steps, values, 'o-', color=MODEL_COLORS.get(model_name, 'gray'),
                       label=MODEL_LABELS.get(model_name, model_name), markersize=4)
        ax.set_xlabel("Training Step")
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.set_xscale("symlog", linthresh=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "spectral_evolution.png", bbox_inches='tight')
    plt.close()
    print(f"Saved spectral_evolution.png")


def plot_perplexity_curves(all_results):
    """Plot perplexity across training for all models."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for model_name, results in all_results.items():
        steps = [r["step"] for r in results if "perplexity" in r]
        ppls = [r["perplexity"] for r in results if "perplexity" in r]
        if steps:
            ax.plot(steps, ppls, 'o-', color=MODEL_COLORS.get(model_name, 'gray'),
                   label=MODEL_LABELS.get(model_name, model_name), markersize=5)

    ax.set_xlabel("Training Step")
    ax.set_ylabel("Perplexity")
    ax.set_title("Perplexity During Training")
    ax.set_xscale("symlog", linthresh=10)
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "perplexity_curves.png", bbox_inches='tight')
    plt.close()
    print(f"Saved perplexity_curves.png")


def plot_topology_vs_perplexity(all_results):
    """Scatter plots of topological features vs. perplexity with correlation."""
    topo_features = [
        "h0_total_persistence", "h1_total_persistence",
        "h0_persistence_entropy", "h1_persistence_entropy",
        "h0_n_features", "h1_n_features",
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    for idx, feature in enumerate(topo_features):
        ax = axes[idx // 3, idx % 3]
        for model_name, results in all_results.items():
            ppls = [r["perplexity"] for r in results if feature in r and "perplexity" in r]
            vals = [r[feature] for r in results if feature in r and "perplexity" in r]
            if ppls and vals:
                ax.scatter(vals, ppls, color=MODEL_COLORS.get(model_name, 'gray'),
                          label=MODEL_LABELS.get(model_name, model_name), alpha=0.7, s=40)

        # Compute overall correlation
        all_vals = []
        all_ppls = []
        for model_name, results in all_results.items():
            for r in results:
                if feature in r and "perplexity" in r:
                    all_vals.append(r[feature])
                    all_ppls.append(r["perplexity"])

        if len(all_vals) > 5:
            rho, p = spearmanr(all_vals, all_ppls)
            ax.set_title(f"{feature}\nSpearman r={rho:.3f}, p={p:.2e}")
        else:
            ax.set_title(feature)

        ax.set_xlabel(feature)
        ax.set_ylabel("Perplexity")
        ax.set_yscale("log")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "topology_vs_perplexity.png", bbox_inches='tight')
    plt.close()
    print(f"Saved topology_vs_perplexity.png")


def plot_wasserstein_rate(all_results):
    """Plot rate of topological change (Wasserstein distance) across training."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for model_name, results in all_results.items():
        steps = [r["step"] for r in results if "wasserstein_h1_from_prev" in r]
        dists = [r["wasserstein_h1_from_prev"] for r in results if "wasserstein_h1_from_prev" in r]
        if steps:
            ax.plot(steps, dists, 'o-', color=MODEL_COLORS.get(model_name, 'gray'),
                   label=MODEL_LABELS.get(model_name, model_name), markersize=5)

    ax.set_xlabel("Training Step")
    ax.set_ylabel("Wasserstein Distance (H1) from Previous Checkpoint")
    ax.set_title("Rate of Topological Change During Training")
    ax.set_xscale("symlog", linthresh=10)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "wasserstein_rate.png", bbox_inches='tight')
    plt.close()
    print(f"Saved wasserstein_rate.png")


def plot_topo_vs_spectral(all_results):
    """Compare topological and spectral feature correlations with perplexity."""
    features_to_compare = [
        # Topological
        "h0_total_persistence", "h1_total_persistence",
        "h0_persistence_entropy", "h1_persistence_entropy",
        "h1_n_features",
        # Spectral
        "effective_rank", "participation_ratio", "eigenspectrum_decay_alpha",
    ]

    # Compute per-model correlations
    correlation_results = {}
    for model_name, results in all_results.items():
        ppls = np.array([r["perplexity"] for r in results])
        model_corrs = {}
        for feat in features_to_compare:
            vals = np.array([r.get(feat, np.nan) for r in results])
            valid = ~np.isnan(vals) & ~np.isnan(ppls)
            if valid.sum() > 5:
                rho, p = spearmanr(vals[valid], ppls[valid])
                model_corrs[feat] = {"rho": rho, "p": p}
            else:
                model_corrs[feat] = {"rho": np.nan, "p": np.nan}
        correlation_results[model_name] = model_corrs

    # Bar plot of correlations
    fig, ax = plt.subplots(figsize=(14, 7))
    x = np.arange(len(features_to_compare))
    width = 0.25

    for i, (model_name, corrs) in enumerate(correlation_results.items()):
        rhos = [corrs[f]["rho"] for f in features_to_compare]
        bars = ax.bar(x + i * width, rhos, width,
                     label=MODEL_LABELS.get(model_name, model_name),
                     color=MODEL_COLORS.get(model_name, 'gray'), alpha=0.8)
        # Mark significant correlations
        for j, f in enumerate(features_to_compare):
            p = corrs[f]["p"]
            if not np.isnan(p) and p < 0.05:
                ax.text(x[j] + i * width, rhos[j] + 0.02 * np.sign(rhos[j]),
                       '*', ha='center', fontsize=14, fontweight='bold')

    ax.set_xlabel("Feature")
    ax.set_ylabel("Spearman Correlation with Perplexity")
    ax.set_title("Topological vs. Spectral Feature Correlations with Perplexity\n(* = p < 0.05)")
    ax.set_xticks(x + width)
    ax.set_xticklabels([f.replace("_", "\n") for f in features_to_compare], fontsize=8, rotation=45, ha='right')
    ax.legend()
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "topo_vs_spectral_correlations.png", bbox_inches='tight')
    plt.close()
    print(f"Saved topo_vs_spectral_correlations.png")

    return correlation_results


def compute_statistics(all_results):
    """Compute comprehensive statistics and save as JSON."""
    stats = {}

    all_features = [
        "h0_n_features", "h1_n_features",
        "h0_total_persistence", "h1_total_persistence",
        "h0_persistence_entropy", "h1_persistence_entropy",
        "h0_max_persistence", "h1_max_persistence",
        "effective_rank", "participation_ratio", "eigenspectrum_decay_alpha",
        "perplexity",
    ]

    for model_name, results in all_results.items():
        model_stats = {}
        ppls = np.array([r["perplexity"] for r in results])

        for feat in all_features:
            vals = np.array([r.get(feat, np.nan) for r in results])
            valid = ~np.isnan(vals)
            if valid.sum() < 2:
                continue

            feat_stats = {
                "mean": float(np.nanmean(vals)),
                "std": float(np.nanstd(vals)),
                "min": float(np.nanmin(vals)),
                "max": float(np.nanmax(vals)),
                "range": float(np.nanmax(vals) - np.nanmin(vals)),
            }

            # Correlation with perplexity
            valid_both = valid & ~np.isnan(ppls)
            if valid_both.sum() > 5 and feat != "perplexity":
                rho, p_rho = spearmanr(vals[valid_both], ppls[valid_both])
                r, p_r = pearsonr(vals[valid_both], ppls[valid_both])
                feat_stats["spearman_rho_vs_ppl"] = float(rho)
                feat_stats["spearman_p_vs_ppl"] = float(p_rho)
                feat_stats["pearson_r_vs_ppl"] = float(r)
                feat_stats["pearson_p_vs_ppl"] = float(p_r)

            model_stats[feat] = feat_stats
        stats[model_name] = model_stats

    # Save stats
    with open(RESULTS_DIR / "statistics.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved statistics.json")

    return stats


def plot_combined_summary(all_results):
    """Create a summary figure with key results."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Top-left: Perplexity curves
    ax = axes[0, 0]
    for model_name, results in all_results.items():
        steps = [r["step"] for r in results]
        ppls = [r["perplexity"] for r in results]
        ax.plot(steps, ppls, 'o-', color=MODEL_COLORS.get(model_name, 'gray'),
               label=MODEL_LABELS.get(model_name, model_name), markersize=4)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Perplexity")
    ax.set_title("(a) Perplexity During Training")
    ax.set_xscale("symlog", linthresh=10)
    ax.set_yscale("log")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Top-right: H1 total persistence
    ax = axes[0, 1]
    for model_name, results in all_results.items():
        steps = [r["step"] for r in results if "h1_total_persistence" in r]
        vals = [r["h1_total_persistence"] for r in results if "h1_total_persistence" in r]
        ax.plot(steps, vals, 'o-', color=MODEL_COLORS.get(model_name, 'gray'),
               label=MODEL_LABELS.get(model_name, model_name), markersize=4)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("H1 Total Persistence")
    ax.set_title("(b) H1 Total Persistence During Training")
    ax.set_xscale("symlog", linthresh=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Bottom-left: Effective rank
    ax = axes[1, 0]
    for model_name, results in all_results.items():
        steps = [r["step"] for r in results if "effective_rank" in r]
        vals = [r["effective_rank"] for r in results if "effective_rank" in r]
        ax.plot(steps, vals, 'o-', color=MODEL_COLORS.get(model_name, 'gray'),
               label=MODEL_LABELS.get(model_name, model_name), markersize=4)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Effective Rank")
    ax.set_title("(c) Effective Rank (RankMe) During Training")
    ax.set_xscale("symlog", linthresh=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Bottom-right: Wasserstein distance
    ax = axes[1, 1]
    for model_name, results in all_results.items():
        steps = [r["step"] for r in results if "wasserstein_h1_from_prev" in r]
        vals = [r["wasserstein_h1_from_prev"] for r in results if "wasserstein_h1_from_prev" in r]
        if steps:
            ax.plot(steps, vals, 'o-', color=MODEL_COLORS.get(model_name, 'gray'),
                   label=MODEL_LABELS.get(model_name, model_name), markersize=4)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Wasserstein Distance (H1)")
    ax.set_title("(d) Rate of Topological Change")
    ax.set_xscale("symlog", linthresh=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.suptitle("Topological Signatures of Resource-Constrained Language Model Training", fontsize=15, y=1.02)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "summary_figure.png", bbox_inches='tight')
    plt.close()
    print(f"Saved summary_figure.png")


def main():
    print("Loading results...")
    all_results = load_results()

    print(f"Models found: {list(all_results.keys())}")
    for model, results in all_results.items():
        print(f"  {model}: {len(results)} checkpoints")

    print("\nGenerating visualizations...")
    plot_perplexity_curves(all_results)
    plot_topological_evolution(all_results)
    plot_spectral_evolution(all_results)
    plot_topology_vs_perplexity(all_results)
    plot_wasserstein_rate(all_results)
    correlation_results = plot_topo_vs_spectral(all_results)
    plot_combined_summary(all_results)

    print("\nComputing statistics...")
    stats = compute_statistics(all_results)

    # Print correlation summary
    print("\n" + "=" * 80)
    print("CORRELATION SUMMARY (Spearman rho with perplexity)")
    print("=" * 80)
    for model_name in all_results:
        print(f"\n{MODEL_LABELS.get(model_name, model_name)}:")
        if model_name in stats:
            for feat, feat_stats in stats[model_name].items():
                if "spearman_rho_vs_ppl" in feat_stats:
                    sig = "*" if feat_stats["spearman_p_vs_ppl"] < 0.05 else " "
                    print(f"  {sig} {feat:35s} rho={feat_stats['spearman_rho_vs_ppl']:+.3f} "
                          f"(p={feat_stats['spearman_p_vs_ppl']:.4f})")

    print("\nAll visualizations saved to:", PLOTS_DIR)
    print("Statistics saved to:", RESULTS_DIR / "statistics.json")


if __name__ == "__main__":
    main()
