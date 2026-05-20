"""
Generate a heatmap of suitability scores across all candidate analogue sites
and all Enceladus target environments.

Reproduces the AstroMatch v3 scoring engine (Jaccard-Gaussian hybrid, weighted
arithmetic mean across active parameters) using equal weighting across the
six parameters as the baseline view.

Output: heatmap_sites_x_targets.png (300 dpi, publication-ready)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# --- 1. CONFIGURATION ---------------------------------------------------------

ANALOGUES_PATH = "analogues_v2.csv"
TARGETS_PATH = "targets_v2.csv"
OUTPUT_PATH = "heatmap_sites_x_targets.png"

# Parameter prefixes match the column-naming convention in the CSVs
PARAMS_CONFIG = {
    "Temperature":     {"col_prefix": "T",     "default_weight": 5},
    "Salinity":        {"col_prefix": "Sal",   "default_weight": 5},
    "pH":              {"col_prefix": "pH",    "default_weight": 5},
    "Pressure":        {"col_prefix": "Pres",  "default_weight": 5},
    "Isolation":       {"col_prefix": "Iso",   "default_weight": 5},
    "Redox Potential": {"col_prefix": "Redox", "default_weight": 5},
}

# Order target environments top-down through Enceladus (plume -> ice -> ocean -> vents)
# so the reader's eye traces the moon from outside in.
TARGET_ORDER = [
    "Plume Ice Grains - Type I",
    "Plume Ice Grains - Type II (VOC)",
    "Plume Ice Grains - Type II (HMOC)",
    "Plume Ice Grains - Type III",
    "Intrashell Mushy Fracture Brines",
    "Ocean Connected Tiger Stripe Brines",
    "Abyssal Ocean",
    "Diffuse Venting Zone",
    "Hydrothermal Vents",
]

# --- 2. SCORING ENGINE (replicated from app.py for consistency) ---------------

def calculate_suitability(site_min, site_max, target_min, target_max):
    """Per-parameter fit: max(Jaccard overlap, Gaussian decay) in [0, 1].
    Returns None if any input is missing."""
    if pd.isna(site_min) or pd.isna(target_min) or pd.isna(site_max) or pd.isna(target_max):
        return None

    s_mid = (site_min + site_max) / 2
    t_mid = (target_min + target_max) / 2
    sigma = max(target_max - target_min, 1.0)
    gaussian = np.exp(-((s_mid - t_mid) ** 2) / (2 * sigma ** 2))

    overlap_min = max(site_min, target_min)
    overlap_max = min(site_max, target_max)
    if overlap_max > overlap_min:
        intersection = overlap_max - overlap_min
        union = max(site_max, target_max) - min(site_min, target_min)
        return max(gaussian, intersection / union)
    return gaussian


def score_site_against_target(site, target_data, user_weights):
    """Returns the weighted-mean suitability score for one site against one
    target environment."""
    fits = {}
    weights = {}
    for param, info in PARAMS_CONFIG.items():
        if param not in user_weights:
            continue
        prefix = info["col_prefix"]
        # Continuous params have _min/_max; ordinal params have _score
        min_col = f"{prefix}_min" if f"{prefix}_min" in site else f"{prefix}_score"
        max_col = f"{prefix}_max" if f"{prefix}_max" in site else f"{prefix}_score"
        t_min_col = f"{prefix}_min" if f"{prefix}_min" in target_data else f"{prefix}_score"
        t_max_col = f"{prefix}_max" if f"{prefix}_max" in target_data else f"{prefix}_score"

        fit = calculate_suitability(
            site.get(min_col), site.get(max_col),
            target_data.get(t_min_col), target_data.get(t_max_col)
        )
        if fit is not None:
            fits[param] = fit
            weights[param] = user_weights[param]

    if not weights:
        return np.nan
    w_sum = sum(weights.values())
    return sum(fits[p] * weights[p] for p in weights) / w_sum


# --- 3. BUILD THE MATRIX ------------------------------------------------------

def build_suitability_matrix(analogues_df, targets_df, user_weights):
    """Return a DataFrame indexed by site, columns are target environments,
    cells are suitability scores in [0, 1]."""
    matrix = {}
    for target_env in targets_df["Target_env"]:
        target_data = targets_df[targets_df["Target_env"] == target_env].iloc[0]
        scores = {}
        for _, site in analogues_df.iterrows():
            scores[site["Site"]] = score_site_against_target(site, target_data, user_weights)
        matrix[target_env] = scores
    df = pd.DataFrame(matrix)
    return df


# --- 4. RENDER ----------------------------------------------------------------

def render_heatmap(matrix_df, output_path, weights_label="equal weighting"):
    """Render the heatmap with top-3 cells per column outlined in red."""

    # Order columns by physical progression through Enceladus
    cols_present = [c for c in TARGET_ORDER if c in matrix_df.columns]
    matrix_df = matrix_df[cols_present]

    # Order rows by mean suitability descending (versatile sites at top,
    # specialists at bottom)
    matrix_df = matrix_df.loc[
        matrix_df.mean(axis=1).sort_values(ascending=False).index
    ]

    fig, ax = plt.subplots(figsize=(11, 8))

    sns.heatmap(
        matrix_df,
        cmap="viridis",
        vmin=0, vmax=1,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 8},
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Suitability score", "shrink": 0.85},
        ax=ax,
    )

    # Outline top-3 cells per column (the visual proof of the portfolio claim)
    top3_counts = {site: 0 for site in matrix_df.index}
    for col_idx, col in enumerate(matrix_df.columns):
        top3_sites = matrix_df[col].nlargest(3).index
        for site in top3_sites:
            row_idx = matrix_df.index.get_loc(site)
            ax.add_patch(plt.Rectangle(
                (col_idx, row_idx), 1, 1,
                fill=False, edgecolor="red", lw=1.8
            ))
            top3_counts[site] += 1

    # Diagnostic: print the top-3 count per site (the figure's central claim
    # is that no site appears in top-3 for more than four targets)
    print("\nTop-3 appearances per site (out of 9 targets):")
    for site, count in sorted(top3_counts.items(), key=lambda x: -x[1]):
        print(f"  {count}  -  {site}")
    max_top3 = max(top3_counts.values())
    print(f"\nMaximum top-3 appearances by any site: {max_top3}")

    ax.set_xlabel("Enceladus target environment", fontsize=11)
    ax.set_ylabel("Terrestrial analogue site", fontsize=11)
    ax.set_title(
        f"Per-target suitability across candidate analogue sites\n"
        f"({weights_label} across six parameters; red outlines = top-3 per target)",
        fontsize=12, pad=14
    )

    plt.setp(ax.get_xticklabels(), rotation=35, ha="right", fontsize=9)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nFigure saved to: {output_path}")
    return matrix_df, top3_counts


# --- 5. MAIN ------------------------------------------------------------------

def main():
    analogues_df = pd.read_csv(ANALOGUES_PATH)
    targets_df = pd.read_csv(TARGETS_PATH)

    # Equal weights baseline
    user_weights = {p: info["default_weight"] for p, info in PARAMS_CONFIG.items()}

    matrix = build_suitability_matrix(analogues_df, targets_df, user_weights)
    print("Suitability matrix shape:", matrix.shape)
    print(f"Sites: {matrix.shape[0]}, Target environments: {matrix.shape[1]}")

    render_heatmap(matrix, OUTPUT_PATH, weights_label="equal weighting")

    # Also save the raw matrix for reproducibility / further analysis
    matrix.to_csv("heatmap_sites_x_targets.csv")
    print("Matrix saved to: heatmap_sites_x_targets.csv")


if __name__ == "__main__":
    main()
