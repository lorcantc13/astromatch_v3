"""
pages/heatmap.py
----------------
Heatmap analysis page for AstroMatch v3-analysis branch.

Renders a 15-sites x 9-targets suitability matrix using the same
Gaussian-Jaccard scoring engine as the main app, with adjustable
per-parameter weights and download buttons for the figure and raw data.
"""

import io
import os

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

# ---------------------------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------------------------

st.set_page_config(page_title="AstroMatch · Heatmap", layout="wide")

# ---------------------------------------------------------------------------
# DATA LOADING
# Looks for CSVs in the repo root (one level up from pages/).
# ---------------------------------------------------------------------------

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

@st.cache_data
def load_data():
    analogues = pd.read_csv(os.path.join(ROOT, "analogues_v2.csv"))
    targets   = pd.read_csv(os.path.join(ROOT, "targets_v2.csv"))
    return analogues, targets

try:
    analogues_df, targets_df = load_data()
except FileNotFoundError:
    st.error(
        "Could not find `analogues_v2.csv` or `targets_v2.csv` in the repo root. "
        "Make sure both files are committed to the v3-analysis branch."
    )
    st.stop()

# ---------------------------------------------------------------------------
# PARAMS CONFIG  (mirrors the main app)
# ---------------------------------------------------------------------------

PARAMS_CONFIG = {
    "Temperature":     {"col_prefix": "T",     "color": "#EF553B"},
    "Salinity":        {"col_prefix": "Sal",   "color": "#F5F5F5"},
    "pH":              {"col_prefix": "pH",    "color": "#00CC96"},
    "Pressure":        {"col_prefix": "Pres",  "color": "#AB63FA"},
    "Isolation":       {"col_prefix": "Iso",   "color": "#6ab7f1"},
    "Redox Potential": {"col_prefix": "Redox", "color": "#FF8C00"},
}

# Physical depth ordering of target environments (plume → seafloor)
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

# ---------------------------------------------------------------------------
# SCORING ENGINE  (verbatim from main app)
# ---------------------------------------------------------------------------

def calculate_suitability(site_min, site_max, target_min, target_max):
    """Per-parameter fit: max(Jaccard, Gaussian) in [0,1], or None if missing."""
    if any(pd.isna(x) for x in (site_min, site_max, target_min, target_max)):
        return None
    s_mid  = (site_min + site_max) / 2
    t_mid  = (target_min + target_max) / 2
    sigma  = max(target_max - target_min, 1.0)
    gaussian = np.exp(-((s_mid - t_mid) ** 2) / (2 * sigma ** 2))

    ov_min = max(site_min, target_min)
    ov_max = min(site_max, target_max)
    if ov_max > ov_min:
        intersection = ov_max - ov_min
        union = max(site_max, target_max) - min(site_min, target_min)
        return max(gaussian, intersection / union)
    return gaussian


def score_site_against_target(site, target_data, weights):
    """Weighted-mean suitability score for one site against one target."""
    fits, active_weights = {}, {}
    for param, info in PARAMS_CONFIG.items():
        if param not in weights:
            continue
        prefix  = info["col_prefix"]
        min_col = f"{prefix}_min"   if f"{prefix}_min"   in site        else f"{prefix}_score"
        max_col = f"{prefix}_max"   if f"{prefix}_max"   in site        else f"{prefix}_score"
        t_min_c = f"{prefix}_min"   if f"{prefix}_min"   in target_data else f"{prefix}_score"
        t_max_c = f"{prefix}_max"   if f"{prefix}_max"   in target_data else f"{prefix}_score"

        fit = calculate_suitability(
            site.get(min_col), site.get(max_col),
            target_data.get(t_min_c), target_data.get(t_max_c),
        )
        if fit is not None:
            fits[param]           = fit
            active_weights[param] = weights[param]

    w_sum = sum(active_weights.values())
    return sum(fits[p] * active_weights[p] for p in active_weights) / w_sum if w_sum else np.nan


@st.cache_data
def build_matrix(weights_tuple):
    """Build the full sites × targets matrix. Cached per weight combination."""
    weights = dict(weights_tuple)          # convert back from hashable tuple
    records = []
    for target_env in TARGET_ORDER:
        row = targets_df[targets_df["Target_env"] == target_env]
        if row.empty:
            continue
        target_data = row.iloc[0]
        col = {
            site["Site"]: score_site_against_target(site, target_data, weights)
            for _, site in analogues_df.iterrows()
        }
        col["_target"] = target_env
        records.append(col)

    matrix = pd.DataFrame(records).set_index("_target").T
    matrix.index.name = "Site"
    # Order rows by mean suitability so generalists appear at the top
    matrix = matrix.loc[matrix.mean(axis=1).sort_values(ascending=False).index]
    return matrix[TARGET_ORDER]            # enforce column order


# ---------------------------------------------------------------------------
# RENDER FUNCTION
# ---------------------------------------------------------------------------

def render_heatmap(matrix, weights):
    fig, ax = plt.subplots(figsize=(12, 9))
    fig.patch.set_facecolor("#0e1117")     # match Streamlit dark background
    ax.set_facecolor("#0e1117")

    sns.heatmap(
        matrix,
        cmap="viridis",
        vmin=0, vmax=1,
        annot=True, fmt=".2f",
        annot_kws={"size": 8, "color": "white"},
        linewidths=0.5, linecolor="#1e1e2e",
        cbar_kws={"label": "Suitability score", "shrink": 0.65},
        ax=ax,
    )

    # Re-colour annotations on bright cells for readability
    for text, (i, j) in zip(ax.texts, [
        (i, j) for i in range(matrix.shape[0]) for j in range(matrix.shape[1])
    ]):
        if pd.notna(matrix.iloc[i, j]) and matrix.iloc[i, j] > 0.6:
            text.set_color("black")

    # Red outlines on top-3 sites per target column
    top3_counts = {site: 0 for site in matrix.index}
    for col_idx, target in enumerate(matrix.columns):
        for site in matrix[target].nlargest(3).index:
            row_idx = matrix.index.get_loc(site)
            ax.add_patch(mpatches.Rectangle(
                (col_idx, row_idx), 1, 1,
                fill=False, edgecolor="#ff3030", lw=2.0, zorder=10,
            ))
            top3_counts[site] += 1

    # Style labels
    weight_label = (
        "equal weighting"
        if len(set(weights.values())) == 1
        else "custom weighting"
    )
    ax.set_title(
        f"Per-target suitability across candidate analogue sites\n"
        f"{weight_label} across six parameters  ·  red outlines = top-3 per target",
        color="white", fontsize=12, pad=14,
    )
    ax.set_xlabel("Enceladus target environment", color="white", fontsize=10, labelpad=10)
    ax.set_ylabel("Terrestrial analogue site",    color="white", fontsize=10, labelpad=10)
    ax.tick_params(colors="white")
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right", fontsize=8.5, color="white")
    plt.setp(ax.get_yticklabels(), rotation=0,  fontsize=8.5, color="white")

    cbar = ax.collections[0].colorbar
    cbar.ax.yaxis.set_tick_params(color="white")
    cbar.set_label("Suitability score", color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    plt.tight_layout()
    return fig, top3_counts


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

st.title("📊 AstroMatch · Sites × Targets Heatmap")
st.markdown(
    "Suitability scores for all 15 analogue sites against all 9 Enceladus target "
    "environments, computed with the same Gaussian–Jaccard engine as the main tool. "
    "Adjust weights in the sidebar and click **Generate** to rerun."
)

# --- Sidebar weights ---
st.sidebar.header("⚖️ Parameter Weights")
st.sidebar.info("Default: equal weighting (5). Adjust to explore sensitivity.")

weights = {}
for param, info in PARAMS_CONFIG.items():
    col1, col2 = st.sidebar.columns([4, 1])
    with col1:
        st.markdown(
            f"**<span style='color:{info['color']}'>{param}</span>**",
            unsafe_allow_html=True,
        )
    with col2:
        active = st.toggle(" ", value=True, key=f"tog_{param}",
                           label_visibility="collapsed")
    if active:
        weights[param] = st.sidebar.slider(
            param, 1, 10, 5, key=f"sld_{param}",
            label_visibility="collapsed",
        )
    st.sidebar.write("")

if not weights:
    st.warning("Enable at least one parameter in the sidebar.")
    st.stop()

# --- Generate button ---
if st.button("🚀 Generate Heatmap", type="primary"):
    with st.spinner("Computing suitability matrix…"):
        matrix = build_matrix(tuple(sorted(weights.items())))

    fig, top3_counts = render_heatmap(matrix, weights)
    st.pyplot(fig)

    # --- Key finding callout ---
    max_site  = max(top3_counts, key=top3_counts.get)
    max_count = top3_counts[max_site]
    st.info(
        f"**Key finding:** Under the current weighting, the most versatile site is "
        f"**{max_site}**, appearing in the top 3 for **{max_count} of 9** target "
        f"environments. "
        + ("No site dominates across all targets, supporting the portfolio framing."
           if max_count <= 4 else
           f"This exceeds the equal-weighting baseline result — try adjusting weights "
           f"to explore whether dominance persists.")
    )

    # --- Top-3 table ---
    with st.expander("Top-3 appearance counts per site"):
        counts_df = (
            pd.Series(top3_counts)
            .sort_values(ascending=False)
            .reset_index()
        )
        counts_df.columns = ["Site", "Top-3 appearances (out of 9)"]
        st.dataframe(counts_df, use_container_width=True, hide_index=True)

    # --- Download buttons ---
    st.divider()
    col_a, col_b = st.columns(2)

    with col_a:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=300, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        st.download_button(
            "📥 Download PNG (300 dpi)",
            data=buf.getvalue(),
            file_name="heatmap_sites_x_targets.png",
            mime="image/png",
        )

    with col_b:
        st.download_button(
            "📥 Download matrix CSV",
            data=matrix.to_csv().encode("utf-8"),
            file_name="heatmap_sites_x_targets.csv",
            mime="text/csv",
        )
else:
    st.info("👈 Set weights in the sidebar, then click **Generate Heatmap**.")
