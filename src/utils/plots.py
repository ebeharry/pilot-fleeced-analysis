import os
import re
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.size"] = 12
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

_DECEPTION_TYPE_ORDER = [
    "Falsehood",
    "Omission",
    "Equivocation",
    "Paltering",
]

_MODEL_COLORS = [
    "#000000", "#E69F00", "#56B4E9", "#009E73",
    "#F0E442", "#0072B2", "#D55E00", "#CC79A7",
]

_MODEL_DISPLAY_NAMES = {
    "gpt-4.1": "GPT-4.1",
    "gpt-4-1": "GPT-4.1",
    "deepseek-v4-flash": "DeepSeek-V4-Flash",
    "deepseek-chat": "DeepSeek-V4-Flash",
    "mistral-small-2506": "Mistral-Small-2506",
    "llama-4-scout": "Llama-4-Scout",
    "gemini-2.5-flash-lite": "Gemini-2.5-Flash-Lite",
    "claude-3-haiku": "Claude-3-Haiku",
    "gpt-4o": "GPT-4o",
}

def _format_model_name(model: str) -> str:
    key = model.lower()
    for pattern, display in _MODEL_DISPLAY_NAMES.items():
        if pattern in key:
            return display
    return model.split("/")[-1]

def save_dstudy_plot(
    dstudy_df,
    plot_dir: str,
    scenario: str,
    target: float = 0.85,
    filename: str | None = None,
) -> None:
    """
    Three-panel line plot of G-coefficient vs n for each varied facet.

    One subplot per facet (n_judges, n_convs, n_subtypes). A horizontal dashed
    line marks the target reliability threshold.

    :param dstudy_df: DataFrame with columns [varied_facet, n, g_coefficient].
    :param plot_dir: Directory to write the output file.
    :param scenario: Scenario label used in the title and default filename.
    :param target: G-coefficient reference threshold.
    :param filename: Output filename; defaults to dstudy_{scenario}.png.
    """
    os.makedirs(plot_dir, exist_ok=True)

    facet_labels = {
        "n_judges": "Number of Judges",
        "n_convs": "Conversations per Sub-scenario",
        "n_subtypes": "Number of Sub-scenarios",
    }
    facets = [f for f in ["n_judges", "n_convs", "n_subtypes"]
              if f in dstudy_df["varied_facet"].values]

    fig, axes = plt.subplots(1, len(facets), figsize=(5 * len(facets), 4), sharey=True)
    if len(facets) == 1:
        axes = [axes]

    for ax, facet in zip(axes, facets):
        sub = dstudy_df[dstudy_df["varied_facet"] == facet].sort_values("n")
        ax.plot(sub["n"], sub["g_coefficient"], marker="o", linewidth=2, markersize=5, color="#000000")
        ax.axhline(target, color="#D55E00", linestyle="--", linewidth=1.2,
                   label=f"Target (Eρ² = {target})")
        ax.set_xlabel(facet_labels.get(facet, facet), fontsize=12)
        ax.set_ylim(0, 1.05)
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.yaxis.grid(True, linestyle="--", alpha=0.5)
        ax.set_axisbelow(True)
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        if ax is axes[0]:
            ax.set_ylabel("Generalizability Coefficient (Eρ²)", fontsize=12)
            ax.legend(fontsize=12, loc="lower right")

    title = f"D-Study: {scenario.replace('_', ' ').title()}"
    fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout()

    out_filename = filename or f"dstudy_{scenario}.png"
    fig.savefig(os.path.join(plot_dir, out_filename), dpi=300, bbox_inches="tight")
    plt.close(fig)

def save_mtmm_heatmap(
    corr_matrix,
    judges: list[str],
    subtypes: list[str],
    output_path: str,
) -> None:
    """
    Save a heatmap of the full MTMM correlation matrix with method-block grid lines.

    Columns and rows are ordered by judge first, then subtype, so each square
    block along the diagonal corresponds to one judge's heterotrait-monomethod
    correlations. Bold grid lines delineate method blocks.

    :param corr_matrix: Symmetric correlation DataFrame from
        :func:`~src.analysis.mtmm.compute_mtmm_correlation_matrix`.
    :param judges: Ordered list of judge identifiers.
    :param subtypes: Ordered list of subtype names.
    :param output_path: Full file path for the saved PNG.
    """
    ordered_cols = [
        f"{j}::{s}"
        for j in judges
        for s in subtypes
        if f"{j}::{s}" in corr_matrix.columns
    ]
    matrix = corr_matrix.loc[ordered_cols, ordered_cols]
    n = len(ordered_cols)
    n_sub = len(subtypes)

    tick_labels = [c.split("::", 1)[1][:4] for c in ordered_cols]

    fig_size = max(6, n * 0.55 + 2)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))

    mask_upper = np.triu(np.ones((n, n), dtype=bool), k=1)

    sns.heatmap(
        matrix,
        mask=mask_upper,
        ax=ax,
        vmin=-1,
        vmax=1,
        center=0,
        cmap="RdBu_r",
        annot=n <= 12,
        fmt=".2f",
        linewidths=0.3,
        linecolor="lightgray",
        xticklabels=tick_labels,
        yticklabels=tick_labels,
        cbar_kws={"label": "Pearson r", "shrink": 0.8},
    )

    for i in range(1, len(judges)):
        ax.axhline(i * n_sub, color="black", linewidth=2)
        ax.axvline(i * n_sub, color="black", linewidth=2)

    for i, judge in enumerate(judges):
        if not any(c.startswith(f"{judge}::") for c in ordered_cols):
            continue
        short = _format_model_name(judge)
        mid = i * n_sub + n_sub / 2
        ax.text(mid, n + 1.2, short, ha="center", va="top", fontsize=12,
                fontweight="bold", rotation=30, transform=ax.transData, clip_on=False)
        ax.text(-0.8, mid, short, ha="right", va="center", fontsize=12,
                fontweight="bold", transform=ax.transData, clip_on=False)

    for j, subtype in enumerate(subtypes):
        ax.text(
            j + 0.5, n + 0.3, subtype[:4], ha="center", va="bottom",
            fontsize=8, color="gray", transform=ax.transData, clip_on=False,
        )
        ax.text(
            -0.3, j + 0.5, subtype[:4], ha="right", va="center",
            fontsize=8, color="gray", transform=ax.transData, clip_on=False,
        )

    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=12)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=12)
    ax.set_title("MTMM Correlation Matrix", fontweight="bold", fontsize=12)

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

def save_mtmm_summary_bar(
    classified: dict[str, list[float]],
    output_path: str,
) -> None:
    """
    Save a bar chart comparing mean correlations across the three MTMM block types.

    Bar height is the mean correlation; error bars show ±1 SD.

    :param classified: Output of
        :func:`~src.analysis.mtmm.classify_mtmm_correlations`.
    :param output_path: Full file path for the saved PNG.
    """
    keys = [
        "monotrait_heteromethod",
        "heterotrait_monomethod",
        "heterotrait_heteromethod",
    ]
    labels = [
        "Monotrait\nHeteromethod",
        "Heterotrait\nMonomethod",
        "Heterotrait\nHeteromethod",
    ]
    means = [
        float(np.mean(classified[k])) if classified.get(k) else 0.0
        for k in keys
    ]
    sds = [
        float(np.std(classified[k])) if classified.get(k) else 0.0
        for k in keys
    ]

    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(keys))
    colors = ["#000000", "#D55E00", "#009E73"]
    ax.bar(x, means, yerr=sds, capsize=5, color=colors, alpha=0.85, width=0.5,
           error_kw={"elinewidth": 1.2})
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel("Mean Pearson r", fontsize=12)
    ax.set_title("MTMM Block Correlations", fontweight="bold", fontsize=12)
    ax.set_ylim(-1, 1)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

def _efa_sort_key(var: str) -> tuple:
    parts = var.split("::", 1)
    judge = parts[0] if parts else var
    subtype = parts[1] if len(parts) > 1 else ""
    type_order = _DECEPTION_TYPE_ORDER.index(subtype) if subtype in _DECEPTION_TYPE_ORDER else len(_DECEPTION_TYPE_ORDER)
    return (type_order, _format_model_name(judge))

def _efa_var_label(var: str) -> str:
    parts = var.split("::", 1)
    judge = parts[0] if parts else var
    subtype = parts[1] if len(parts) > 1 else var
    return f"{_format_model_name(judge)} · {subtype}"

def save_efa_scree_plot(
    observed_evals: list[float],
    threshold_95: list[float],
    n_factors_retained: int,
    output_path: str,
) -> None:
    """
    Save a parallel analysis scree plot.

    Plots observed eigenvalues alongside the 95th-percentile simulated baseline.
    A shaded region marks the retained factors and a horizontal reference line
    at y=1.0 shows the Kaiser criterion for comparison.

    :param observed_evals: Observed eigenvalues in descending order.
    :param threshold_95: 95th-percentile simulated eigenvalues, same length.
    :param n_factors_retained: Number of factors retained by parallel analysis.
    :param output_path: Full file path for the saved PNG.
    """
    n = len(observed_evals)
    x = np.arange(1, n + 1)

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(x, observed_evals, "o-", color="#000000", linewidth=2,
            label="Observed eigenvalues")
    ax.plot(x, threshold_95, "--", color="#999999", linewidth=1.5,
            label="95th pct random (parallel analysis)")
    if n_factors_retained > 0:
        ax.axvline(n_factors_retained + 0.5, color="#D55E00", linewidth=1.5,
                   linestyle=":", label=f"Retain {n_factors_retained} factor(s)")

    ax.set_xlabel("Factor number", fontsize=12)
    ax.set_ylabel("Eigenvalue", fontsize=12)
    ax.set_title("Parallel Analysis Scree Plot", fontweight="bold", fontsize=12)
    ax.set_xticks(x)
    ax.legend(fontsize=8)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

def save_efa_loading_bar_chart(
    loadings_df: "pd.DataFrame",
    cross_loading_threshold: float,
    output_path: str,
) -> None:
    """
    Save a per-factor horizontal bar chart of factor loadings.

    Each subplot shows one factor. Bars are coloured by deception subtype
    (the trait dimension of each judge::subtype column). Dashed reference lines
    at ±cross_loading_threshold mark the simple-structure boundary.

    :param loadings_df: Rotated loadings DataFrame, shape (n_vars, n_factors),
        with index entries in ``"{judge}::{subtype}"`` format.
    :param cross_loading_threshold: Threshold value drawn as dashed reference lines.
    :param output_path: Full file path for the saved PNG.
    """
    import pandas as pd

    import matplotlib.patches as mpatches

    sorted_vars = sorted(loadings_df.index.tolist(), key=_efa_sort_key)
    unique_models = list(dict.fromkeys(
        v.split("::", 1)[0] for v in sorted_vars
    ))
    model_color = {m: _MODEL_COLORS[i % len(_MODEL_COLORS)] for i, m in enumerate(unique_models)}

    factor_cols = list(loadings_df.columns)
    n_factors = len(factor_cols)
    fig_width = max(4.5 * n_factors, 6)
    fig, axes = plt.subplots(1, n_factors, figsize=(fig_width, max(4, len(loadings_df) * 0.3)),
                             sharey=True)
    if n_factors == 1:
        axes = [axes]

    for k, ax in enumerate(axes):
        col = factor_cols[k]
        vals = loadings_df.loc[sorted_vars, col]
        colors = [model_color[v.split("::", 1)[0]] for v in vals.index]
        tick_labels = [_efa_var_label(v) for v in vals.index]
        ax.barh(tick_labels, vals.values, color=colors, edgecolor="white", height=0.6)
        ax.axvline(0, color="#222222", linewidth=0.8)
        ax.axvline(cross_loading_threshold, color="#999999", linewidth=0.8,
                   linestyle="--", alpha=0.7,
                   label=f"|λ| = {cross_loading_threshold}")
        ax.axvline(-cross_loading_threshold, color="#999999", linewidth=0.8,
                   linestyle="--", alpha=0.7)
        ax.set_xlim(-1, 1)
        ax.set_xlabel("Loading", fontsize=10)
        ax.set_title(col, fontsize=11, fontweight="bold")
        if k == 0:
            ax.legend(fontsize=7)

    handles = [
        mpatches.Patch(color=model_color[m], label=_format_model_name(m))
        for m in unique_models
    ]
    n_legend_cols = min(len(unique_models), 4)
    fig.legend(handles=handles, loc="lower center", ncol=n_legend_cols, fontsize=8,
               bbox_to_anchor=(0.5, -0.04))
    fig.suptitle(
        "Factor Loadings (varimax rotation)\n"
        "High loading = column captures this factor; "
        "Low cross-loading = discriminant validity",
        fontsize=10,
    )

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

def save_efa_loading_heatmap(
    loadings_df: "pd.DataFrame",
    cross_loading_threshold: float,
    output_path: str,
    figsize: tuple[float, float] | None = None,
) -> None:
    """
    Save a heatmap of the full rotated factor loadings matrix.

    Rows are judge::subtype variables grouped and sorted by primary factor.
    Cells are annotated with loading values; cells whose absolute value exceeds
    ``cross_loading_threshold`` on the primary factor are outlined to mark
    simple structure. A diverging colormap centred at 0 makes positive and
    negative loadings immediately readable.

    :param loadings_df: Rotated loadings DataFrame, shape (n_vars, n_factors),
        with index entries in ``"{judge}::{subtype}"`` format.
    :param cross_loading_threshold: Threshold used to outline primary loadings.
    :param output_path: Full file path for the saved PNG.
    :param figsize: Optional (width, height) override; defaults to auto-sizing.
    """
    import pandas as pd

    primary_factor = loadings_df.abs().idxmax(axis=1)
    sorted_idx = sorted(
        loadings_df.index.tolist(),
        key=lambda v: (primary_factor[v], -abs(loadings_df.loc[v, primary_factor[v]]), _efa_sort_key(v)),
    )
    df_sorted = loadings_df.loc[sorted_idx]

    n_vars, n_factors = df_sorted.shape
    fig_h = max(5, n_vars * 0.38)
    fig_w = max(5, n_factors * 1.1)
    fig, ax = plt.subplots(figsize=figsize or (fig_w, fig_h))

    sns.heatmap(
        df_sorted,
        ax=ax,
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        center=0,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 7},
        linewidths=0.4,
        linecolor="#dddddd",
        cbar_kws={"label": "Loading", "shrink": 0.6},
    )

    for i, var in enumerate(df_sorted.index):
        pf = primary_factor[var]
        j = list(df_sorted.columns).index(pf)
        if abs(df_sorted.loc[var, pf]) >= cross_loading_threshold:
            ax.add_patch(plt.Rectangle(
                (j, i), 1, 1,
                fill=False, edgecolor="#000000", linewidth=1.8,
            ))

    short_labels = [_efa_var_label(v) for v in df_sorted.index]

    ax.set_yticklabels(short_labels, fontsize=8, rotation=0)
    ax.set_xticklabels(df_sorted.columns, fontsize=9, rotation=30, ha="right")
    ax.set_title(
        "Factor Loadings Heatmap",
        fontsize=10,
        fontweight="bold",
    )

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
