import json
import os
import numpy as np
import pandas as pd
from sklearn.decomposition import FactorAnalysis
from sklearn.preprocessing import StandardScaler
from ..utils.claim_matrix import (
    _find_flagging_files,
    _load_flagging_results,
    _merge_flagging_results,
    build_claim_matrix,
    compute_mtmm_correlation_matrix,
)
from ..utils.plots import save_efa_loading_bar_chart, save_efa_loading_heatmap, save_efa_scree_plot

def _drop_constant_cols(claim_matrix: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Drop columns whose correlation with every other column is NaN (constant columns).

    Constant columns contribute no variance and cannot be factored.

    :param claim_matrix: Binary claim matrix from :func:`build_claim_matrix`.
    :return: Tuple of (filtered claim_matrix, retained column names).
    """
    corr = claim_matrix.corr(method="pearson")
    valid_cols = [c for c in corr.columns if not corr[c].isna().all()]
    return claim_matrix[valid_cols], valid_cols

def _scale_claim_matrix(claim_matrix: pd.DataFrame) -> np.ndarray:
    """
    Fill NaN with 0 and apply StandardScaler to the claim matrix.

    NaN arises when a claim was not evaluated by a particular judge; treating it
    as 0 (not flagged) is a conservative imputation consistent with the binary
    coding scheme.

    :param claim_matrix: Filtered claim matrix with no all-NaN columns.
    :return: Scaled array of shape (n_claims, n_cols).
    """
    X = claim_matrix.fillna(0.0).to_numpy(dtype=float)
    return StandardScaler().fit_transform(X)

def _run_parallel_analysis(
    X: np.ndarray,
    n_random: int,
    seed: int,
) -> dict:
    """
    Compare observed eigenvalues against the 95th percentile from random matrices.

    Generates ``n_random`` random normal matrices of the same shape as ``X``,
    computes their correlation matrices, extracts eigenvalues, and retains factors
    whose observed eigenvalue exceeds the simulated 95th-percentile threshold
    (Horn, 1965).

    :param X: Scaled claim matrix of shape (n_claims, n_cols).
    :param n_random: Number of random matrices to simulate.
    :param seed: Seed for the random number generator.
    :return: Dict with observed_eigenvalues, simulated_95th_percentile,
             and n_factors_retained.
    """
    n_cols = X.shape[1]
    rng = np.random.default_rng(seed)

    cov_real = np.corrcoef(X.T)
    # eigvalsh exploits symmetry of the correlation matrix for speed and
    # numerical stability; eigvals on a symmetric matrix can return small
    # spurious imaginary components.
    observed_evals = np.sort(np.linalg.eigvalsh(cov_real))[::-1]

    sim_evals = np.zeros((n_random, n_cols))
    for i in range(n_random):
        rand_data = rng.standard_normal(X.shape)
        sim_evals[i] = np.sort(np.linalg.eigvalsh(np.corrcoef(rand_data.T)))[::-1]  # symmetric solver

    threshold_95 = np.percentile(sim_evals, 95, axis=0)
    n_factors = int((observed_evals > threshold_95).sum())

    return {
        "observed_eigenvalues": observed_evals.tolist(),
        "simulated_95th_percentile": threshold_95.tolist(),
        "n_factors_retained": n_factors,
    }

def _fit_efa_loadings(X: np.ndarray, n_factors: int, seed: int) -> np.ndarray:
    """
    Fit a factor model with varimax rotation using sklearn's FactorAnalysis.

    FactorAnalysis estimates unique variances via EM and analyses only shared
    variance, unlike PCA which decomposes total variance.

    :param X: Scaled claim matrix of shape (n_claims, n_cols).
    :param n_factors: Number of factors to extract.
    :param seed: Random seed for FactorAnalysis initialisation.
    :return: Rotated loadings of shape (n_cols, n_factors).
    """
    fa = FactorAnalysis(n_components=n_factors, rotation="varimax", random_state=seed)
    fa.fit(X)
    return fa.components_.T

def _check_simple_loading_structure(
    loadings: np.ndarray,
    col_names: list[str],
    cross_loading_threshold: float,
) -> dict:
    """
    Assess whether each variable shows a simple loading pattern.

    A variable is simple if its maximum absolute loading is on a single factor and
    all remaining absolute loadings fall below ``cross_loading_threshold``.

    :param loadings: Rotated loadings of shape (n_vars, n_factors).
    :param col_names: Variable names corresponding to rows of loadings.
    :param cross_loading_threshold: Maximum permitted cross-loading magnitude.
    :return: Dict with per-variable results and overall simple_structure_fraction.
    """
    n_vars, n_factors = loadings.shape
    per_variable = []
    n_simple = 0
    for i, name in enumerate(col_names):
        abs_row = np.abs(loadings[i])
        primary_factor = int(np.argmax(abs_row))
        primary_loading = float(abs_row[primary_factor])
        cross_loadings = [float(abs_row[j]) for j in range(n_factors) if j != primary_factor]
        is_simple = all(c < cross_loading_threshold for c in cross_loadings)
        if is_simple:
            n_simple += 1
        per_variable.append(
            {
                "variable": name,
                "primary_factor": primary_factor + 1,
                "primary_loading": round(primary_loading, 4),
                "max_cross_loading": round(max(cross_loadings), 4) if cross_loadings else 0.0,
                "is_simple": is_simple,
            }
        )
    return {
        "per_variable": per_variable,
        "n_simple": n_simple,
        "n_vars": n_vars,
        "simple_structure_fraction": round(n_simple / n_vars, 4) if n_vars > 0 else 0.0,
    }


def _compute_variance_explained(loadings: np.ndarray, X: np.ndarray, col_names: list[str]) -> dict:
    """
    Compute per-factor variance explained and per-variable communalities.

    Communality for each variable is the sum of squared loadings across all factors,
    representing the proportion of that variable's variance captured by the factor model.
    Per-factor variance is the sum of squared loadings across all variables for that factor.
    Total variance equals the trace of the correlation matrix (n_vars for standardised data).

    :param loadings: Rotated loadings of shape (n_vars, n_factors).
    :param X: Scaled claim matrix of shape (n_claims, n_vars).
    :param col_names: Variable names corresponding to rows of loadings.
    :return: Dict with total_variance, factor_variance, factor_variance_pct,
             total_explained_pct, and communalities (keyed by variable name).
    """
    total_variance = float(np.trace(np.corrcoef(X.T)))
    factor_variance = np.sum(loadings ** 2, axis=0)
    communalities = np.sum(loadings ** 2, axis=1)
    return {
        "total_variance": round(total_variance, 4),
        "factor_variance": [round(float(v), 4) for v in factor_variance],
        "factor_variance_pct": [round(float(v) / total_variance * 100, 2) for v in factor_variance],
        "total_explained_pct": round(float(factor_variance.sum()) / total_variance * 100, 2),
        "communalities": {name: round(float(h2), 4) for name, h2 in zip(col_names, communalities)},
    }

def save_efa_results(results: dict, output_dir: str) -> None:
    """
    Write EFA outputs to disk.

    Produces ``efa_results.json`` with parallel analysis and loading structure
    statistics, and ``efa_loadings.csv`` with the rotated factor loadings matrix.

    :param results: Output dict from :func:`run_efa`.
    :param output_dir: Directory to write outputs into.
    """
    os.makedirs(output_dir, exist_ok=True)

    json_data = {
        "n_claims": results["n_claims"],
        "n_vars": results["n_vars"],
        "n_factors_retained": results["parallel"]["n_factors_retained"],
        "observed_eigenvalues": results["parallel"]["observed_eigenvalues"],
        "simulated_95th_percentile": results["parallel"]["simulated_95th_percentile"],
        "simple_structure": results.get("simple_structure"),
        "variance_explained": results.get("variance_explained"),
    }
    json_path = os.path.join(output_dir, "efa_results.json")
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)

    if results.get("loadings_df") is not None:
        csv_path = os.path.join(output_dir, "efa_loadings.csv")
        results["loadings_df"].to_csv(csv_path)

def run_efa(
    flagging_results_path: str,
    output_dir: str,
    n_random: int = 500,
    seed: int = 42,
    cross_loading_threshold: float = 0.40,
    verbose: bool = False,
) -> dict:
    """
    Run EFA and parallel analysis on a flagging_results.json file or directory tree.

    Applies the same data pipeline as MTMM: load/merge flagging results and build
    the binary claim matrix. The matrix is then standardised and used for parallel
    analysis (Horn, 1965) to determine the number of retained factors. Factor
    loadings are extracted via sklearn's FactorAnalysis with varimax rotation,
    which estimates unique variances and analyses only shared variance. Each
    variable is assessed for simple loading structure.

    :param flagging_results_path: Path to a flagging_results.json file, or to a
        directory that will be searched recursively for such files.
    :param output_dir: Directory to write JSON, CSV, and plot outputs.
    :param n_random: Number of random matrices for the parallel analysis simulation.
    :param seed: Seed for reproducible random matrix generation and FactorAnalysis.
    :param cross_loading_threshold: Maximum permitted cross-loading for simple
        structure classification.
    :param verbose: Print progress messages when True.
    :return: Dict with keys: claim_matrix, parallel, loadings_df, simple_structure,
             variance_explained, n_claims, n_vars, col_names.
    """
    if os.path.isdir(flagging_results_path):
        paths = _find_flagging_files(flagging_results_path)
        if not paths:
            raise ValueError(
                f"No flagging_results.json files found under: {flagging_results_path}"
            )
        if verbose:
            print(f"Found {len(paths)} flagging_results.json files under {flagging_results_path}")
        flagging_results = _merge_flagging_results(paths)
    else:
        flagging_results = _load_flagging_results(flagging_results_path)

    if verbose:
        print(f"Loaded {len(flagging_results.get('claim_evaluations', []))} claim evaluations")

    claim_matrix = build_claim_matrix(flagging_results)
    if claim_matrix.empty:
        raise ValueError("No claim evaluations found in flagging_results.json")

    claim_matrix, col_names = _drop_constant_cols(claim_matrix)
    n_claims, n_vars = claim_matrix.shape

    if n_vars == 0:
        raise ValueError(
            "All columns are constant after removing invariant columns; cannot perform EFA."
        )

    if verbose:
        print(f"Claim matrix: {n_claims} claims × {n_vars} variables (after dropping constant columns)")

    X = _scale_claim_matrix(claim_matrix)
    parallel = _run_parallel_analysis(X, n_random, seed)
    n_factors = parallel["n_factors_retained"]

    if verbose:
        print(f"Parallel analysis suggests retaining {n_factors} factor(s).")

    loadings_df = None
    simple_structure = None
    variance_explained = None

    if n_factors > 0:
        loadings = _fit_efa_loadings(X, n_factors, seed)
        factor_cols = [f"Factor {i + 1}" for i in range(n_factors)]
        loadings_df = pd.DataFrame(loadings, index=col_names, columns=factor_cols)
        simple_structure = _check_simple_loading_structure(loadings, col_names, cross_loading_threshold)
        variance_explained = _compute_variance_explained(loadings, X, col_names)
        if verbose:
            frac = simple_structure["simple_structure_fraction"]
            print(f"Simple structure fraction: {frac:.3f} ({simple_structure['n_simple']}/{n_vars} variables)")
            total = variance_explained["total_variance"]
            for k, (v, pct) in enumerate(zip(variance_explained["factor_variance"], variance_explained["factor_variance_pct"])):
                print(f"  Factor {k + 1}: {v:.2f} / {total:.2f} = {pct:.1f}%")
            print(f"  Total explained: {variance_explained['total_explained_pct']:.1f}%")
            for name, h2 in variance_explained["communalities"].items():
                print(f"  {name}: {h2:.3f}")
    else:
        if verbose:
            print("No factors retained; skipping loading extraction.")

    results = {
        "claim_matrix": claim_matrix,
        "parallel": parallel,
        "loadings_df": loadings_df,
        "simple_structure": simple_structure,
        "variance_explained": variance_explained,
        "n_claims": n_claims,
        "n_vars": n_vars,
        "col_names": col_names,
    }

    os.makedirs(output_dir, exist_ok=True)
    save_efa_results(results, output_dir)
    save_efa_scree_plot(
        observed_evals=parallel["observed_eigenvalues"],
        threshold_95=parallel["simulated_95th_percentile"],
        n_factors_retained=n_factors,
        output_path=os.path.join(output_dir, "efa_scree_plot.png"),
    )
    if loadings_df is not None:
        save_efa_loading_bar_chart(
            loadings_df=loadings_df,
            cross_loading_threshold=cross_loading_threshold,
            output_path=os.path.join(output_dir, "efa_loadings_bar.png"),
        )
        save_efa_loading_heatmap(
            loadings_df=loadings_df,
            cross_loading_threshold=cross_loading_threshold,
            output_path=os.path.join(output_dir, "efa_loadings_heatmap.png"),
        )

    print(f"Results written to {output_dir}")

    return results
