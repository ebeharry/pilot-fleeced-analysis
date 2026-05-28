from itertools import combinations
import json 
import os
import numpy as np
import pandas as pd
import scipy.stats
from ..utils.claim_matrix import (
    SUBTYPES,
    _find_flagging_files,
    _load_flagging_results,
    _merge_flagging_results,
    build_claim_matrix,
    compute_mtmm_correlation_matrix,
)
from ..utils.plots import save_mtmm_heatmap, save_mtmm_summary_bar

CONV_THRESHOLD = 0.3

def _parse_column(col: str) -> tuple[str, str]:
    """
    Split a ``"{judge}::{subtype}"`` column name into its components.

    :param col: Column name with ``::`` separator.
    :return: Tuple of (judge, subtype).
    """
    judge, subtype = col.split("::", 1)
    return judge, subtype

def classify_mtmm_correlations(
    corr_matrix: pd.DataFrame,
) -> dict[str, list[float]]:
    """
    Categorise each off-diagonal upper-triangle correlation into one of the
    three canonical MTMM block types.

    - monotrait_heteromethod: same subtype, different judge
    - heterotrait_monomethod: different subtype, same judge
    - heterotrait_heteromethod: different subtype, different judge

    NaN correlations (from constant columns) are silently skipped.

    :param corr_matrix: Output of :func:`compute_mtmm_correlation_matrix`.
    :return: Dict mapping block-type name to list of correlation values.
    """
    result: dict[str, list[float]] = {
        "monotrait_heteromethod": [],
        "heterotrait_monomethod": [],
        "heterotrait_heteromethod": [],
    }
    cols = list(corr_matrix.columns)
    n = len(cols)
    for i in range(n):
        for j in range(i + 1, n):
            r = corr_matrix.iloc[i, j]
            if pd.isna(r):
                continue
            judge_i, subtype_i = _parse_column(cols[i])
            judge_j, subtype_j = _parse_column(cols[j])
            same_judge = judge_i == judge_j
            same_subtype = subtype_i == subtype_j
            if same_judge:
                result["heterotrait_monomethod"].append(float(r))
            elif same_subtype:
                result["monotrait_heteromethod"].append(float(r))
            else:
                result["heterotrait_heteromethod"].append(float(r))
    return result

def _hthm_row_col_neighbours(corr_matrix: pd.DataFrame, col_i: str, col_j: str) -> list[float]:
    """
    Return HTHM values sharing a row or column with the MTHM coefficient at (col_i, col_j).

    For each endpoint, collect r(endpoint, col_k) where judge_k differs from the
    endpoint's judge and subtype_k differs from the endpoint's subtype. Deduplicates
    pairs across both endpoints.

    :param corr_matrix: Full MTMM correlation matrix.
    :param col_i: First column of the MTHM pair (format ``"{judge}::{subtype}"``).
    :param col_j: Second column of the MTHM pair.
    :return: List of HTHM correlation values neighbouring this MTHM entry.
    """
    judge_i, subtype_i = _parse_column(col_i)
    judge_j, subtype_j = _parse_column(col_j)
    cols = list(corr_matrix.columns)
    pairs_seen: set[tuple[str, str]] = set()
    result = []
    for anchor, judge_a, subtype_a in [
        (col_i, judge_i, subtype_i),
        (col_j, judge_j, subtype_j),
    ]:
        for col_k in cols:
            if col_k == anchor:
                continue
            judge_k, subtype_k = _parse_column(col_k)
            if judge_k != judge_a and subtype_k != subtype_a:
                pair = (min(anchor, col_k), max(anchor, col_k))
                if pair not in pairs_seen:
                    pairs_seen.add(pair)
                    r = corr_matrix.loc[anchor, col_k]
                    if not pd.isna(r):
                        result.append(float(r))
    return result

def _htmonm_local_values(corr_matrix: pd.DataFrame, col_i: str, col_j: str) -> list[float]:
    """
    Return HTMonoM values from the method blocks of both endpoints of an MTHM coefficient.

    For each endpoint, collect r(endpoint, col_k) where judge_k equals the endpoint's
    judge and subtype_k differs (same-judge, different-subtype pairs).

    :param corr_matrix: Full MTMM correlation matrix.
    :param col_i: First column of the MTHM pair.
    :param col_j: Second column of the MTHM pair.
    :return: List of HTMonoM correlation values from both method blocks.
    """
    judge_i, subtype_i = _parse_column(col_i)
    judge_j, subtype_j = _parse_column(col_j)
    cols = list(corr_matrix.columns)
    result = []
    for anchor, judge_a, subtype_a in [
        (col_i, judge_i, subtype_i),
        (col_j, judge_j, subtype_j),
    ]:
        for col_k in cols:
            if col_k == anchor:
                continue
            judge_k, subtype_k = _parse_column(col_k)
            if judge_k == judge_a and subtype_k != subtype_a:
                r = corr_matrix.loc[anchor, col_k]
                if not pd.isna(r):
                    result.append(float(r))
    return result

def assess_construct_validity(classified: dict[str, list[float]], corr_matrix: pd.DataFrame) -> dict:
    """
    Assess construct validity following Campbell and Fiske (1959).

    :param classified: Output of :func:`classify_mtmm_correlations`.
    :param corr_matrix: Full MTMM correlation matrix from
        :func:`compute_mtmm_correlation_matrix`, used for per-coefficient
        structural comparisons.
    :return: Dict with mean values per block, boolean validity flags, the
             fraction of individual MTHM correlations above the HTHM mean,
             a threshold gate (>= CONV_THRESHOLD), a one-sample t-test
             significance flag, and exact per-coefficient structural checks.
    """
    mthm = classified.get("monotrait_heteromethod", [])
    htmonm = classified.get("heterotrait_monomethod", [])
    hthm = classified.get("heterotrait_heteromethod", [])

    mean_mthm = float(np.mean(mthm)) if mthm else float("nan")
    mean_htmonm = float(np.mean(htmonm)) if htmonm else float("nan")
    mean_hthm = float(np.mean(hthm)) if hthm else float("nan")

    mthm_is_positive = bool(mean_mthm > 0) if mthm else False
    mthm_all_positive = bool(all(r > 0 for r in mthm)) if mthm else False
    mthm_pct_positive = float(sum(1 for r in mthm if r > 0) / len(mthm)) if mthm else float("nan")
    mean_mthm_exceeds_mean_hthm = bool(mean_mthm > mean_hthm) if (mthm and hthm) else False
    mean_mthm_exceeds_mean_htmonm = bool(mean_mthm > mean_htmonm) if (mthm and htmonm) else False

    if mthm and hthm:
        pct = float(sum(1 for r in mthm if r > mean_hthm) / len(mthm))
    else:
        pct = float("nan")

    mthm_meets_threshold = bool(mean_mthm >= CONV_THRESHOLD) if mthm else False

    if len(mthm) >= 2:
        t_result = scipy.stats.ttest_1samp(mthm, 0, alternative="greater")
        mthm_significant_gt_zero = bool(t_result.pvalue < 0.05)
    else:
        mthm_significant_gt_zero = False

    block_ordering_holds = bool(1.0 > mean_mthm > mean_htmonm > mean_hthm) if (mthm and htmonm and hthm) else False

    mthm_exceed_row_col_hthm_flags = []
    mthm_exceed_local_htmonm_flags = []
    cols = list(corr_matrix.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            judge_ci, subtype_ci = _parse_column(cols[i])
            judge_cj, subtype_cj = _parse_column(cols[j])
            if judge_ci == judge_cj or subtype_ci != subtype_cj:
                continue
            r_val = corr_matrix.iloc[i, j]
            if pd.isna(r_val):
                continue
            r_val = float(r_val)
            hthm_nb = _hthm_row_col_neighbours(corr_matrix, cols[i], cols[j])
            htmonm_nb = _htmonm_local_values(corr_matrix, cols[i], cols[j])
            mthm_exceed_row_col_hthm_flags.append(bool(hthm_nb and all(r_val > v for v in hthm_nb)))
            mthm_exceed_local_htmonm_flags.append(bool(htmonm_nb and all(r_val > v for v in htmonm_nb)))

    mthm_all_exceed_row_col_hthm = bool(mthm_exceed_row_col_hthm_flags and all(mthm_exceed_row_col_hthm_flags))
    mthm_all_exceed_local_htmonm = bool(mthm_exceed_local_htmonm_flags and all(mthm_exceed_local_htmonm_flags))
    if mthm_exceed_row_col_hthm_flags:
        mthm_pct_exceed_row_col_hthm = float(sum(mthm_exceed_row_col_hthm_flags) / len(mthm_exceed_row_col_hthm_flags))
    else:
        mthm_pct_exceed_row_col_hthm = float("nan")
    if mthm_exceed_local_htmonm_flags:
        mthm_pct_exceed_local_htmonm = float(sum(mthm_exceed_local_htmonm_flags) / len(mthm_exceed_local_htmonm_flags))
    else:
        mthm_pct_exceed_local_htmonm = float("nan")

    return {
        "mean_mthm": mean_mthm,
        "mean_htmonm": mean_htmonm,
        "mean_hthm": mean_hthm,
        "mthm_is_positive": mthm_is_positive,
        "mthm_all_positive": mthm_all_positive,
        "mthm_pct_positive": mthm_pct_positive,
        "mthm_exceeds_hthm": mean_mthm_exceeds_mean_hthm,
        "mthm_exceeds_htmonm": mean_mthm_exceeds_mean_htmonm,
        "mthm_pct_above_hthm_mean": pct,
        "mthm_meets_threshold": mthm_meets_threshold,
        "mthm_significant_gt_zero": mthm_significant_gt_zero,
        "mthm_all_exceed_row_col_hthm": mthm_all_exceed_row_col_hthm,
        "mthm_pct_exceed_row_col_hthm": mthm_pct_exceed_row_col_hthm,
        "mthm_all_exceed_local_htmonm": mthm_all_exceed_local_htmonm,
        "mthm_pct_exceed_local_htmonm": mthm_pct_exceed_local_htmonm,
        "block_ordering_holds": block_ordering_holds,
    }

def _get_heterotrait_monomethod_block(
    corr_matrix: pd.DataFrame, judge: str, subtypes: list[str]
) -> np.ndarray:
    """
    Extract the square subtype-intercorrelation block for a single judge.

    :param corr_matrix: Full MTMM correlation matrix.
    :param judge: Judge identifier.
    :param subtypes: Ordered list of subtype names.
    :return: NumPy array of shape (n_subtypes, n_subtypes).
    """
    cols = [f"{judge}::{s}" for s in subtypes]
    return corr_matrix.loc[cols, cols].values

def check_trait_pattern_consistency(
    corr_matrix: pd.DataFrame, judges: list[str], subtypes: list[str]
) -> dict:
    """
    Assess whether trait intercorrelation patterns are consistent across judges.

    For each judge, the upper-triangle of the heterotrait-monomethod block is
    extracted as a vector of C(n_subtypes, 2) values. Pairwise Spearman
    correlations between these vectors measure method-block consistency.

    :param corr_matrix: Full MTMM correlation matrix.
    :param judges: List of judge identifiers present in corr_matrix.
    :param subtypes: Ordered list of subtype names.
    :return: Dict with ``pairwise_spearman`` (judge-pair keys) and
             ``mean_consistency`` (mean Spearman r across all pairs).
    """
    triu_idx = np.triu_indices(len(subtypes), k=1)
    judge_vectors: dict[str, np.ndarray] = {}
    for judge in judges:
        cols = [f"{judge}::{s}" for s in subtypes]
        if not all(c in corr_matrix.columns for c in cols):
            continue
        block = _get_heterotrait_monomethod_block(corr_matrix, judge, subtypes)
        judge_vectors[judge] = block[triu_idx]

    pairwise_spearman: dict[str, float] = {}
    for j1, j2 in combinations(list(judge_vectors.keys()), 2):
        v1 = judge_vectors[j1]
        v2 = judge_vectors[j2]
        mask = ~(np.isnan(v1) | np.isnan(v2))
        if mask.sum() < 2:
            r = float("nan")
        else:
            try:
                result = scipy.stats.spearmanr(v1[mask], v2[mask])
                r = float(result.statistic)
            except Exception:
                r = float("nan")
        pairwise_spearman[f"{j1} vs {j2}"] = r

    values = [v for v in pairwise_spearman.values() if not np.isnan(v)]
    mean_consistency = float(np.mean(values)) if values else float("nan")

    pattern_meets_threshold = bool(mean_consistency >= CONV_THRESHOLD) if values else False

    return {
        "pairwise_spearman": pairwise_spearman,
        "mean_consistency": mean_consistency,
        "pattern_meets_threshold": pattern_meets_threshold,
    }

def save_mtmm_results(results: dict, output_dir: str) -> None:
    """
    Write MTMM analysis outputs to disk.

    Produces ``mtmm_results.json`` with validity and consistency statistics
    and ``mtmm_corr_matrix.csv`` with the full correlation matrix.

    :param results: Output dict from :func:`run_mtmm`.
    :param output_dir: Directory to write outputs into.
    """
    os.makedirs(output_dir, exist_ok=True)

    json_data = {
        "judges": results["judges"],
        "subtypes": results["subtypes"],
        "validity": results["validity"],
        "consistency": {
            "pairwise_spearman": results["consistency"]["pairwise_spearman"],
            "mean_consistency": results["consistency"]["mean_consistency"],
        },
        "classified_means": {
            k: float(np.mean(v)) if v else None
            for k, v in results["classified"].items()
        },
        "classified_sds": {
            k: float(np.std(v)) if v else None
            for k, v in results["classified"].items()
        },
    }

    json_path = os.path.join(output_dir, "mtmm_results.json")
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)

    csv_path = os.path.join(output_dir, "mtmm_corr_matrix.csv")
    results["corr_matrix"].to_csv(csv_path)

def run_mtmm(
    flagging_results_path: str,
    output_dir: str,
    verbose: bool = False,
) -> dict:
    """
    Run the full MTMM analysis pipeline from a flagging_results.json file or
    a directory tree containing multiple such files.

    When a directory is supplied, all ``flagging_results.json`` files found
    recursively are merged into a single claim matrix, enabling analysis across
    multiple agent models and scenario subtypes simultaneously.

    Steps: load/merge → build claim matrix → compute correlation matrix →
    classify correlations → assess convergent validity →
    check trait pattern consistency → save results and plots.

    :param flagging_results_path: Path to a flagging_results.json file, or to
        a directory that will be searched recursively for such files.
    :param output_dir: Directory to write JSON, CSV, and plot outputs.
    :param verbose: Print progress messages when True.
    :return: Dict with keys: claim_matrix, corr_matrix, classified, validity,
             consistency, judges, subtypes.
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

    if verbose:
        print(f"Claim matrix: {claim_matrix.shape[0]} claims × {claim_matrix.shape[1]} columns")

    corr_matrix = compute_mtmm_correlation_matrix(claim_matrix)

    if verbose:
        print(f"Correlation matrix: {corr_matrix.shape[0]} × {corr_matrix.shape[1]}")

    cols = list(claim_matrix.columns)
    judges = sorted(set(_parse_column(c)[0] for c in cols))
    subtypes_present = [s for s in SUBTYPES if any(_parse_column(c)[1] == s for c in cols)]

    if len(judges) < 2:
        raise ValueError(
            f"MTMM requires at least 2 judges; found {len(judges)}: {judges}"
        )

    classified = classify_mtmm_correlations(corr_matrix)
    validity = assess_construct_validity(classified, corr_matrix)
    consistency = check_trait_pattern_consistency(corr_matrix, judges, subtypes_present)

    if verbose:
        print(f"Judges ({len(judges)}): {judges}")
        print(f"Subtypes: {subtypes_present}")
        print(
            f"Mean MTHM={validity['mean_mthm']:.3f}  "
            f"Mean HTMonoM={validity['mean_htmonm']:.3f}  "
            f"Mean HTHM={validity['mean_hthm']:.3f}"
        )
        threshold_label = f"Mean MTHM meets threshold (>={CONV_THRESHOLD}):"
        print(f"{threshold_label:<43}{validity['mthm_meets_threshold']}")
        print(f"{'Mean MTHM significantly > 0:':<43}{validity['mthm_significant_gt_zero']}")
        print(f"{'All MTHM coefficients > 0:':<43}{validity['mthm_all_positive']}  ({_fmt_pct(validity['mthm_pct_positive'])})")
        print(f"{'All MTHM > row/col HTHM neighbours:':<43}{validity['mthm_all_exceed_row_col_hthm']}  ({_fmt_pct(validity['mthm_pct_exceed_row_col_hthm'])})")
        print(f"{'All MTHM > local HTMonoM:':<43}{validity['mthm_all_exceed_local_htmonm']}  ({_fmt_pct(validity['mthm_pct_exceed_local_htmonm'])})")
        print(f"{'Block ordering (diag>MTHM>HTMonoM>HTHM):':<43}{validity['block_ordering_holds']}")
        print(f"Mean trait-pattern consistency: {consistency['mean_consistency']:.3f}  "
              f"meets threshold: {consistency['pattern_meets_threshold']}")

    results = {
        "claim_matrix": claim_matrix,
        "corr_matrix": corr_matrix,
        "classified": classified,
        "validity": validity,
        "consistency": consistency,
        "judges": judges,
        "subtypes": subtypes_present,
    }

    os.makedirs(output_dir, exist_ok=True)
    save_mtmm_results(results, output_dir)
    save_mtmm_heatmap(
        corr_matrix,
        judges,
        subtypes_present,
        os.path.join(output_dir, "mtmm_heatmap.png"),
    )
    save_mtmm_summary_bar(
        classified,
        os.path.join(output_dir, "mtmm_summary.png"),
    )

    print(f"Results written to {output_dir}")

    return results
