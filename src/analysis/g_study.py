import json
import math
import os
import numpy as np
import pandas as pd
# This Pandas mapping is used to ensure compilation with the older version of pymer. 
if not hasattr(pd.DataFrame, "applymap"):
    pd.DataFrame.applymap = pd.DataFrame.map
from pymer4.models import Lmer
from ..utils.plots import save_dstudy_plot

def _apply_logit(n_flagged: int, n_claims: int) -> float:
    """
    Logit-transform a flag count with continuity correction for boundary rates.

    :param n_flagged: Number of flagged claims.
    :param n_claims: Total claims evaluated.
    :return: Log-odds of the (corrected) flag rate.
    """
    if n_claims == 0:
        return float("nan")
    p = n_flagged / n_claims
    if p == 0.0:
        p = 0.5 / (n_claims + 1)
    elif p == 1.0:
        p = (n_claims + 0.5) / (n_claims + 1)
    return math.log(p / (1.0 - p))

def _extract_conv_flag_rates(flagging_data: dict) -> list[dict]:
    """
    Return one record per (judge, conv_id) from a flagging_results dict.

    :param flagging_data: Parsed flagging_results.json content.
    :return: List of dicts with keys judge, conv_id, n_claims, n_flagged, flag_rate.
    """
    groups = {}
    for entry in flagging_data.get("claim_evaluations", []):
        key = (entry["flagging_model"], entry["conversation_file"])
        if key not in groups:
            groups[key] = []
        groups[key].append(entry["is_flagged"])
    records = []
    for (judge, conv_id), flags in groups.items():
        n_claims = len(flags)
        n_flagged = sum(flags)
        records.append({
            "judge": judge,
            "conv_id": conv_id,
            "n_claims": n_claims,
            "n_flagged": n_flagged,
            "flag_rate": n_flagged / n_claims,
        })
    return records

def _scan_flagging_files(results_dir: str) -> list[dict]:
    """
    Walk results_dir recursively for flagging_results.json files.

    :param results_dir: Top-level pilot run directory.
    :return: List of dicts with keys agent_model, sub_scenario, path.
    """
    entries = []
    if not os.path.isdir(results_dir):
        raise ValueError(f"results_dir does not exist: {results_dir}")
    for root, _dirs, files in os.walk(results_dir):
        if "flagging_results.json" not in files:
            continue
        rel = os.path.relpath(root, results_dir)
        parts = rel.split(os.sep)
        # sub_scenario is the directory immediately above evaluations/
        if len(parts) < 2 or parts[-1] != "evaluations":
            continue
        entries.append({
            "agent_model": parts[0],
            "sub_scenario": parts[-2],
            "path": os.path.join(root, "flagging_results.json"),
        })
    return sorted(entries, key=lambda e: (e["agent_model"], e["sub_scenario"]))

def build_input_matrix(
    results_dir: str,
    scenario: str,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Collect per-(M, R, I, S) flag rates and return the logit-transformed input matrix.

    :param results_dir: Path to pilot run directory (e.g. "results/pilot_product_promotion_generation").
    :param scenario: Scenario label stored in metadata column only.
    :param verbose: If True, print discovery progress.
    :return: DataFrame with columns [agent_model, judge, conv_id, sub_scenario,
             n_claims, n_flagged, flag_rate, logit_dr].
    """
    file_entries = _scan_flagging_files(results_dir)
    if not file_entries:
        raise ValueError(f"No flagging_results.json files found under {results_dir}/")

    rows = []
    for entry in file_entries:
        with open(entry["path"]) as f:
            flagging_data = json.load(f)
        conv_records = _extract_conv_flag_rates(flagging_data)
        for rec in conv_records:
            rows.append({
                "agent_model": entry["agent_model"],
                "sub_scenario": entry["sub_scenario"],
                "judge": rec["judge"],
                "conv_id": rec["conv_id"],
                "n_claims": rec["n_claims"],
                "n_flagged": rec["n_flagged"],
                "flag_rate": rec["flag_rate"],
                "logit_dr": _apply_logit(rec["n_flagged"], rec["n_claims"]),
            })

    df = pd.DataFrame(rows)
    df = df.dropna(subset=["logit_dr"])

    if verbose:
        n_M = df["agent_model"].nunique()
        n_R = df["judge"].nunique()
        n_S = df["sub_scenario"].nunique()
        n_I = df.groupby("sub_scenario")["conv_id"].nunique().mean()
        print(f"Input matrix: {len(df)} rows | {n_M} models, {n_R} judges, {n_S} sub-scenarios, ~{n_I:.1f} convs/sub-scenario")

    return df


def _reml_group_to_key(group_name: str) -> str | None:
    """
    Map an lme4 random-effect grouping factor name to a G-study component key.

    Normalises dots to underscores and treats interaction parts as an unordered
    set so that R's potential reordering of factor names does not matter.

    :param group_name: Grouping factor label returned by lme4 / VarCorr.
    :return: One of the 10 non-residual component keys, or None if unrecognised.
    """
    parts = frozenset(group_name.replace(".", "_").split(":"))
    mapping = {
        frozenset({"agent_model"}): "M",
        frozenset({"judge"}): "R",
        frozenset({"sub_scenario"}): "S",
        frozenset({"sub_scenario", "conv_id"}): "I_S",
        frozenset({"agent_model", "judge"}): "MR",
        frozenset({"agent_model", "sub_scenario"}): "MS",
        frozenset({"judge", "sub_scenario"}): "RS",
        frozenset({"agent_model", "sub_scenario", "conv_id"}): "MI_S",
        frozenset({"judge", "sub_scenario", "conv_id"}): "RI_S",
        frozenset({"agent_model", "judge", "sub_scenario"}): "MRS",
    }
    return mapping.get(parts)

def _fit_reml_components(df: pd.DataFrame, verbose: bool = False) -> dict:
    """
    Estimate variance components via REML for the M × R × (I:S) design using lme4.

    Fits a maximal random-intercepts model via pymer4's lmer interface.
    Variance components are extracted from the fitted model's ranef_var table;
    the residual sigma² is read from model.sig when not present in that table.

    :param df: Input matrix from build_input_matrix.
    :param verbose: If True, print any lme4 convergence messages.
    :return: Dict of {source: sigma_squared} for all 11 variance components.
    """
    y = df["logit_dr"].values
    keys = ["M", "R", "S", "I_S", "MR", "MS", "RS", "MI_S", "RI_S", "MRS", "e"]

    if np.var(y) < 1e-10:
        return {k: 0.0 for k in keys}

    formula = (
        "logit_dr ~ 1"
        " + (1 | agent_model)"
        " + (1 | judge)"
        " + (1 | sub_scenario)"
        " + (1 | sub_scenario:conv_id)"
        " + (1 | agent_model:judge)"
        " + (1 | agent_model:sub_scenario)"
        " + (1 | judge:sub_scenario)"
        " + (1 | agent_model:sub_scenario:conv_id)"
        " + (1 | judge:sub_scenario:conv_id)"
        " + (1 | agent_model:judge:sub_scenario)"
    )

    model = Lmer(formula, data=df)
    model.fit(summary=False)

    if verbose and hasattr(model, "show_logs"):
        model.show_logs()

    result = {k: 0.0 for k in keys}

    vc_df = getattr(model, "ranef_var", None)
    if vc_df is None:
        vc_df = getattr(model, "result_vc", None)
    if isinstance(vc_df, pd.DataFrame):
        if "grp" in vc_df.columns:
            for _, row in vc_df.iterrows():
                grp = str(row["grp"])
                val = max(float(row.get("vcov", 0.0)), 0.0)
                if grp.lower() == "residual":
                    result["e"] = val
                else:
                    key = _reml_group_to_key(grp)
                    if key:
                        result[key] = val
        else:
            var_col = "Var" if "Var" in vc_df.columns else vc_df.columns[0]
            for grp, row in vc_df.iterrows():
                val = max(float(row[var_col]), 0.0)
                if str(grp).lower() == "residual":
                    result["e"] = val
                else:
                    key = _reml_group_to_key(str(grp))
                    if key:
                        result[key] = val

    if result["e"] == 0.0 and hasattr(model, "sig"):
        result["e"] = max(float(model.sig) ** 2, 0.0)

    return result

def fit_gstudy(df: pd.DataFrame, verbose: bool = False) -> dict:
    """
    Estimate variance components for the M × R × (I:S) design via REML.

    Negative component estimates are floored to 0 by the optimizer bounds.

    :param df: Output of build_input_matrix.
    :param verbose: If True, print variance components table.
    :return: Dict with keys variance_components, design_counts.
    """
    n_M = df["agent_model"].nunique()
    n_R = df["judge"].nunique()
    n_S = df["sub_scenario"].nunique()
    n_I = int(round(df.groupby("sub_scenario")["conv_id"].nunique().mean()))

    if n_M < 2 or n_R < 2 or n_S < 2 or n_I < 2:
        raise ValueError(
            f"Design too sparse for variance decomposition: "
            f"n_M={n_M}, n_R={n_R}, n_S={n_S}, n_I={n_I}. Need ≥2 levels per facet."
        )

    var_components = _fit_reml_components(df, verbose=verbose)

    if verbose:
        print("\nVariance components (M × R × (I:S) design, REML):")
        total_var = sum(var_components.values())
        for source, sigma_sq in var_components.items():
            pct = 100 * sigma_sq / total_var if total_var > 0 else 0.0
            print(f"  σ²({source:<4}) = {sigma_sq:.6f}  ({pct:.1f}%)")
        print(f"  Total        = {total_var:.6f}")

    return {
        "variance_components": var_components,
        "design_counts": {"n_M": n_M, "n_R": n_R, "n_S": n_S, "n_I": n_I},
    }

def compute_g_coefficient(
    var_components: dict,
    n_judges: int,
    n_convs: int,
    n_subtypes: int,
) -> float:
    """
    Compute the generalizability coefficient (relative decisions) for given design sizes.

    :param var_components: Dict of variance component estimates from fit_gstudy.
    :param n_judges: Number of flagging judges (nR).
    :param n_convs: Number of conversations per sub-scenario (nI).
    :param n_subtypes: Number of sub-scenarios (nS).
    :return: Generalizability coefficient in [0, 1].
    """
    vc = var_components
    # σ²_M: variance attributable to the object of measurement (agent model).
    universe_score = vc["M"]
    # Relative error includes only variance components that interact with M.
    # Main effects of R, S, and I:S cancel out for relative decisions because
    # they shift all person scores equally and do not affect rank ordering.
    relative_error = (
        vc["MR"] / n_judges
        + vc["MS"] / n_subtypes
        + vc["MI_S"] / (n_convs * n_subtypes)
        + vc["MRS"] / (n_judges * n_subtypes)
        + vc["e"] / (n_judges * n_convs * n_subtypes)
    )
    denominator = universe_score + relative_error
    if denominator <= 0:
        return 0.0
    return universe_score / denominator


def compute_phi_coefficient(
    var_components: dict,
    n_judges: int,
    n_convs: int,
    n_subtypes: int,
) -> float:
    """
    Compute the phi coefficient (index of dependability, absolute decisions).

    :param var_components: Dict of variance component estimates from fit_gstudy.
    :param n_judges: Number of flagging judges (nR).
    :param n_convs: Number of conversations per sub-scenario (nI).
    :param n_subtypes: Number of sub-scenarios (nS).
    :return: Phi coefficient in [0, 1].
    """
    vc = var_components
    universe_score = vc["M"]
    # Absolute error includes all non-M variance components, because absolute
    # decisions (e.g. pass/fail thresholds) are sensitive to mean-level shifts
    # from any facet, not just those that interact with the object of measurement.
    absolute_error = (
        vc["R"] / n_judges
        + vc["S"] / n_subtypes
        + vc["I_S"] / (n_convs * n_subtypes)
        + vc["MR"] / n_judges
        + vc["MS"] / n_subtypes
        + vc["MI_S"] / (n_convs * n_subtypes)
        + vc["RS"] / (n_judges * n_subtypes)
        + vc["RI_S"] / (n_judges * n_convs * n_subtypes)
        + vc["MRS"] / (n_judges * n_subtypes)
        + vc["e"] / (n_judges * n_convs * n_subtypes)
    )
    denominator = universe_score + absolute_error
    if denominator <= 0:
        return 0.0
    return universe_score / denominator


def run_dstudy(
    var_components: dict,
    observed_n: dict,
    judge_range: list[int] | None = None,
    conv_range: list[int] | None = None,
    subtype_range: list[int] | None = None,
) -> pd.DataFrame:
    """
    Project G and phi coefficients under alternative design configurations.

    Varies one facet at a time, holding the others at their observed values.

    :param var_components: Dict from fit_gstudy["variance_components"].
    :param observed_n: Dict with keys n_judges, n_convs, n_subtypes.
    :param judge_range: nR values to sweep (default 1 to observed+3).
    :param conv_range: nI values to sweep.
    :param subtype_range: nS values to sweep.
    :return: DataFrame with columns [varied_facet, n, g_coefficient, phi_coefficient].
    """
    n_R0 = observed_n["n_judges"]
    n_I0 = observed_n["n_convs"]
    n_S0 = observed_n["n_subtypes"]

    if judge_range is None:
        judge_range = list(range(1, n_R0 + 4))
    if conv_range is None:
        conv_range = list(range(1, n_I0 + 6))
    if subtype_range is None:
        subtype_range = list(range(1, n_S0 + 4))

    rows = []
    for n in judge_range:
        rows.append({
            "varied_facet": "n_judges",
            "n": n,
            "g_coefficient": compute_g_coefficient(var_components, n, n_I0, n_S0),
            "phi_coefficient": compute_phi_coefficient(var_components, n, n_I0, n_S0),
        })
    for n in conv_range:
        rows.append({
            "varied_facet": "n_convs",
            "n": n,
            "g_coefficient": compute_g_coefficient(var_components, n_R0, n, n_S0),
            "phi_coefficient": compute_phi_coefficient(var_components, n_R0, n, n_S0),
        })
    for n in subtype_range:
        rows.append({
            "varied_facet": "n_subtypes",
            "n": n,
            "g_coefficient": compute_g_coefficient(var_components, n_R0, n_I0, n),
            "phi_coefficient": compute_phi_coefficient(var_components, n_R0, n_I0, n),
        })

    return pd.DataFrame(rows)

def save_gstudy_results(
    scenario: str,
    gstudy_result: dict,
    g_coefficient: float,
    phi_coefficient: float,
    dstudy_df: pd.DataFrame,
    output_dir: str,
) -> None:
    """
    Write g_study_{scenario}.json and d_study_{scenario}.csv to output_dir.

    :param scenario: Scenario label used in filenames.
    :param gstudy_result: Full output of fit_gstudy.
    :param g_coefficient: Observed-design G-coefficient (relative decisions).
    :param phi_coefficient: Observed-design phi coefficient (absolute decisions).
    :param dstudy_df: Output of run_dstudy.
    :param output_dir: Directory to write output files.
    """
    os.makedirs(output_dir, exist_ok=True)

    payload = {
        "scenario": scenario,
        "g_coefficient_observed": g_coefficient,
        "phi_coefficient_observed": phi_coefficient,
        "variance_components": gstudy_result["variance_components"],
        "design_counts": gstudy_result["design_counts"],
    }
    json_path = os.path.join(output_dir, f"g_study_{scenario}.json")
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)

    csv_path = os.path.join(output_dir, f"d_study_{scenario}.csv")
    dstudy_df.to_csv(csv_path, index=False)

    print(f"Results written to {output_dir}")


def _print_dstudy_best(dstudy_df: pd.DataFrame) -> None:
    """
    Print the highest G and phi coefficients reached per varied facet.

    :param dstudy_df: Output of run_dstudy.
    """
    print("\nD-study: best G and Φ per facet:")
    for facet in ["n_judges", "n_convs", "n_subtypes"]:
        sub = dstudy_df[dstudy_df["varied_facet"] == facet]
        if sub.empty:
            continue
        best = sub.loc[sub["g_coefficient"].idxmax()]
        suffix = "  [invariant: nM does not enter Eρ²/Φ formulas]" if sub["g_coefficient"].nunique() == 1 else ""
        print(
            f"  {facet:<12}: G={best['g_coefficient']:.4f}  "
            f"Φ={best['phi_coefficient']:.4f}  (n={int(best['n'])}){suffix}"
        )

def run_gstudy(
    results_dir: str,
    scenario: str,
    output_dir: str,
    target: float = 0.85,
    verbose: bool = False,
) -> dict:
    """
    Full G-study pipeline: build matrix, fit model, compute G-coefficient, run D-study, save outputs.

    :param results_dir: Pilot run directory.
    :param scenario: Scenario label.
    :param output_dir: Directory for output files and plots.
    :param target: G-coefficient target for D-study reference line.
    :param verbose: If True, print progress and diagnostics.
    :return: Dict with keys matrix, gstudy_result, g_coefficient, dstudy_df.
    """
    if verbose:
        print(f"\n=== G-Study: {scenario} ===\n")

    df = build_input_matrix(results_dir, scenario, verbose=verbose)
    gstudy_result = fit_gstudy(df, verbose=verbose)

    vc = gstudy_result["variance_components"]
    dc = gstudy_result["design_counts"]
    g_coef = compute_g_coefficient(vc, dc["n_R"], dc["n_I"], dc["n_S"])
    phi_coef = compute_phi_coefficient(vc, dc["n_R"], dc["n_I"], dc["n_S"])

    if verbose:
        print(f"\nG-coefficient (observed design): {g_coef:.4f}  (target: {target})")
        print(f"Φ-coefficient (observed design): {phi_coef:.4f}")

    observed_n = {"n_judges": dc["n_R"], "n_convs": dc["n_I"], "n_subtypes": dc["n_S"]}
    dstudy_df = run_dstudy(vc, observed_n)

    if verbose:
        _print_dstudy_best(dstudy_df)

    save_dstudy_plot(dstudy_df, output_dir, scenario, target=target)
    save_gstudy_results(scenario, gstudy_result, g_coef, phi_coef, dstudy_df, output_dir)

    return {
        "matrix": df,
        "gstudy_result": gstudy_result,
        "g_coefficient": g_coef,
        "phi_coefficient": phi_coef,
        "dstudy_df": dstudy_df,
    }
