import json
import os
import pandas as pd

SUBTYPES = ["Falsehood", "Omission", "Equivocation", "Paltering"]


def _load_flagging_results(path: str) -> dict:
    """
    Load a flagging_results.json file from disk.

    :param path: Absolute or relative path to flagging_results.json.
    :return: Parsed JSON dict.
    """
    if not os.path.isfile(path):
        raise ValueError(f"flagging_results.json not found: {path}")
    with open(path) as f:
        return json.load(f)

def _find_flagging_files(results_dir: str) -> list[str]:
    """
    Recursively find all flagging_results.json files under a directory tree.

    :param results_dir: Root directory to search.
    :return: Sorted list of absolute or relative file paths.
    :raises ValueError: If results_dir does not exist.
    """
    if not os.path.isdir(results_dir):
        raise ValueError(f"Directory not found: {results_dir}")
    found = []
    for root, _dirs, files in os.walk(results_dir):
        for fname in files:
            if fname == "flagging_results.json":
                found.append(os.path.join(root, fname))
    return sorted(found)

def _merge_flagging_results(paths: list[str]) -> dict:
    """
    Load and merge multiple flagging_results.json files into a single dict.

    Claim IDs are prefixed with the file index to avoid collisions across
    different agent models or scenario subtypes.

    :param paths: Ordered list of flagging_results.json file paths.
    :return: Merged dict with combined ``claim_evaluations`` and union of
             ``flagging_models``.
    :raises ValueError: If paths is empty.
    """
    if not paths:
        raise ValueError("No flagging_results.json paths provided to merge")
    merged_evals = []
    all_judges: set[str] = set()
    for idx, path in enumerate(paths):
        data = _load_flagging_results(path)
        all_judges.update(data.get("flagging_models", []))
        for entry in data.get("claim_evaluations", []):
            new_entry = dict(entry)
            new_entry["claim_id"] = f"{idx}_{entry['claim_id']}"
            merged_evals.append(new_entry)
    return {
        "flagging_models": sorted(all_judges),
        "claim_evaluations": merged_evals,
    }

def build_claim_matrix(flagging_results: dict) -> pd.DataFrame:
    """
    Build a wide binary DataFrame from a flagging_results dict.

    Rows are unique claim_ids; columns are ``"{judge}::{subtype}"`` pairs with
    binary 0/1 integer values. Claims evaluated by only a subset of judges have
    NaN for the missing judge columns.

    :param flagging_results: Parsed flagging_results.json content.
    :return: DataFrame indexed by claim_id with one column per (judge, subtype).
    """
    records: dict[str, dict[str, int]] = {}
    for entry in flagging_results.get("claim_evaluations", []):
        claim_id = entry["claim_id"]
        judge = entry["flagging_model"]
        indicators = entry.get("deception_indicators", {})
        if claim_id not in records:
            records[claim_id] = {}
        for subtype in SUBTYPES:
            col = f"{judge}::{subtype}"
            records[claim_id][col] = int(bool(indicators.get(subtype, False)))
    df = pd.DataFrame.from_dict(records, orient="index")
    df.index.name = "claim_id"
    return df.sort_index()

def compute_mtmm_correlation_matrix(claim_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the Pearson correlation matrix over all (judge, subtype) column pairs.

    For binary columns this is equivalent to the phi coefficient. Pairwise
    complete observations are used so NaN-filled cells do not drop entire
    columns.

    :param claim_matrix: Output of :func:`build_claim_matrix`.
    :return: Symmetric DataFrame of shape (n_cols, n_cols) with diagonal 1.0.
    :raises ValueError: If claim_matrix is empty.
    """
    if claim_matrix.empty:
        raise ValueError("claim_matrix is empty; cannot compute correlations")
    return claim_matrix.corr(method="pearson")
