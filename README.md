# FLEECED Analysis Toolkit

This repo provides a post-hoc reliability and validity analysis for the FLEECED benchmark, a framework for measuring LLM deception propensity in non-adversarial settings. This toolkit runs three psychometric analyses on bundled pilot data to assess the quality of the LLM-as-judge flagging judgments.

## System Requirements

- **OS**: macOS (Apple Silicon / arm64) — tested. Linux x86_64 likely works but is untested. Windows is not supported (rpy2 dependency).
- **RAM**: >= 8 GB recommended (lme4 REML fitting for G-study peaks ~2 GB)
- **Disk**: ~2.5 GB (~2 GB conda environment + data + results)
- **GPU**: not required — all analyses are CPU-only

Expected runtimes on an Apple M-series or equivalent modern x86 laptop:

| Analysis | product_promotion | loan_qa |
|---|---|---|
| G-study | ~2–5 min | ~2–5 min |
| MTMM | < 30 sec | < 30 sec |
| EFA | < 30 sec | < 30 sec |

## Pilot Data

**All data is bundled in this repository — no download or external access is required.**

Two scenarios are included in `src/data/`:

| Dataset | Subtypes | Models |
|---|---|---|
| `product_promotion_pilot_data` | Shoes (50%, 60% benign), Books (70%, 80% benign) | claude-3-haiku, deepseek-chat, gemini-2.5-flash-lite, gpt-4.1, llama-4-scout, mistral-small-2506 |
| `loan_qa_pilot_data` | Loans (50%, 60%, 70%, 78% benign) | same 6 models |

Data layout: `{scenario}_pilot_data/{model}/default/{subtype}/evaluations/flagging_results.json`

Each scenario directory also contains:
- `deception_results.json` — aggregated deception classification results across all models and subtypes
- `trial_manifest.json` — metadata for all trials (conversation IDs, subtype labels, benign/deceptive ground truth)

## Installation

```bash
conda env create -f environment.yml
conda activate fleeced_pilot
```

## Reproducing All Results

Complete step-by-step instructions to reproduce every result from scratch:

1. **Get the repository**

   ```bash
   # unzip the archive, or clone:
   git clone https://github.com/ebeharry/pilot-fleeced-analysis
   cd pilot-fleeced-analysis
   ```

2. **Create and activate the conda environment** (pins all exact package versions)

   ```bash
   conda env create -f environment.yml
   conda activate fleeced_pilot
   ```

   > If you are on a different machine, update or remove the `prefix:` line at the bottom of `environment.yml` before running this command — it does not affect package resolution.

3. **Run all six analyses** (both scenarios × three analysis types)

   ```bash
   python -m src gstudy --scenario product_promotion --output_dir reproduction/g-study/product_promotion
   python -m src gstudy --scenario loan_qa           --output_dir reproduction/g-study/loan_qa

   python -m src mtmm   --scenario product_promotion --output_dir reproduction/mtmm/product_promotion
   python -m src mtmm   --scenario loan_qa           --output_dir reproduction/mtmm/loan_qa

   python -m src efa    --scenario product_promotion --output_dir reproduction/efa/product_promotion
   python -m src efa    --scenario loan_qa           --output_dir reproduction/efa/loan_qa
   ```

4. **Find the outputs** in `reproduction/` — see [Script -> Output Mapping](#script--output-mapping) below for the exact file list.

## Quick Start

All commands default to the bundled pilot data when no path arguments are given.

### G-Study / D-Study

```bash
# Product promotion scenario  ->  results/g-study/product_promotion/
python -m src gstudy --scenario product_promotion

# Loan Q&A scenario  ->  results/g-study/loan_qa/
python -m src gstudy --scenario loan_qa
```

### MTMM

```bash
# Product promotion scenario  ->  results/mtmm/product_promotion/
python -m src mtmm --scenario product_promotion

# Loan Q&A scenario  ->  results/mtmm/loan_qa/
python -m src mtmm --scenario loan_qa
```

### EFA

```bash
# Product promotion scenario  ->  results/efa/product_promotion/
python -m src efa --scenario product_promotion

# Loan Q&A scenario  ->  results/efa/loan_qa/
python -m src efa --scenario loan_qa
```

## CLI Reference

### `gstudy`

Estimates reliability of flagging judgments using Generalizability Theory (G-study and D-study).

| Argument | Default | Required | Description |
|---|---|---|---|
| `--scenario` | — | Yes | `product_promotion` or `loan_qa` |
| `--results_dir` | bundled pilot data | No | Path to results directory with `{model}/{condition}/{subtype}/` layout |
| `--output_dir` | `results/g-study/{scenario}` | No | Directory to write outputs |
| `--condition` | `default` | No | Condition subdirectory name |
| `--target` | `0.85` | No | G-coefficient reliability target for D-study reference line |
| `--verbose` | `False` | No | Print progress |

**Outputs:** `g_study_{scenario}.json`, `d_study_{scenario}.csv`, `dstudy_{scenario}.png`
The `g_study_{scenario}.json` file is used to build the variance component tables and extract the reliability metrics. The `d_study_{scenario}.csv` is used to build the projected D-study tables. 

### `mtmm`

Assesses convergent and discriminant validity of deception type classifications using Multi-Trait Multi-Method analysis.

| Argument | Default | Required | Description |
|---|---|---|---|
| `--scenario` | — | Yes | `product_promotion` or `loan_qa` |
| `--flagging_results` | bundled pilot data | No | Path to `flagging_results.json` or directory to search recursively |
| `--output_dir` | `results/mtmm/{scenario}` | No | Directory to write outputs |
| `--verbose` | `False` | No | Print progress |

**Outputs:** `mtmm_results.json`, `mtmm_corr_matrix.csv`, `mtmm_heatmap.png`, `mtmm_summary.png`. The `mtmm_results.json` is used to build the MTMM summary statistics table. The `mtmm_heatmap.png` is visualized in the paper appendix. 

### `efa`

Identifies latent factor structure in deception indicator correlations using Exploratory Factor Analysis with parallel analysis for factor count selection.

| Argument | Default | Required | Description |
|---|---|---|---|
| `--scenario` | — | Yes | `product_promotion` or `loan_qa` |
| `--flagging_results` | bundled pilot data | No | Path to `flagging_results.json` or directory to search recursively |
| `--output_dir` | `results/efa/{scenario}` | No | Directory to write outputs |
| `--n_random` | `500` | No | Number of random matrices for parallel analysis simulation |
| `--seed` | `42` | No | Random seed for parallel analysis |
| `--cross_loading_threshold` | `0.40` | No | Maximum permitted cross-loading for simple structure |
| `--verbose` | `False` | No | Print progress |

**Outputs:** `efa_results.json`, `efa_loadings.csv`, `efa_scree_plot.png`, `efa_loadings_bar.png`, `efa_loadings_heatmap.png`

The `efa_loadings_heatmap.png` is visualized in the results section, and the `efa_scree_plot.png` is visualized in the paper appendix. The `efa_results.json` is used to extract the percent of variables with a simple structure. 

## Project Structure

```
fleeced-benchmark/
├── environment.yml                  # Pinned conda environment (Python 3.14 + R 4.5)
├── src/
│   ├── __main__.py                  # CLI entry point (gstudy / mtmm / efa)
│   ├── analysis/
│   │   ├── g_study.py               # G-study and D-study (REML via pymer4/lme4)
│   │   ├── mtmm.py                  # MTMM correlation analysis
│   │   └── efa.py                   # EFA with varimax rotation and parallel analysis
│   ├── utils/
│   │   └── plots.py                 # Shared plotting utilities
│   └── data/
│       ├── product_promotion_pilot_data/   # Bundled pilot data — product promotion scenario
│       └── loan_qa_pilot_data/             # Bundled pilot data — loan Q&A scenario
└── results/                         # Generated outputs written here at runtime
    ├── g-study/
    │   ├── product_promotion/
    │   └── loan_qa/
    ├── mtmm/
    │   ├── product_promotion/
    │   └── loan_qa/
    └── efa/
        ├── product_promotion/
        └── loan_qa/
```

## Reproducibility Notes

- **Pinned environment**: `environment.yml` specifies every package at an exact build hash. Always use `conda env create -f environment.yml` rather than installing packages ad hoc.
- **Random seeds**: EFA parallel analysis is the only stochastic step. The default `--seed 42` is used in all commands above. Pass a different value to `--seed` to verify stability.
- **Deterministic analyses**: G-study (REML via lme4) and MTMM are fully deterministic given the same input data and environment.
- **R backend**: lme4 REML estimation runs through rpy2. Exact numerical results depend on the R (`4.5.3`) and lme4 (`2.0-1`) versions pinned in `environment.yml`.

## Script -> Output Mapping

The table below maps each command to the files it writes under `results/`.

| Command | Output files |
|---|---|
| `gstudy --scenario product_promotion` | `results/g-study/product_promotion/g_study_product_promotion.json`, `d_study_product_promotion.csv`, `dstudy_product_promotion.png` |
| `gstudy --scenario loan_qa` | `results/g-study/loan_qa/g_study_loan_qa.json`, `d_study_loan_qa.csv`, `dstudy_loan_qa.png` |
| `mtmm --scenario product_promotion` | `results/mtmm/product_promotion/mtmm_results.json`, `mtmm_corr_matrix.csv`, `mtmm_heatmap.png`, `mtmm_summary.png` |
| `mtmm --scenario loan_qa` | `results/mtmm/loan_qa/mtmm_results.json`, `mtmm_corr_matrix.csv`, `mtmm_heatmap.png`, `mtmm_summary.png` |
| `efa --scenario product_promotion` | `results/efa/product_promotion/efa_results.json`, `efa_loadings.csv`, `efa_scree_plot.png`, `efa_loadings_bar.png`, `efa_loadings_heatmap.png` |
| `efa --scenario loan_qa` | `results/efa/loan_qa/efa_results.json`, `efa_loadings.csv`, `efa_scree_plot.png`, `efa_loadings_bar.png`, `efa_loadings_heatmap.png` |
