import argparse
from pathlib import Path

from .analysis.efa import run_efa
from .analysis.g_study import run_gstudy
from .analysis.mtmm import run_mtmm

_DATA_DIR = Path(__file__).parent / "data"
_RESULTS_DIR = Path(__file__).parent.parent / "results"


def main():
    parser = argparse.ArgumentParser(description="Run reliability and validity analyses on pilot data")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    ##### G-Study Mode  - Argparse ########
    gstudy_parser = subparsers.add_parser("gstudy", help="Run G-study and D-study reliability analysis")
    gstudy_parser.add_argument(
        "--scenario",
        choices=["product_promotion", "loan_qa"],
        required=True,
        help="Scenario type to analyse"
    )
    gstudy_parser.add_argument(
        "--results_dir",
        type=str,
        default=None,
        help="Path to trial results directory (contains {model}/{condition}/{subtype}/ subdirs); defaults to bundled pilot data"
    )
    gstudy_parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to write g_study JSON, d_study CSV, and plot; defaults to results/g-study/{scenario}"
    )
    gstudy_parser.add_argument(
        "--condition",
        type=str,
        default="default",
        help="Condition subdirectory to read (default: default)"
    )
    gstudy_parser.add_argument(
        "--target",
        type=float,
        default=0.85,
        help="G-coefficient reliability target for D-study reference line"
    )
    gstudy_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbosity"
    )

    ##### MTMM Mode  - Argparse ########
    mtmm_parser = subparsers.add_parser("mtmm", help="Run Multi-Trait Multi-Method validity analysis")
    mtmm_parser.add_argument(
        "--scenario",
        choices=["product_promotion", "loan_qa"],
        required=True,
        help="Scenario type to analyse"
    )
    mtmm_parser.add_argument(
        "--flagging_results",
        type=str,
        default=None,
        help="Path to a flagging_results.json file, or a directory to search recursively; defaults to bundled pilot data"
    )
    mtmm_parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to write MTMM outputs (JSON, CSV, plots); defaults to results/mtmm/{scenario}"
    )
    mtmm_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbosity"
    )

    ##### EFA Mode  - Argparse ########
    efa_parser = subparsers.add_parser("efa", help="Run EFA and parallel analysis on flagging results")
    efa_parser.add_argument(
        "--scenario",
        choices=["product_promotion", "loan_qa"],
        required=True,
        help="Scenario type to analyse"
    )
    efa_parser.add_argument(
        "--flagging_results",
        type=str,
        default=None,
        help="Path to a flagging_results.json file, or a directory to search recursively; defaults to bundled pilot data"
    )
    efa_parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to write EFA outputs (JSON, CSV, plot); defaults to results/efa/{scenario}"
    )
    efa_parser.add_argument(
        "--n_random",
        type=int,
        default=500,
        help="Number of random matrices for parallel analysis simulation"
    )
    efa_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for parallel analysis simulation"
    )
    efa_parser.add_argument(
        "--cross_loading_threshold",
        type=float,
        default=0.40,
        help="Maximum permitted cross-loading for simple structure classification"
    )
    efa_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbosity"
    )

    args = parser.parse_args()

    ##### G-Study Mode - Engine ########
    if args.mode == "gstudy":
        results_dir = args.results_dir or str(_DATA_DIR / f"{args.scenario}_pilot_data")
        output_dir = args.output_dir or str(_RESULTS_DIR / "g-study" / args.scenario)
        run_gstudy(
            results_dir=results_dir,
            scenario=args.scenario,
            output_dir=output_dir,
            condition=args.condition,
            target=args.target,
            verbose=args.verbose,
        )

    ##### MTMM Mode - Engine ########
    elif args.mode == "mtmm":
        flagging_results = args.flagging_results or str(_DATA_DIR / f"{args.scenario}_pilot_data")
        output_dir = args.output_dir or str(_RESULTS_DIR / "mtmm" / args.scenario)
        run_mtmm(
            flagging_results_path=flagging_results,
            output_dir=output_dir,
            verbose=args.verbose,
        )

    ##### EFA Mode - Engine ########
    elif args.mode == "efa":
        flagging_results = args.flagging_results or str(_DATA_DIR / f"{args.scenario}_pilot_data")
        output_dir = args.output_dir or str(_RESULTS_DIR / "efa" / args.scenario)
        run_efa(
            flagging_results_path=flagging_results,
            output_dir=output_dir,
            n_random=args.n_random,
            seed=args.seed,
            cross_loading_threshold=args.cross_loading_threshold,
            verbose=args.verbose,
        )


if __name__ == "__main__":
    main()
