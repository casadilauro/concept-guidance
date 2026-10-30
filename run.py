"""Run the three paper strategies, preserving their original request protocol."""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("strategy", choices=("nonce", "prior-titles", "breadth-first", "all"))
    parser.add_argument("--output", type=Path, required=True, help="Study directory")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Validate and resume existing request records")
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if output == ROOT or output in ROOT.parents or output.is_relative_to(ROOT / "data"):
        parser.error("Choose a new study directory outside the bundled evidence")
    from recipe_diversity.generation import generate_nonce, generate_prior_titles, generate_wide

    strategies = {
        "nonce": ("nonce_1000", generate_nonce),
        "prior-titles": ("prior_titles_1000", generate_prior_titles),
        "breadth-first": ("wide_1000", generate_wide),
    }
    selected = strategies if args.strategy == "all" else {args.strategy: strategies[args.strategy]}
    for strategy, (folder, _) in selected.items():
        target = output / "runs" / folder
        if target.exists() and not args.resume:
            parser.error(f"{target} already exists; use --resume to validate and continue")
        print(json.dumps({"strategy": strategy, "output": str(target), "recipes": 1000,
                          "planned_calls": 101 if strategy == "breadth-first" else 1000}))
    if args.dry_run:
        return 0
    from rich.progress import Progress

    with Progress() as progress:
        for folder, generate in selected.values():
            generate(output / "runs" / folder, progress=progress)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
