"""Reconstruct the paper's numerical results from recorded evidence."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
REFERENCE_FILES = (
    "manifest.json", "population_summary.csv", "manifold_coverage.csv",
    "rho_curve.csv", "covering_curve.csv", "stratification.csv",
    "inference.csv", "lexical_diversity.csv",
    "fig01_claim_statistics.csv", "fig02_claim_statistics.csv",
    "fig04_claim_statistics.csv", "fig16_claim_statistics.csv",
)


def check_inputs() -> None:
    manifest = json.loads((ROOT / "data/SHA256SUMS.json").read_text())
    for name, expected in manifest.items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Evidence checksum mismatch: {name}")


def recorded_encoder(texts: list[str]):
    import numpy as np
    from recipe_diversity import EMBEDDING_MODEL

    with np.load(ROOT / "data/embeddings.npz", allow_pickle=False) as cache:
        # Missing or altered text raises KeyError; this path cannot call an API.
        vectors = np.stack([
            cache[hashlib.sha256(f"{EMBEDDING_MODEL}\0{text}".encode()).hexdigest()]
            for text in texts
        ])
    if vectors.shape != (len(texts), 3072) or not np.isfinite(vectors).all():
        raise ValueError("Invalid recorded embeddings")
    return vectors


def check_results(output: Path) -> dict[str, str]:
    import numpy as np

    def equal(actual, expected):
        if isinstance(expected, dict):
            return actual.keys() == expected.keys() and all(equal(actual[k], v) for k, v in expected.items())
        if isinstance(expected, (float, int)):
            return bool(np.isclose(actual, expected, rtol=1e-7, atol=1e-9))
        try:
            return bool(np.isclose(float(actual), float(expected), rtol=1e-7, atol=1e-9, equal_nan=True))
        except (ValueError, TypeError):
            return actual == expected

    checks = {}
    for name in REFERENCE_FILES:
        actual, expected = output / name, ROOT / "reference" / name
        if actual.read_bytes() == expected.read_bytes():
            checks[name] = "byte-identical"
            continue
        if name.endswith(".json"):
            matches = equal(json.loads(actual.read_text()), json.loads(expected.read_text()))
        else:
            with actual.open(newline="") as a, expected.open(newline="") as b:
                a_rows, b_rows = list(csv.reader(a)), list(csv.reader(b))
            matches = len(a_rows) == len(b_rows) and all(
                len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
                for a, b in zip(a_rows, b_rows)
            )
        if not matches:
            raise ValueError(f"Reference mismatch: {name}")
        checks[name] = "matches within rtol=1e-7, atol=1e-9"
    return checks


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Compare all twelve archived result files")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs")
    parser.add_argument("--runs", type=Path, help="New study directory written by run.py")
    parser.add_argument("--embed", action="store_true", help="Allow paid embeddings for a new study")
    args = parser.parse_args(argv)
    if args.runs and (args.check or not args.embed):
        parser.error("New runs require --embed and cannot use --check against the historical results")
    if args.embed and not args.runs:
        parser.error("--embed requires --runs")
    output = args.output.resolve()
    runs = args.runs.resolve() if args.runs else None
    check_inputs()
    output.mkdir(parents=True, exist_ok=False)
    os.environ.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MPLCONFIGDIR=str(output / ".matplotlib"))
    from recipe_diversity.analysis import compare_populations

    sources = {
        "baseline": Path("data/original_recipes"),
        "nonce": Path("runs/nonce_1000"),
        "prior_titles": Path("runs/prior_titles_1000"),
        "hierarchical": Path("runs/wide_1000"),
    }
    encoder = recorded_encoder
    if runs:
        from recipe_diversity.embedding import EmbeddingSettings, embed_texts
        sources.update({name: runs / path for name, path in sources.items() if name != "baseline"})
        settings = EmbeddingSettings(cache_dir=runs / ".embedding_cache")
        encoder = lambda texts: embed_texts(texts, settings).vectors
    previous = Path.cwd()
    try:
        os.chdir(ROOT / "data")
        compare_populations(sources, output, encoder=encoder)
        commands = [
            ("inference.py", "--results", str(output)),
            ("lexical_table.py", "--out", str(output), *(["--runs", str(runs / "runs")] if runs else [])),
            ("make_figures.py", "--results", str(output), "--out", str(output / "figures")),
            ("make_tables.py", "--results", str(output)),
            ("paper_statistics.py", "--results", str(output)),
        ]
        for name, *arguments in commands:
            subprocess.run([sys.executable, str(ROOT / "scripts" / name), *arguments], check=True)
    finally:
        os.chdir(previous)
    if args.check:
        checks = check_results(output)
        (output / "verification.json").write_text(json.dumps(checks, indent=2) + "\n")
        print(json.dumps(checks, indent=2))
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
