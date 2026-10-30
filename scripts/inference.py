"""Bootstrap intervals, permutation tests and Holm adjustment for the paper comparison.

Estimands are differences in within-population rho between the hierarchical
strategy and each baseline, on the shared min-max scale written by
`compare_populations`.

Both procedures resample saved per-recipe rho values, without recomputing neighbours.
These are descriptive within-population resampling summaries. Recipes share
neighbours and generation histories; these intervals do not quantify uncertainty
across independent runs of a strategy. The historical calculation is preserved.
"""

import argparse
import csv
from pathlib import Path

import numpy as np
from sklearn.neighbors import NearestNeighbors

BOOTSTRAPS = 5000
PERMUTATIONS = 10_000
SEED = 123
KS = (1, 2, 3, 5, 10, 15, 25, 50, 100, 200)
BASELINES = ("nonce", "prior_titles")
TREATMENT = "hierarchical"


def rho_values(vectors: np.ndarray, k: int) -> np.ndarray:
    neighbours = min(k, len(vectors) - 1)
    distances, _ = NearestNeighbors(n_neighbors=neighbours + 1).fit(vectors).kneighbors(vectors)
    return (distances[:, 1:] ** 2).mean(axis=1)


def bootstrap_interval(
    treatment: np.ndarray,
    baseline: np.ndarray,
    statistic,
    generator: np.random.Generator,
    resamples: int = BOOTSTRAPS,
) -> tuple[float, float]:
    """Percentile interval for statistic(treatment) - statistic(baseline)."""
    draws = np.empty(resamples)
    for index in range(resamples):
        a = treatment[generator.integers(0, len(treatment), len(treatment))]
        b = baseline[generator.integers(0, len(baseline), len(baseline))]
        draws[index] = statistic(a) - statistic(b)
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def permutation_p(
    treatment: np.ndarray,
    baseline: np.ndarray,
    statistic,
    generator: np.random.Generator,
    permutations: int = PERMUTATIONS,
) -> float:
    """One-sided p for H1: statistic(treatment) > statistic(baseline)."""
    observed = statistic(treatment) - statistic(baseline)
    pooled = np.concatenate([treatment, baseline])
    cut = len(treatment)
    at_least = 0
    for _ in range(permutations):
        shuffled = generator.permutation(pooled)
        if statistic(shuffled[:cut]) - statistic(shuffled[cut:]) >= observed:
            at_least += 1
    # Add-one keeps p strictly positive: with 10,000 permutations the floor is
    # 1e-4, which is a resolution limit, not evidence of a smaller p.
    return (at_least + 1) / (permutations + 1)


def holm(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values))
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(p_values) - rank) * p_values[index])
        adjusted[index] = min(1.0, running)
    return [float(value) for value in adjusted]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="results/paper_v2")
    results = Path(parser.parse_args().results)

    scaled = np.load(results / "scaled_embeddings.npy")
    with (results / "records.csv").open(encoding="utf-8", newline="") as handle:
        populations = np.array([row["population"] for row in csv.DictReader(handle)])

    generator = np.random.default_rng(SEED)
    rows: list[dict] = []

    for k in KS:
        rho = {
            name: rho_values(scaled[populations == name], k)
            for name in (TREATMENT, *BASELINES)
        }
        for baseline in BASELINES:
            for label, statistic in (("mean", np.mean), ("median", np.median)):
                treated, control = rho[TREATMENT], rho[baseline]
                low, high = bootstrap_interval(treated, control, statistic, generator)
                rows.append({
                    "k": k,
                    "comparison": f"{TREATMENT} - {baseline}",
                    "statistic": label,
                    "treatment": round(float(statistic(treated)), 4),
                    "baseline": round(float(statistic(control)), 4),
                    "difference": round(float(statistic(treated) - statistic(control)), 4),
                    "ci_low": round(low, 4),
                    "ci_high": round(high, 4),
                    "ratio": round(float(statistic(treated) / statistic(control)), 4),
                    "p_one_sided": permutation_p(treated, control, statistic, generator),
                })

    primary = [row for row in rows if row["k"] == 15]
    for row in primary:
        row["holm_family"] = "primary (k=15)"
    for row, adjusted in zip(primary, holm([row["p_one_sided"] for row in primary]), strict=True):
        row["p_holm"] = adjusted

    secondary = [row for row in rows if row["k"] != 15]
    for row in secondary:
        row["holm_family"] = "k-robustness"
    for row, adjusted in zip(secondary, holm([row["p_one_sided"] for row in secondary]), strict=True):
        row["p_holm"] = adjusted

    fields = ["k", "comparison", "statistic", "treatment", "baseline", "difference",
              "ci_low", "ci_high", "ratio", "p_one_sided", "p_holm", "holm_family"]
    with (results / "inference.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f"PRIMARY family (k=15), {BOOTSTRAPS} bootstraps, {PERMUTATIONS} permutations, seed {SEED}\n")
    print(f"{'comparison':<28}{'stat':<8}{'difference':>12}{'95% CI':>26}{'ratio':>8}{'p_holm':>10}")
    print("-" * 92)
    for row in primary:
        ci = f"[{row['ci_low']:.1f}, {row['ci_high']:.1f}]"
        print(f"{row['comparison']:<28}{row['statistic']:<8}{row['difference']:>12.1f}"
              f"{ci:>26}{row['ratio']:>8.2f}{row['p_holm']:>10.4f}")

    print(f"\nK-ROBUSTNESS (mean), Holm across {len(secondary)} tests")
    print(f"{'k':>5}{'vs nonce':>26}{'vs prior_titles':>30}")
    for k in KS:
        if k == 15:
            continue
        cells = []
        for baseline in BASELINES:
            row = next(r for r in secondary
                       if r["k"] == k and r["statistic"] == "mean" and baseline in r["comparison"])
            cells.append(f"{row['difference']:>9.1f} (x{row['ratio']:.2f}, p={row['p_holm']:.4f})")
        print(f"{k:>5}{cells[0]:>26}{cells[1]:>30}")

    print(f"\nWrote {results / 'inference.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
