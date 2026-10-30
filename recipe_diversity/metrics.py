"""Diversity measures that a single rho value cannot express.

rho at one k answers "how far apart are a recipe's nearest neighbours". That is
local spacing. A scaffold that partitions a population into well-separated
groups changes the answer at large k while leaving it unchanged, or worse, at
small k, because siblings inside a group are near-duplicates of each other. The
measures here separate those scales:

- `rho_curve`     local-to-global spacing as a function of k
- `covering_curve` how many distinct modes the population resolves
- `stratification` whether a scaffold's groups are real, against a shuffled null
- `manifold_coverage` how much of a reference population is actually reached
"""

from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.neighbors import NearestNeighbors

K_CURVE = (1, 2, 3, 5, 10, 15, 25, 50, 100, 200)
EPSILONS = (0.1, 0.2, 0.3, 0.4, 0.5)
COVERAGE_K = 5
NULL_REPEATS = 20


def rho_at(vectors: np.ndarray, k: int) -> np.ndarray:
    """Mean squared distance to each point's k nearest neighbours."""
    if k > len(vectors) - 1:
        raise ValueError(f"k={k} needs at least {k + 1} vectors, got {len(vectors)}.")
    distances, _ = NearestNeighbors(n_neighbors=k + 1).fit(vectors).kneighbors(vectors)
    return (distances[:, 1:] ** 2).mean(axis=1)


def rho_curve(vectors: np.ndarray, ks: Sequence[int] = K_CURVE) -> list[dict[str, float]]:
    """rho across a range of k.

    Only k values the population can actually support are returned, so a small
    population never reports a saturated value that looks comparable to a large
    one. Read the curve, not one point: a crossover between two populations
    means the scaffold moved diversity between scales rather than adding it.
    """
    return [
        {
            "k": k,
            "mean_rho": float(np.mean(values)),
            "median_rho": float(np.median(values)),
        }
        for k in ks
        if k <= len(vectors) - 1
        for values in (rho_at(vectors, k),)
    ]


def covering_number(unit_vectors: np.ndarray, epsilon: float) -> int:
    """Greedy epsilon-net size: distinct modes at cosine radius `epsilon`.

    Deterministic in the input order, which the analysis fixes by sorting
    records. This is an upper bound on the true covering number; the greedy
    approximation is standard and adequate for comparing populations of equal
    size. Counts are not comparable across populations of different n.
    """
    if not len(unit_vectors):
        return 0
    centers = unit_vectors[:1]
    for vector in unit_vectors[1:]:
        if float(np.max(centers @ vector)) < 1.0 - epsilon:
            centers = np.vstack((centers, vector))
    return len(centers)


def covering_curve(
    unit_vectors: np.ndarray,
    epsilons: Sequence[float] = EPSILONS,
) -> list[dict[str, float]]:
    return [
        {"epsilon": float(epsilon), "modes": covering_number(unit_vectors, epsilon)}
        for epsilon in epsilons
    ]


def _between_within(unit_vectors: np.ndarray, labels: np.ndarray) -> float:
    groups = np.unique(labels)
    members = [unit_vectors[labels == group] for group in groups]
    within = float(np.mean([np.var(block, axis=0).sum() for block in members]))
    between = float(np.var(np.array([block.mean(axis=0) for block in members]), axis=0).sum())
    return between / within if within > 0 else 0.0


def stratification(
    unit_vectors: np.ndarray,
    labels: Sequence[str],
    *,
    repeats: int = NULL_REPEATS,
    seed: int = 0,
) -> dict[str, Any]:
    """Between-group over within-group variance, against a shuffled-label null.

    The null is the same statistic with the group assignments permuted, so it
    measures what this many groups of this size would score by chance. A ratio
    far above the null means the scaffold created real, separated regions —
    the thing a hierarchy is for, and the thing rho cannot report.
    """
    labels = np.asarray(labels)
    if len(labels) != len(unit_vectors):
        raise ValueError("stratification needs one label per vector.")
    if len(np.unique(labels)) < 2:
        raise ValueError("stratification needs at least two groups.")
    observed = _between_within(unit_vectors, labels)
    generator = np.random.default_rng(seed)
    null = float(
        np.mean([
            _between_within(unit_vectors, generator.permutation(labels))
            for _ in range(repeats)
        ])
    )
    return {
        "groups": int(len(np.unique(labels))),
        "between_within": observed,
        "null_between_within": null,
        "ratio_to_null": observed / null if null > 0 else float("inf"),
    }


def manifold_coverage(
    reference: np.ndarray,
    generated: np.ndarray,
    k: int = COVERAGE_K,
) -> dict[str, float]:
    """Fraction of reference points with a generated point inside their k-NN radius.

    This is the coverage measure of Naeem et al. It asks whether a population
    reached the reference population's space at all, which is a different
    question from whether its own members are spread out. A population can be
    highly dispersed and still miss most of what it is meant to cover.
    """
    if k > len(reference) - 1:
        raise ValueError(f"coverage k={k} needs at least {k + 1} reference vectors.")
    if not len(generated):
        raise ValueError("coverage needs at least one generated vector.")
    within, _ = NearestNeighbors(n_neighbors=k + 1, metric="cosine").fit(reference).kneighbors(reference)
    radius = within[:, -1]
    nearest, _ = NearestNeighbors(n_neighbors=1, metric="cosine").fit(generated).kneighbors(reference)
    return {
        "k": k,
        "coverage": float(np.mean(nearest[:, 0] <= radius)),
        "median_distance": float(np.median(nearest[:, 0])),
    }


def scaffold_levels(metadata: Sequence[Mapping[str, Any]]) -> dict[str, list[str]]:
    """Scaffold label columns present on every record, coarsest level first."""
    return {
        level: [str(entry[level]) for entry in metadata]
        for level in ("cuisine_id", "dish_id")
        if metadata and all(entry.get(level) is not None for entry in metadata)
    }
