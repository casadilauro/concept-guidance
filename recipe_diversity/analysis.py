import csv


from dataclasses import dataclass


import json


import os


from pathlib import Path


from typing import Any, Callable, Mapping


import matplotlib


import numpy as np


from rich.progress import Progress




from sklearn.decomposition import PCA




from sklearn.neighbors import NearestNeighbors


from sklearn.preprocessing import normalize


from . import EMBEDDING_BACKEND, EMBEDDING_MODEL, K_NEIGHBORS


from .metrics import (
    COVERAGE_K,
    covering_curve,
    manifold_coverage,
    rho_curve,
    scaffold_levels,
    stratification,
)


from matplotlib import pyplot as plt


@dataclass(frozen=True)
class RecipeRecord:
    identifier: str
    title: str
    text: str
    population: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class AnalysisArtifacts:
    output_dir: Path
    summary: dict[str, Any]


Encoder = Callable[[list[str]], np.ndarray]


REQUIRED_POPULATIONS = ("baseline", "nonce", "prior_titles", "hierarchical")


OPTIONAL_POPULATIONS = ()


def _title_from(text: str) -> str:
    """Mirrors generation._title: skip an echoed nonce line."""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("Nonce:"):
            return stripped
    return ""


def _record(data: dict[str, Any], population: str, fallback_identifier: str) -> RecipeRecord:
    text = data.get("text")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Each response record must contain non-whitespace text.")
    identifier = str(data.get("recipe_id", fallback_identifier))
    title = data.get("title") or _title_from(text)
    metadata = {key: value for key, value in data.items() if key not in {"recipe_id", "title", "text"}}
    return RecipeRecord(identifier, str(title), text, population, metadata)


def load_response_folder(folder: Path, population: str) -> list[RecipeRecord]:
    jsonl = folder / "recipes.jsonl"
    if jsonl.exists():
        rows = [json.loads(line) for line in jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
        return [_record(row, population, str(index)) for index, row in sorted(enumerate(rows), key=lambda item: (str(item[1].get("recipe_id", "")), item[0]))]

    return [
        _record({"recipe_id": path.stem, "text": path.read_text(encoding="utf-8")}, population, path.stem)
        for path in sorted(folder.glob("*.txt"))
    ]


def minmax_scale(vectors: np.ndarray, low: float = 0.0, high: float = 3.0) -> np.ndarray:
    mins = vectors.min(axis=0)
    spans = vectors.max(axis=0) - mins
    scaled = np.zeros_like(vectors, dtype=np.float32)
    variable = spans > 1e-12
    scaled[:, variable] = (vectors[:, variable] - mins[variable]) / spans[variable]
    return scaled * (high - low) + low


def compute_rho(vectors: np.ndarray, k: int = K_NEIGHBORS) -> np.ndarray:
    if len(vectors) < 2:
        raise ValueError("At least two response records are required to compute rho.")
    distances, _ = NearestNeighbors(n_neighbors=min(k, len(vectors) - 1) + 1).fit(vectors).kneighbors(vectors)
    return (distances[:, 1:] ** 2).mean(axis=1)


def encode_remotely(texts: list[str], progress: Progress | None = None) -> np.ndarray:
    from .embedding import EmbeddingSettings, embed_texts

    result = embed_texts(texts, EmbeddingSettings(model=EMBEDDING_MODEL), progress=progress)
    message = (
        f"Embedded {len(texts)} texts with {EMBEDDING_MODEL} "
        f"({result.requested} requested, {result.cached} cached, "
        f"{result.prompt_tokens} tokens)"
    )
    print(message) if progress is None else progress.console.log(message)
    return result.vectors


def encode_texts(texts: list[str], progress: Progress | None = None) -> np.ndarray:
    return encode_remotely(texts, progress)


def explained_variance(vectors: np.ndarray) -> np.ndarray:
    if len(vectors) < 2:
        return np.zeros(0, dtype=np.float64)
    centered = np.asarray(vectors, dtype=np.float64) - np.mean(vectors, axis=0, dtype=np.float64)
    return np.linalg.svd(centered, compute_uv=False) ** 2 / (len(vectors) - 1)


def embedding_diagnostics(vectors: np.ndarray) -> dict[str, Any]:
    """Report how much of the embedding space this population actually occupies.

    `effective_rank` is the participation ratio of the covariance spectrum. It
    is the direct measure of whether a wider encoder buys real information: a
    3072-dimension model whose effective rank matches a 384-dimension one is
    not resolving any additional distinctions between recipes.
    """
    eigenvalues = explained_variance(vectors)
    total = float(eigenvalues.sum())
    cumulative = np.cumsum(eigenvalues)
    return {
        "model": EMBEDDING_MODEL,
        "backend": EMBEDDING_BACKEND,
        "dimension": int(vectors.shape[1]),
        "effective_rank": total**2 / float((eigenvalues**2).sum()) if total > 0 else 0.0,
        "variance_explained": {
            str(count): float(cumulative[count - 1] / total)
            for count in (10, 50, 100, 200)
            if total > 0 and count <= len(eigenvalues)
        },
    }


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plot_rho_curve(path: Path, curves: Mapping[str, list[dict[str, float]]]) -> None:
    figure, axis = plt.subplots()
    for population, rows in curves.items():
        axis.plot(
            [row["k"] for row in rows],
            [row["mean_rho"] for row in rows],
            marker="o",
            label=population,
        )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("k (neighbours)")
    axis.set_ylabel("mean rho")
    axis.legend()
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def scale_rows(
    records: list[RecipeRecord],
    scaled_embeddings: np.ndarray,
    raw_embeddings: np.ndarray,
    population: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Per-population rho curve, covering curve, and scaffold stratification rows.

    The rho curve uses the same scaled space as rho itself so the values are
    directly comparable; the covering and stratification measures use cosine on
    unit-normalised raw vectors, where an angular radius is meaningful.
    """
    units = normalize(raw_embeddings)
    curve = [{"population": population, **row} for row in rho_curve(scaled_embeddings)]
    covering = [{"population": population, **row} for row in covering_curve(units)]
    strata = [
        {"population": population, "level": level, **stratification(units, labels)}
        for level, labels in scaffold_levels([record.metadata for record in records]).items()
        if len(set(labels)) > 1
    ]
    return curve, covering, strata


def write_comparison_artifacts(
    records: list[RecipeRecord],
    raw_embeddings: np.ndarray,
    scaled_embeddings: np.ndarray,
    rho: np.ndarray,
    output_dir: Path,
    source_folders: Mapping[str, Path],
) -> AnalysisArtifacts:
    output_dir.mkdir(parents=True, exist_ok=True)
    populations = tuple(source_folders)
    pca = PCA(n_components=2, random_state=0).fit_transform(raw_embeddings)
    record_rows = [
        {
            "identifier": record.identifier,
            "title": record.title,
            "population": record.population,
            "rho": float(rho[index]),
            "pca_x": float(pca[index, 0]),
            "pca_y": float(pca[index, 1]),
        }
        for index, record in enumerate(records)
    ]
    summary_rows = []
    for population in populations:
        values = rho[[record.population == population for record in records]]
        summary_rows.append(
            {
                "population": population,
                "n": len(values),
                "mean_rho": float(np.mean(values)),
                "median_rho": float(np.median(values)),
                "min_rho": float(np.min(values)),
                "max_rho": float(np.max(values)),
            }
        )

    _write_csv(
        output_dir / "records.csv",
        record_rows,
        ["identifier", "title", "population", "rho", "pca_x", "pca_y"],
    )
    _write_csv(
        output_dir / "population_summary.csv",
        summary_rows,
        ["population", "n", "mean_rho", "median_rho", "min_rho", "max_rho"],
    )

    members = {
        population: np.array([record.population == population for record in records])
        for population in populations
    }
    curves: dict[str, list[dict[str, Any]]] = {}
    covering: list[dict[str, Any]] = []
    strata: list[dict[str, Any]] = []
    for population, mask in members.items():
        curve, population_covering, population_strata = scale_rows(
            [record for record in records if record.population == population],
            scaled_embeddings[mask],
            raw_embeddings[mask],
            population,
        )
        curves[population] = curve
        covering.extend(population_covering)
        strata.extend(population_strata)

    units = normalize(raw_embeddings)
    reference = units[members["baseline"]]
    # A tiny baseline cannot support the default neighbourhood; clamp rather
    # than lose every other artifact in the comparison.
    coverage_k = min(COVERAGE_K, len(reference) - 1)
    coverage_rows = [
        {"population": population, **manifold_coverage(reference, units[mask], coverage_k)}
        for population, mask in members.items()
        if population != "baseline" and coverage_k >= 1
    ]

    _write_csv(
        output_dir / "rho_curve.csv",
        [row for rows in curves.values() for row in rows],
        ["population", "k", "mean_rho", "median_rho"],
    )
    _write_csv(output_dir / "covering_curve.csv", covering, ["population", "epsilon", "modes"])
    _write_csv(
        output_dir / "stratification.csv",
        strata,
        ["population", "level", "groups", "between_within", "null_between_within", "ratio_to_null"],
    )
    _write_csv(
        output_dir / "manifold_coverage.csv",
        coverage_rows,
        ["population", "k", "coverage", "median_distance"],
    )
    _plot_rho_curve(output_dir / "rho_curve.png", curves)

    np.save(output_dir / "raw_embeddings.npy", raw_embeddings)
    np.save(output_dir / "scaled_embeddings.npy", scaled_embeddings)

    figure, axis = plt.subplots()
    for population in populations:
        indices = np.array([record.population == population for record in records])
        axis.scatter(pca[indices, 0], pca[indices, 1], label=population)
    axis.set_xlabel("PCA 1")
    axis.set_ylabel("PCA 2")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "populations_pca.png")
    plt.close(figure)

    figure, axis = plt.subplots()
    axis.boxplot(
        [rho[[record.population == population for record in records]] for population in populations],
        tick_labels=populations,
    )
    axis.set_xlabel("Population")
    axis.set_ylabel("rho")
    figure.tight_layout()
    figure.savefig(output_dir / "rho_by_population.png")
    plt.close(figure)

    manifest = {
        "source_folders": {population: str(folder) for population, folder in source_folders.items()},
        "shared_scaling": True,
        "scaling_bounds": {"low": 0.0, "high": 3.0},
        "model": EMBEDDING_MODEL,
        "k": K_NEIGHBORS,
        "embedding": embedding_diagnostics(raw_embeddings),
        "embedding_by_population": {
            population: embedding_diagnostics(
                raw_embeddings[[record.population == population for record in records]]
            )
            for population in populations
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return AnalysisArtifacts(output_dir, {"populations": summary_rows})


def compare_populations(
    folders: Mapping[str, Path],
    output_dir: Path,
    encoder: Encoder | None = None,
    progress: Progress | None = None,
) -> AnalysisArtifacts:
    missing = [population for population in REQUIRED_POPULATIONS if population not in folders]
    unknown = [
        population
        for population in folders
        if population not in REQUIRED_POPULATIONS + OPTIONAL_POPULATIONS
    ]
    if missing or unknown:
        raise ValueError(
            f"Expected exactly these comparison folders: {REQUIRED_POPULATIONS}, "
            f"optionally {OPTIONAL_POPULATIONS}."
        )
    expected = tuple(
        population
        for population in REQUIRED_POPULATIONS + OPTIONAL_POPULATIONS
        if population in folders
    )
    source_folders = {population: folders[population] for population in expected}
    records = [
        record
        for population, folder in source_folders.items()
        for record in load_response_folder(folder, population)
    ]
    encode = encoder or (lambda texts: encode_texts(texts, progress))
    raw_embeddings = np.asarray(encode([record.text for record in records]))
    scaled_embeddings = minmax_scale(raw_embeddings)
    rho = np.full(len(records), np.nan, dtype=np.float64)
    for population in expected:
        indices = np.array([index for index, record in enumerate(records) if record.population == population])
        rho[indices] = compute_rho(scaled_embeddings[indices])
    return write_comparison_artifacts(
        records,
        raw_embeddings,
        scaled_embeddings,
        rho,
        output_dir,
        source_folders,
    )
