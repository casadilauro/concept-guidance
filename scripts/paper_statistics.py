"""Reconstruct the current manuscript's four geometry claim tables.

Extracted from figure_review/{setup,geometry,selected_statistics}.py.
Preserves float64 normalization, seed 42 pooled PCA, and seed 0 matched draws
in the order human, breadth-first, prior titles, nonce. No notebook dependency.
"""

import argparse
import csv
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

POPS = ("baseline", "nonce", "prior_titles", "hierarchical")


def write_rows(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def geometry(units, human_direction):
    center = units.mean(axis=0)
    distance = max(0., 1 - (center / np.linalg.norm(center)) @ human_direction)
    spread = 1 - (len(units) * np.dot(center, center) - 1) / (len(units) - 1)
    return distance, spread


def reconstruct(results: Path):
    with (results / "records.csv").open(newline="") as handle:
        records = list(csv.DictReader(handle))
    populations = np.array([r["population"] for r in records])
    rho = np.array([float(r["rho"]) for r in records])
    raw = np.load(results / "raw_embeddings.npy", allow_pickle=False).astype(np.float64)
    units = raw / np.linalg.norm(raw, axis=1, keepdims=True)
    indices = {p: np.flatnonzero(populations == p) for p in POPS}
    cosine = np.clip(1 - units @ units.T, 0, 2)
    np.fill_diagonal(cosine, 0)
    human = indices["baseline"]
    human_direction = units[human].mean(axis=0)
    human_direction /= np.linalg.norm(human_direction)

    pca = PCA(n_components=2, svd_solver="randomized", random_state=42)
    xy = pca.fit_transform(raw)
    human_center = xy[human].mean(axis=0)
    pca_rows = []
    for population in POPS:
        ids = indices[population]
        raw_centered = raw[ids] - raw[ids].mean(axis=0)
        projected_centered = xy[ids] - xy[ids].mean(axis=0)
        pca_rows.append(dict(population=population, n=len(ids),
            pooled_pc1_fraction=pca.explained_variance_ratio_[0],
            pooled_pc2_fraction=pca.explained_variance_ratio_[1],
            pooled_two_pc_fraction=pca.explained_variance_ratio_.sum(),
            within_population_variance_retained=np.sum(projected_centered ** 2) / np.sum(raw_centered ** 2),
            projected_centroid_distance_to_human=np.linalg.norm(xy[ids].mean(axis=0) - human_center)))

    generator = np.random.default_rng(0)
    matched = len(human)
    geometries, spectrum_rows = {}, {}
    for population in ("baseline", "hierarchical", "prior_titles", "nonce"):
        ids = indices[population]
        fractions, prs, drawn_geometry = [], [], []
        for _ in range(1 if population == "baseline" else 50):
            local = np.arange(matched) if population == "baseline" else generator.choice(len(ids), matched, replace=False)
            block = units[ids[local]]
            centered = block - block.mean(axis=0)
            eigenvalues = np.linalg.eigvalsh(centered @ centered.T / (matched - 1))[::-1]
            eigenvalues = np.clip(eigenvalues[:matched - 1], 0, None)
            fraction = eigenvalues / eigenvalues.sum()
            fractions.append(fraction)
            prs.append(1 / np.sum(fraction ** 2))
            drawn_geometry.append(geometry(block, human_direction))
        fractions, prs = np.array(fractions), np.array(prs)
        geometries[population] = np.array(drawn_geometry)
        cumulative = np.cumsum(fractions, axis=1)
        spectrum_rows[population] = dict(population=population, sample_n=matched, draws=len(prs),
            participation_ratio_mean=prs.mean(), participation_ratio_sd=prs.std(ddof=1) if len(prs) > 1 else 0.,
            participation_ratio_q05=np.quantile(prs, .05), participation_ratio_q95=np.quantile(prs, .95),
            top10_variance_share_mean=cumulative[:, 9].mean(),
            components_to_90pct_mean=(np.argmax(cumulative >= .9, axis=1) + 1).mean())
    for row in spectrum_rows.values():
        row["participation_ratio_fraction_of_human"] = row["participation_ratio_mean"] / spectrum_rows["baseline"]["participation_ratio_mean"]

    geometry_rows = []
    for population in POPS:
        ids = indices[population]
        distance, _ = geometry(units[ids], human_direction)
        spread = cosine[np.ix_(ids, ids)].sum() / (len(ids) * (len(ids) - 1))
        row = dict(population=population, n=len(ids), centroid_cosine_from_baseline=distance,
                   mean_pairwise_cosine_distance=spread)
        for index, short in enumerate(("centroid", "spread")):
            draws = geometries[population][:, index]
            row.update({f"matched_{short}_mean": draws.mean(),
                        f"matched_{short}_q05": np.quantile(draws, .05),
                        f"matched_{short}_q95": np.quantile(draws, .95)})
        geometry_rows.append(row)
    for row in geometry_rows:
        row["full_spread_fraction_of_human"] = row["mean_pairwise_cosine_distance"] / geometry_rows[0]["mean_pairwise_cosine_distance"]

    human_distances = cosine[np.ix_(human, human)].copy()
    np.fill_diagonal(human_distances, np.inf)
    radius = np.sort(human_distances, axis=1)[:, 4]
    proximity_rows = []
    for population in POPS[1:]:
        ids = indices[population]
        cross = cosine[np.ix_(ids, human)]
        generated_to_human, human_to_generated = cross.min(axis=1), cross.min(axis=0)
        values, covered = rho[ids], human_to_generated <= radius
        proximity_rows.append(dict(population=population, generated_n=len(ids), human_n=matched,
            rho_mean=values.mean(), rho_median=np.median(values), rho_q25=np.quantile(values, .25), rho_q75=np.quantile(values, .75),
            generated_to_human_mean=generated_to_human.mean(), generated_to_human_median=np.median(generated_to_human),
            generated_to_human_q25=np.quantile(generated_to_human, .25), generated_to_human_q75=np.quantile(generated_to_human, .75),
            human_to_generated_mean=human_to_generated.mean(), human_to_generated_median=np.median(human_to_generated),
            humans_covered=int(covered.sum()), human_coverage=covered.mean()))

    for name, rows in (("fig01", pca_rows), ("fig02", [spectrum_rows[p] for p in POPS]),
                       ("fig04", geometry_rows), ("fig16", proximity_rows)):
        write_rows(results / f"{name}_claim_statistics.csv", rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    reconstruct(parser.parse_args().results)


if __name__ == "__main__":
    main()
