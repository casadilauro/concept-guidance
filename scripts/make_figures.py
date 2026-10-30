"""Publication figures for the three-strategy comparison.

Writes vector PDF (for \\includegraphics) and 400-dpi PNG side by side.

Palette is the validated categorical order (blue, orange, aqua, yellow),
assigned to entities in a fixed order so a figure that drops a series never
repaints the survivors. Aqua and yellow fall below 3:1 against the surface, so
every series carries a visible direct label rather than relying on colour alone.
"""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
from matplotlib import pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

# Fixed entity -> slot assignment. Never reordered, never cycled.
COLOR = {
    "baseline": "#2a78d6",
    "hierarchical": "#eb6834",
    "prior_titles": "#1baf7a",
    "nonce": "#eda100",
}
LABEL = {
    "baseline": "human corpus",
    "hierarchical": "breadth-first",
    "prior_titles": "prior titles",
    "nonce": "nonce",
}
ORDER = ["baseline", "hierarchical", "prior_titles", "nonce"]
KS = (1, 2, 3, 5, 10, 15, 25, 50, 100, 200)
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#dedcd6"


def style() -> None:
    plt.rcParams.update({
        "figure.dpi": 400,
        "savefig.dpi": 400,
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def recessive(axis) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.set_axisbelow(True)


def save(figure, out: Path, name: str) -> None:
    for suffix in ("pdf", "png"):
        figure.savefig(out / f"{name}.{suffix}")
    plt.close(figure)
    print(f"  {out/name}.pdf  +  .png")


def rho_values(vectors: np.ndarray, k: int) -> np.ndarray:
    neighbours = min(k, len(vectors) - 1)
    distances, _ = NearestNeighbors(n_neighbors=neighbours + 1).fit(vectors).kneighbors(vectors)
    return (distances[:, 1:] ** 2).mean(axis=1)


def covering_number(units: np.ndarray, epsilon: float) -> int:
    centers = units[:1]
    for vector in units[1:]:
        if float(np.max(centers @ vector)) < 1.0 - epsilon:
            centers = np.vstack((centers, vector))
    return len(centers)


def figure_rho_curve(scaled, populations, out: Path) -> None:
    """Headline: the ordering holds at every neighbourhood size."""
    figure, axis = plt.subplots(figsize=(5.6, 3.6))
    for name in ORDER:
        block = scaled[populations == name]
        ks = [k for k in KS if k <= len(block) - 1]
        means = [rho_values(block, k).mean() for k in ks]
        axis.plot(ks, means, marker="o", markersize=4, linewidth=1.8,
                  color=COLOR[name], label=LABEL[name], clip_on=False, zorder=3)
        axis.annotate(LABEL[name], (ks[-1], means[-1]), textcoords="offset points",
                      xytext=(7, -1), color=COLOR[name], fontsize=8.5,
                      va="center")
    axis.set_xscale("log"); axis.set_yscale("log")
    axis.set_xlabel("k (neighborhood size)")
    axis.set_ylabel(r"mean $\rho$")
    axis.set_xticks(KS); axis.set_xticklabels([str(k) for k in KS])
    axis.set_xlim(0.9, 260)
    axis.margins(y=0.12)
    recessive(axis)
    save(figure, out, "fig1_rho_curve")


def figure_rho_distribution(scaled, populations, out: Path) -> None:
    """Distributions at k=15, log scale — the populations span three decades."""
    figure, axis = plt.subplots(figsize=(5.6, 3.4))
    data = [rho_values(scaled[populations == name], 15) for name in ORDER]
    boxes = axis.boxplot(data, patch_artist=True, widths=0.55, showfliers=False,
                         medianprops={"color": "#fcfcfb", "linewidth": 1.4},
                         whiskerprops={"color": MUTED, "linewidth": 0.9},
                         capprops={"color": MUTED, "linewidth": 0.9})
    for patch, name in zip(boxes["boxes"], ORDER, strict=True):
        patch.set_facecolor(COLOR[name])
        patch.set_edgecolor("#fcfcfb")
        patch.set_linewidth(1.2)
    # Beside the box, not above it — on a log axis the whiskers reach far enough
    # that a label offset vertically lands on top of them.
    for index, (name, values) in enumerate(zip(ORDER, data, strict=True), start=1):
        axis.annotate(f"{np.median(values):,.0f}", (index + 0.33, np.median(values)),
                      ha="left", va="center", fontsize=8, color=INK)
    axis.set_yscale("log")
    axis.set_xticklabels([LABEL[name] for name in ORDER])
    axis.set_ylabel(r"$\rho$ at $k=15$")
    axis.grid(axis="x", visible=False)
    recessive(axis)
    save(figure, out, "fig2_rho_distribution")


def effective_rank(block: np.ndarray) -> float:
    centered = block - block.mean(axis=0)
    spectrum = np.linalg.svd(centered, compute_uv=False) ** 2 / (len(block) - 1)
    return float(spectrum.sum() ** 2 / (spectrum ** 2).sum())


def figure_measures(units, populations, out: Path, draws: int = 50) -> None:
    """Small multiples — three measures on three scales, never one dual axis.

    Effective rank and the greedy epsilon-net count both grow with sample size, so comparing
    a 1,000-recipe population against the 168-recipe human corpus at full n
    would flatter the generated populations. Both panels subsample every
    population to the size of the human corpus and average over `draws`.
    Coverage is exempt: it is defined against the human corpus and the three
    generated populations already share one n.
    """
    base = units[populations == "baseline"]
    matched = len(base)
    within, _ = NearestNeighbors(n_neighbors=6, metric="cosine").fit(base).kneighbors(base)
    radius = within[:, -1]
    generator = np.random.default_rng(0)

    values = {}
    errors = {}
    for name in ORDER:
        block = units[populations == name]
        if len(block) == matched:
            rank, modes = effective_rank(block), covering_number(block, 0.10)
            rank_sd, modes_sd = np.nan, np.nan
        else:
            samples = [block[generator.choice(len(block), matched, replace=False)]
                       for _ in range(draws)]
            ranks = np.array([effective_rank(s) for s in samples])
            mode_counts = np.array([covering_number(s, 0.10) for s in samples])
            rank, rank_sd = float(ranks.mean()), float(ranks.std(ddof=1))
            modes, modes_sd = float(mode_counts.mean()), float(mode_counts.std(ddof=1))
        if name == "baseline":
            coverage = np.nan
        else:
            nearest, _ = NearestNeighbors(n_neighbors=1, metric="cosine").fit(block).kneighbors(base)
            coverage = float(np.mean(nearest[:, 0] <= radius)) * 100
        values[name] = (rank, modes, coverage)
        errors[name] = (rank_sd, modes_sd, 0.0)

    panels = [
        (f"effective rank (n={matched})", 0, "{:.1f}", ORDER),
        (f"greedy ε-net count, ε=0.1 (n={matched})", 1, "{:,.0f}", ORDER),
        ("covers human space (%)", 2, "{:.1f}%", [n for n in ORDER if n != "baseline"]),
    ]
    figure, axes = plt.subplots(1, 3, figsize=(7.4, 2.7))
    for axis, (title, index, fmt, names) in zip(axes, panels, strict=True):
        heights = [values[name][index] for name in names]
        xerr = [errors[name][index] for name in names] if index < 2 else None
        positions = np.arange(len(names))
        axis.barh(positions, heights, xerr=xerr, height=0.62,
                  color=[COLOR[name] for name in names],
                  edgecolor="#fcfcfb", linewidth=1.2, zorder=3,
                  error_kw={"ecolor": INK, "elinewidth": 0.8, "capsize": 2})
        for position, height in zip(positions, heights, strict=True):
            axis.annotate(fmt.format(height), (height, position), textcoords="offset points",
                          xytext=(4, 0), va="center", fontsize=8, color=INK)
        axis.set_yticks(positions)
        axis.set_yticklabels([LABEL[name] for name in names])
        axis.invert_yaxis()
        axis.set_title(title, pad=8)
        axis.grid(axis="y", visible=False)
        axis.margins(x=0.22)
        recessive(axis)
    figure.tight_layout(w_pad=2.0)
    save(figure, out, "fig3_diversity_measures")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="results/paper_v2")
    parser.add_argument("--out", default="results/paper_v2/figures")
    arguments = parser.parse_args()
    results, out = Path(arguments.results), Path(arguments.out)
    out.mkdir(parents=True, exist_ok=True)
    style()

    scaled = np.load(results / "scaled_embeddings.npy")
    units = normalize(np.load(results / "raw_embeddings.npy"))
    with (results / "records.csv").open(encoding="utf-8", newline="") as handle:
        populations = np.array([row["population"] for row in csv.DictReader(handle)])

    figure_rho_curve(scaled, populations, out)
    figure_rho_distribution(scaled, populations, out)
    figure_measures(units, populations, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
