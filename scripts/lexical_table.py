"""Lexical and surface-form diversity, as a check that embedding rho is not alone.

Embedding-space diversity is a proxy. These measures need no embedding model at
all, so agreement between them and rho is evidence the ordering is a property of
the text rather than of one encoder.

Corpus-level distinct-n and unique-item rates depend on sample size, so
every population is subsampled to the size of the human corpus and averaged.
Rates that are already per-recipe means (unique-unigram ratio) and pairwise
Jaccard are size-stable and reported at full n.

Writes CSV and a booktabs LaTeX table.
"""

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np
from scipy import sparse

TOKEN = re.compile(r"[a-z0-9]+")
DRAWS = 50
SEED = 0
POPULATIONS = [
    ("baseline", "data/original_recipes", "Human corpus"),
    ("hierarchical", "runs/wide_1000", "Breadth-first"),
    ("prior_titles", "runs/prior_titles_1000", "Prior titles"),
    ("nonce", "runs/nonce_1000", "Nonce"),
]


def tokens(text: str) -> list[str]:
    return TOKEN.findall(text.lower())


def ngrams(items: list[str], size: int) -> list[tuple[str, ...]]:
    return [tuple(items[i:i + size]) for i in range(len(items) - size + 1)]


def distinct_rate(documents: list[list[str]], size: int) -> float:
    """Corpus-level distinct-n: unique n-grams over total n-grams."""
    total, seen = 0, set()
    for document in documents:
        grams = ngrams(document, size)
        total += len(grams)
        seen.update(grams)
    return len(seen) / total if total else 0.0


def mean_pairwise_jaccard_distance(documents: list[list[str]]) -> float:
    """Exact mean over all pairs, via a sparse incidence matrix."""
    vocabulary: dict[str, int] = {}
    rows, cols = [], []
    for index, document in enumerate(documents):
        for token in set(document):
            rows.append(index)
            cols.append(vocabulary.setdefault(token, len(vocabulary)))
    matrix = sparse.csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)),
        shape=(len(documents), len(vocabulary)),
    )
    intersection = (matrix @ matrix.T).toarray()
    sizes = np.asarray(matrix.sum(axis=1)).ravel()
    union = sizes[:, None] + sizes[None, :] - intersection
    with np.errstate(divide="ignore", invalid="ignore"):
        similarity = np.where(union > 0, intersection / union, 0.0)
    upper = np.triu_indices(len(documents), k=1)
    return float(1.0 - similarity[upper].mean())


def load(folder: Path) -> tuple[list[str], list[str]]:
    rows = [json.loads(line) for line in (folder / "recipes.jsonl").read_text(encoding="utf-8").splitlines() if line]
    rows.sort(key=lambda row: str(row.get("recipe_id", "")))
    titles = [str(row.get("title") or "") for row in rows]
    texts = [str(row["text"]) for row in rows]
    return titles, texts


def measure(titles: list[str], texts: list[str]) -> dict[str, float]:
    title_tokens = [tokens(t) for t in titles]
    text_tokens = [tokens(t) for t in texts]
    return {
        "unique_title_rate": len(set(titles)) / len(titles),
        "unique_text_rate": len(set(texts)) / len(texts),
        "distinct_1": distinct_rate(text_tokens, 1),
        "distinct_2": distinct_rate(text_tokens, 2),
        "distinct_3": distinct_rate(text_tokens, 3),
        "per_recipe_unique_unigram": float(np.mean([
            len(set(d)) / len(d) for d in text_tokens if d
        ])),
        "title_jaccard_distance": mean_pairwise_jaccard_distance(title_tokens),
        "text_jaccard_distance": mean_pairwise_jaccard_distance(text_tokens),
    }


SIZE_SENSITIVE = ("unique_title_rate", "unique_text_rate", "distinct_1", "distinct_2", "distinct_3")

HEADERS = [
    ("unique_title_rate", "Unique title rate"),
    ("unique_text_rate", "Unique text rate"),
    ("distinct_1", "Distinct-1"),
    ("distinct_2", "Distinct-2"),
    ("distinct_3", "Distinct-3"),
    ("per_recipe_unique_unigram", "Per-recipe unique unigram"),
    ("title_jaccard_distance", "Mean pairwise title Jaccard dist."),
    ("text_jaccard_distance", "Mean pairwise text Jaccard dist."),
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="results/paper_v2")
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    loaded = {name: load(Path(folder) if name == "baseline" else args.runs / Path(folder).name)
              for name, folder, _ in POPULATIONS}
    matched = len(loaded["baseline"][0])
    generator = np.random.default_rng(SEED)
    results: dict[str, dict[str, float]] = {}
    standard_deviations: dict[str, dict[str, float]] = {}

    for name, _, _ in POPULATIONS:
        titles, texts = loaded[name]
        full = measure(titles, texts)
        standard_deviations[name] = {}
        if len(titles) > matched:
            draws = []
            for _ in range(DRAWS):
                pick = generator.choice(len(titles), matched, replace=False)
                draws.append(measure([titles[i] for i in pick], [texts[i] for i in pick]))
            for key in SIZE_SENSITIVE:
                full[key] = float(np.mean([draw[key] for draw in draws]))
                standard_deviations[name][key] = float(np.std([draw[key] for draw in draws], ddof=1))
        full["n"] = len(titles)
        results[name] = full

    with (out / "lexical_diversity.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["measure", *(name for name, _, _ in POPULATIONS)])
        writer.writerow(["n", *(int(results[name]["n"]) for name, _, _ in POPULATIONS)])
        for key, label in HEADERS:
            writer.writerow([label, *(round(results[name][key], 4) for name, _, _ in POPULATIONS)])

    labels = {name: label for name, _, label in POPULATIONS}
    lines = [
        r"\begin{table}[!htbp]", r"  \centering", r"  \small",
        r"  \caption{Lexical and surface-form diversity. Rates marked $\dagger$ grow with"
        r" sample size and are averaged over 50 subsamples of $n=168$ to match the human"
        r" corpus. Parentheses report standard deviations across those subsamples for the"
        r" generated populations; the remainder are size-stable and computed at full $n$.}",
        r"  \label{tab:lexical-diversity}",
        r"  \begin{tabular}{l" + "r" * len(POPULATIONS) + "}", r"    \toprule",
        "    Measure & " + " & ".join(labels[n] for n, _, _ in POPULATIONS) + r" \\",
        r"    \midrule",
    ]
    for key, label in HEADERS:
        mark = r"$\dagger$" if key in SIZE_SENSITIVE else ""
        cells = " & ".join(
            f"{results[name][key]:.3f} ({standard_deviations[name][key]:.3f})"
            if key in standard_deviations[name] else f"{results[name][key]:.3f}"
            for name, _, _ in POPULATIONS
        )
        lines.append(f"    {label}{mark} & {cells} " + r"\\")
    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
    (out / "table_lexical_diversity.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    width = max(len(label) for _, label in HEADERS) + 2
    print(f"{'measure':<{width}}" + "".join(f"{labels[n]:>15}" for n, _, _ in POPULATIONS))
    print("-" * (width + 15 * len(POPULATIONS)))
    print(f"{'n':<{width}}" + "".join(f"{int(results[n]['n']):>15}" for n, _, _ in POPULATIONS))
    for key, label in HEADERS:
        mark = " †" if key in SIZE_SENSITIVE else ""
        print(f"{label + mark:<{width}}" + "".join(f"{results[n][key]:>15.3f}" for n, _, _ in POPULATIONS))
    print(f"\n† averaged over {DRAWS} subsamples of n={matched}")
    print(f"Wrote {out/'lexical_diversity.csv'} and {out/'table_lexical_diversity.tex'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
