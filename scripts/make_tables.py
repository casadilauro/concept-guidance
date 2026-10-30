"""Create the two statistical LaTeX tables used by the paper."""

import argparse
import csv
from pathlib import Path

import numpy as np
from sklearn.preprocessing import normalize

from make_figures import ORDER, covering_number, effective_rank


LABEL = {
    "baseline": "Human corpus",
    "hierarchical": "Breadth-first",
    "prior_titles": "Prior titles",
    "nonce": "Nonce",
}


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def population_table(results: Path) -> str:
    summaries = {row["population"]: row for row in rows(results / "population_summary.csv")}
    coverage = {row["population"]: float(row["coverage"]) for row in rows(results / "manifold_coverage.csv")}
    raw = normalize(np.load(results / "raw_embeddings.npy"))
    populations = np.array([row["population"] for row in rows(results / "records.csv")])
    matched = int(np.sum(populations == "baseline"))
    generator = np.random.default_rng(0)
    supplemental: dict[str, tuple[float, float, float, float]] = {}
    for name in ORDER:
        block = raw[populations == name]
        if name == "baseline":
            supplemental[name] = (effective_rank(block), 0.0, covering_number(block, 0.1), 0.0)
            continue
        samples = [block[generator.choice(len(block), matched, replace=False)] for _ in range(50)]
        ranks = np.array([effective_rank(sample) for sample in samples])
        nets = np.array([covering_number(sample, 0.1) for sample in samples])
        supplemental[name] = (ranks.mean(), ranks.std(ddof=1), nets.mean(), nets.std(ddof=1))

    body = []
    for name in ORDER:
        summary = summaries[name]
        rank, rank_sd, net, net_sd = supplemental[name]
        n = f"{int(summary['n']):,}".replace(",", "{,}")
        if name == "baseline":
            values = (
                f"{float(summary['mean_rho']):.1f}",
                f"{float(summary['median_rho']):.1f}",
                f"{rank:.1f}",
                f"{net:.0f}",
                "---",
            )
        else:
            values = (
                f"{float(summary['mean_rho']):.1f}",
                f"{float(summary['median_rho']):.1f}",
                f"{rank:.1f} ({rank_sd:.1f})",
                f"{net:.1f} ({net_sd:.1f})",
                f"{100 * coverage[name]:.1f}\\%",
            )
        if name == "hierarchical":
            values = tuple(r"\textbf{" + value + "}" for value in values)
        body.append(f"    {LABEL[name]} & {n} & " + " & ".join(values) + r" \\")

    return "\n".join([
        r"\begin{table}[!htbp]",
        r"  \centering",
        r"  \caption{Population summaries. $\rho$ is within-population at $k=15$ on one",
        r"  shared min--max scale. Effective rank and the greedy $\varepsilon$-net count grow with sample",
        r"  size, so both are computed on subsamples of $n=168$ matched to the human",
        r"  corpus and averaged over 50 draws. Parentheses report standard deviations",
        r"  across those draws for generated populations; the human corpus is evaluated",
        r"  once. Coverage is the fraction of human recipes with a generated recipe inside",
        r"  their own $k{=}5$ neighborhood radius.}",
        r"  \label{tab:population-summary}",
        r"  \begin{tabular}{lrrrrrr}",
        r"    \toprule",
        r"    Population & $n$ & Mean $\rho$ & Median $\rho$ & Eff.\ rank & $\varepsilon$-net & Coverage \\",
        r"    \midrule",
        body[0],
        r"    \midrule",
        *body[1:],
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
        "",
    ])


def hypothesis_table(results: Path) -> str:
    primary = {
        (row["comparison"], row["statistic"]): row
        for row in rows(results / "inference.csv")
        if row["k"] == "15"
    }
    body = []
    for baseline, comparison in (("nonce", "hierarchical - nonce"), ("prior titles", "hierarchical - prior_titles")):
        for statistic in ("mean", "median"):
            row = primary[(comparison, statistic)]
            p = float(row["p_holm"])
            p_text = "$<0.001$" if p < 0.001 else f"${p:.3f}$"
            body.append(
                f"    vs.\\ {baseline} & {statistic} & {float(row['difference']):.1f} & "
                f"[{float(row['ci_low']):.1f}, {float(row['ci_high']):.1f}] & "
                f"{float(row['ratio']):.2f}$\\times$ & {p_text} " + r"\\"
            )
    return "\n".join([
        r"\begin{table}[!htbp]",
        r"  \centering",
        r"  \caption{Breadth-first scaffolding against each baseline at $k=15$. Differences are in within-population $\rho$ on the shared scale. Each ratio divides the breadth-first mean or median by the corresponding baseline statistic. Intervals are 5{,}000-resample percentile bootstraps; $p$-values are one-sided permutation tests ($10{,}000$ permutations) with Holm adjustment across the four-test family. Both procedures treat per-recipe values as exchangeable and summarize the realized generation runs rather than repeated-run performance.}",
        r"  \label{tab:hypothesis-tests}",
        r"  \begin{tabular}{llrrrr}",
        r"    \toprule",
        r"    Comparison & Statistic & Difference & 95\% CI & Ratio & $p$ (Holm) \\",
        r"    \midrule",
        *body,
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
        "",
    ])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("build/results/paper_v2"))
    args = parser.parse_args()
    (args.results / "table_population_summary.tex").write_text(population_table(args.results), encoding="utf-8")
    (args.results / "table_hypothesis_tests.tex").write_text(hypothesis_table(args.results), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
