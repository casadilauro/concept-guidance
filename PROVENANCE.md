# Source and evidence provenance

This is a standalone extraction of the Arborea recipe experiment, following the
small algorithm / runner / offline analysis structure of
[EliteTable](https://github.com/StuartFarmer/elitetable). It is derived from the
author's implementation, not an independently invented replacement algorithm.

The source manuscript is `paper_workspace/paper_v2/main.tex` in the Arborea
workspace (called `paper_workshop` in the request). The evidence is its frozen
`reproducibility/` package. The original workspace HEAD was
`26ed0a05c20302943bda73b8ba19a80cd1194185`, but uncommitted files contain the actual
experiment; HEAD alone is not sufficient provenance. `SOURCE_PROVENANCE.json`
records source file hashes and retained definitions. All 5,484 original frozen
manifest entries passed verification before extraction.

## Preserved and removed

Generation retains the exact functions reachable from `generate_nonce`,
`generate_prior_titles` and `generate_wide`, with their original constants,
prompts, parsing, retries, logging and persistence. Only unused strategies and
their prompt-map entries were removed. The replay test validates all 2,104
historical requests and all 3,000 final recipe records, including targeted
repairs and resumption. No Slick dependency was introduced: the original
experiment used HTTPX streaming directly.

Comparison retains shared coordinate-wise scaling to [0,3], within-population
squared neighbour distances, cosine coverage, covariance diagnostics and the
original seeded resampling algorithms. The unused local sentence-transformers
encoder and single-population clustering path were removed. Figure/table and
inference computations were retained; the inference module's explanation was
corrected to avoid implying independent experimental replications.

The new CLI uses explicit output paths and separates generation, paid embeddings
and offline reconstruction. It rejects accidental overwrites. The offline
encoder requires recorded whole-text hashes and has no API fallback. The
archived CSV/JSON results remain unchanged.

No research-tree framework, alternative cohorts, GPU libraries, notebook stack,
scraper, manuscript drafts, credentials, author notes or PDF build is included.
The generated empirical figures and tables are retained; synthetic explanatory
illustrations and manuscript typesetting are outside this repository's scope.

The later manuscript adds four claim tables for pooled PCA, matched covariance
spectra, centroid/spread geometry, and bidirectional human proximity.
`scripts/paper_statistics.py` extracts those numerical sections from the paper
review code, preserving float64 normalization, PCA seed 42, and 50 matched
draws with seed 0 in the original population order. It removes pandas, notebook
display, and narrative generation. All four tables are checked alongside the
eight original references; manuscript-specific TikZ layout is not rebuilt.

## Portable evidence

`data/embeddings.npz` losslessly compresses the original 3,168 float vectors,
keyed by SHA-256 of `model + NUL + full recipe text`. Each
`data/runs/*/requests.jsonl.gz` contains every original raw response object,
ordered by request filename; compression changes storage, not JSON values.
The four `recipes.jsonl` files, human attribution index, generation configs,
prompt files and twelve reference result files are copied byte-for-byte.
`data/SHA256SUMS.json` verifies the distributed evidence and references.

The human source texts are included under the author's documented research
reproduction distribution assessment in [DATA_PROVENANCE.md](data/DATA_PROVENANCE.md).
They are not represented as public-domain or relicensed software. Attribution
and original source URLs are retained in `data/data/original_recipes/index.csv`.

The original provider route, model revision, request timestamps and complete
billing metadata are unavailable. This release does not reconstruct them.
Temperature zero does not guarantee identical fresh responses.
