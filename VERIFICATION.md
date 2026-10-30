# Verification — 2026-10-30

Verified on macOS ARM64 with CPython 3.14.0, using an isolated environment
created from this repository's `uv.lock` and the `analysis` extra.

- All 5,484 entries in the originating frozen package's integrity manifest passed.
- All 52 retained generation functions/classes have identical Python ASTs to
  the frozen implementation. Unused strategies and prompt-map entries were removed.
- The 77-test offline suite passed in the isolated environment. The additional
  current-paper geometry test also passed (78 tests total).
- Historical replay validated 2,104 request prompts/settings and reproduced all
  3,000 final recipe records exactly, including the three repair calls and
  idempotent resume. Unexpected generation calls fail the replay test.
- The eight original reference artifacts reproduced byte-for-byte, both in the
  source environment and the new isolated environment.
- All four current-manuscript claim tables also matched with relative tolerance
  `1e-7` and absolute tolerance `1e-9`. These include pooled PCA retention,
  matched participation ratio/spectral shape, full and matched centroid/spread
  geometry, and bidirectional human proximity and coverage. CSV serialization
  and floating-point rounding differ from the notebook exports.
- The CLI dry run succeeded from a different working directory without an API
  key or creating an output directory.
- Source and wheel distributions built successfully. The source distribution
  includes evidence, runners and reference results; the wheel is the importable
  generation/analysis library. The full reproduction commands use the source tree.
- Release contents were checked for credential patterns and exclude virtual
  environments, generated outputs, notebooks, drafts and unrelated cohorts.

No new generation or embedding API calls were made. Fresh provider availability,
billing, model output equivalence and independent-run repeatability were not
tested. The manuscript PDF and its synthetic/TikZ teaching figures were not rebuilt.

To repeat the checks:

```sh
uv sync --locked --extra analysis
uv run python -m unittest discover -s tests
uv run python analyze.py --check --output outputs/verification
```
