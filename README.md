# Concept guidance

Companion code for **Breadth-first scaffolding increases diversity in LLM-generated
recipe populations**, by Stuart Farmer, Casa Dilauro Independent Research House.
One call generates 100 cuisine concepts, then one call per concept generates ten
recipes. The experiment compares this population with identifier variation
(nonce), accumulated prior-title prompting, and 168 human recipes.

## Reproduce the recorded results

Use Python 3.14 (the verified version). From this repository:

```sh
python -m venv .venv
. .venv/bin/activate
pip install -r requirements-analysis.txt
python analyze.py --check
```

Alternatively, `uv sync --locked --extra analysis` installs the committed lock.
The analysis is offline after installation: it needs no API key, model download,
GPU, notebook, or TeX installation. It verifies the evidence hashes, recomputes
the measurements, resampling summaries, tables and figures, and checks all twelve
archived CSV/JSON results (eight original outputs and four current-manuscript
claim tables). Outputs go to `outputs/`. For another reconstruction,
choose a new directory with `--output outputs/check-2`; existing directories are
never overwritten. Numerical comparisons allow `rtol=1e-7, atol=1e-9` for platform
rounding and report which files are byte-identical.

The evidence contains 3,168 whole-text embeddings with 3,072 coordinates each,
3,000 generated recipes and all 2,104 recorded generation responses. The paper's
`hierarchical` analysis label means `wide_1000` (100 concepts × 10 recipes).
It does not mean the older ten-by-ten-by-ten experiment.

## Run a new experiment

```sh
pip install -e '.[analysis]'
export OPENROUTER_API_KEY=your-key
python run.py all --output runs/new-study --dry-run
```

Remove `--dry-run` to make paid OpenRouter requests. You can select `nonce`,
`prior-titles`, or `breadth-first` individually instead of `all`. Resume an
interrupted study explicitly:

```sh
python run.py all --output runs/new-study --resume
python analyze.py --runs runs/new-study --embed --output outputs/new-study
```

`--embed` explicitly permits paid embedding requests for a new study, including
the human reference; subsequent requests use its local cache. The bundled
historical evidence is never the destination of new generation.

The paper settings are fixed: `openai/gpt-oss-120b:nitro`, temperature 0, top-p 1,
1,000 recipes per strategy, four workers, 1,200 output tokens for each baseline
and 16,000 for breadth-first calls. Prior-title generation is sequential; its
100,000 estimated-token budget retains the newest titles. Breadth-first uses
101 planned calls plus up to three repairs per batch for missing sections;
the recorded run needed three repair calls in total. Prompts, deterministic
SHA-256 nonces, request order, delimiter parsing, transport retries, raw response
logging and resume validation are preserved from the actual experiment.

Fresh model calls replicate the procedure, not the recorded bytes or scores.
Provider routing and model revisions were not recorded, and later availability
of the original `:nitro` identifier is not guaranteed. There is no silent model
substitution.

## Use the generation code

```python
from pathlib import Path
from recipe_diversity import generate_wide

generate_wide(Path("runs/my-breadth-first-population"))
```

The implementation is in `recipe_diversity/generation.py`. This experiment used
direct streaming HTTP requests through HTTPX, not Slick. Retaining that boundary
avoids changing the paper's request and failure behavior. Prompts remain the
original text constants; each run also saves its exact templates and settings.
Importing the generation package requires only HTTPX and Rich and makes no calls.

## Files and checks

- `run.py`: fresh generation and explicit resume.
- `analyze.py`: offline reconstruction, or explicit embedding of new runs.
- `recipe_diversity/`: the three strategies, embeddings and numerical methods.
- `scripts/`: inference, lexical measurements, figures, tables and current-paper geometry.
- `data/`, `reference/`: selected evidence and archived expected results.
- `tests/`: transport, repairs, persistence, metrics, CLI and historical replay.

```sh
python -m unittest discover -s tests
python analyze.py --check --output outputs/verification
```

The replay test reconstructs all three populations from recorded responses,
compares every generated record, checks every request against its original
prompt/settings, and verifies that resume makes no further model calls.
See [source provenance](PROVENANCE.md) and [data provenance](data/DATA_PROVENANCE.md).

The measurements describe one realized population per strategy. Recipe-level
resampling does not establish repeatability across independent generation runs.
Whole texts include echoed nonces and length-limited outputs, as in the paper;
the package does not silently clean them or claim to measure recipe quality.
