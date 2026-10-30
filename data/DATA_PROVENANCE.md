# Data provenance and fair-use assessment

Record date: September 2, 2026

## Corpus identity

- Source: TextOnlyRecipes, <https://www.textonlyrecipes.com/>
- Record pattern: `https://www.textonlyrecipes.com/recipes/<slug>`
- Records: 168 published recipe pages
- Original local snapshot metadata: June 10, 2026, 12:42:44 +03:00
- Earlier license audit: June 24, 2026
- Updated provenance and fair-use audit: September 2, 2026
- Archived index: <https://web.archive.org/web/20260902111341/https://www.textonlyrecipes.com/>
- Archive capture time: September 2, 2026, 11:13:41 UTC
- Archive access date: September 2, 2026
- Collection code: `scripts/prepare_recipe_data.py`

The archived homepage contains 168 recipe links. Its normalized URL inventory
is identical to `data/original_recipes/index.csv`. The archived homepage was
also checked for text or links labeled license, copyright, terms, privacy,
legal, Creative Commons, or CC-BY; none were present. This observation is
limited to the archived homepage and does not assert that the material is
unprotected or in the public domain.

`data/original_recipes/index.csv` provides the title, original URL, and local
filename for every record. `INPUT_SHA256SUMS` provides a SHA-256 digest for each
local source file and every other frozen package input.

## Author's fair-use assessment

The author relies on fair use under 17 U.S.C. § 107 to retain and distribute
the source texts in this research reproducibility package. This is the author's
case-specific assessment, not legal advice or a claim that every use of the
material would be fair. The four-factor assessment is:

1. **Purpose and character.** The use is scholarly research and reproducibility.
   Recipe text is analyzed as input to whole-document embeddings and aggregate
   diversity statistics; the package does not republish the material as a
   competing recipe-reading service.
2. **Nature of the work.** The pages were published and are substantially
   functional or factual recipes, although particular selection, arrangement,
   and expressive directions may contain protectable authorship.
3. **Amount and substantiality.** Complete recipe texts are retained because
   truncation would change the embeddings and prevent exact audit and
   reproduction. Because the full texts are used, this factor does not
   independently favor fair use; the research necessity is part of the overall
   balancing assessment.
4. **Market effect.** The package preserves attribution and source URLs, is
   organized for computational research, and is not intended to substitute for
   the source website or a commercial recipe product. No known current or
   potential market is displaced by the reported aggregate analysis.

The U.S. Copyright Office explains that scholarship and research may qualify
as fair use, that all four statutory factors must be weighed, and that the
determination is fact-specific:
<https://www.copyright.gov/fair-use/> (accessed September 2, 2026).

## Generated populations

The three generated populations retain their prompts, request settings, parsed
records, and raw responses. They use OpenRouter's
`openai/gpt-oss-120b:nitro` identifier, temperature 0, top-p 1, and deterministic
SHA-256 nonces. The nonce and prior-title strategies use a 1,200-token cap per
response. The breadth-first strategy uses a 16,000-token cap for batched
responses and contains 101 planned calls plus three targeted repair calls in
the realized run. The records do not preserve a request timestamp, the actual
provider route selected by `:nitro`, or a provider model revision.

## Reproducibility boundary

The offline workflow reproduces the analysis of the checked-in realized runs
using the checked-in embedding vectors. It does not claim that fresh model or
embedding API requests will reproduce the same bytes. Fresh calls are a new
experimental realization and must be reported as such.

