from dataclasses import dataclass


import hashlib


import json


import math


import os


import re


import time


from collections.abc import Callable, Iterable


from concurrent.futures import ThreadPoolExecutor


from pathlib import Path


from threading import Lock


from typing import Any


import httpx


from rich.progress import Progress, TaskID


from . import MODEL_NAME


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


TOTAL_RECIPES = 1000


PRIOR_TITLE_TOKEN_BUDGET = 100_000


WIDE_CUISINES = 100


WIDE_CUISINE_BATCH = 100


WIDE_RECIPES_PER_CUISINE = 10


RECIPE_OUTPUT_CONTRACT = """Title

Ingredients:
- ingredient

Instructions:
1. instruction"""


PRIOR_TITLES_TEMPLATE = """Write one original recipe in this exact format:
""" + RECIPE_OUTPUT_CONTRACT + """
Do not duplicate any of these previously generated titles:
{existing_titles}

Nonce: {nonce}
Return only the recipe."""


NONCE_TEMPLATE = """Write one original recipe in this exact format:
""" + RECIPE_OUTPUT_CONTRACT + """
Nonce: {nonce}
Return only the recipe."""


BATCHED_CUISINE_TEMPLATE = """Create exactly {count} distinct cuisine contexts, each with a short title and description.

Make them as different from one another as possible.

Use each delimiter and nonce exactly once:
{labels}

Return only the {count} delimiter-labelled cuisine contexts."""


WIDE_RECIPE_TEMPLATE = """Write exactly {recipe_count} original recipes using this cuisine context.

Cuisine:
{cuisine}

Make them as different from one another as possible.

Use each delimiter and nonce exactly once:
{recipe_labels}

For each delimiter-labelled recipe, use this exact recipe output format:
""" + RECIPE_OUTPUT_CONTRACT + """

Return only the {recipe_count} delimiter-labelled recipes."""


PROMPT_TEMPLATES = {
    "nonce_prompt.txt": NONCE_TEMPLATE,
    "prior_titles_prompt.txt": PRIOR_TITLES_TEMPLATE,
    "batched_cuisine_prompt.txt": BATCHED_CUISINE_TEMPLATE,
    "wide_recipe_prompt.txt": WIDE_RECIPE_TEMPLATE,
}


class OpenRouterError(RuntimeError):
    pass


@dataclass(frozen=True)
class GenerationSettings:
    model: str = MODEL_NAME
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 1200
    timeout_seconds: float = 1800.0
    max_workers: int = 4


@dataclass(frozen=True)
class Completion:
    text: str
    finish_reason: str | None
    reasoning_characters: int


@dataclass(frozen=True)
class RecipeRequest:
    request_id: str
    recipe_ids: tuple[str, ...]
    nonces: tuple[str, ...]
    prompt: str
    strategy: str
    cuisine_id: str | None = None
    cuisine_title: str | None = None
    dish_id: str | None = None
    dish_title: str | None = None


def require_api_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not key:
        raise OpenRouterError("OPENROUTER_API_KEY is not set.")
    return key


def stable_nonce(kind: str, *parts: int) -> str:
    material = ":".join(
        ["textonlyrecipes-nonce-v1", kind, *(f"{part:04d}" for part in parts)]
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:40]


def build_prior_titles_prompt(
    nonce: str,
    titles: list[str],
    *,
    title_token_budget: int = PRIOR_TITLE_TOKEN_BUDGET,
) -> str:
    recent_titles: list[str] = []
    used_tokens = 0
    for title in reversed(titles):
        estimated_tokens = max(1, math.ceil(len(title.encode("utf-8")) / 4))
        if used_tokens + estimated_tokens > title_token_budget:
            break
        recent_titles.append(title)
        used_tokens += estimated_tokens
    title_block = "\n".join(f"- {title}" for title in reversed(recent_titles)) or "- None yet"
    return PRIOR_TITLES_TEMPLATE.format(existing_titles=title_block, nonce=nonce)


def parse_labelled_batch(
    text: str,
    expected_ids: Iterable[str],
    *,
    complete: bool = True,
    duplicates: str = "reject",
) -> dict[str, str]:
    """Split one response into its delimiter-labelled sections.

    The id pattern is deliberately looser than the expected set so a section the
    model invented is reported as unexpected rather than silently ignored.

    `complete=False` returns whatever sections arrived instead of raising on a
    short response. Models do drop items from a large batch while still
    reporting a clean stop, and the caller repairs that by asking again for the
    missing ids only.

    `duplicates="first"` keeps the earliest section for a repeated label instead
    of rejecting the response. A model that emits ten recipes but labels the
    fifth with the first one's id has produced usable work under one wrong
    heading; keeping the first occurrence is deterministic, and the id it failed
    to emit is then simply missing and repairable.
    """
    expected = list(expected_ids)
    markers = list(re.finditer(r"(?m)^=== ([a-z]+(?:_\d{3})+) ===\s*$", text))
    found = {match.group(1) for match in markers}
    if len(markers) != len(found) and duplicates != "first":
        raise ValueError("Duplicate delimiter.")
    missing = [recipe_id for recipe_id in expected if recipe_id not in found]
    if missing and complete:
        raise ValueError(f"Missing delimiters: {', '.join(missing)}")
    unexpected = found.difference(expected)
    if unexpected:
        raise ValueError(f"Unexpected delimiters: {', '.join(sorted(unexpected))}")
    parts: dict[str, str] = {}
    for index, marker in enumerate(markers):
        identifier = marker.group(1)
        if identifier in parts:
            continue
        end = markers[index + 1].start() if index + 1 < len(markers) else len(text)
        parts[identifier] = text[marker.end():end].strip()
    return parts


def parse_recipe_batch(text: str, expected_ids: Iterable[str]) -> dict[str, str]:
    return parse_labelled_batch(text, expected_ids)


def identifier_nonce(kind: str, identifier: str, *extra: int) -> str:
    """Deterministic nonce for any scaffold identifier, at any level.

    The level prefix is folded into the kind so `cuisine_007` and `recipe_007`
    never collide.
    """
    parts = tuple(int(chunk) for chunk in re.findall(r"\d{3}", identifier))
    return stable_nonce(f"{kind}-{identifier.split('_')[0]}", *parts, *extra)


def batch_labels(identifiers: Iterable[str], nonces: Iterable[str]) -> str:
    return "\n".join(
        f"=== {identifier} ===\nNonce: {nonce}"
        for identifier, nonce in zip(identifiers, nonces, strict=True)
    )


def resume_prefix_length(records: list[dict[str, Any]], total: int) -> int:
    if len(records) > total:
        raise ValueError("Existing records exceed the deterministic run length.")
    expected = [f"recipe_{index:03d}" for index in range(total)]
    actual = [str(record.get("recipe_id", "")) for record in records]
    if actual != expected[:len(actual)]:
        raise ValueError("Existing records are not a contiguous deterministic prefix.")
    return len(actual)


def _validate_experiment_settings(
    settings: GenerationSettings, max_tokens: int = 1200
) -> None:
    if settings.model != MODEL_NAME:
        raise ValueError(f"Generation requires model={MODEL_NAME}.")
    actual = (
        settings.temperature,
        settings.top_p,
        settings.max_tokens,
        settings.max_workers,
    )
    if actual != (0.0, 1.0, max_tokens, 4):
        raise ValueError(
            f"Generation requires temperature=0, top_p=1, max_tokens={max_tokens}, "
            "and max_workers=4."
        )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid JSONL in {path} at line {line_number}.") from error
        if not isinstance(record, dict):
            raise ValueError(f"Invalid JSONL record in {path} at line {line_number}.")
        records.append(record)
    return records


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def _atomic_json(path: Path, record: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _config(
    strategy: str,
    settings: GenerationSettings,
    total_recipes: int,
) -> dict[str, Any]:
    return {
        "strategy": strategy,
        "total_recipes": total_recipes,
        "model": settings.model,
        "temperature": settings.temperature,
        "top_p": settings.top_p,
        "max_tokens": settings.max_tokens,
        "timeout_seconds": settings.timeout_seconds,
        "max_workers": settings.max_workers,
        "nonce_scheme": "sha256:textonlyrecipes-nonce-v1",
    }


def _prepare_run(
    output_dir: Path,
    strategy: str,
    settings: GenerationSettings,
    template_names: Iterable[str],
    max_tokens: int = 1200,
    total_recipes: int = TOTAL_RECIPES,
) -> None:
    _validate_experiment_settings(settings, max_tokens)
    require_api_key()
    config = _config(strategy, settings, total_recipes)
    config_path = output_dir / "generation_config.json"
    if config_path.exists():
        if json.loads(config_path.read_text(encoding="utf-8")) != config:
            raise ValueError("Existing generation config conflicts with this run.")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "raw").mkdir(exist_ok=True)
    if not config_path.exists():
        _atomic_json(config_path, config)
    for name in template_names:
        path = output_dir / name
        template = PROMPT_TEMPLATES[name]
        if path.exists() and path.read_text(encoding="utf-8") != template:
            raise ValueError(f"Existing prompt template conflicts: {name}")
        if not path.exists():
            path.write_text(template, encoding="utf-8")
    _repair_raw_index(output_dir)


def _raw_path(output_dir: Path, request_id: str) -> Path:
    return output_dir / "raw" / f"{request_id}.json"


def _raw_index_row(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        key: payload[key]
        for key in (
            "request_id",
            "raw_path",
            "text",
            "model",
            "request_settings",
            "nonces",
            "prompt_sha256",
            "elapsed_seconds",
            "finish_reason",
            "reasoning_characters",
            "strategy",
            "recipe_ids",
            "cuisine_id",
            "cuisine_title",
            "dish_id",
            "dish_title",
        )
        if key in payload
    }


def _repair_raw_index(output_dir: Path) -> None:
    index_path = output_dir / "raw_responses.jsonl"
    rows = _read_jsonl(index_path)
    by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        request_id = str(row.get("request_id", ""))
        if not request_id or request_id in by_id:
            raise ValueError("Raw response index has a missing or duplicate request ID.")
        expected_path = f"raw/{request_id}.json"
        if row.get("raw_path") != expected_path:
            raise ValueError(f"Raw response index path conflict: {request_id}")
        if not (output_dir / expected_path).is_file():
            raise ValueError(f"Raw response index has no corresponding raw file: {request_id}")
        by_id[request_id] = row
    for raw_path in sorted((output_dir / "raw").glob("*.json")):
        payload = json.loads(raw_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid raw response file: {raw_path.name}")
        request_id = str(payload.get("request_id", ""))
        if raw_path.name != f"{request_id}.json":
            raise ValueError(f"Raw response filename conflict: {raw_path.name}")
        expected = _raw_index_row(payload)
        existing = by_id.get(request_id)
        if existing is not None and existing != expected:
            raise ValueError(f"Raw response index conflict: {request_id}")
        if existing is None:
            _append_jsonl(index_path, expected)
            by_id[request_id] = expected


def _request_payload(
    request: RecipeRequest,
    settings: GenerationSettings,
    completion: Completion,
    elapsed_seconds: float,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "request_id": request.request_id,
        "raw_path": f"raw/{request.request_id}.json",
        "text": completion.text,
        "model": settings.model,
        "request_settings": {
            "temperature": settings.temperature,
            "top_p": settings.top_p,
            "max_tokens": settings.max_tokens,
            "timeout_seconds": settings.timeout_seconds,
        },
        "nonces": list(request.nonces),
        "prompt": request.prompt,
        "prompt_sha256": hashlib.sha256(request.prompt.encode("utf-8")).hexdigest(),
        "elapsed_seconds": elapsed_seconds,
        "finish_reason": completion.finish_reason,
        "reasoning_characters": completion.reasoning_characters,
        "strategy": request.strategy,
        "recipe_ids": list(request.recipe_ids),
    }
    for name in ("cuisine_id", "cuisine_title", "dish_id", "dish_title"):
        value = getattr(request, name)
        if value is not None:
            payload[name] = value
    return payload


def _validate_raw_payload(
    payload: dict[str, Any],
    request: RecipeRequest,
    settings: GenerationSettings,
) -> None:
    expected_hash = hashlib.sha256(request.prompt.encode("utf-8")).hexdigest()
    expected = {
        "request_id": request.request_id,
        "raw_path": f"raw/{request.request_id}.json",
        "recipe_ids": list(request.recipe_ids),
        "nonces": list(request.nonces),
        "prompt": request.prompt,
        "prompt_sha256": expected_hash,
        "strategy": request.strategy,
        "model": settings.model,
        "request_settings": {
            "temperature": settings.temperature,
            "top_p": settings.top_p,
            "max_tokens": settings.max_tokens,
            "timeout_seconds": settings.timeout_seconds,
        },
    }
    for name in ("cuisine_id", "cuisine_title", "dish_id", "dish_title"):
        expected[name] = getattr(request, name)
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(
                f"Raw response conflicts with deterministic request: {request.request_id}"
            )


def _obtain_raw(
    output_dir: Path,
    request: RecipeRequest,
    settings: GenerationSettings,
) -> dict[str, Any]:
    path = _raw_path(output_dir, request.request_id)
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid raw response: {request.request_id}")
        _validate_raw_payload(payload, request, settings)
        return payload
    started = time.perf_counter()
    completion = stream_completion(request.prompt, settings)
    payload = _request_payload(
        request,
        settings,
        completion,
        time.perf_counter() - started,
    )
    _atomic_json(path, payload)
    return payload


_INDEX_LOCK = Lock()


def _ensure_indexed(output_dir: Path, payload: dict[str, Any]) -> None:
    """Append one row to the raw-response index, atomically.

    Strategies that repair or index from worker threads would otherwise
    interleave a read-modify-append on a single shared file.
    """
    with _INDEX_LOCK:
        _ensure_indexed_locked(output_dir, payload)


def _ensure_indexed_locked(output_dir: Path, payload: dict[str, Any]) -> None:
    request_id = str(payload["request_id"])
    rows = _read_jsonl(output_dir / "raw_responses.jsonl")
    matching = [row for row in rows if row.get("request_id") == request_id]
    expected = _raw_index_row(payload)
    if matching and matching != [expected]:
        raise ValueError(f"Raw response index conflict: {request_id}")
    if not matching:
        _append_jsonl(output_dir / "raw_responses.jsonl", expected)


def _title(text: str) -> str:
    """First real line of a section.

    Batched prompts label each section with a delimiter and a nonce, and models
    routinely echo the `Nonce:` line back at the top of the section body. Taking
    it as the title made every record in a batched run carry a hash string.
    """
    for line in text.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("Nonce:"):
            return stripped
    return ""


def _recipe_validation(text: str) -> dict[str, bool]:
    return {
        "has_title": bool(_title(text)),
        "has_ingredients": "Ingredients:" in text,
        "has_instructions": "Instructions:" in text,
    }


def _recipe_records(
    request: RecipeRequest,
    payload: dict[str, Any],
    texts: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    if texts is None:
        if len(request.recipe_ids) == 1:
            texts = {request.recipe_ids[0]: str(payload["text"]).strip()}
        else:
            texts = parse_recipe_batch(str(payload["text"]), request.recipe_ids)
    records: list[dict[str, Any]] = []
    for recipe_id, nonce in zip(request.recipe_ids, request.nonces, strict=True):
        text = texts[recipe_id]
        validation = _recipe_validation(text)
        record: dict[str, Any] = {
            "recipe_id": recipe_id,
            "text": text,
            "title": _title(text),
            "valid_format": all(validation.values()),
            "validation": validation,
            "strategy": request.strategy,
            "request_id": request.request_id,
            "raw_path": payload["raw_path"],
            "model": payload["model"],
            "request_settings": payload["request_settings"],
            "nonce": nonce,
            "prompt_sha256": payload["prompt_sha256"],
            "elapsed_seconds": payload["elapsed_seconds"],
            "finish_reason": payload["finish_reason"],
            "reasoning_characters": payload["reasoning_characters"],
        }
        for name in ("cuisine_id", "cuisine_title", "dish_id", "dish_title"):
            if name in payload:
                record[name] = payload[name]
        records.append(record)
    return records


def _persist_request(
    output_dir: Path,
    request: RecipeRequest,
    payload: dict[str, Any],
    completed_recipe_ids: set[str],
    texts: dict[str, str] | None = None,
) -> int:
    _ensure_indexed(output_dir, payload)
    written = 0
    for record in _recipe_records(request, payload, texts):
        if record["recipe_id"] not in completed_recipe_ids:
            _append_jsonl(output_dir / "recipes.jsonl", record)
            completed_recipe_ids.add(str(record["recipe_id"]))
            written += 1
    return written


def _validate_recipe_provenance(
    record: dict[str, Any],
    expected: dict[str, Any],
) -> None:
    if any(record.get(key) != value for key, value in expected.items()):
        raise ValueError(
            f"Persisted recipe provenance conflicts: {expected['recipe_id']}"
        )


def _assert_raw_prefix(
    output_dir: Path,
    requests: list[RecipeRequest],
    first_incomplete: int,
) -> None:
    exists = [_raw_path(output_dir, request.request_id).exists() for request in requests]
    missing_seen = False
    for present in exists[first_incomplete:]:
        if not present:
            missing_seen = True
        elif missing_seen:
            raise ValueError("Existing raw responses are not a contiguous deterministic prefix.")


def _run_request_pool(
    output_dir: Path,
    requests: list[RecipeRequest],
    settings: GenerationSettings,
    completed_ids: set[str],
    max_workers: int,
    progress: Progress | None = None,
    task_id: TaskID | None = None,
) -> None:
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for start in range(0, len(requests), max_workers):
            chunk = requests[start:start + max_workers]
            payloads = list(
                executor.map(
                    lambda request: _obtain_raw(output_dir, request, settings),
                    chunk,
                )
            )
            for request, payload in zip(chunk, payloads, strict=True):
                written = _persist_request(output_dir, request, payload, completed_ids)
                if progress is not None and task_id is not None:
                    progress.advance(task_id, written)


def nonce_recipe_requests(total: int = TOTAL_RECIPES) -> list[RecipeRequest]:
    requests = []
    for index in range(total):
        recipe_id = f"recipe_{index:03d}"
        nonce = stable_nonce("nonce", index)
        requests.append(
            RecipeRequest(
                request_id=f"nonce_{recipe_id}",
                recipe_ids=(recipe_id,),
                nonces=(nonce,),
                prompt=NONCE_TEMPLATE.format(nonce=nonce),
                strategy="nonce",
            )
        )
    return requests


def run_recipe_requests(
    output_dir: Path,
    requests: list[RecipeRequest],
    settings: GenerationSettings,
    max_workers: int,
    progress: Progress | None = None,
) -> Path:
    records = _read_jsonl(output_dir / "recipes.jsonl")
    prefix = resume_prefix_length(records, len(requests))
    task_id = (
        progress.add_task("Recipes", total=len(requests), completed=prefix)
        if progress is not None
        else None
    )
    completed_ids = {str(record["recipe_id"]) for record in records}
    for index, record in enumerate(records):
        request = requests[index]
        if not _raw_path(output_dir, request.request_id).exists():
            raise ValueError(f"Recipe record has no raw provenance: {request.request_id}")
        payload = _obtain_raw(output_dir, request, settings)
        _validate_recipe_provenance(record, _recipe_records(request, payload)[0])
    _assert_raw_prefix(output_dir, requests, prefix)

    recovery_end = prefix
    while recovery_end < len(requests) and _raw_path(
        output_dir,
        requests[recovery_end].request_id,
    ).exists():
        recovery_end += 1
    for request in requests[prefix:recovery_end]:
        payload = _obtain_raw(output_dir, request, settings)
        written = _persist_request(output_dir, request, payload, completed_ids)
        if progress is not None and task_id is not None:
            progress.advance(task_id, written)
    _run_request_pool(
        output_dir,
        requests[recovery_end:],
        settings,
        completed_ids,
        max_workers,
        progress,
        task_id,
    )
    return output_dir


def _prior_request(index: int, titles: list[str]) -> RecipeRequest:
    recipe_id = f"recipe_{index:03d}"
    nonce = stable_nonce("prior-title", index)
    return RecipeRequest(
        request_id=f"prior_{recipe_id}",
        recipe_ids=(recipe_id,),
        nonces=(nonce,),
        prompt=build_prior_titles_prompt(nonce, titles),
        strategy="prior_titles",
    )


def run_prior_title_requests(
    output_dir: Path,
    settings: GenerationSettings,
    progress: Progress | None = None,
) -> Path:
    records = _read_jsonl(output_dir / "recipes.jsonl")
    prefix = resume_prefix_length(records, TOTAL_RECIPES)
    task_id = (
        progress.add_task("Recipes", total=TOTAL_RECIPES, completed=prefix)
        if progress is not None
        else None
    )
    titles: list[str] = []
    completed_ids = {str(record["recipe_id"]) for record in records}
    for index, record in enumerate(records):
        request = _prior_request(index, titles)
        if not _raw_path(output_dir, request.request_id).exists():
            raise ValueError(f"Recipe record has no raw provenance: recipe_{index:03d}")
        payload = _obtain_raw(output_dir, request, settings)
        expected = _recipe_records(request, payload)[0]
        _validate_recipe_provenance(record, expected)
        titles.append(str(expected["title"]))
    for index in range(prefix, TOTAL_RECIPES):
        request = _prior_request(index, titles)
        if not _raw_path(output_dir, request.request_id).exists() and any(
            _raw_path(output_dir, f"prior_recipe_{later:03d}").exists()
            for later in range(index + 1, TOTAL_RECIPES)
        ):
            raise ValueError("Existing raw responses are not a contiguous deterministic prefix.")
        payload = _obtain_raw(output_dir, request, settings)
        written = _persist_request(output_dir, request, payload, completed_ids)
        if progress is not None and task_id is not None:
            progress.advance(task_id, written)
        titles.append(_title(str(payload["text"])))
    return output_dir


MAX_BATCH_REPAIRS = 3


def _batched_repair_request(
    request: RecipeRequest,
    prompt_for: Callable[[tuple[str, ...], tuple[str, ...]], str],
    missing: tuple[str, ...],
    attempt: int,
) -> RecipeRequest:
    """Ask again for only the sections the model dropped.

    Temperature is zero, so re-sending the original prompt reproduces the same
    omission. Asking for the missing ids alone is a different prompt, which is
    both deterministic and far more likely to succeed at a smaller batch size.
    """
    nonces = tuple(identifier_nonce("repair", identifier, attempt) for identifier in missing)
    return RecipeRequest(
        request_id=f"{request.request_id}_repair_{attempt:02d}",
        recipe_ids=missing,
        nonces=nonces,
        prompt=prompt_for(missing, nonces),
        strategy=request.strategy,
        cuisine_id=request.cuisine_id,
        cuisine_title=request.cuisine_title,
        dish_id=request.dish_id,
        dish_title=request.dish_title,
    )


def _batched_leaf(
    output_dir: Path,
    settings: GenerationSettings,
    request: RecipeRequest,
    prompt_for: Callable[[tuple[str, ...], tuple[str, ...]], str],
    identifiers: tuple[str, ...] | None = None,
) -> tuple[dict[str, Any], dict[str, str]]:
    """Obtain one batched response, repairing any sections the model dropped.

    `identifiers` defaults to the request's recipe ids, but any batched layer
    can pass its own — context requests carry no recipe ids at all.
    """
    expected = identifiers if identifiers is not None else request.recipe_ids
    payload = _obtain_raw(output_dir, request, settings)
    texts = parse_labelled_batch(
        str(payload["text"]), expected, complete=False, duplicates="first"
    )
    for attempt in range(1, MAX_BATCH_REPAIRS + 1):
        missing = tuple(rid for rid in expected if rid not in texts)
        if not missing:
            return payload, texts
        repair = _batched_repair_request(request, prompt_for, missing, attempt)
        repaired = _obtain_raw(output_dir, repair, settings)
        _ensure_indexed(output_dir, repaired)
        texts.update(
            parse_labelled_batch(
                str(repaired["text"]), missing, complete=False, duplicates="first"
            )
        )
    missing = tuple(rid for rid in expected if rid not in texts)
    if missing:
        raise ValueError(
            f"{request.request_id}: model omitted {', '.join(missing)} after "
            f"{MAX_BATCH_REPAIRS} repair attempts."
        )
    return payload, texts


def wide_cuisine_ids() -> list[str]:
    return [f"cuisine_{index:03d}" for index in range(WIDE_CUISINES)]


def _wide_cuisine_request(batch_index: int) -> RecipeRequest:
    start = batch_index * WIDE_CUISINE_BATCH
    identifiers = tuple(
        f"cuisine_{start + offset:03d}" for offset in range(WIDE_CUISINE_BATCH)
    )
    nonces = tuple(
        stable_nonce("wide-cuisine", start + offset) for offset in range(WIDE_CUISINE_BATCH)
    )
    return RecipeRequest(
        request_id=f"wide_cuisines_{batch_index:03d}",
        recipe_ids=(),
        nonces=nonces,
        prompt=BATCHED_CUISINE_TEMPLATE.format(
            count=WIDE_CUISINE_BATCH,
            labels=batch_labels(identifiers, nonces),
        ),
        strategy="wide_context",
    )


def _wide_leaf_prompt(cuisine: str, recipe_ids: tuple[str, ...], nonces: tuple[str, ...]) -> str:
    return WIDE_RECIPE_TEMPLATE.format(
        cuisine=cuisine,
        recipe_labels=batch_labels(recipe_ids, nonces),
        recipe_count=len(recipe_ids),
    )


def _wide_contexts(
    output_dir: Path,
    settings: GenerationSettings,
    progress: Progress | None = None,
) -> list[str]:
    """`WIDE_CUISINES` cuisines, generated `WIDE_CUISINE_BATCH` per response."""
    batches = WIDE_CUISINES // WIDE_CUISINE_BATCH
    task_id = progress.add_task("Cuisines", total=batches) if progress is not None else None
    sections: dict[str, str] = {}

    def build(batch_index: int) -> dict[str, str]:
        request = _wide_cuisine_request(batch_index)
        start = batch_index * WIDE_CUISINE_BATCH
        identifiers = tuple(
            f"cuisine_{start + offset:03d}" for offset in range(WIDE_CUISINE_BATCH)
        )
        payload, texts = _batched_leaf(
            output_dir,
            settings,
            request,
            lambda ids, nonces: BATCHED_CUISINE_TEMPLATE.format(
                count=len(ids), labels=batch_labels(ids, nonces)
            ),
            identifiers,
        )
        _ensure_indexed(output_dir, payload)
        if progress is not None and task_id is not None:
            progress.advance(task_id)
        return texts

    with ThreadPoolExecutor(max_workers=settings.max_workers) as executor:
        for texts in executor.map(build, range(batches)):
            sections.update(texts)
    return [sections[identifier] for identifier in wide_cuisine_ids()]


def wide_jobs() -> list[tuple[int, tuple[str, ...]]]:
    """Each cuisine owns one contiguous block of recipes."""
    return [
        (
            cuisine_index,
            tuple(
                f"recipe_{index:03d}"
                for index in range(
                    cuisine_index * WIDE_RECIPES_PER_CUISINE,
                    (cuisine_index + 1) * WIDE_RECIPES_PER_CUISINE,
                )
            ),
        )
        for cuisine_index in range(WIDE_CUISINES)
    ]


def _wide_recipe_requests(cuisines: list[str]) -> list[RecipeRequest]:
    requests = []
    for cuisine_index, recipe_ids in wide_jobs():
        cuisine = cuisines[cuisine_index]
        nonces = tuple(
            stable_nonce("wide-recipe", int(recipe_id.removeprefix("recipe_")))
            for recipe_id in recipe_ids
        )
        requests.append(
            RecipeRequest(
                request_id=f"wide_leaf_{cuisine_index:03d}",
                recipe_ids=recipe_ids,
                nonces=nonces,
                prompt=_wide_leaf_prompt(cuisine, recipe_ids, nonces),
                strategy="wide",
                cuisine_id=f"cuisine_{cuisine_index:03d}",
                cuisine_title=_title(cuisine),
            )
        )
    return requests


def run_wide_requests(
    output_dir: Path,
    settings: GenerationSettings,
    progress: Progress | None = None,
) -> Path:
    """100 cuisines x 10 recipes in 101 planned calls, plus any repairs."""
    records = _read_jsonl(output_dir / "recipes.jsonl")
    prefix = resume_prefix_length(records, TOTAL_RECIPES)
    cuisines = _wide_contexts(output_dir, settings, progress)
    requests = _wide_recipe_requests(cuisines)
    completed_ids = {str(record["recipe_id"]) for record in records}
    task_id = (
        progress.add_task("Recipes", total=TOTAL_RECIPES, completed=prefix)
        if progress is not None
        else None
    )

    def leaf(index: int) -> tuple[dict[str, Any], dict[str, str]]:
        cuisine = cuisines[index]
        return _batched_leaf(
            output_dir,
            settings,
            requests[index],
            lambda ids, nonces: _wide_leaf_prompt(cuisine, ids, nonces),
        )

    settled = prefix // WIDE_RECIPES_PER_CUISINE
    for index in range(settled):
        if not _raw_path(output_dir, requests[index].request_id).exists():
            raise ValueError(f"Recipe record has no raw provenance: {requests[index].request_id}")
        payload, texts = leaf(index)
        for offset, expected in enumerate(_recipe_records(requests[index], payload, texts)):
            _validate_recipe_provenance(records[index * WIDE_RECIPES_PER_CUISINE + offset], expected)

    pending = list(range(settled, len(requests)))
    with ThreadPoolExecutor(max_workers=settings.max_workers) as executor:
        for index, (payload, texts) in zip(pending, executor.map(leaf, pending), strict=True):
            written = _persist_request(output_dir, requests[index], payload, completed_ids, texts)
            if progress is not None and task_id is not None:
                progress.advance(task_id, written)
    return output_dir


def generate_wide(
    output_dir: Path,
    settings: GenerationSettings = GenerationSettings(max_tokens=16000),
    progress: Progress | None = None,
) -> Path:
    _prepare_run(
        output_dir,
        "wide",
        settings,
        ("batched_cuisine_prompt.txt", "wide_recipe_prompt.txt"),
        max_tokens=16000,
        total_recipes=TOTAL_RECIPES,
    )
    return run_wide_requests(output_dir, settings, progress)


def generate_nonce(
    output_dir: Path,
    settings: GenerationSettings = GenerationSettings(),
    progress: Progress | None = None,
) -> Path:
    _prepare_run(
        output_dir,
        "nonce",
        settings,
        ("nonce_prompt.txt",),
        total_recipes=TOTAL_RECIPES,
    )
    return run_recipe_requests(
        output_dir,
        nonce_recipe_requests(total=TOTAL_RECIPES),
        settings,
        max_workers=settings.max_workers,
        progress=progress,
    )


def generate_prior_titles(
    output_dir: Path,
    settings: GenerationSettings = GenerationSettings(),
    progress: Progress | None = None,
) -> Path:
    _prepare_run(
        output_dir,
        "prior_titles",
        settings,
        ("prior_titles_prompt.txt",),
        total_recipes=TOTAL_RECIPES,
    )
    return run_prior_title_requests(output_dir, settings, progress)


def parse_sse_events(lines: Iterable[str]) -> Completion:
    text: list[str] = []
    finish_reason = None
    reasoning_characters = 0

    for line in lines:
        line = line.strip()
        if not line or line.startswith(":") or not line.startswith("data:"):
            continue
        data = line.removeprefix("data:").strip()
        if data == "[DONE]":
            break
        try:
            event = json.loads(data)
        except json.JSONDecodeError:
            continue
        choices = event.get("choices") if isinstance(event, dict) else None
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            continue
        choice = choices[0]
        delta = choice.get("delta", {})
        if not isinstance(delta, dict):
            continue
        reasoning = delta.get("reasoning")
        if isinstance(reasoning, str):
            reasoning_characters += len(reasoning)
        content = delta.get("content")
        if isinstance(content, str) and content:
            text.append(content)
        if isinstance(choice.get("finish_reason"), str):
            finish_reason = choice["finish_reason"]

    if not text:
        raise OpenRouterError("OpenRouter stream returned no content.")
    return Completion("".join(text), finish_reason, reasoning_characters)


def _retry_delay(response: httpx.Response | None, attempt: int) -> float:
    if response is not None:
        try:
            retry_after = float(response.headers.get("Retry-After", ""))
            if retry_after > 0 and math.isfinite(retry_after):
                return retry_after
        except ValueError:
            pass
    return float(2 ** (attempt + 1))


def stream_completion(prompt: str, settings: GenerationSettings) -> Completion:
    key = require_api_key()
    payload = {
        "stream": True,
        "model": settings.model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": settings.temperature,
        "top_p": settings.top_p,
        "max_tokens": settings.max_tokens,
    }

    with httpx.Client(timeout=settings.timeout_seconds) as client:
        for attempt in range(3):
            response = None
            try:
                with client.stream(
                    "POST",
                    OPENROUTER_URL,
                    headers={"Authorization": f"Bearer {key}"},
                    json=payload,
                ) as response:
                    if response.status_code not in (429, 503):
                        response.raise_for_status()
                        return parse_sse_events(response.iter_lines())
                    if attempt == 2:
                        response.raise_for_status()
            except (httpx.TransportError, OpenRouterError):
                if attempt == 2:
                    raise
            time.sleep(_retry_delay(response, attempt))

    raise AssertionError("unreachable")
