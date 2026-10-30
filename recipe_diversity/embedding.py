"""OpenRouter embeddings backend.

Recipes are sent whole. `openai/text-embedding-3-large` has an 8192-token
context and the longest recipe in this corpus is 657 tokens, so no text is
truncated. The full 3072-dimension vector is kept: the `dimensions` parameter
is deliberately not sent.
"""

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any

import httpx
import numpy as np
from rich.progress import Progress

from . import EMBEDDING_MODEL
from .generation import OpenRouterError, _retry_delay, require_api_key

EMBEDDINGS_URL = "https://openrouter.ai/api/v1/embeddings"
CACHE_DIR = Path(".embedding_cache")


@dataclass(frozen=True)
class EmbeddingSettings:
    model: str = EMBEDDING_MODEL
    batch_size: int = 256
    timeout_seconds: float = 600.0
    max_attempts: int = 3
    provider_order: tuple[str, ...] = ("openai",)
    allow_fallbacks: bool = False
    cache_dir: Path | None = CACHE_DIR


@dataclass(frozen=True)
class EmbeddingResult:
    vectors: np.ndarray
    prompt_tokens: int
    requested: int
    cached: int


def text_digest(model: str, text: str) -> str:
    return hashlib.sha256(f"{model}\x00{text}".encode("utf-8")).hexdigest()


def _cache_path(cache_dir: Path, model: str, text: str) -> Path:
    digest = text_digest(model, text)
    return cache_dir / model.replace("/", "__") / digest[:2] / f"{digest}.npy"


def _read_cached(cache_dir: Path | None, model: str, text: str) -> np.ndarray | None:
    if cache_dir is None:
        return None
    path = _cache_path(cache_dir, model, text)
    if not path.is_file():
        return None
    try:
        return np.load(path)
    except (OSError, ValueError):
        return None


def _write_cached(cache_dir: Path | None, model: str, text: str, vector: np.ndarray) -> None:
    if cache_dir is None:
        return
    path = _cache_path(cache_dir, model, text)
    path.parent.mkdir(parents=True, exist_ok=True)
    # np.save appends .npy unless the name already ends in it, so the temporary
    # name has to keep the suffix or the write lands on a different path.
    temporary = path.with_name(f".{path.stem}.{os.getpid()}.tmp.npy")
    np.save(temporary, vector)
    temporary.replace(path)


def parse_embedding_response(payload: Any, expected: int) -> tuple[np.ndarray, int]:
    """Return vectors ordered by the response `index` field, plus token usage."""
    if not isinstance(payload, dict):
        raise OpenRouterError("Embedding response is not a JSON object.")
    data = payload.get("data")
    if not isinstance(data, list) or len(data) != expected:
        raise OpenRouterError(
            f"Embedding response returned {len(data) if isinstance(data, list) else 'no'} "
            f"vectors for {expected} inputs."
        )
    ordered: list[list[float]] = [[] for _ in range(expected)]
    for item in data:
        if not isinstance(item, dict):
            raise OpenRouterError("Embedding response contains a malformed entry.")
        index = item.get("index")
        vector = item.get("embedding")
        if not isinstance(index, int) or not 0 <= index < expected:
            raise OpenRouterError("Embedding response has a missing or out-of-range index.")
        if not isinstance(vector, list) or not vector:
            raise OpenRouterError("Embedding response has a missing or empty vector.")
        if ordered[index]:
            raise OpenRouterError(f"Embedding response repeated index {index}.")
        ordered[index] = vector
    widths = {len(vector) for vector in ordered}
    if len(widths) != 1:
        raise OpenRouterError(f"Embedding response mixes vector widths: {sorted(widths)}.")
    usage = payload.get("usage")
    tokens = usage.get("total_tokens") if isinstance(usage, dict) else None
    return np.asarray(ordered, dtype=np.float32), int(tokens) if isinstance(tokens, int) else 0


def _post_batch(
    texts: list[str],
    settings: EmbeddingSettings,
    client: httpx.Client,
) -> tuple[np.ndarray, int]:
    key = require_api_key()
    body: dict[str, Any] = {
        "model": settings.model,
        "input": texts,
        "encoding_format": "float",
        "provider": {
            "order": list(settings.provider_order),
            "allow_fallbacks": settings.allow_fallbacks,
        },
    }
    for attempt in range(settings.max_attempts):
        response = None
        try:
            response = client.post(
                EMBEDDINGS_URL,
                headers={"Authorization": f"Bearer {key}"},
                json=body,
            )
            if response.status_code in (400, 413) and len(texts) > 1:
                # The provider rejected the batch itself; halve it and retry.
                half = len(texts) // 2
                head, head_tokens = _post_batch(texts[:half], settings, client)
                tail, tail_tokens = _post_batch(texts[half:], settings, client)
                return np.vstack((head, tail)), head_tokens + tail_tokens
            if response.status_code not in (429, 500, 502, 503, 504):
                response.raise_for_status()
                return parse_embedding_response(response.json(), len(texts))
            if attempt == settings.max_attempts - 1:
                response.raise_for_status()
        except (httpx.TransportError, json.JSONDecodeError, OpenRouterError):
            if attempt == settings.max_attempts - 1:
                raise
        time.sleep(_retry_delay(response, attempt))

    raise AssertionError("unreachable")


def embed_texts(
    texts: list[str],
    settings: EmbeddingSettings = EmbeddingSettings(),
    client: httpx.Client | None = None,
    progress: Progress | None = None,
    description: str = "Embedding",
) -> EmbeddingResult:
    """Embed every text in full, batching uncached texts `batch_size` at a time.

    Only uncached texts are counted on the progress bar, so a resumed run shows
    the work it will actually do rather than restarting from zero.
    """
    if settings.batch_size < 1:
        raise ValueError("Embedding batch_size must be at least 1.")
    vectors: list[np.ndarray | None] = [
        _read_cached(settings.cache_dir, settings.model, text) for text in texts
    ]
    pending = [index for index, vector in enumerate(vectors) if vector is None]
    cached = len(texts) - len(pending)
    prompt_tokens = 0

    if pending:
        owned = client is None
        active = client or httpx.Client(timeout=settings.timeout_seconds)
        task_id = progress.add_task(description, total=len(pending)) if progress else None
        try:
            for start in range(0, len(pending), settings.batch_size):
                chunk = pending[start:start + settings.batch_size]
                batch, tokens = _post_batch([texts[index] for index in chunk], settings, active)
                prompt_tokens += tokens
                for offset, index in enumerate(chunk):
                    vectors[index] = batch[offset]
                    _write_cached(settings.cache_dir, settings.model, texts[index], batch[offset])
                if progress is not None and task_id is not None:
                    progress.advance(task_id, len(chunk))
        finally:
            if owned:
                active.close()
            if progress is not None and task_id is not None:
                progress.remove_task(task_id)

    if any(vector is None for vector in vectors):
        raise OpenRouterError("Embedding backend returned no vector for some texts.")
    return EmbeddingResult(
        np.asarray(vectors, dtype=np.float32),
        prompt_tokens,
        len(pending),
        cached,
    )
