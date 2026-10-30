import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx
import numpy as np
from rich.progress import Progress

from recipe_diversity.embedding import (
    EMBEDDINGS_URL,
    EmbeddingSettings,
    embed_texts,
    parse_embedding_response,
    text_digest,
)
from recipe_diversity.generation import OpenRouterError

KEY = {"OPENROUTER_API_KEY": "test-key"}


def vector_for(text: str) -> list[float]:
    return [float(len(text)), float(sum(map(ord, text)) % 97), 1.0]


def responder(
    batches: list[list[str]],
    *,
    statuses: list[int] | None = None,
    reverse: bool = False,
) -> httpx.MockTransport:
    """Record every batch of inputs and answer with deterministic vectors."""
    codes = list(statuses or [])

    def handle(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        texts = body["input"]
        batches.append(list(texts))
        if codes:
            code = codes.pop(0)
            if code != 200:
                return httpx.Response(code, json={"error": "retry"})
        entries = [
            {"object": "embedding", "index": index, "embedding": vector_for(text)}
            for index, text in enumerate(texts)
        ]
        if reverse:
            entries.reverse()
        return httpx.Response(
            200,
            json={
                "object": "list",
                "data": entries,
                "model": body["model"],
                "usage": {"prompt_tokens": len(texts), "total_tokens": len(texts)},
            },
        )

    return httpx.MockTransport(handle)


class ResponseParsingTests(unittest.TestCase):
    def test_orders_vectors_by_the_response_index_field(self) -> None:
        payload = {
            "data": [
                {"index": 1, "embedding": [3.0, 4.0]},
                {"index": 0, "embedding": [1.0, 2.0]},
            ],
            "usage": {"total_tokens": 11},
        }
        vectors, tokens = parse_embedding_response(payload, 2)

        np.testing.assert_array_equal(vectors, np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
        self.assertEqual(tokens, 11)

    def test_rejects_a_response_that_drops_or_repeats_inputs(self) -> None:
        with self.assertRaisesRegex(OpenRouterError, "vectors for 2 inputs"):
            parse_embedding_response({"data": [{"index": 0, "embedding": [1.0]}]}, 2)
        with self.assertRaisesRegex(OpenRouterError, "repeated index 0"):
            parse_embedding_response(
                {"data": [{"index": 0, "embedding": [1.0]}, {"index": 0, "embedding": [2.0]}]},
                2,
            )

    def test_rejects_ragged_vector_widths(self) -> None:
        with self.assertRaisesRegex(OpenRouterError, "mixes vector widths"):
            parse_embedding_response(
                {"data": [{"index": 0, "embedding": [1.0]}, {"index": 1, "embedding": [1.0, 2.0]}]},
                2,
            )


class BatchingTests(unittest.TestCase):
    def test_sends_full_texts_in_batches_and_preserves_corpus_order(self) -> None:
        texts = [f"recipe {index}" for index in range(11)]
        batches: list[list[str]] = []
        settings = EmbeddingSettings(batch_size=4, cache_dir=None)
        with patch.dict(os.environ, KEY):
            client = httpx.Client(transport=responder(batches))
            result = embed_texts(texts, settings, client=client)

        self.assertEqual([len(batch) for batch in batches], [4, 4, 3])
        self.assertEqual([text for batch in batches for text in batch], texts)
        np.testing.assert_allclose(
            result.vectors,
            np.array([vector_for(text) for text in texts], dtype=np.float32),
        )
        self.assertEqual(result.prompt_tokens, 11)
        self.assertEqual((result.requested, result.cached), (11, 0))

    def test_reorders_a_response_whose_entries_arrive_shuffled(self) -> None:
        texts = ["alpha", "beta", "gamma"]
        with patch.dict(os.environ, KEY):
            client = httpx.Client(transport=responder([], reverse=True))
            result = embed_texts(texts, EmbeddingSettings(cache_dir=None), client=client)

        np.testing.assert_allclose(
            result.vectors,
            np.array([vector_for(text) for text in texts], dtype=np.float32),
        )

    def test_pins_the_provider_so_reruns_hit_the_same_deployment(self) -> None:
        captured: list[dict] = []

        def handle(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            captured.append(body)
            self.assertEqual(str(request.url), EMBEDDINGS_URL)
            self.assertEqual(request.headers["Authorization"], "Bearer test-key")
            return httpx.Response(
                200,
                json={"data": [{"index": 0, "embedding": [1.0, 2.0]}], "usage": {"total_tokens": 3}},
            )

        with patch.dict(os.environ, KEY):
            client = httpx.Client(transport=httpx.MockTransport(handle))
            embed_texts(["one"], EmbeddingSettings(cache_dir=None), client=client)

        self.assertEqual(captured[0]["provider"], {"order": ["openai"], "allow_fallbacks": False})
        self.assertEqual(captured[0]["encoding_format"], "float")
        self.assertNotIn("dimensions", captured[0])

    def test_retries_a_rate_limited_batch_then_succeeds(self) -> None:
        batches: list[list[str]] = []
        with patch.dict(os.environ, KEY), patch("recipe_diversity.embedding.time.sleep"):
            client = httpx.Client(transport=responder(batches, statuses=[429, 200]))
            result = embed_texts(["one", "two"], EmbeddingSettings(cache_dir=None), client=client)

        self.assertEqual(len(batches), 2)
        self.assertEqual(result.vectors.shape, (2, 3))

    def test_raises_after_the_retry_budget_is_spent(self) -> None:
        with patch.dict(os.environ, KEY), patch("recipe_diversity.embedding.time.sleep"):
            client = httpx.Client(transport=responder([], statuses=[429, 429, 429]))
            with self.assertRaises(httpx.HTTPStatusError):
                embed_texts(["one"], EmbeddingSettings(cache_dir=None), client=client)

    def test_halves_a_batch_the_provider_rejects_as_too_large(self) -> None:
        texts = [f"recipe {index}" for index in range(8)]
        batches: list[list[str]] = []
        with patch.dict(os.environ, KEY):
            client = httpx.Client(transport=responder(batches, statuses=[413, 200, 200]))
            result = embed_texts(texts, EmbeddingSettings(batch_size=8, cache_dir=None), client=client)

        self.assertEqual([len(batch) for batch in batches], [8, 4, 4])
        np.testing.assert_allclose(
            result.vectors,
            np.array([vector_for(text) for text in texts], dtype=np.float32),
        )

    def test_rejects_a_batch_size_below_one(self) -> None:
        with self.assertRaisesRegex(ValueError, "batch_size must be at least 1"):
            embed_texts(["one"], EmbeddingSettings(batch_size=0, cache_dir=None))


class ProgressTests(unittest.TestCase):
    def test_advances_once_per_text_and_clears_the_bar_when_done(self) -> None:
        texts = [f"recipe {index}" for index in range(11)]
        progress = Progress(disable=True)
        with patch.dict(os.environ, KEY), progress:
            client = httpx.Client(transport=responder([]))
            embed_texts(
                texts,
                EmbeddingSettings(batch_size=4, cache_dir=None),
                client=client,
                progress=progress,
                description="Embedding nonce",
            )
            live = [(task.description, task.total, task.completed) for task in progress.tasks]

        self.assertEqual(live, [])

    def test_counts_only_uncached_texts_so_a_resume_shows_real_work(self) -> None:
        seen: list[tuple[str, float | None]] = []

        class Recorder(Progress):
            def add_task(self, description, **kwargs):
                seen.append((description, kwargs.get("total")))
                return super().add_task(description, **kwargs)

        with tempfile.TemporaryDirectory() as directory:
            settings = EmbeddingSettings(batch_size=8, cache_dir=Path(directory))
            progress = Recorder(disable=True)
            with patch.dict(os.environ, KEY), progress:
                embed_texts(
                    ["one", "two"],
                    settings,
                    client=httpx.Client(transport=responder([])),
                    progress=progress,
                )
                embed_texts(
                    ["one", "two", "three", "four"],
                    settings,
                    client=httpx.Client(transport=responder([])),
                    progress=progress,
                )

        self.assertEqual(seen, [("Embedding", 2), ("Embedding", 2)])

    def test_adds_no_task_when_every_text_is_already_cached(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            settings = EmbeddingSettings(cache_dir=Path(directory))
            progress = Progress(disable=True)
            with patch.dict(os.environ, KEY), progress:
                embed_texts(["one"], settings, client=httpx.Client(transport=responder([])))
                before = len(progress.tasks)
                embed_texts(
                    ["one"],
                    settings,
                    client=httpx.Client(transport=responder([])),
                    progress=progress,
                )

                self.assertEqual(len(progress.tasks), before)


class CacheTests(unittest.TestCase):
    def test_reuses_cached_vectors_and_only_requests_the_missing_texts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            settings = EmbeddingSettings(batch_size=4, cache_dir=Path(directory))
            first: list[list[str]] = []
            second: list[list[str]] = []
            with patch.dict(os.environ, KEY):
                embed_texts(["one", "two"], settings, client=httpx.Client(transport=responder(first)))
                result = embed_texts(
                    ["one", "two", "three"],
                    settings,
                    client=httpx.Client(transport=responder(second)),
                )

            self.assertEqual(first, [["one", "two"]])
            self.assertEqual(second, [["three"]])
            self.assertEqual((result.requested, result.cached), (1, 2))
            np.testing.assert_allclose(
                result.vectors,
                np.array([vector_for(text) for text in ("one", "two", "three")], dtype=np.float32),
            )

    def test_makes_no_request_at_all_when_every_text_is_cached(self) -> None:
        def refuse(_request: httpx.Request) -> httpx.Response:
            raise AssertionError("cached texts must not be re-requested")

        with tempfile.TemporaryDirectory() as directory:
            settings = EmbeddingSettings(cache_dir=Path(directory))
            with patch.dict(os.environ, KEY):
                embed_texts(["one"], settings, client=httpx.Client(transport=responder([])))
                result = embed_texts(
                    ["one"],
                    settings,
                    client=httpx.Client(transport=httpx.MockTransport(refuse)),
                )

            self.assertEqual((result.requested, result.cached), (0, 1))

    def test_cache_keys_separate_models_so_a_swap_never_mixes_spaces(self) -> None:
        self.assertNotEqual(
            text_digest("openai/text-embedding-3-large", "Recipe"),
            text_digest("sentence-transformers/all-MiniLM-L6-v2", "Recipe"),
        )
        self.assertEqual(
            text_digest("openai/text-embedding-3-large", "Recipe"),
            text_digest("openai/text-embedding-3-large", "Recipe"),
        )

    def test_requires_an_api_key_before_contacting_the_endpoint(self) -> None:
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": ""}):
            with self.assertRaisesRegex(OpenRouterError, "OPENROUTER_API_KEY is not set"):
                embed_texts(["one"], EmbeddingSettings(cache_dir=None))
