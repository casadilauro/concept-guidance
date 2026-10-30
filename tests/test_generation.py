import json
import os
import re
import tempfile
import threading
import time
import unittest
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import call, patch

import httpx
from rich.progress import Progress

from recipe_diversity.generation import (
    GenerationSettings,
    Completion,
    OpenRouterError,
    PROMPT_TEMPLATES,
    TOTAL_RECIPES,
    build_prior_titles_prompt,
    generate_nonce,
    generate_prior_titles,
    nonce_recipe_requests,
    parse_recipe_batch,
    parse_sse_events,
    require_api_key,
    resume_prefix_length,
    stable_nonce,
    stream_completion,
)


class FakeResponse:
    def __init__(
        self,
        lines: list[str] | None = None,
        *,
        status_code: int = 200,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.lines = lines or []
        self.status_code = status_code
        self.headers = headers or {}
        self.request = httpx.Request("POST", "https://openrouter.ai")

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def iter_lines(self) -> Iterator[str]:
        return iter(self.lines)

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                f"HTTP {self.status_code}",
                request=self.request,
                response=httpx.Response(self.status_code, request=self.request),
            )


class FakeClient:
    def __init__(self, outcomes: list[FakeResponse | Exception]) -> None:
        self.outcomes = outcomes
        self.calls: list[tuple[str, str, dict[str, object]]] = []

    def __enter__(self) -> "FakeClient":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def stream(self, method: str, url: str, **kwargs: object) -> FakeResponse:
        self.calls.append((method, url, kwargs))
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class OpenRouterBoundaryTests(unittest.TestCase):
    def test_collects_content_and_ignores_reasoning_events(self) -> None:
        completion = parse_sse_events([
            'data: {"choices":[{"delta":{"reasoning":"hidden"}}]}',
            'data: {"choices":[{"delta":{"content":"Tea"}}]}',
            'data: {"choices":[{"delta":{"content":" Cake"},"finish_reason":"stop"}]}',
            "data: [DONE]",
        ])

        self.assertEqual(completion.text, "Tea Cake")
        self.assertEqual(completion.reasoning_characters, 6)
        self.assertEqual(completion.finish_reason, "stop")

    def test_rejects_missing_api_key_before_requesting(self) -> None:
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": ""}, clear=False):
            with self.assertRaisesRegex(OpenRouterError, "OPENROUTER_API_KEY"):
                require_api_key()

    def test_ignores_noise_and_rejects_stream_without_content(self) -> None:
        with self.assertRaisesRegex(OpenRouterError, "no content"):
            parse_sse_events([
                "",
                ": keepalive",
                "event: message",
                "data: not-json",
                'data: {"choices":[1]}',
                'data: {"choices":[{"delta":{"reasoning":7}}]}',
                'data: {"choices":[{"delta":{"reasoning":"private"}}]}',
                'data: {"choices":[{"delta":{"content":""}}]}',
            ])

    def test_streams_expected_request_and_returns_completion(self) -> None:
        response = FakeResponse([
            'data: {"choices":[{"delta":{"content":"Soup"},"finish_reason":"stop"}]}',
            "data: [DONE]",
        ])
        client = FakeClient([response])
        settings = GenerationSettings(
            model="test/model",
            temperature=0.4,
            top_p=0.8,
            max_tokens=99,
            timeout_seconds=12.0,
        )

        with (
            patch.dict(os.environ, {"OPENROUTER_API_KEY": " secret "}, clear=False),
            patch("recipe_diversity.generation.httpx.Client", return_value=client) as client_class,
        ):
            completion = stream_completion("Make soup", settings)

        self.assertEqual(completion.text, "Soup")
        client_class.assert_called_once_with(timeout=12.0)
        self.assertEqual(
            client.calls,
            [(
                "POST",
                "https://openrouter.ai/api/v1/chat/completions",
                {
                    "headers": {"Authorization": "Bearer secret"},
                    "json": {
                        "stream": True,
                        "model": "test/model",
                        "messages": [{"role": "user", "content": "Make soup"}],
                        "temperature": 0.4,
                        "top_p": 0.8,
                        "max_tokens": 99,
                    },
                },
            )],
        )

    def test_missing_key_fails_before_constructing_client(self) -> None:
        with (
            patch.dict(os.environ, {"OPENROUTER_API_KEY": "  "}, clear=False),
            patch("recipe_diversity.generation.httpx.Client") as client_class,
            self.assertRaises(OpenRouterError),
        ):
            stream_completion("prompt", GenerationSettings())

        client_class.assert_not_called()

    def test_retries_transport_errors_with_exponential_delays(self) -> None:
        request = httpx.Request("POST", "https://openrouter.ai")
        client = FakeClient([
            httpx.ReadTimeout("slow", request=request),
            httpx.ConnectError("offline", request=request),
            FakeResponse(['data: {"choices":[{"delta":{"content":"ok"}}]}']),
        ])

        with (
            patch.dict(os.environ, {"OPENROUTER_API_KEY": "key"}, clear=False),
            patch("recipe_diversity.generation.httpx.Client", return_value=client),
            patch("recipe_diversity.generation.time.sleep") as sleep,
        ):
            completion = stream_completion("prompt", GenerationSettings())

        self.assertEqual(completion.text, "ok")
        self.assertEqual(sleep.call_args_list, [call(2.0), call(4.0)])
        self.assertEqual(len(client.calls), 3)

    def test_retries_empty_streams(self) -> None:
        client = FakeClient([
            FakeResponse(['data: {"choices":[{"delta":{"reasoning":"hidden"}}]}']),
            FakeResponse(['data: {"choices":[{"delta":{"content":"ok"}}]}']),
        ])

        with (
            patch.dict(os.environ, {"OPENROUTER_API_KEY": "key"}, clear=False),
            patch("recipe_diversity.generation.httpx.Client", return_value=client),
            patch("recipe_diversity.generation.time.sleep") as sleep,
        ):
            completion = stream_completion("prompt", GenerationSettings())

        self.assertEqual(completion.text, "ok")
        sleep.assert_called_once_with(2.0)
        self.assertEqual(len(client.calls), 2)

    def test_uses_positive_retry_after_for_transient_http_status(self) -> None:
        client = FakeClient([
            FakeResponse(status_code=429, headers={"Retry-After": "1.5"}),
            FakeResponse(['data: {"choices":[{"delta":{"content":"ok"}}]}']),
        ])

        with (
            patch.dict(os.environ, {"OPENROUTER_API_KEY": "key"}, clear=False),
            patch("recipe_diversity.generation.httpx.Client", return_value=client),
            patch("recipe_diversity.generation.time.sleep") as sleep,
        ):
            stream_completion("prompt", GenerationSettings())

        sleep.assert_called_once_with(1.5)
        self.assertEqual(len(client.calls), 2)

    def test_does_not_retry_non_transient_http_status(self) -> None:
        client = FakeClient([FakeResponse(status_code=400)])

        with (
            patch.dict(os.environ, {"OPENROUTER_API_KEY": "key"}, clear=False),
            patch("recipe_diversity.generation.httpx.Client", return_value=client),
            patch("recipe_diversity.generation.time.sleep") as sleep,
            self.assertRaises(httpx.HTTPStatusError),
        ):
            stream_completion("prompt", GenerationSettings())

        sleep.assert_not_called()
        self.assertEqual(len(client.calls), 1)


class StrategyTests(unittest.TestCase):
    def test_prior_titles_prompt_contains_titles_and_no_recipe_body_argument(self) -> None:
        prompt = build_prior_titles_prompt("abc123", ["Tea Cake", "Rice Soup"])

        self.assertIn("- Tea Cake", prompt)
        self.assertIn("- Rice Soup", prompt)
        self.assertIn("Nonce: abc123", prompt)
        self.assertIn("Ingredients:", prompt)

    def test_prior_titles_prompt_keeps_the_newest_titles_within_budget(self) -> None:
        older = "a" * 40
        newer = "b" * 20

        prompt = build_prior_titles_prompt(
            "abc123", [older, newer], title_token_budget=10
        )

        self.assertNotIn(f"- {older}", prompt)
        self.assertIn(f"- {newer}", prompt)



    def test_recipe_batch_requires_every_expected_delimiter(self) -> None:
        with self.assertRaisesRegex(ValueError, "Missing"):
            parse_recipe_batch(
                "=== recipe_000 ===\nTea Cake",
                ["recipe_000", "recipe_001"],
            )

    def test_recipe_batch_rejects_duplicate_expected_delimiter(self) -> None:
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            parse_recipe_batch(
                "=== recipe_000 ===\nFirst\n=== recipe_000 ===\nSecond",
                ["recipe_000"],
            )

    def test_resume_only_accepts_a_contiguous_recipe_prefix(self) -> None:
        self.assertEqual(
            resume_prefix_length(
                [{"recipe_id": "recipe_000"}, {"recipe_id": "recipe_001"}],
                TOTAL_RECIPES,
            ),
            2,
        )
        with self.assertRaisesRegex(ValueError, "contiguous"):
            resume_prefix_length(
                [{"recipe_id": "recipe_000"}, {"recipe_id": "recipe_002"}],
                TOTAL_RECIPES,
            )

    def test_generation_validates_api_key_before_writing_run_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "nonce"
            with (
                patch(
                    "recipe_diversity.generation.require_api_key",
                    side_effect=OpenRouterError("missing"),
                ),
                self.assertRaises(OpenRouterError),
            ):
                generate_nonce(output)

            self.assertFalse(output.exists())

    def test_generation_rejects_non_experiment_model_before_writing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "nonce"
            with (
                patch("recipe_diversity.generation.require_api_key", return_value="key"),
                patch(
                    "recipe_diversity.generation.stream_completion",
                    return_value=Completion("Title", "stop", 0),
                ),
                self.assertRaisesRegex(ValueError, "model"),
            ):
                generate_nonce(output, GenerationSettings(model="other/model"))

            self.assertFalse(output.exists())

    def test_nonce_prompt_varies_only_by_opaque_nonce(self) -> None:
        first, second = nonce_recipe_requests(2)

        self.assertNotIn("recipe_000", first.prompt)
        self.assertNotIn("recipe_001", second.prompt)
        self.assertNotIn("Request ID:", first.prompt)
        self.assertEqual(
            re.sub(r"Nonce: [0-9a-f]+", "Nonce: <opaque>", first.prompt),
            re.sub(r"Nonce: [0-9a-f]+", "Nonce: <opaque>", second.prompt),
        )

    def test_nonce_generation_writes_deterministic_records_and_raw_provenance(self) -> None:
        def complete(prompt: str, settings: GenerationSettings) -> Completion:
            return Completion(
                "Tea title\n\nIngredients:\n- tea\n\nInstructions:\n1. Brew.",
                "stop",
                7,
            )

        with (
            tempfile.TemporaryDirectory() as tmp,
            patch("recipe_diversity.generation.require_api_key", return_value="key"),
            patch("recipe_diversity.generation.stream_completion", side_effect=complete),
        ):
            progress = Progress(disable=True)
            with progress:
                output = generate_nonce(Path(tmp) / "nonce", progress=progress)
            recipes = self._read_jsonl(output / "recipes.jsonl")
            raw_index = self._read_jsonl(output / "raw_responses.jsonl")
            config = json.loads(
                (output / "generation_config.json").read_text(encoding="utf-8")
            )

            self.assertEqual(len(recipes), TOTAL_RECIPES)
            self.assertEqual(len(raw_index), TOTAL_RECIPES)
            self.assertEqual(
                [row["recipe_id"] for row in recipes],
                [f"recipe_{index:03d}" for index in range(TOTAL_RECIPES)],
            )
            self.assertEqual(len({row["nonce"] for row in recipes}), TOTAL_RECIPES)
            self.assertTrue(all(row["valid_format"] for row in recipes))
            self.assertTrue(all((output / row["raw_path"]).is_file() for row in raw_index))
            task = next(task for task in progress.tasks if task.description == "Recipes")
            self.assertEqual((task.total, task.completed), (TOTAL_RECIPES, TOTAL_RECIPES))
            self.assertEqual(
                {
                    "total_recipes": config["total_recipes"],
                    "temperature": config["temperature"],
                    "top_p": config["top_p"],
                    "max_tokens": config["max_tokens"],
                    "max_workers": config["max_workers"],
                },
                {
                    "total_recipes": TOTAL_RECIPES,
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "max_tokens": 1200,
                    "max_workers": 4,
                },
            )

    def test_resume_rejects_corrupt_completed_recipe_before_requesting(self) -> None:
        completion = Completion(
            "Tea title\n\nIngredients:\n- tea\n\nInstructions:\n1. Brew.",
            "stop",
            0,
        )
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "nonce"
            with (
                patch("recipe_diversity.generation.require_api_key", return_value="key"),
                patch(
                    "recipe_diversity.generation.stream_completion",
                    return_value=completion,
                ),
            ):
                generate_nonce(output)

            records = self._read_jsonl(output / "recipes.jsonl")
            records[0]["nonce"] = "corrupt"
            (output / "recipes.jsonl").write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )
            with (
                patch("recipe_diversity.generation.require_api_key", return_value="key"),
                patch("recipe_diversity.generation.stream_completion") as complete,
                self.assertRaisesRegex(ValueError, "provenance"),
            ):
                generate_nonce(output)

            complete.assert_not_called()

    def test_nonce_failure_does_not_schedule_beyond_four_request_batch(self) -> None:
        calls = 0

        def fail(prompt: str, settings: GenerationSettings) -> Completion:
            nonlocal calls
            calls += 1
            raise RuntimeError("interrupted")

        with (
            tempfile.TemporaryDirectory() as tmp,
            patch("recipe_diversity.generation.require_api_key", return_value="key"),
            patch("recipe_diversity.generation.stream_completion", side_effect=fail),
            self.assertRaisesRegex(RuntimeError, "interrupted"),
        ):
            generate_nonce(Path(tmp) / "nonce")

        self.assertLessEqual(calls, 4)

    def test_prior_titles_resumes_raw_response_without_repeating_request(self) -> None:
        nonces = {
            stable_nonce("prior-title", index): index
            for index in range(TOTAL_RECIPES)
        }

        def complete(prompt: str, settings: GenerationSettings) -> Completion:
            nonce = re.search(r"Nonce: ([0-9a-f]+)", prompt).group(1)
            index = nonces[nonce]
            if index == 2:
                raise RuntimeError("interrupted")
            return Completion(
                f"Title {index}\n\nIngredients:\n- item {index}\n\nInstructions:\n1. Cook {index}.",
                "stop",
                0,
            )

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "prior"
            with (
                patch("recipe_diversity.generation.require_api_key", return_value="key"),
                patch(
                    "recipe_diversity.generation.stream_completion",
                    side_effect=complete,
                ),
                self.assertRaisesRegex(RuntimeError, "interrupted"),
            ):
                generate_prior_titles(output)

            index_rows = self._read_jsonl(output / "raw_responses.jsonl")
            (output / "raw_responses.jsonl").write_text(
                json.dumps(index_rows[0]) + "\n",
                encoding="utf-8",
            )
            resumed_prompts: list[str] = []

            def resume_complete(
                prompt: str,
                settings: GenerationSettings,
            ) -> Completion:
                resumed_prompts.append(prompt)
                nonce = re.search(r"Nonce: ([0-9a-f]+)", prompt).group(1)
                index = nonces[nonce]
                return Completion(
                    f"Title {index}\n\nIngredients:\n- item {index}\n\nInstructions:\n1. Cook {index}.",
                    "stop",
                    0,
                )

            with (
                patch("recipe_diversity.generation.require_api_key", return_value="key"),
                patch(
                    "recipe_diversity.generation.stream_completion",
                    side_effect=resume_complete,
                ),
            ):
                generate_prior_titles(output)

            recipes = self._read_jsonl(output / "recipes.jsonl")
            self.assertEqual(len(recipes), TOTAL_RECIPES)
            self.assertEqual(
                len(self._read_jsonl(output / "raw_responses.jsonl")), TOTAL_RECIPES
            )
            self.assertEqual(len(resumed_prompts), TOTAL_RECIPES - 2)
            self.assertIn("- Title 0", resumed_prompts[0])
            self.assertIn("- Title 1", resumed_prompts[0])
            self.assertNotIn("item 0", resumed_prompts[0])
            self.assertNotIn("Cook 1", resumed_prompts[0])

    def test_generation_rejects_existing_recipe_gap_without_requesting(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "prior"
            output.mkdir()
            (output / "recipes.jsonl").write_text(
                "\n".join(
                    json.dumps({"recipe_id": recipe_id})
                    for recipe_id in ("recipe_000", "recipe_002")
                )
                + "\n",
                encoding="utf-8",
            )
            with (
                patch("recipe_diversity.generation.require_api_key", return_value="key"),
                patch("recipe_diversity.generation.stream_completion") as complete,
                self.assertRaisesRegex(ValueError, "contiguous"),
            ):
                generate_prior_titles(output)

            complete.assert_not_called()

    def test_raw_index_without_raw_file_stops_before_requesting(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "prior"
            output.mkdir()
            (output / "raw_responses.jsonl").write_text(
                json.dumps(
                    {
                        "request_id": "prior_recipe_000",
                        "raw_path": "raw/prior_recipe_000.json",
                        "text": "Title 0",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with (
                patch("recipe_diversity.generation.require_api_key", return_value="key"),
                patch("recipe_diversity.generation.stream_completion") as complete,
                self.assertRaisesRegex(ValueError, "raw file"),
            ):
                generate_prior_titles(output)

            complete.assert_not_called()






    @staticmethod
    def _read_jsonl(path: Path) -> list[dict[str, object]]:
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line
        ]
