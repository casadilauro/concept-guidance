import json
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from recipe_diversity.generation import (
    PROMPT_TEMPLATES,
    TOTAL_RECIPES,
    WIDE_CUISINE_BATCH,
    WIDE_CUISINES,
    WIDE_RECIPES_PER_CUISINE,
    Completion,
    GenerationSettings,
    generate_wide,
    wide_cuisine_ids,
    wide_jobs,
)

RECIPE_BODY = "Ingredients:\n- item\n\nInstructions:\n1. Cook."


def sections(prompt: str, body: str) -> Completion:
    identifiers = re.findall(r"^=== ([a-z]+(?:_\d{3})+) ===$", prompt, flags=re.MULTILINE)
    return Completion(
        "\n".join(f"=== {name} ===\n{name} title\n{body}" for name in identifiers), "stop", 0
    )


def responder(calls: list[str]):
    def complete(prompt: str, _settings: GenerationSettings) -> Completion:
        calls.append(prompt)
        if prompt.startswith("Create exactly"):
            return sections(prompt, "Cuisine description.")
        return sections(prompt, f"\n{RECIPE_BODY}")

    return complete


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


class WideScaffoldTests(unittest.TestCase):
    def test_budget_moves_from_depth_to_breadth(self) -> None:
        jobs = wide_jobs()

        self.assertEqual(len(jobs), WIDE_CUISINES)
        self.assertEqual(sum(len(ids) for _, ids in jobs), TOTAL_RECIPES)
        self.assertEqual({len(ids) for _, ids in jobs}, {WIDE_RECIPES_PER_CUISINE})

    def test_each_cuisine_owns_one_contiguous_block(self) -> None:
        jobs = wide_jobs()

        self.assertEqual(jobs[0][1][0], "recipe_000")
        self.assertEqual(jobs[0][1][-1], "recipe_009")
        self.assertEqual(jobs[99][1][0], "recipe_990")
        self.assertEqual(jobs[99][1][-1], "recipe_999")
        self.assertEqual(len(wide_cuisine_ids()), WIDE_CUISINES)

    def test_recipe_template_has_no_dish_layer_and_no_exclusion_list(self) -> None:
        template = PROMPT_TEMPLATES["wide_recipe_prompt.txt"]

        self.assertIn("{cuisine}", template)
        self.assertNotIn("{dish}", template)
        self.assertNotIn("Do not duplicate", template)
        self.assertNotIn("{existing_titles}", template)
        self.assertIn("as different from one another as possible", template)


class WideGenerationTests(unittest.TestCase):
    def _generate(self, tmp: Path) -> tuple[list[str], list[dict]]:
        calls: list[str] = []
        with (
            patch("recipe_diversity.generation.require_api_key", return_value="key"),
            patch("recipe_diversity.generation.stream_completion", side_effect=responder(calls)),
        ):
            output = generate_wide(tmp / "wide")
        return calls, read_jsonl(output / "recipes.jsonl")

    def test_uses_the_same_call_budget_as_the_batched_scaffold(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            calls, recipes = self._generate(Path(directory))

        # 10 cuisine batches + 100 recipe batches.
        self.assertEqual(len(calls), WIDE_CUISINES // WIDE_CUISINE_BATCH + WIDE_CUISINES)
        self.assertEqual(len(recipes), TOTAL_RECIPES)
        self.assertTrue(all(row["valid_format"] for row in recipes))
        self.assertEqual({row["strategy"] for row in recipes}, {"wide"})

    def test_produces_a_hundred_cuisines_ten_per_response(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            calls, recipes = self._generate(Path(directory))

        cuisine_calls = [c for c in calls if c.startswith("Create exactly")]
        self.assertEqual(len(cuisine_calls), WIDE_CUISINES // WIDE_CUISINE_BATCH)
        for call in cuisine_calls:
            self.assertEqual(call.count("=== cuisine_"), WIDE_CUISINE_BATCH)
        self.assertEqual(
            {row["cuisine_id"] for row in recipes}, set(wide_cuisine_ids())
        )

    def test_records_carry_no_dish_level_so_only_cuisine_strata_are_reported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _, recipes = self._generate(Path(directory))

        self.assertTrue(all("dish_id" not in row for row in recipes))
        self.assertTrue(all(row["cuisine_id"] for row in recipes))

    def test_each_leaf_prompt_carries_only_its_own_cuisine(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            calls, _ = self._generate(Path(directory))

        leaf_calls = [c for c in calls if c.startswith("Write exactly")]
        self.assertEqual(len(leaf_calls), WIDE_CUISINES)
        for call in leaf_calls:
            self.assertEqual(call.count("=== cuisine_"), 0)
            self.assertEqual(call.count("=== recipe_"), WIDE_RECIPES_PER_CUISINE)

    def test_records_are_written_in_recipe_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _, recipes = self._generate(Path(directory))

        self.assertEqual(
            [row["recipe_id"] for row in recipes],
            [f"recipe_{index:03d}" for index in range(TOTAL_RECIPES)],
        )

    def test_resuming_a_finished_run_requests_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, first = self._generate(root)

            replay: list[str] = []
            with (
                patch("recipe_diversity.generation.require_api_key", return_value="key"),
                patch(
                    "recipe_diversity.generation.stream_completion",
                    side_effect=responder(replay),
                ),
            ):
                output = generate_wide(root / "wide")
            resumed = read_jsonl(output / "recipes.jsonl")

        self.assertEqual(replay, [])
        self.assertEqual(resumed, first)

    def test_repairs_a_dropped_cuisine_as_well_as_a_dropped_recipe(self) -> None:
        """Both batched layers must survive the model omitting an item."""
        remaining = {"cuisine_007": 1, "recipe_044": 1}
        calls: list[str] = []

        def complete(prompt: str, _settings: GenerationSettings) -> Completion:
            calls.append(prompt)
            body = "Cuisine description." if prompt.startswith("Create exactly") else f"\n{RECIPE_BODY}"
            identifiers = re.findall(r"^=== ([a-z]+(?:_\d{3})+) ===$", prompt, flags=re.MULTILINE)
            kept = []
            for name in identifiers:
                if remaining.get(name, 0) > 0:
                    remaining[name] -= 1
                    continue
                kept.append(name)
            return Completion(
                "\n".join(f"=== {n} ===\n{n} title\n{body}" for n in kept), "stop", 0
            )

        with (
            tempfile.TemporaryDirectory() as directory,
            patch("recipe_diversity.generation.require_api_key", return_value="key"),
            patch("recipe_diversity.generation.stream_completion", side_effect=complete),
        ):
            output = generate_wide(Path(directory) / "wide")
            recipes = read_jsonl(output / "recipes.jsonl")

        self.assertEqual(len(recipes), TOTAL_RECIPES)
        self.assertEqual({row["cuisine_id"] for row in recipes}, set(wide_cuisine_ids()))
        self.assertEqual(len([c for c in calls if c.startswith("Create exactly 1 ")]), 1)
        self.assertEqual(len([c for c in calls if c.startswith("Write exactly 1 ")]), 1)


class NonceEchoTitleTests(unittest.TestCase):
    """Batched sections routinely echo their `Nonce:` label into the body."""

    def test_title_skips_an_echoed_nonce_line(self) -> None:
        from recipe_diversity.generation import _title

        self.assertEqual(
            _title("Nonce: c7bcefcf46c4\nArctic Icefish Feast\n\nIngredients:"),
            "Arctic Icefish Feast",
        )
        self.assertEqual(_title("Coconut Taro Wedges\n\nIngredients:"), "Coconut Taro Wedges")
        self.assertEqual(_title("Nonce: abc"), "")
        self.assertEqual(_title(""), "")

    def test_loader_derives_the_same_title_as_generation(self) -> None:
        from recipe_diversity.analysis import _title_from
        from recipe_diversity.generation import _title

        text = "Nonce: deadbeef\nSpace Colony Hydroponics\n\nIngredients:"
        self.assertEqual(_title_from(text), _title(text))

    def test_generated_records_carry_real_titles_not_nonces(self) -> None:
        calls: list[str] = []

        def complete(prompt: str, _settings: GenerationSettings) -> Completion:
            calls.append(prompt)
            body = "Cuisine description." if prompt.startswith("Create exactly") else f"\n{RECIPE_BODY}"
            out = []
            for name, nonce in re.findall(
                r"^=== ([a-z]+(?:_\d{3})+) ===\n(Nonce: \w+)$", prompt, flags=re.MULTILINE
            ):
                # Echo the nonce back, exactly as the real model does.
                out.append(f"=== {name} ===\n{nonce}\n{name} title\n{body}")
            return Completion("\n".join(out), "stop", 0)

        with (
            tempfile.TemporaryDirectory() as directory,
            patch("recipe_diversity.generation.require_api_key", return_value="key"),
            patch("recipe_diversity.generation.stream_completion", side_effect=complete),
        ):
            output = generate_wide(Path(directory) / "wide")
            recipes = read_jsonl(output / "recipes.jsonl")

        self.assertTrue(all(not row["title"].startswith("Nonce:") for row in recipes))
        self.assertTrue(all(not row["cuisine_title"].startswith("Nonce:") for row in recipes))
        self.assertEqual(recipes[0]["title"], "recipe_000 title")
        self.assertEqual(recipes[0]["cuisine_title"], "cuisine_000 title")


class SingleCallRootTests(unittest.TestCase):
    def test_all_cuisines_are_requested_in_one_response(self) -> None:
        self.assertEqual(WIDE_CUISINE_BATCH, WIDE_CUISINES)

        calls: list[str] = []
        with (
            tempfile.TemporaryDirectory() as directory,
            patch("recipe_diversity.generation.require_api_key", return_value="key"),
            patch("recipe_diversity.generation.stream_completion", side_effect=responder(calls)),
        ):
            generate_wide(Path(directory) / "wide")

        cuisine_calls = [c for c in calls if c.startswith("Create exactly")]
        self.assertEqual(len(cuisine_calls), 1)
        self.assertEqual(cuisine_calls[0].count("=== cuisine_"), WIDE_CUISINES)
        self.assertIn("=== cuisine_099 ===", cuisine_calls[0])
        # 1 cuisine call + 100 recipe calls.
        self.assertEqual(len(calls), 1 + WIDE_CUISINES)
