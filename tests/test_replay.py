"""Replay every recorded request through the extracted generation procedure."""

import gzip
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from recipe_diversity import generate_nonce, generate_prior_titles, generate_wide

ROOT = Path(__file__).resolve().parents[1]


class ReplayTests(unittest.TestCase):
    def test_all_three_realized_populations_and_repairs(self):
        for name, generate in (
            ("nonce_1000", generate_nonce),
            ("prior_titles_1000", generate_prior_titles),
            ("wide_1000", generate_wide),
        ):
            with self.subTest(population=name), tempfile.TemporaryDirectory() as tmp:
                folder = Path(tmp)
                source = ROOT / "data/runs" / name
                (folder / "raw").mkdir()
                with gzip.open(source / "requests.jsonl.gz", "rt") as handle:
                    requests = [json.loads(line) for line in handle]
                for request in requests:
                    (folder / "raw" / f"{request['request_id']}.json").write_text(json.dumps(request))
                with (
                    patch("recipe_diversity.generation.require_api_key", return_value="offline-replay"),
                    patch("recipe_diversity.generation.stream_completion", side_effect=AssertionError("Unexpected API call")),
                ):
                    generate(folder)
                    expected = [json.loads(line) for line in (source / "recipes.jsonl").read_text().splitlines()]
                    actual = [json.loads(line) for line in (folder / "recipes.jsonl").read_text().splitlines()]
                    self.assertEqual(actual, expected)
                    before = (folder / "recipes.jsonl").read_bytes()
                    generate(folder)
                    self.assertEqual((folder / "recipes.jsonl").read_bytes(), before)
                self.assertEqual(len(requests), 104 if name == "wide_1000" else 1000)


if __name__ == "__main__":
    unittest.main()
