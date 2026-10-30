import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import analyze
import run


class CliTests(unittest.TestCase):
    def test_dry_run_needs_no_key_and_writes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "study"
            with contextlib.redirect_stdout(io.StringIO()) as capture:
                self.assertEqual(run.main(["all", "--output", str(output), "--dry-run"]), 0)
            self.assertEqual(len(capture.getvalue().splitlines()), 3)
            self.assertFalse(output.exists())

    def test_existing_run_requires_explicit_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            (output / "runs/nonce_1000").mkdir(parents=True)
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                run.main(["nonce", "--output", str(output), "--dry-run"])

    def test_evidence_cannot_be_generation_output(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            run.main(["all", "--output", str(run.ROOT / "data"), "--dry-run"])

    def test_recorded_encoder_cannot_fall_back_to_paid_embeddings(self):
        with patch("recipe_diversity.embedding.embed_texts", side_effect=AssertionError("API called")):
            with self.assertRaises(KeyError):
                analyze.recorded_encoder(["not a recorded recipe"])

    def test_new_run_analysis_requires_explicit_embedding_request(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            analyze.main(["--runs", "missing-study"])
