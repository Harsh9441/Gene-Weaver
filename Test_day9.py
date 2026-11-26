"""Day 9-10 tests for the Textual terminal search pipeline."""
import json
import os
import tempfile
import unittest
from unittest import mock

import fasta as f


class TextualProgress(unittest.TestCase):
    def test_textual_is_available_or_reported_as_optional(self):
        if f.HAVE_TEXTUAL:
            self.assertTrue(callable(f.run_textual_app))
            self.assertTrue(issubclass(f.GeneWeaverApp, f.App))
        else:
            with self.assertRaisesRegex(RuntimeError, "Textual"):
                f.run_textual_app()

    def test_progress_callback_records_progress(self):
        view = f.TextualProgressView(None)
        text = view(5, 10, 7, 1.5)
        self.assertEqual((view.completed, view.total, view.hits, view.elapsed), (5, 10, 7, 1.5))
        self.assertIn("50%", text)

    def test_fake_app_receives_progress_updates(self):
        class FakeApp:
            def __init__(self):
                self.calls = []

            def call_from_thread(self, fn, *args):
                fn(*args)

            def update_progress(self, completed, total, hits, elapsed):
                self.calls.append((completed, total, hits, elapsed))

        app = FakeApp()
        f.TextualProgressView(app)(3, 12, 4, 0.25)
        self.assertEqual(app.calls, [(3, 12, 4, 0.25)])


class ConnectedPipeline(unittest.TestCase):
    def test_pipeline_runs_search_verifies_and_persists_the_record(self):
        spacer = "GACGTCTAGCATCGATCGAA"
        genome = {"chr1": spacer + "TGG"}
        with tempfile.TemporaryDirectory() as td:
            result = f.run_search_pipeline(
                genome, [("g1", spacer)], "C1", backend="emulated",
                runs=1, warmup=0, load_s=0.25, out_dir=td,
            )

            record, hits, saved_path = result
            with open(saved_path) as fh:
                saved = json.load(fh)
            self.assertTrue(os.path.isfile(saved_path))

        self.assertEqual(len(hits), 1)
        self.assertTrue(record["correctness"]["matches_baseline"])
        self.assertEqual(record["load_s"], 0.25)
        self.assertEqual(saved["total_hits"], len(hits))

    def test_textual_search_uses_the_connected_pipeline(self):
        if not f.HAVE_TEXTUAL:
            self.skipTest("Textual is not installed")
        status = mock.Mock()
        record = {"correctness": {"note": "matches baseline"}}
        hits = ["hit"]
        app = f.GeneWeaverApp(
            {"chr1": "ACGT"}, [("g1", "ACGT")], f.CONFIGS["C1"],
            backend="emulated", out_dir="out",
        )
        with mock.patch.object(f, "run_search_pipeline", return_value=(record, hits, "out/result.json")) as run, \
                mock.patch.object(app, "query_one", return_value=status), \
                mock.patch.object(app, "call_from_thread", side_effect=lambda fn, *args: fn(*args)):
            app._run_search()

        run.assert_called_once()
        self.assertEqual(app.results, hits)
        self.assertIs(app.record, record)
        self.assertEqual(app.saved_path, "out/result.json")
        self.assertIn("saved out/result.json", status.update.call_args.args[0])

    def test_textual_search_runs_as_a_thread_worker(self):
        if not f.HAVE_TEXTUAL:
            self.skipTest("Textual is not installed")
        app = f.GeneWeaverApp({"chr1": "ACGT"}, [("g1", "ACGT")])
        with mock.patch.object(app, "run_worker") as run_worker:
            app.action_run_search()
        run_worker.assert_called_once_with(app._run_search, exclusive=True, thread=True)


class TuiCommand(unittest.TestCase):
    def test_tui_command_is_registered(self):
        if f.HAVE_TEXTUAL:
            with mock.patch.object(f.GeneWeaverApp, "run", return_value=None):
                rc = f.main(["tui", "--n-guides", "1", "--scale", "0.0001"])
            self.assertEqual(rc, 0)

    def test_tui_command_forwards_file_metadata_and_pipeline_options(self):
        genome = {"chr1": "ACGT"}
        guides = [("g1", "ACGT")]
        with mock.patch.object(f, "read_genome", return_value=genome), \
                mock.patch.object(f, "read_guides_any", return_value=guides), \
                mock.patch.object(f, "sha256", return_value="abc123"), \
                mock.patch.object(f, "run_textual_app") as launch:
            rc = f.main([
                "tui", "--genome", "genome.fa", "--guides", "guides.tsv",
                "--runs", "2", "--warmup", "1", "--out-dir", "out",
            ])

        self.assertEqual(rc, 0)
        args = launch.call_args.args
        self.assertEqual(args[0:2], (genome, guides))
        self.assertEqual(args[7:11], (None, "genome.fa", "guides.tsv", "abc123"))
        self.assertIsInstance(args[11], float)
        self.assertEqual(args[12:], (2, 1, "out"))
