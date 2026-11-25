"""Day 9 tests for the Textual terminal interface."""
import io
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


class TuiCommand(unittest.TestCase):
    def test_tui_command_is_registered(self):
        if f.HAVE_TEXTUAL:
            with mock.patch.object(f.GeneWeaverApp, "run", return_value=None):
                rc = f.main(["tui", "--n-guides", "1", "--scale", "0.0001"])
            self.assertEqual(rc, 0)
