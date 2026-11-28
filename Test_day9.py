"""Day 9 tests: the text interface (live dashboard with a progress bar per stage)."""
import io, os, random, re, unittest
from unittest import mock

import fasta as f

STAGES = ("setup", "inputs", "search")


class FakeTTY(io.StringIO):
    def isatty(self):
        return True


class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t

    def tick(self, seconds):
        self.t += seconds


def replay(text):
    """A tiny terminal model: applies cursor-up (ESC[nA), clear-line (ESC[K) and newlines, returns the screen lines."""
    screen, row, i = [""], 0, 0
    while i < len(text):
        m = re.match(r"\x1b\[(\d*)([AK])", text[i:])
        if m:
            if m.group(2) == "A":
                row = max(0, row - int(m.group(1) or 1))
            else:
                screen[row] = ""
            i += m.end()
            continue
        if text[i] == "\n":
            row += 1
            while len(screen) <= row:
                screen.append("")
        else:
            screen[row] += text[i]
        i += 1
    return screen


def make(ansi=True, width=70, **kw):
    clock = Clock()
    out = FakeTTY() if ansi else io.StringIO()
    return f.Dashboard("Title", STAGES, out=out, ansi=ansi, width=width, clock=clock, **kw), out, clock


class ProgressBar(unittest.TestCase):
    def test_bar_shapes(self):
        self.assertEqual(f.make_bar(0, 10), " " * 10)
        self.assertEqual(f.make_bar(1, 10), "=" * 10)
        self.assertEqual(f.make_bar(0.5, 10), "====>     ")
        self.assertEqual(f.make_bar(0.06, 10), ">" + " " * 9)                          # one cell: just the head
        self.assertEqual(f.make_bar(0.9, 10), "=" * 8 + "> ")                          # nine cells: the head is the last filled one
        for frac in (-3, 0, 0.01, 0.33, 0.5, 0.99, 1, 7):
            self.assertEqual(len(f.make_bar(frac, 17)), 17, frac)
        self.assertEqual(f.make_bar(-3, 10), f.make_bar(0, 10))                        # out-of-range values are clamped
        self.assertEqual(f.make_bar(7, 10), f.make_bar(1, 10))

    def test_day_6_progress_line_still_uses_the_same_bar(self):
        self.assertEqual(f.format_progress(5, 10, 7, 1.5), "[" + f.make_bar(0.5, 30) + "]  50%  chunk 5/10  hits 7  1.5s")


class Rendering(unittest.TestCase):
    def test_layout_is_fixed(self):
        d, _, _ = make()
        lines = d.render()
        self.assertEqual(len(lines), len(STAGES) + 4)                                  # title, rule, rows, rule, metrics
        self.assertTrue(all(len(l) == 70 for l in lines))
        self.assertTrue(lines[0].startswith("Title"))
        self.assertEqual(lines[1], "-" * 70)
        for stage, line in zip(STAGES, lines[2:5]):
            self.assertIn(stage, line)
            self.assertIn("waiting", line)
            self.assertIn("0%", line)
        self.assertTrue(lines[-1].strip().startswith("elapsed 0.0s"))

    def test_stage_life_cycle(self):
        d, _, clock = make(ansi=False)
        d.start("setup")
        clock.tick(1.5)
        self.assertIn("running", d.render()[2])
        self.assertIn("1.50s", d.render()[2])                                          # a running stage shows live time
        d.progress("setup", 0.5, "half way")
        row = d.render()[2]
        self.assertIn("50%", row)
        self.assertIn("half way", row)
        clock.tick(0.5)
        d.finish("setup", True, "ready")
        clock.tick(10)
        row = d.render()[2]
        self.assertIn("done", row)
        self.assertIn("100%", row)
        self.assertIn("2.00s", row)                                                    # frozen at the finish time
        d.start("inputs")
        d.progress("inputs", 0.4)
        clock.tick(1)
        d.finish("inputs", False, "bad file")
        row = d.render()[3]
        self.assertIn("FAILED", row)
        self.assertIn("40%", row)                                                      # a failed bar stays where it stopped
        d.skip("search", "an earlier stage failed")
        self.assertIn("skipped", d.render()[4])
        self.assertIn("0.00s", d.render()[4])

    def test_progress_is_clamped_and_unknown_stages_are_rejected(self):
        d, _, _ = make(ansi=False)
        d.start("setup")
        d.progress("setup", 9)
        self.assertEqual(d.rows[0]["frac"], 1.0)
        d.progress("setup", -9)
        self.assertEqual(d.rows[0]["frac"], 0.0)
        with self.assertRaises(KeyError):
            d.start("nope")

    def test_long_details_are_cut_to_the_width(self):
        d, _, _ = make(width=60)
        d.start("setup")
        d.progress("setup", 0.5, "x" * 500)
        self.assertTrue(all(len(l) == 60 for l in d.render()))
        d.metric("a", "y" * 500)
        self.assertEqual(len(d.render()[-1]), 60)

    def test_metrics_line(self):
        d, _, clock = make(ansi=False)
        d.metric("hits", 12)
        d.metric("up", "0.5MB")
        clock.tick(3.25)
        self.assertEqual(d.render()[-1].strip(), "hits 12  up 0.5MB  elapsed 3.2s")
        d.metric("hits", 13)                                                           # updating keeps the position
        self.assertTrue(d.render()[-1].startswith("hits 13  up 0.5MB"))

    def test_default_width_follows_the_terminal(self):
        for columns, want in ((60, 59), (300, 120), (20, 40)):
            with mock.patch("shutil.get_terminal_size", return_value=os.terminal_size((columns, 24))):
                self.assertEqual(f.Dashboard("t", STAGES, out=io.StringIO(), ansi=False).width, want, columns)


class LiveRedraw(unittest.TestCase):
    def test_redraws_in_place_with_a_constant_height(self):
        d, out, clock = make()
        d.start("setup"); clock.tick(0.2)
        d.progress("setup", 0.5, "working"); clock.tick(0.2)
        d.finish("setup", True, "ready"); clock.tick(0.2)
        d.start("inputs"); clock.tick(0.2)
        d.finish("inputs", True, "ok"); clock.tick(0.2)
        d.start("search"); clock.tick(0.2)
        d.finish("search", True, "found it")
        text = out.getvalue()
        height = len(STAGES) + 4
        self.assertNotRegex(text.split("\n")[0], r"\x1b\[\d+A")                      # the first frame is not preceded by a move
        ups = re.findall(r"\x1b\[(\d+)A", text)
        self.assertGreaterEqual(len(ups), 5)
        self.assertEqual(set(ups), {str(height)})                                      # always back up exactly one frame
        self.assertEqual(text.count("\x1b[K"), height * (len(ups) + 1))                # every line is cleared before it is drawn

    def test_the_screen_a_person_sees_is_the_last_frame(self):
        d, out, clock = make()
        d.start("setup"); clock.tick(0.3)
        d.finish("setup", True, "ready"); clock.tick(0.3)
        d.start("inputs"); clock.tick(0.3)
        d.progress("inputs", 0.75, "loading"); clock.tick(0.3)
        d.refresh(True)                                                                # the next redraw shows the new time
        screen = replay(out.getvalue())
        want = d.render()
        self.assertEqual([l for l in screen[:len(want)]], want)
        self.assertEqual(screen[len(want):], [""])                                     # nothing stray under the frame
        self.assertIn("done", screen[2])
        self.assertIn("running", screen[3])

    def test_redraws_are_throttled_but_stage_changes_are_not(self):
        d, out, clock = make()
        d.start("setup")
        drawn = out.getvalue().count("\x1b[K")
        for i in range(50):                                                            # a burst of updates within 50 ms
            clock.tick(0.0005)
            d.progress("setup", i / 50)
        self.assertEqual(out.getvalue().count("\x1b[K"), drawn)
        clock.tick(0.06)
        d.progress("setup", 0.9)
        self.assertGreater(out.getvalue().count("\x1b[K"), drawn)
        before = out.getvalue().count("\x1b[K")
        d.finish("setup", True, "done")                                                # forced regardless of timing
        self.assertGreater(out.getvalue().count("\x1b[K"), before)

    def test_close_draws_the_final_frame_once(self):
        d, out, _ = make()
        d.start("setup")
        before = out.getvalue()
        d.close()
        self.assertGreater(len(out.getvalue()), len(before))
        quiet, qout, _ = make(ansi=False)
        quiet.close()
        self.assertEqual(qout.getvalue(), "")

    def test_the_windows_console_gets_ansi_switched_on(self):
        with mock.patch.object(f.os, "name", "nt"), mock.patch.object(f.os, "system") as system:
            f.Dashboard("t", STAGES, out=FakeTTY(), ansi=True)
            system.assert_called_once_with("")
        with mock.patch.object(f.os, "name", "nt"), mock.patch.object(f.os, "system") as system:
            f.Dashboard("t", STAGES, out=io.StringIO(), ansi=False)
            system.assert_not_called()

    def test_ansi_defaults_to_whether_the_output_is_a_terminal(self):
        self.assertTrue(f.Dashboard("t", STAGES, out=FakeTTY()).ansi)
        self.assertFalse(f.Dashboard("t", STAGES, out=io.StringIO()).ansi)


class PlainMode(unittest.TestCase):
    def test_one_line_per_finished_stage_and_no_escape_codes(self):
        d, out, clock = make(ansi=False)
        d.start("setup"); d.progress("setup", 0.5); d.metric("hits", 3)
        self.assertEqual(out.getvalue(), "")                                           # nothing until a stage ends
        clock.tick(0.25)
        d.finish("setup", True, "ready")
        d.start("inputs"); clock.tick(1)
        d.finish("inputs", False, "bad file")
        d.skip("search", "an earlier stage failed")
        lines = out.getvalue().strip().split("\n")
        self.assertEqual(len(lines), 3)
        self.assertTrue(lines[0].startswith("[1/3] setup"))
        self.assertIn("done", lines[0])
        self.assertIn("0.25s", lines[0])
        self.assertIn("FAILED", lines[1])
        self.assertIn("skipped", lines[2])
        self.assertNotIn("\x1b", out.getvalue())


class FeedingTheDashboard(unittest.TestCase):
    def test_chunk_progress_callback(self):
        d, _, _ = make(ansi=False)
        d.start("search")
        cb = d.chunk_progress("search")
        cb(1, 4, 7, 0.1)
        self.assertEqual(d.rows[2]["frac"], 0.25)
        self.assertEqual(d.rows[2]["detail"], "chunk 1/4  hits 7")
        self.assertEqual(d.metrics["hits"], "7")
        cb(4, 4, 12, 0.4)
        self.assertEqual(d.rows[2]["frac"], 1.0)
        cb(0, 0, 0, 0.0)                                                               # nothing to do counts as complete
        self.assertEqual(d.rows[2]["frac"], 1.0)

    def test_a_real_search_drives_the_bar(self):
        rng = random.Random(91)
        genome = {"a": "".join(rng.choice("ACGT") for _ in range(400)), "b": "".join(rng.choice("ACGT") for _ in range(150))}
        d, out, _ = make()
        d.start("search")
        fracs = []
        cb = d.chunk_progress("search")
        hits = f.gpu_search(genome, [("g", "".join(rng.choice("ACGT") for _ in range(20)))], f.CONFIGS["C2"], 100, "emulated",
                            scheme="multi", progress=lambda *a: (cb(*a), fracs.append(d.rows[2]["frac"])))
        self.assertEqual(fracs, [1 / 6, 2 / 6, 3 / 6, 4 / 6, 5 / 6, 1.0])             # 4 chunks of "a" + 2 of "b"
        self.assertEqual(d.metrics["hits"], str(len(hits)))


if __name__ == "__main__":
    unittest.main(verbosity=2)