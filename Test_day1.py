"""Day 1 tests: setup, benchmark rules, and the core types in fasta.py.

Run from the folder that contains fasta.py:
    python -m unittest test_day1 -v
"""
import contextlib, io, json, os, tempfile, unittest

import fasta as f


def _read(path):
    with open(path) as fh:
        return fh.read()


def _write(path, text):
    with open(path, "w") as fh:
        fh.write(text)


def quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)


class Day1Rules(unittest.TestCase):
    def setUp(self):
        self.md = f.rules_markdown()

    def test_version_and_sections(self):
        self.assertIn(f"v{f.RULES_VERSION}", self.md.split("\n")[0])
        for i in range(1, 13):
            self.assertIn(f"## {i}. ", self.md)

    def test_every_configuration_is_listed_with_its_budget(self):
        for k, c in f.CONFIGS.items():
            self.assertIn(f"| {k} | {c['pam']} | up to {c['mm']} |", self.md)

    def test_protocol_numbers(self):
        self.assertIn("at least 5 timed runs", self.md)
        self.assertIn("20%", self.md)
        self.assertIn("median and interquartile range", self.md)

    def test_genomes_guide_sets_and_targets_listed(self):
        for k in list(f.GENOMES) + list(f.GUIDE_SETS):
            self.assertIn(f"| {k} |", self.md)
        for target, _ in f.ACCEPTANCE:
            self.assertIn(target, self.md)

    def test_bulge_convention_is_fixed(self):
        self.assertIn("not allowed at either end", self.md)

    def test_setup_creates_layout_rules_and_manifest(self):
        with tempfile.TemporaryDirectory() as td:
            root = os.path.join(td, "proj")
            made = f.setup_project(root)
            self.assertEqual(len(made), 7)
            self.assertTrue(all(os.path.exists(p) for p in made))
            self.assertEqual(_read(os.path.join(root, "BENCHMARK_RULES.md")), self.md)
            mf = json.loads(_read(os.path.join(root, "m.json")))
            for key in ("genomes", "workdir", "tool_cmd", "sizes"):
                self.assertIn(key, mf)
            self.assertIn("{genome}", mf["tool_cmd"])


class Day1Core(unittest.TestCase):
    def test_six_configurations(self):
        self.assertEqual(sorted(f.CONFIGS), ["C1", "C2", "C3", "C4", "C5", "C6"])

    def test_revcomp(self):
        self.assertEqual(f.revcomp("ACGTN"), "NACGT")

    def test_pam_matching(self):
        self.assertTrue(f.pam_matches("TGG", "NGG"))
        self.assertFalse(f.pam_matches("NGG", "NGG"))      # genome N never matches
        self.assertTrue(f.pam_matches("AAG", "NRG"))
        self.assertFalse(f.pam_matches("ATG", "NGG"))

    def test_fasta_reading(self):
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "t.fa")
            _write(p, ">x some description\nacgtnRY\nAC\n")
            self.assertEqual(f.read_fasta(p), {"x": "ACGTNNNAC"})

    def test_hit_and_guide_round_trip(self):
        h = f.Hit("g", "c", 5, "+", 1, 0, 0, "ACGTAGG")
        with tempfile.TemporaryDirectory() as td:
            hp, gp = os.path.join(td, "h.tsv"), os.path.join(td, "g.tsv")
            f.write_hits(f.normalize([h, h]), hp)
            self.assertEqual(f.read_hits(hp), [h])
            f.write_guides([("g0", "ACGT")], gp)
            self.assertEqual(f.read_guides(gp), [("g0", "ACGT")])


class Day1CommandLine(unittest.TestCase):
    def test_rules_command_writes_file(self):
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "r.md")
            self.assertEqual(quiet(f.main, ["rules", "--out", p]), 0)
            self.assertEqual(_read(p), f.rules_markdown())

    def test_setup_command_creates_project(self):
        with tempfile.TemporaryDirectory() as td:
            root = os.path.join(td, "proj")
            self.assertEqual(quiet(f.main, ["setup", "--dir", root]), 0)
            for name in ("BENCHMARK_RULES.md", "m.json", "data", "results", "mock", "logs"):
                self.assertTrue(os.path.exists(os.path.join(root, name)), name)


if __name__ == "__main__":
    unittest.main(verbosity=2)