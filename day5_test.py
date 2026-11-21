"""Day 5 tests: the pure-Python packed baseline and its recorded timing in fasta.py.

Run from the folder that contains fasta.py:
    python -m unittest test_day5 -v
"""
import contextlib, io, json, os, random, tempfile, unittest
from unittest.mock import patch

import fasta as f

SPACER = "GACGTCTAGCATCGATCGAA"


def quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)


def rnd(rng, n):
    return "".join(rng.choice("ACGT") for _ in range(n))


def mutate(rng, seq, k):
    seq = list(seq)
    for p in rng.sample(range(len(seq)), k):
        seq[p] = rng.choice([b for b in "ACGT" if b != seq[p]])
    return "".join(seq)


def random_genome(rng, guides, pams=("TGG", "AGG", "CGG", "GGG", "TAG", "AAG"), with_n=False):
    genome = {}
    for name in ("a", "b"):
        seq = rnd(rng, rng.randint(150, 500))
        for _ in range(rng.randint(2, 5)):                      # plant near-matches, either strand
            gid, g = rng.choice(guides)
            site = mutate(rng, g, rng.randint(0, 4)) + rng.choice(pams)
            site = site if rng.random() < 0.5 else f.revcomp(site)
            p = rng.randrange(0, len(seq) - len(site))
            seq = seq[:p] + site + seq[p + len(site):]
        if with_n:
            for _ in range(3):
                p = rng.randrange(0, len(seq) - 4)
                seq = seq[:p] + "N" * rng.randint(1, 3) + seq[p + 3:]
        genome[name] = seq
    return genome


class PackedMatchesOracle(unittest.TestCase):
    """The fast baseline must give exactly the same hits as the simple Day 2 reference."""
    def test_random_genomes_all_mismatch_configs(self):
        rng = random.Random(2026)
        configs = [f.CONFIGS["C1"], f.CONFIGS["C2"], f.CONFIGS["C3"], f.CONFIGS["C6"]]
        total = 0
        for trial in range(30):
            guides = [(f"g{i}", rnd(rng, 20)) for i in range(rng.randint(1, 4))]
            genome = random_genome(rng, guides)
            for cfg in configs:
                want = f.oracle_search(genome, guides, cfg)
                got = f.packed_search(genome, guides, cfg)
                self.assertEqual(got, want, (trial, cfg))
                total += len(want)
        self.assertGreater(total, 100)                           # not a vacuous comparison

    def test_genome_n_bases_in_spacer_and_pam(self):
        rng = random.Random(11)
        total = 0
        for trial in range(25):
            guides = [("g0", rnd(rng, 20))]
            genome = random_genome(rng, guides, with_n=True)
            for cfg in (f.CONFIGS["C2"], dict(f.CONFIGS["C2"], mm=1)):
                want = f.oracle_search(genome, guides, cfg)
                self.assertEqual(f.packed_search(genome, guides, cfg), want, trial)
                total += len(want)
        self.assertGreater(total, 10)

    def test_iupac_guide_takes_the_slow_path_and_still_matches(self):
        rng = random.Random(3)
        guide = SPACER[:10] + "R" + SPACER[11:]
        genome = {"c": rnd(rng, 100) + SPACER + "TGG" + rnd(rng, 100)}
        cfg = dict(f.CONFIGS["C2"], mm=1)
        want = f.oracle_search(genome, [("g", guide)], cfg)
        self.assertTrue(want)
        self.assertEqual(f.packed_search(genome, [("g", guide)], cfg), want)

    def test_guides_of_different_lengths(self):
        rng = random.Random(8)
        guides = [("a", rnd(rng, 20)), ("b", rnd(rng, 18)), ("c", rnd(rng, 23))]
        genome = {"c": rnd(rng, 200) + guides[1][1] + "TGG" + rnd(rng, 100) + guides[2][1] + "AGG" + rnd(rng, 100)}
        cfg = dict(f.CONFIGS["C2"], mm=2)
        got = f.packed_search(genome, guides, cfg)
        self.assertEqual(got, f.oracle_search(genome, guides, cfg))
        self.assertEqual({h.guide_id for h in got if h.mm == 0}, {"b", "c"})

    def test_reported_targets_are_really_in_the_genome(self):
        rng = random.Random(21)
        guides = [("g0", rnd(rng, 20))]
        genome = random_genome(rng, guides)
        hits = f.packed_search(genome, guides, f.CONFIGS["C2"])
        self.assertTrue(hits)
        for h in hits:
            win = genome[h.chrom][h.start:h.start + len(h.target)]
            self.assertEqual(f.revcomp(win) if h.strand == "-" else win, h.target)

    def test_rejects_bulge_configs(self):
        for name in ("C4", "C5"):
            with self.assertRaises(ValueError):
                f.packed_search({"c": "ACGT" * 20}, [("g", SPACER)], f.CONFIGS[name])

    def test_is_much_faster_than_the_naive_oracle(self):
        rng = random.Random(4)
        guides = [(f"g{i}", rnd(rng, 20)) for i in range(5)]
        genome = {"c": rnd(rng, 30_000)}
        cfg = f.CONFIGS["C2"]
        fast, _ = f.time_runs(lambda: f.packed_search(genome, guides, cfg), runs=2, warmup=0)
        slow, _ = f.time_runs(lambda: f.oracle_search(genome, guides, cfg), runs=2, warmup=0)
        self.assertLess(min(fast) * 3, min(slow))


class TimingHelpers(unittest.TestCase):
    def test_summarize_times(self):
        st = f.summarize_times([1.0, 1.1, 0.9, 1.0, 1.0])
        self.assertEqual(st["median"], 1.0)
        self.assertFalse(st["unstable"])
        self.assertTrue(f.summarize_times([1, 1, 1, 1, 2])["unstable"])     # 100% from the median
        self.assertFalse(f.summarize_times([1.0, 1.19, 0.81])["unstable"])  # inside the 20% rule
        self.assertTrue(f.summarize_times([1.0, 1.0, 1.21])["unstable"])
        self.assertEqual(f.summarize_times([2.5])["median"], 2.5)           # a single run works

    def test_throughput_formula(self):
        self.assertEqual(f.gps(10, 1_000_000, 2.0), 1e7)                   # G x 2L / t
        self.assertEqual(f.gps(1, 50, 1.0), 100)

    def test_time_runs_does_warmup_then_timed_runs(self):
        calls = []
        times, result = f.time_runs(lambda: calls.append(1) or "r", runs=5, warmup=1)
        self.assertEqual((len(calls), len(times), result), (6, 5, "r"))
        self.assertTrue(all(t >= 0 for t in times))
        for bad in ({"runs": 0}, {"warmup": -1}):
            with self.assertRaises(ValueError):
                f.time_runs(lambda: None, **bad)

    def test_peak_memory_is_measured_untimed(self):
        self.assertGreater(f.peak_memory_mb(lambda: [0] * 200_000), 1.0)

    def test_hardware_info(self):
        info = f.hardware_info()
        for key in ("os", "cpu", "cores", "python"):
            self.assertIn(key, info)


class BaselineRecord(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        mock = f.generate_mock(scale=0.0001, chroms=["chr21", "chr22"], n_guides=3, copies=4, max_mm=4, bulge_copies=2)
        cls.genome = {c: ba.decode() for c, ba in mock["genome"].items()}
        cls.mock = mock

    def run_it(self, config="C2", **kw):
        return f.run_baseline(self.genome, self.mock["guides"], config, runs=3, warmup=0, truth=self.mock["truth"], **kw)

    def test_record_has_the_rules_fields_and_is_json(self):
        rec = self.run_it()
        for key in ("tool", "implementation", "rules_version", "genome", "genome_sha256", "genome_bp", "guide_set",
                    "n_guides", "config", "runs_s", "median_search_s", "iqr_s", "unstable", "gps", "total_hits",
                    "peak_python_mem_mb", "correctness", "hardware", "created"):
            self.assertIn(key, rec)
        json.dumps(rec)                                                    # must be serialisable
        self.assertEqual(len(rec["runs_s"]), 3)
        self.assertEqual(rec["genome_bp"], sum(len(s) for s in self.genome.values()))
        self.assertEqual(len(rec["genome_sha256"]), 64)
        self.assertAlmostEqual(rec["gps"], f.gps(3, rec["genome_bp"], rec["median_search_s"]))

    def test_every_planted_site_within_budget_is_found(self):
        rec = self.run_it("C2")
        chk = rec["correctness"]
        self.assertGreater(chk["planted_total"], 0)
        self.assertEqual(chk["planted_found"], chk["planted_total"])

    def test_smaller_budget_counts_only_eligible_planted_sites(self):
        c1, c2 = self.run_it("C1")["correctness"], self.run_it("C2")["correctness"]
        self.assertLessEqual(c1["planted_total"], c2["planted_total"])
        self.assertEqual(c1["planted_found"], c1["planted_total"])

    def test_oracle_implementation_is_recorded_separately(self):
        rec = self.run_it("C2", impl="oracle")
        self.assertEqual(rec["implementation"], "oracle")
        self.assertEqual(rec["correctness"]["planted_found"], rec["correctness"]["planted_total"])

    def test_digest_changes_when_the_genome_changes(self):
        other = dict(self.genome)
        k = next(iter(other))
        other[k] = other[k][:-1] + ("A" if other[k][-1] != "A" else "C")
        self.assertNotEqual(f.genome_digest(self.genome), f.genome_digest(other))
        self.assertEqual(f.genome_digest(self.genome), f.genome_digest(dict(self.genome)))

    def test_save_writes_json_and_appends_a_log_line(self):
        rec = self.run_it()
        with tempfile.TemporaryDirectory() as td:
            path = f.save_baseline(rec, os.path.join(td, "results"))
            with open(path) as fh:
                self.assertEqual(json.load(fh)["median_search_s"], rec["median_search_s"])
            f.save_baseline(rec, os.path.join(td, "results"))
            with open(os.path.join(td, "results", "baseline_log.jsonl")) as fh:
                lines = [json.loads(l) for l in fh if l.strip()]
            self.assertEqual(len(lines), 2)
            self.assertEqual(lines[0]["config"], "C2")

    def test_report_text(self):
        rec = self.run_it()
        text = f.format_baseline(rec, saved="x.json")
        for part in ("pure-Python baseline", "median", "throughput", "planted sites found", "saved       : x.json"):
            self.assertIn(part, text)
        rec["unstable"] = True
        self.assertIn("UNSTABLE", f.format_baseline(rec))


class CommandLine(unittest.TestCase):
    def test_debug_flag_works_before_or_after_baseline_command(self):
        with tempfile.TemporaryDirectory() as td:
            common = ["--scale", "0.0001", "--chroms", "chr21", "--n-guides", "1",
                      "--runs", "1", "--warmup", "0", "--out-dir", td]
            for args in (["--debug", "baseline"] + common, ["baseline", "-d"] + common):
                diagnostics = io.StringIO()
                with contextlib.redirect_stderr(diagnostics):
                    self.assertEqual(quiet(f.main, args), 0)
                self.assertIn("[DEBUG] enabled by command-line flag; command=baseline", diagnostics.getvalue())
                self.assertIn("[DEBUG] search complete:", diagnostics.getvalue())

    def test_debug_environment_variable_enables_selftest_diagnostics(self):
        diagnostics = io.StringIO()
        with patch.dict(os.environ, {"GENEWEAVER_DEBUG": "1"}):
            with contextlib.redirect_stderr(diagnostics):
                self.assertEqual(quiet(f.main, ["selftest"]), 0)
        self.assertIn("[DEBUG] enabled by GENEWEAVER_DEBUG; command=selftest", diagnostics.getvalue())
        self.assertIn("[DEBUG] built-in self-test exit code=0", diagnostics.getvalue())

    def test_records_of_different_genomes_never_overwrite_each_other(self):
        with tempfile.TemporaryDirectory() as td:
            for scale in ("0.0005", "0.001"):                                   # 23 kb vs 47 kb chromosomes, same names otherwise
                self.assertEqual(quiet(f.main, ["baseline", "--scale", scale, "--chroms", "chr21,chr22", "--n-guides", "2",
                                                "--runs", "1", "--warmup", "0", "--out-dir", td]), 0)
            files = [n for n in os.listdir(td) if n.endswith(".json")]
            self.assertEqual(len(files), 2)
            with open(os.path.join(td, "baseline_log.jsonl")) as fh:
                self.assertEqual(len([l for l in fh if l.strip()]), 2)

    def test_baseline_on_a_generated_mock_genome(self):
        with tempfile.TemporaryDirectory() as td:
            args = ["baseline", "--scale", "0.0001", "--chroms", "chr21,chr22", "--n-guides", "3", "--runs", "2",
                    "--warmup", "0", "--out-dir", td]
            self.assertEqual(quiet(f.main, args), 0)
            self.assertEqual(quiet(f.main, args), 0)
            files = [n for n in os.listdir(td) if n.startswith("baseline_G-mock_") and n.endswith(".json")]
            self.assertEqual(len(files), 1)                                     # same genome -> same record, not two
            with open(os.path.join(td, files[0])) as fh:
                rec = json.load(fh)
            self.assertIn(rec["genome_sha256"][:8], files[0])
            self.assertEqual(rec["correctness"]["planted_found"], rec["correctness"]["planted_total"])
            with open(os.path.join(td, "baseline_log.jsonl")) as fh:
                self.assertEqual(len([l for l in fh if l.strip()]), 2)           # timings accumulate

    def test_baseline_on_a_fasta_file(self):
        rng = random.Random(5)
        with tempfile.TemporaryDirectory() as td:
            fa = os.path.join(td, "g.fa")
            f.write_fasta({"c1": rnd(rng, 3000), "c2": rnd(rng, 2000)}, fa)
            rc = quiet(f.main, ["baseline", "--genome", fa, "--n-guides", "2", "--runs", "2", "--warmup", "0",
                                "--parser", "builtin", "--out-dir", td])
            self.assertEqual(rc, 0)
            files = [n for n in os.listdir(td) if n.startswith("baseline_g.fa_") and n.endswith("_random-2_C2_packed.json")]
            self.assertEqual(len(files), 1)
            with open(os.path.join(td, files[0])) as fh:
                rec = json.load(fh)
            self.assertEqual(rec["genome_bp"], 5000)
            self.assertIsNotNone(rec["load_s"])
            self.assertIn("not checked", rec["correctness"]["note"])

    def test_errors_are_clean(self):
        with tempfile.TemporaryDirectory() as td:
            base = ["baseline", "--out-dir", td]
            self.assertEqual(quiet(f.main, base + ["--config", "C4"]), 2)                          # bulges need the oracle
            self.assertEqual(quiet(f.main, base + ["--impl", "oracle"]), 2)                         # big genome, no --force
            self.assertEqual(quiet(f.main, base + ["--scale", "0.0001", "--chroms", "chr21", "--runs", "0"]), 2)
            self.assertEqual(quiet(f.main, base + ["--genome", os.path.join(td, "missing.fa"), "--parser", "builtin"]), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)