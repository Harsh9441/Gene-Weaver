"""Day 10 tests: the whole pipeline, connected (rules -> inputs -> QC -> plan -> baseline -> CUDA search -> verify -> report)."""
import contextlib, io, json, os, random, re, tempfile, unittest
from unittest import mock

import fasta as f

STAGE_NAMES = ["setup", "inputs", "inspect", "plan", "baseline", "search", "verify", "report"]


class FakeTTY(io.StringIO):
    def isatty(self):
        return True


def capture(fn, *a, **k):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        return fn(*a, **k), out.getvalue()


def read_json(path):
    with open(path) as fh:
        return json.load(fh)


def rnd(rng, n):
    return "".join(rng.choice("ACGT") for _ in range(n))


def small(td, **kw):
    base = dict(out_dir=os.path.join(td, "run"), runs=1, warmup=0, n_guides=6)
    base.update(kw)
    return f.pipeline_options(**base)


class Options(unittest.TestCase):
    def test_defaults_and_overrides(self):
        o = f.pipeline_options()
        self.assertEqual((o.config, o.scheme, o.backend, o.out_dir), ("C2", "multi", "auto", "pipeline_run"))
        self.assertEqual(f.pipeline_options(config="C6", runs=9).runs, 9)

    def test_unknown_options_are_refused(self):
        with self.assertRaises(ValueError) as cm:
            f.pipeline_options(colour="blue")
        self.assertIn("colour", str(cm.exception))

    def test_stage_list(self):
        self.assertEqual(list(f.PIPELINE_STAGES), STAGE_NAMES)


class WholeRun(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._td = tempfile.TemporaryDirectory()
        cls.dir = cls._td.name
        cls.opts = small(cls.dir)
        cls.result = f.run_pipeline(cls.opts)
        cls.results_dir = os.path.join(cls.opts.out_dir, "results")

    @classmethod
    def tearDownClass(cls):
        cls._td.cleanup()

    def test_every_stage_runs_in_order_and_passes(self):
        self.assertTrue(self.result["ok"])
        self.assertEqual([s["name"] for s in self.result["stages"]], STAGE_NAMES)
        self.assertTrue(all(s["status"] == "ok" for s in self.result["stages"]), self.result["stages"])
        self.assertTrue(all(s["seconds"] >= 0 and s["detail"] for s in self.result["stages"]))

    def test_files_on_disk(self):
        root = self.opts.out_dir
        for rel in ("BENCHMARK_RULES.md", "m.json", os.path.join("data", "genome.fa"), os.path.join("data", "guides.tsv"),
                    os.path.join("results", "pipeline_report.json"), os.path.join("results", "pipeline_report.md"),
                    os.path.join("results", "baseline_log.jsonl")):
            self.assertTrue(os.path.exists(os.path.join(root, rel)), rel)
        records = [n for n in os.listdir(self.results_dir) if n.startswith("baseline_G-mock_") and n.endswith(".json")]
        self.assertEqual(len(records), 2)                                              # the baseline and the CUDA search
        self.assertEqual(sorted(r.rsplit("_", 1)[1] for r in records), ["cuda-emulated-multi.json", "packed.json"])
        with open(os.path.join(self.results_dir, "baseline_log.jsonl")) as fh:
            self.assertEqual(len([l for l in fh if l.strip()]), 2)

    def test_report_json(self):
        rep = read_json(os.path.join(self.results_dir, "pipeline_report.json"))
        for key in ("title", "rules_version", "created", "ok", "backend", "scheme", "config", "plan", "genome", "guides",
                    "stages", "baseline", "search", "speedup", "speedup_note", "verification", "strands", "transfers",
                    "per_guide", "hardware"):
            self.assertIn(key, rep)
        self.assertTrue(rep["ok"])
        self.assertEqual([s["name"] for s in rep["stages"]], STAGE_NAMES)              # the saved report lists all 8 stages
        self.assertTrue(all(v["ok"] for v in rep["verification"]))
        self.assertEqual(rep["backend"], "emulated")
        self.assertIn("NOT a GPU speed", rep["speedup_note"])
        self.assertEqual(rep["guides"]["n"], 6)
        self.assertEqual(len(rep["per_guide"]), 6)
        self.assertEqual(rep["genome"]["sha256"], f.sha256(os.path.join(self.opts.out_dir, "data", "genome.fa")))
        self.assertEqual(rep["strands"]["forward"] + rep["strands"]["reverse"], rep["search"]["hits"])
        self.assertEqual(sum(p["total"] for p in rep["per_guide"]), rep["search"]["hits"])
        self.assertEqual(rep["search"]["hits"], rep["baseline"]["hits"])
        self.assertGreater(rep["transfers"]["h2d_bytes"], 0)

    def test_report_markdown(self):
        with open(os.path.join(self.results_dir, "pipeline_report.md")) as fh:
            md = fh.read()
        md.encode("ascii")
        for part in ("# GeneWeaver pipeline report", "ALL STAGES PASSED", "## Stages", "## Timing", "## Verification",
                     "## Host/device transfers", "## Hits per guide (both strands)", "NOT a GPU speed"):
            self.assertIn(part, md)
        self.assertEqual(len(re.findall(r"^\| \d \| ", md, re.M)), 8)                  # one table row per stage

    def test_the_pipeline_result_matches_an_independent_search(self):
        genome = f.read_genome(os.path.join(self.opts.out_dir, "data", "genome.fa"), "builtin")
        guides = f.read_guides_any(os.path.join(self.opts.out_dir, "data", "guides.tsv"), "builtin")
        self.assertEqual(self.result["state"]["hits"], f.oracle_search(genome, guides, f.CONFIGS["C2"]))

    def test_verification_covers_baseline_oracle_planted_sites_and_strands(self):
        names = [c for c, _ in self.result["state"]["checks"]]
        self.assertEqual(len(names), 4)
        self.assertTrue(any("pure-Python baseline" in n for n in names))
        self.assertTrue(any("CPU oracle" in n for n in names))
        self.assertTrue(any("planted sites found" in n for n in names))
        self.assertTrue(any("both strands" in n for n in names))

    def test_text_summary(self):
        text = f.format_pipeline(self.result)
        for part in ("pipeline COMPLETE - every stage passed", "backend     : emulated", "speedup", "[ok] search hits equal",
                     "per-guide results (both strands):", "report      :"):
            self.assertIn(part, text)


class Repeatability(unittest.TestCase):
    def test_same_seed_same_results_and_log_grows(self):
        with tempfile.TemporaryDirectory() as td:
            first = f.run_pipeline(small(td))
            second = f.run_pipeline(small(td))
            self.assertTrue(first["ok"] and second["ok"])
            r1 = read_json(os.path.join(td, "run", "results", "pipeline_report.json"))
            self.assertEqual(first["state"]["hits"], second["state"]["hits"])
            self.assertEqual(first["state"]["sha"], second["state"]["sha"])
            with open(os.path.join(td, "run", "results", "baseline_log.jsonl")) as fh:
                self.assertEqual(len([l for l in fh if l.strip()]), 4)                 # two runs x (baseline + search)
            self.assertTrue(r1["ok"])

    def test_an_edited_manifest_survives_a_rerun(self):
        with tempfile.TemporaryDirectory() as td:
            o = small(td)
            f.run_pipeline(o)
            with open(os.path.join(o.out_dir, "m.json"), "w") as fh:
                fh.write('{"mine": true}')
            f.run_pipeline(o)
            with open(os.path.join(o.out_dir, "m.json")) as fh:
                self.assertEqual(fh.read(), '{"mine": true}')


class Failures(unittest.TestCase):
    def test_a_wrong_search_result_fails_verification_and_skips_the_report(self):
        real = f.gpu_search

        def dropping_a_hit(*args, **kwargs):
            return real(*args, **kwargs)[:-1]
        with tempfile.TemporaryDirectory() as td:
            with mock.patch.object(f, "gpu_search", dropping_a_hit):
                result = f.run_pipeline(small(td))
            self.assertFalse(result["ok"])
            self.assertEqual([s["status"] for s in result["stages"]], ["ok"] * 6 + ["failed", "skipped"])
            self.assertIn("FAILED", result["stages"][6]["detail"])
            self.assertIn("Day 5 pure-Python baseline", result["stages"][6]["detail"])
            self.assertEqual(result["stages"][7]["detail"], "an earlier stage failed")
            self.assertFalse(os.path.exists(os.path.join(td, "run", "results", "pipeline_report.json")))
            text = f.format_pipeline(result)
            self.assertIn("pipeline FAILED", text)
            self.assertIn("verify: failed", text)
            self.assertIn("report: skipped", text)

    def test_a_missing_genome_fails_the_inputs_stage_and_skips_everything_after_it(self):
        with tempfile.TemporaryDirectory() as td:
            result = f.run_pipeline(small(td, genome=os.path.join(td, "missing.fa"), parser="builtin"))
            self.assertFalse(result["ok"])
            statuses = [s["status"] for s in result["stages"]]
            self.assertEqual(statuses, ["ok", "failed"] + ["skipped"] * 6)
            self.assertIn("missing.fa", result["stages"][1]["detail"])

    def test_a_search_error_is_reported_not_raised(self):
        with tempfile.TemporaryDirectory() as td:
            result = f.run_pipeline(small(td, threads=0))
            self.assertFalse(result["ok"])
            self.assertEqual(result["stages"][5]["status"], "failed")
            self.assertIn("threads per block", result["stages"][5]["detail"])

    def test_command_exit_codes(self):
        with tempfile.TemporaryDirectory() as td:
            out = os.path.join(td, "run")
            self.assertEqual(capture(f.main, ["pipeline", "--config", "C4", "--out-dir", out])[0], 2)
            self.assertFalse(os.path.exists(out))                                      # refused before anything was created
            with mock.patch.object(f, "HAVE_NUMBA", False):
                self.assertEqual(capture(f.main, ["pipeline", "--backend", "gpu", "--out-dir", out])[0], 2)
            real = f.gpu_search
            with mock.patch.object(f, "gpu_search", lambda *a, **k: real(*a, **k)[:-1]):
                rc, text = capture(f.main, ["pipeline", "--plain", "--runs", "1", "--warmup", "0", "--out-dir", out])
            self.assertEqual(rc, 1)
            self.assertIn("pipeline FAILED", text)


class Variations(unittest.TestCase):
    def test_all_three_schemes_work_end_to_end(self):
        ups = {}
        with tempfile.TemporaryDirectory() as td:
            for scheme in f.SCHEMES:
                result = f.run_pipeline(small(td, scheme=scheme, out_dir=os.path.join(td, scheme)))
                self.assertTrue(result["ok"], (scheme, result["stages"]))
                rep = result["state"]["report"]
                self.assertEqual(rep["search"]["implementation"], "cuda-emulated" + ("" if scheme == "matrix" else "-" + scheme))
                ups[scheme] = rep["transfers"]["d2h_bytes"]
        self.assertGreater(ups["matrix"], 10 * ups["compact"])

    def test_chunking_is_visible_and_harmless(self):
        with tempfile.TemporaryDirectory() as td:
            result = f.run_pipeline(small(td, chunk_size=15_000))
            self.assertTrue(result["ok"])
            self.assertIn("4 chunks of 15,000 bp", result["stages"][3]["detail"])      # two 20 kb chromosomes
            self.assertEqual(result["state"]["plan"]["chunks"], 4)

    def test_the_oracle_is_skipped_when_asked_or_when_the_genome_is_large(self):
        with tempfile.TemporaryDirectory() as td:
            names = lambda r: [c for c, _ in r["state"]["checks"]]
            self.assertFalse(any("oracle" in n for n in names(f.run_pipeline(small(td, no_oracle=True)))))
            with mock.patch.object(f, "ORACLE_LIMIT_BP", 1000):
                self.assertFalse(any("oracle" in n for n in names(f.run_pipeline(small(td)))))
            self.assertTrue(any("oracle" in n for n in names(f.run_pipeline(small(td)))))

    def test_your_own_fasta_and_guide_files(self):
        rng = random.Random(92)
        with tempfile.TemporaryDirectory() as td:
            guides = [(f"mine{i}", rnd(rng, 20)) for i in range(5)]
            genome = {"chrA": rnd(rng, 700), "chrB_random": rnd(rng, 300)}
            for i, (gid, g) in enumerate(guides):                                      # one near-match per guide, 120 bp apart
                seq, p = genome["chrA"], 40 + 120 * i
                site = g[:3] + ("A" if g[3] != "A" else "C") + g[4:] + "TGG"
                genome["chrA"] = seq[:p] + site + seq[p + len(site):]
            fa, gt = os.path.join(td, "genome.fa"), os.path.join(td, "guides.tsv")
            f.write_fasta(genome, fa)
            f.write_guides(guides, gt)
            result = f.run_pipeline(small(td, genome=fa, guides=gt, parser="builtin"))
            self.assertTrue(result["ok"], result["stages"])
            self.assertEqual(result["state"]["report"]["genome"]["id"], "genome.fa")
            self.assertIn("WARNING 1 non-primary contigs", result["stages"][2]["detail"])   # chrB_random
            self.assertFalse(any("planted" in c for c, _ in result["state"]["checks"]))      # no truth table for your files
            self.assertGreaterEqual(result["state"]["report"]["search"]["hits"], 5)
            self.assertEqual(result["state"]["genome_path"], fa)
            self.assertFalse(os.path.exists(os.path.join(td, "run", "data", "genome.fa")))   # your file is used where it is


class TextInterface(unittest.TestCase):
    def test_plain_command_output(self):
        with tempfile.TemporaryDirectory() as td:
            rc, text = capture(f.main, ["pipeline", "--plain", "--runs", "1", "--warmup", "0", "--n-guides", "5",
                                        "--out-dir", os.path.join(td, "run")])
            self.assertEqual(rc, 0, text)
            self.assertNotIn("\x1b", text)
            for i, name in enumerate(STAGE_NAMES, 1):
                self.assertRegex(text, rf"\[{i}/8\] {name}\s+done")
            self.assertIn("pipeline COMPLETE", text)

    def test_live_dashboard_on_a_terminal(self):
        with tempfile.TemporaryDirectory() as td:
            tty = FakeTTY()
            with contextlib.redirect_stdout(tty):
                rc = f.main(["pipeline", "--runs", "1", "--warmup", "0", "--n-guides", "5", "--out-dir", os.path.join(td, "run")])
            text = tty.getvalue()
            self.assertEqual(rc, 0)
            self.assertIn("\x1b[K", text)
            self.assertRegex(text, r"\x1b\[\d+A")                                      # redrawn in place
            self.assertIn("stage details:", text)                                     # full details printed under the dashboard
            self.assertIn("pipeline COMPLETE", text)
            from test_day9 import replay
            screen = replay(text)
            rows = [l for l in screen if re.match(r"\s+\d\s+\w+\s+\[", l)]
            self.assertEqual(len(rows), 8)
            self.assertTrue(all("done" in l and "100%" in l for l in rows), rows)

    def test_dashboard_inside_run_pipeline(self):
        with tempfile.TemporaryDirectory() as td:
            out = FakeTTY()
            dash = f.Dashboard("t", f.PIPELINE_STAGES, out=out, ansi=True, width=100)
            result = f.run_pipeline(small(td), dash)
            dash.close()
            self.assertTrue(result["ok"])
            self.assertTrue(all(r["status"] == "ok" and r["frac"] == 1.0 for r in dash.rows))
            self.assertTrue({"hits", "baseline", "up", "down"} <= set(dash.metrics))
            self.assertEqual(dash.metrics["hits"], str(result["state"]["search"]["total_hits"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)