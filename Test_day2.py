"""Day 2 tests: CPU oracle (verification) and the mock human genome generator in fasta.py.

Run from the folder that contains fasta.py:
    python -m unittest test_day2 -v
"""
import contextlib, io, os, random, tempfile, unittest

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


class Day2Oracle(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = random.Random(f.SEED)
        rand = lambda n: "".join(rng.choice("ACGT") for _ in range(n))
        cls.rand = staticmethod(rand)
        cls.spacer = s = "GACGTCTAGCATCGATCGAA"
        mm1 = s[:5] + ("A" if s[5] != "A" else "C") + s[6:]
        cls.genome = {"c1": rand(300) + mm1 + "AGG" + rand(300) + f.revcomp(s + "TGG") + rand(200),
                      "c2": rand(200) + s[:10] + "T" + s[10:] + "CGG" + rand(200),
                      "c3": rand(200) + s[:10] + s[11:] + "CGG" + rand(200)}
        cls.guides = [("g0", s)]

    def test_fast_path_equals_alignment_path(self):
        c = dict(f.CONFIGS["C2"], mm=2)
        fast = f.oracle_search(self.genome, self.guides, c)
        dp = f.oracle_search(self.genome, self.guides, c, force_dp=True)
        self.assertEqual({h.key() for h in fast}, {h.key() for h in dp})

    def test_mismatch_and_reverse_strand_sites(self):
        hits = f.oracle_search(self.genome, self.guides, dict(f.CONFIGS["C2"], mm=2))
        self.assertTrue({("+", 1), ("-", 0)} <= {(h.strand, h.mm) for h in hits})

    def test_dna_and_rna_bulges(self):
        hits = f.oracle_search(self.genome, self.guides, dict(f.CONFIGS["C4"], mm=2))
        self.assertTrue(any(h.chrom == "c2" and h.db == 1 for h in hits))
        self.assertTrue(any(h.chrom == "c3" and h.rb == 1 for h in hits))

    def test_bulge_free_search_misses_bulge_sites(self):
        hits = f.oracle_search(self.genome, self.guides, dict(f.CONFIGS["C2"], mm=2))
        self.assertFalse(any(h.chrom in ("c2", "c3") for h in hits))

    def test_genome_n_never_matches(self):
        s = self.spacer
        g = {"c": "A" * 10 + s[:8] + "N" + s[9:] + "TGG" + "A" * 10}
        self.assertFalse(f.oracle_search(g, self.guides, dict(pam="NGG", mm=0, db=0, rb=0)))
        self.assertTrue(f.oracle_search(g, self.guides, dict(pam="NGG", mm=1, db=0, rb=0)))

    def test_sites_at_chromosome_ends(self):
        s = self.spacer
        g = {"c": s + "TGG" + "A" * 30, "d": "A" * 30 + s + "TGG"}
        hits = f.oracle_search(g, self.guides, dict(f.CONFIGS["C2"], mm=2))
        self.assertEqual({h.chrom for h in hits}, {"c", "d"})

    def test_make_guides_seeded_and_valid(self):
        src = {"c": self.rand(5000)}
        g1, g2 = f.make_guides(src, 5), f.make_guides(src, 5)
        self.assertEqual(g1, g2)
        self.assertEqual(len(g1), 5)
        exact = dict(pam="NGG", mm=0, db=0, rb=0)
        for g in g1:
            self.assertTrue(any(h.mm == 0 for h in f.oracle_search(src, [g], exact)))


class Day2MockGenome(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        kw = dict(scale=0.0002, chroms=["chr21", "chr22"], n_guides=3, copies=4, max_mm=3, bulge_copies=2)
        cls.a, cls.b = f.generate_mock(**kw), f.generate_mock(**kw)

    def test_deterministic_for_same_seed(self):
        self.assertEqual(self.a["genome"], self.b["genome"])
        self.assertEqual(self.a["truth"], self.b["truth"])

    def test_human_like_composition(self):
        st = self.a["stats"]
        self.assertTrue(36 <= st["gc_percent"] <= 46, st)
        self.assertTrue(0.2 <= st["masked_fraction"] <= 0.6, st)
        self.assertLess(st["cpg_obs_exp"], 0.5)
        self.assertGreater(st["n_fraction"], 0)

    def test_planted_sites_are_exactly_where_the_truth_table_says(self):
        r = f.verify_sites(self.a["genome"], self.a["guides"], self.a["truth"], 3)
        self.assertEqual(r["found"], r["planted"])

    def test_oracle_search_finds_every_planted_site(self):
        tiny = f.generate_mock(scale=0.0001, chroms=["chr21", "chr22"], n_guides=2, copies=3, max_mm=3, bulge_copies=2)
        r = f.verify_planted(tiny["genome"], tiny["guides"], tiny["truth"], f.oracle_search, dict(f.CONFIGS["C4"], mm=3))
        self.assertEqual(r["found"], r["planted"])

    def test_all_24_chromosomes_by_default(self):
        m = f.generate_mock(scale=0.0001, n_guides=1, copies=1, bulge_copies=0)
        self.assertEqual(len(m["genome"]), 24)

    def test_a_corrupted_truth_table_is_caught(self):
        bad = list(self.a["truth"])
        g, c, s, st, mm, db, rb, t = bad[0]
        bad[0] = (g, c, s + 3, st, mm, db, rb, t)
        r = f.verify_sites(self.a["genome"], self.a["guides"], bad, 3)
        self.assertLess(r["found"], r["planted"])


class Day2CommandLine(unittest.TestCase):
    def test_mockgenome_command_with_verify(self):
        with tempfile.TemporaryDirectory() as td:
            rc = quiet(f.main, ["mockgenome", "--scale", "0.0001", "--chroms", "chr21,chr22",
                                "--n-guides", "2", "--out-dir", td, "--verify"])
            self.assertEqual(rc, 0)
            for name in ("genome.fa", "guides.tsv", "planted_truth.tsv", "m.json"):
                self.assertTrue(os.path.exists(os.path.join(td, name)), name)

    def test_selftest_command_passes(self):
        self.assertEqual(quiet(f.main, ["selftest"]), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)