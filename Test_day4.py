"""Day 4 tests: chunking with border overlap in fasta.py.

Run from the folder that contains fasta.py:
    python -m unittest test_day4 -v
"""
import contextlib, io, os, random, tempfile, unittest

import fasta as f

SPACER = "GACGTCTAGCATCGATCGAA"
SITE = SPACER + "TGG"                       # 23 nt window (spacer + PAM)
C2 = dict(f.CONFIGS["C2"], mm=2)
C4 = dict(f.CONFIGS["C4"], mm=2)


def quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)


def rnd(rng, n):
    return "".join(rng.choice("ACGT") for _ in range(n))


class RequiredOverlap(unittest.TestCase):
    def test_formula(self):
        self.assertEqual(f.required_overlap(f.CONFIGS["C2"], 20), 22)    # 20 + 0 + 3 - 1
        self.assertEqual(f.required_overlap(f.CONFIGS["C4"], 20), 23)    # one DNA bulge
        self.assertEqual(f.required_overlap(f.CONFIGS["C5"], 20), 24)    # two DNA bulges
        self.assertEqual(f.required_overlap(f.CONFIGS["C2"], 23), 25)    # longer guide
        self.assertEqual(f.required_overlap(f.CONFIGS["C6"], 20), 22)    # NRG is also 3 long


class DeterministicOrder(unittest.TestCase):
    def test_hits_sharing_a_start_are_ordered_deterministically(self):
        hits = [f.Hit("g", "c", 149, "-", 1, 0, 1, "ACGTCTAGCATCGATCGAATGG"),
                f.Hit("g", "c", 149, "-", 0, 0, 0, "GACGTCTAGCATCGATCGAATGG"),
                f.Hit("g", "c", 149, "-", 1, 1, 0, "GACGTCTAGCATTCGATCGAATGG")]
        want = f.normalize(hits)
        for _ in range(20):
            random.shuffle(hits)
            self.assertEqual(f.normalize(hits), want)
        self.assertEqual([h.mm for h in want], [0, 1, 1])


class PlanChunks(unittest.TestCase):
    def test_owned_ranges_tile_exactly_once(self):
        for length in (0, 1, 99, 100, 101, 1000):
            for size in (1, 7, 100, 5000):
                plan = f.plan_chunks(length, size, 22)
                covered = [i for o_s, o_e, _, _ in plan for i in range(o_s, o_e)]
                self.assertEqual(covered, list(range(length)), (length, size))

    def test_halo_on_both_sides_clipped_at_ends(self):
        plan = f.plan_chunks(1000, 300, 22)
        self.assertEqual([p[2:] for p in plan], [(0, 322), (278, 622), (578, 922), (878, 1000)])

    def test_zero_overlap_and_oversize_chunk(self):
        self.assertEqual(f.plan_chunks(10, 100, 5), [(0, 10, 0, 10)])
        self.assertEqual(f.plan_chunks(10, 5, 0), [(0, 5, 0, 5), (5, 10, 5, 10)])

    def test_invalid_arguments(self):
        for args in ((10, 0, 5), (10, 5, -1), (-1, 5, 5)):
            with self.assertRaises(ValueError):
                f.plan_chunks(*args)

    def test_stats_overhead(self):
        st = f.chunk_stats([1000], 100, 20)
        self.assertEqual((st["chunks"], st["owned_bp"]), (10, 1000))
        self.assertEqual(st["read_bp"], 1000 + 2 * 20 * 10 - 2 * 20)       # halo clipped at both chromosome ends
        self.assertAlmostEqual(st["overhead"], 0.36)


class BorderSites(unittest.TestCase):
    """Sites planted so they straddle chunk borders (chunk size 50: borders at 50, 100, 150, ...)."""
    @classmethod
    def setUpClass(cls):
        rng = random.Random(f.SEED)
        cls.genome = {"c": rnd(rng, 49) + SITE + rnd(rng, 77) + f.revcomp(SITE) + rnd(rng, 100)
                      + SPACER[:10] + "T" + SPACER[10:] + "CGG" + rnd(rng, 60)}
        cls.guides = [("g0", SPACER)]
        cls.full2 = f.oracle_search(cls.genome, cls.guides, C2)
        cls.full4 = f.oracle_search(cls.genome, cls.guides, C4)

    def test_the_unchunked_search_sees_the_border_sites(self):
        self.assertTrue({(h.strand, h.start) for h in self.full2} >= {("+", 49), ("-", 149)})
        self.assertTrue(any(h.db == 1 for h in self.full4))

    def test_chunked_equals_unchunked_mismatch(self):
        for size in (7, 23, 50, 97, 150, 1000):
            self.assertEqual(f.chunked_search(self.genome, self.guides, C2, size), self.full2, size)

    def test_chunked_equals_unchunked_with_bulges(self):
        for size in (50, 97, 150):
            self.assertEqual(f.chunked_search(self.genome, self.guides, C4, size), self.full4, size)

    def test_no_duplicates_across_borders(self):
        hits = f.chunked_search(self.genome, self.guides, C2, 50)
        self.assertEqual(len(hits), len({(h.key()) for h in hits}))

    def test_exact_overlap_is_enough_and_one_less_is_not(self):
        need = f.required_overlap(C2)
        ok = f.chunked_search(self.genome, self.guides, C2, 50, overlap=need)
        short = f.chunked_search(self.genome, self.guides, C2, 50, overlap=need - 1)
        self.assertEqual(ok, self.full2)
        self.assertTrue({h.key() for h in short} < {h.key() for h in self.full2})

    def test_no_overlap_loses_sites(self):
        none = f.chunked_search(self.genome, self.guides, C2, 50, overlap=0)
        self.assertLess(len(none), len(self.full2))

    def test_bulge_site_needs_the_extra_base_of_overlap(self):
        rng = random.Random(1)
        bulged = SPACER[:10] + "T" + SPACER[10:] + "CGG"                 # 24 nt window (one DNA bulge)
        g = {"c": rnd(rng, 49) + bulged + rnd(rng, 80)}
        full = f.oracle_search(g, self.guides, C4)
        self.assertTrue(any(h.db == 1 for h in full))
        self.assertEqual(f.chunked_search(g, self.guides, C4, 50), full)
        self.assertNotEqual(f.chunked_search(g, self.guides, C4, 50, overlap=f.required_overlap(C2)), full)

    def test_n_bases_at_a_border(self):
        g = {"c": "A" * 40 + SPACER[:9] + "N" + SPACER[10:] + "TGG" + "A" * 60}
        for cfg in (dict(C2, mm=0), dict(C2, mm=1)):
            self.assertEqual(f.chunked_search(g, self.guides, cfg, 50), f.oracle_search(g, self.guides, cfg))


class Randomized(unittest.TestCase):
    """Property test: many random genomes, guides and chunk sizes; chunked must always equal unchunked."""
    def test_mismatch_search(self):
        rng = random.Random(2026)
        for trial in range(25):
            g = {"a": rnd(rng, rng.randint(150, 400)), "b": rnd(rng, rng.randint(50, 200))}
            guide = rnd(rng, 20)
            for _ in range(rng.randint(2, 5)):                            # plant near-matches anywhere, any strand
                site = guide[:3] + rng.choice("ACGT") + guide[4:] + rng.choice("ACGT") + "GG"
                site = site if rng.random() < 0.5 else f.revcomp(site)
                c = rng.choice("ab")
                p = rng.randrange(0, len(g[c]) - 23)
                g[c] = g[c][:p] + site + g[c][p + 23:]
            guides = [("g", guide)]
            full = f.oracle_search(g, guides, C2)
            size = rng.choice([5, 11, 40, 64, 101, 333])
            self.assertEqual(f.chunked_search(g, guides, C2, size), full, (trial, size))

    def test_bulge_search(self):
        rng = random.Random(7)
        for trial in range(8):
            g = {"a": rnd(rng, rng.randint(200, 320))}
            guide = rnd(rng, 20)
            kind = rng.choice(["dna", "rna", "mm"])
            body = {"dna": guide[:9] + rng.choice("ACGT") + guide[9:], "rna": guide[:9] + guide[10:],
                    "mm": guide[:5] + rng.choice("ACGT") + guide[6:]}[kind]
            site = body + "TGG"
            site = site if rng.random() < 0.5 else f.revcomp(site)
            p = rng.randrange(10, len(g["a"]) - 40)
            g["a"] = g["a"][:p] + site + g["a"][p + len(site):]
            guides = [("g", guide)]
            full = f.oracle_search(g, guides, C4)
            size = rng.choice([13, 37, 50, 96])
            self.assertEqual(f.chunked_search(g, guides, C4, size), full, (trial, kind, size))


class MockGenomeEndToEnd(unittest.TestCase):
    def test_chunked_search_finds_every_planted_site(self):
        mock = f.generate_mock(scale=0.0001, chroms=["chr21", "chr22"], n_guides=2, copies=3, max_mm=3, bulge_copies=0)
        genome = {c: ba.decode() for c, ba in mock["genome"].items()}
        cfg = dict(f.CONFIGS["C2"], mm=3)
        hits = f.chunked_search(genome, mock["guides"], cfg, chunk_size=4000)
        self.assertEqual(hits, f.oracle_search(genome, mock["guides"], cfg))
        found = {h.key() for h in hits}
        planted = [t for t in mock["truth"]]
        self.assertTrue(all((t[0], t[1], t[2], t[3]) in found for t in planted))

    def test_custom_search_function_is_used(self):
        calls = []

        def fake(genome, guides, cfg):
            calls.append(sum(len(s) for s in genome.values()))
            return []
        f.chunked_search({"c": "A" * 100}, [("g", SPACER)], C2, 40, search_fn=fake)
        self.assertEqual(len(calls), 3)


class CommandLine(unittest.TestCase):
    def test_chunks_command_prints_plan(self):
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "g.fa")
            with open(p, "w") as fh:
                fh.write(">c1\n" + "ACGT" * 100 + "\n>c2\n" + "TTGCA" * 40 + "\n")
            self.assertEqual(quiet(f.main, ["chunks", p, "--chunk-size", "150", "--config", "C4", "--parser", "builtin"]), 0)
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                f.main(["chunks", p, "--chunk-size", "150", "--config", "C4", "--parser", "builtin"])
            text = out.getvalue()
            self.assertIn("overlap     : 23 bp", text)
            self.assertIn("chunks      : 5", text)                       # ceil(400/150)=3 and 200/150 -> 2

    def test_chunks_verify_on_mock_genome(self):
        with tempfile.TemporaryDirectory() as td:
            quiet(f.main, ["mockgenome", "--scale", "0.0001", "--chroms", "chr21", "--n-guides", "2", "--out-dir", td])
            rc = quiet(f.main, ["chunks", os.path.join(td, "genome.fa"), "--chunk-size", "3000", "--config", "C2",
                                "--guides", os.path.join(td, "guides.tsv"), "--verify", "--parser", "builtin"])
            self.assertEqual(rc, 0)

    def test_errors(self):
        self.assertEqual(quiet(f.main, ["chunks", "missing.fa", "--parser", "builtin"]), 2)
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "g.fa")
            with open(p, "w") as fh:
                fh.write(">c\nACGT\n")
            self.assertEqual(quiet(f.main, ["chunks", p, "--verify", "--parser", "builtin"]), 2)       # needs --guides
            self.assertEqual(quiet(f.main, ["chunks", p, "--chunk-size", "0", "--parser", "builtin"]), 2)


class EdgeCases(unittest.TestCase):
    def test_no_guides_gives_no_hits(self):
        self.assertEqual(f.chunked_search({"c": "ACGT" * 50}, [], C2, 40), [])

    def test_empty_chromosome_and_tiny_chromosome(self):
        g = {"empty": "", "tiny": "ACGT"}
        self.assertEqual(f.chunked_search(g, [("g", SPACER)], C2, 10), [])

    def test_chunk_size_larger_than_genome_equals_one_chunk(self):
        rng = random.Random(5)
        g = {"c": rnd(rng, 40) + SITE + rnd(rng, 40)}
        self.assertEqual(f.chunked_search(g, [("g", SPACER)], C2, 10_000), f.oracle_search(g, [("g", SPACER)], C2))


if __name__ == "__main__":
    unittest.main(verbosity=2)