"""Day 8 tests: multi-guide search and both strands in ONE launch (multi_kernel)."""
import contextlib, io, json, os, random, tempfile, unittest
from unittest import mock

import numpy as np

import fasta as f

SPACER = "GACGTCTAGCATCGATCGAA"


def capture(fn, *a, **k):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        return fn(*a, **k), out.getvalue()


def rnd(rng, n, alphabet="ACGT"):
    return "".join(rng.choice(alphabet) for _ in range(n))


def planted(rng, guides, n_chroms=2, per_chrom=(3, 8), lower=False, with_n=False):
    genome = {}
    for c in range(n_chroms):
        seq = rnd(rng, rng.randint(200, 500))
        for _ in range(rng.randint(*per_chrom)):
            gid, g = rng.choice(guides)
            site = list(g)
            for p in rng.sample(range(len(g)), rng.randint(0, 3)):
                site[p] = rng.choice([b for b in "ACGT" if b != site[p]])
            site = "".join(site) + rng.choice(["TGG", "AGG", "CGG", "GGG"])
            site = site if rng.random() < 0.5 else f.revcomp(site)
            p = rng.randrange(0, len(seq) - len(site))
            seq = seq[:p] + site + seq[p + len(site):]
        if with_n:
            for _ in range(3):
                p = rng.randrange(0, len(seq) - 4)
                seq = seq[:p] + "N" * rng.randint(1, 3) + seq[p + 3:]
        if lower:
            a = rng.randrange(0, len(seq) - 60)
            seq = seq[:a] + seq[a:a + 50].lower() + seq[a + 50:]
        genome[f"c{c}"] = seq
    return genome


class PackedGuides(unittest.TestCase):
    def test_words_and_low_bits(self):
        words = f.pack_guides(np.array([[0, 1, 2, 3], [3, 3, 3, 3]], np.uint8))
        self.assertEqual(words.tolist(), [0 | 1 << 2 | 2 << 4 | 3 << 6, 0b11111111])
        self.assertEqual(words.dtype, np.int64)
        self.assertEqual(f.low_bits(4), 0b01010101)
        self.assertEqual(f.low_bits(1), 1)
        self.assertEqual(f.low_bits(20), sum(1 << (2 * k) for k in range(20)))

    def test_length_limit(self):
        f.pack_guides(np.zeros((1, f.MAX_PACKED_GUIDE), np.uint8))
        with self.assertRaises(ValueError) as cm:
            f.pack_guides(np.zeros((1, f.MAX_PACKED_GUIDE + 1), np.uint8))
        self.assertIn("at most 31", str(cm.exception))

    def test_xor_and_bit_count_equal_the_mismatch_count(self):
        rng = random.Random(81)
        for _ in range(200):
            glen = rng.choice([5, 12, 20, 25, 31])
            a, b = [np.array([rng.randrange(4) for _ in range(glen)], np.uint8) for _ in range(2)]
            wa, wb = int(f.pack_guides(a[None, :])[0]), int(f.pack_guides(b[None, :])[0])
            x = wa ^ wb
            self.assertEqual(bin(((x | (x >> 1)) & f.low_bits(glen))).count("1"), int((a != b).sum()))


class MultiKernel(unittest.TestCase):
    def launch(self, codes, guide_codes, glen, mm, g_offset=0, threads=128, capacity=500):
        sess = f.DeviceSession("emulated")
        masks = f.pam_masks_for("NGG")
        d_codes, d_masks = sess.put(codes), sess.put(masks)
        d_words = sess.put(f.pack_guides(guide_codes))
        d_hits, d_count = sess.alloc((capacity, 4), np.int32), sess.alloc((1,), np.int32)
        n_pos = len(codes) - glen - 3 + 1
        blocks = (2 * n_pos + threads - 1) // threads
        sess.launch(f.multi_kernel, blocks, threads, d_codes, len(codes), n_pos, d_words, glen, f.low_bits(glen), d_masks, 3,
                    mm, g_offset, d_hits, d_count, capacity)
        count = int(d_count[0])
        return count, {tuple(r) for r in d_hits[:min(count, capacity)].tolist()}

    def reference(self, codes, guide_codes, glen, mm, g_offset=0):
        masks = f.pam_masks_for("NGG")
        want = set()
        for strand, c in ((0, codes), (1, f.revcomp_codes(codes))):
            m = f.scan_codes(c, guide_codes, glen, masks, mm, "emulated")
            want |= {(int(i), strand, int(g) + g_offset, int(m[i, g])) for i, g in zip(*np.nonzero(m != 255))}
        return want

    def test_one_launch_equals_two_day_6_scans(self):
        rng = random.Random(82)
        seen = 0
        for _ in range(35):
            glen = rng.choice([8, 12, 20, 25])
            guides = np.array([f.encode_guide(rnd(rng, glen)) for _ in range(rng.randint(1, 6))], np.uint8)
            codes = f.encode_bases(rnd(rng, rng.randint(60, 200), "ACGGGTN"))
            for mm in (1, 5):
                off = rng.choice([0, 7])
                count, rows = self.launch(codes, guides, glen, mm, g_offset=off, threads=rng.choice([1, 13, 128]))
                want = self.reference(codes, guides, glen, mm, g_offset=off)
                self.assertEqual(rows, want)
                self.assertEqual(count, len(want))
                seen += len(want)
        self.assertGreater(seen, 20)                                              # the comparison was not vacuous

    def test_guide_offset_is_added_to_the_reported_index(self):
        guides = np.array([f.encode_guide("ACGTACGT")], np.uint8)
        codes = f.encode_bases("ACGTACGTTGG")
        for off in (0, 5, 4000):
            count, rows = self.launch(codes, guides, 8, 0, g_offset=off)
            self.assertIn((0, 0, off, 0), rows)                                   # forward strand, position 0, exact
            self.assertEqual({r[2] for r in rows}, {off})                         # one guide: its index is exactly the offset

    def test_both_strands_come_from_one_grid(self):
        site = SPACER + "TGG"
        codes = f.encode_bases("AAAAA" + site + "AAAAA" + f.revcomp(site) + "AAAAA")
        guides = np.array([f.encode_guide(SPACER)], np.uint8)
        count, rows = self.launch(codes, guides, 20, 0)
        self.assertEqual(sorted((r[1], r[3]) for r in rows), [(0, 0), (1, 0)])
        self.assertEqual(count, 2)

    def test_n_bases_always_count_as_mismatches(self):
        guides = np.array([f.encode_guide("AAAAAAAA")], np.uint8)
        masks_ok = "TGG"
        n_in_spacer = f.encode_bases("AAANAAAA" + masks_ok)
        self.assertEqual(self.launch(n_in_spacer, guides, 8, 0)[0], 0)           # N is not an A
        self.assertEqual({r[3] for r in self.launch(n_in_spacer, guides, 8, 1)[1] if r[1] == 0}, {1})
        self.assertEqual(self.launch(f.encode_bases("AAAAAAAA" + "TNG"), guides, 8, 8)[0], 0)   # N never fits a PAM

    def test_result_does_not_depend_on_the_thread_layout(self):
        rng = random.Random(83)
        codes = f.encode_bases(rnd(rng, 257, "ACGGGT"))
        guides = np.array([f.encode_guide(rnd(rng, 10)) for _ in range(4)], np.uint8)
        base = self.launch(codes, guides, 10, 4, threads=1)
        for threads in (2, 7, 32, 100, 128, 513, 1024):
            self.assertEqual(self.launch(codes, guides, 10, 4, threads=threads), base, threads)

    def test_counter_counts_past_the_buffer(self):
        guides = np.array([f.encode_guide("A" * 6)], np.uint8)
        codes = f.encode_bases("AAAAAATGG" * 10)
        full_count, full = self.launch(codes, guides, 6, 6, capacity=500)
        count, stored = self.launch(codes, guides, 6, 6, capacity=2)
        self.assertGreater(full_count, 2)
        self.assertEqual(count, full_count)
        self.assertEqual(len(stored), 2)
        self.assertTrue(stored <= full)

    def test_kernel_declaration(self):
        with open(f.__file__) as fh:
            src = fh.read()
        self.assertIn("@cuda.jit\ndef multi_kernel(", src)
        self.assertIn("cuda.atomic.add(counter, 0, 1)", src)
        self.assertIn("if t >= 2 * n_pos:", src)                                  # one grid for both strands


class MultiGuideSearch(unittest.TestCase):
    def test_many_guides_equal_the_oracle(self):
        rng = random.Random(84)
        guides = [(f"g{i:03d}", rnd(rng, 20)) for i in range(40)]
        genome = planted(rng, guides, per_chrom=(15, 25))
        for cfg in (f.CONFIGS["C1"], f.CONFIGS["C2"], f.CONFIGS["C6"]):
            want = f.oracle_search(genome, guides, cfg)
            self.assertGreater(len(want), 10)
            for size in (90, 1_000_000):
                self.assertEqual(f.gpu_search(genome, guides, cfg, size, "emulated", scheme="multi"), want, (cfg["pam"], size))

    def test_batches_do_not_change_the_answer_or_the_guide_ids(self):
        rng = random.Random(85)
        guides = [(f"g{i:03d}", rnd(rng, 20)) for i in range(23)]
        genome = planted(rng, guides, per_chrom=(10, 15))
        cfg = f.CONFIGS["C2"]
        want = f.oracle_search(genome, guides, cfg)
        for batch in (0, 1, 2, 5, 22, 23, 24, 1000):                              # 0 means the automatic batch size
            self.assertEqual(f.gpu_search(genome, guides, cfg, 150, "emulated", scheme="multi", guide_batch=batch), want, batch)
        with self.assertRaises(ValueError):
            f.gpu_search(genome, guides, cfg, 150, "emulated", scheme="multi", guide_batch=-3)

    def test_mixed_guide_lengths_n_bases_and_soft_masking(self):
        rng = random.Random(86)
        guides = [("a", rnd(rng, 20)), ("b", rnd(rng, 18)), ("c", rnd(rng, 23)), ("d", rnd(rng, 20))]
        genome = planted(rng, guides, per_chrom=(8, 12), lower=True, with_n=True)
        cfg = dict(f.CONFIGS["C2"], mm=3)
        self.assertEqual(f.gpu_search(genome, guides, cfg, 110, "emulated", scheme="multi", guide_batch=1),
                         f.oracle_search(genome, guides, cfg))

    def test_hits_appear_on_both_strands_without_duplicates(self):
        rng = random.Random(87)
        guides = [(f"g{i}", rnd(rng, 20)) for i in range(10)]
        genome = planted(rng, guides, per_chrom=(20, 30))
        hits = f.gpu_search(genome, guides, f.CONFIGS["C2"], 130, "emulated", scheme="multi")
        self.assertTrue({"+", "-"} <= {h.strand for h in hits})
        self.assertEqual(len(hits), len({(h.key(), h.mm, h.target) for h in hits}))
        for h in hits:
            win = genome[h.chrom][h.start:h.start + len(h.target)].upper()
            self.assertEqual(f.revcomp(win) if h.strand == "-" else win, h.target)

    def test_launch_counts(self):
        rng = random.Random(88)
        guides = [(f"g{i}", rnd(rng, 20)) for i in range(7)] + [("long", rnd(rng, 23))]
        genome = {"a": rnd(rng, 500), "b": rnd(rng, 250)}
        for scheme, per_chunk in (("compact", 2 * 2), ("multi", 2)):        # 2 length groups; compact also needs 2 strands
            stats = f.TransferStats()
            f.gpu_search(genome, guides, f.CONFIGS["C2"], 200, "emulated", scheme=scheme, stats=stats)
            self.assertEqual(stats.launches, 5 * per_chunk)                  # 5 chunks (3 of "a", 2 of "b")
        stats = f.TransferStats()
        f.gpu_search(genome, guides, f.CONFIGS["C2"], 200, "emulated", scheme="multi", guide_batch=3, stats=stats)
        self.assertEqual(stats.launches, 5 * (3 + 1))                        # 7 guides in batches of 3 -> 3, plus 1 long guide

    def test_progress_per_chunk(self):
        rng = random.Random(89)
        calls = []
        f.gpu_search({"a": rnd(rng, 250), "b": rnd(rng, 90)}, [("g", rnd(rng, 20))], f.CONFIGS["C2"], 100, "emulated",
                     scheme="multi", progress=lambda *a: calls.append(a))
        self.assertEqual([c[0] for c in calls], [1, 2, 3, 4])
        self.assertTrue(all(c[1] == 4 for c in calls))

    def test_every_planted_site_of_a_mock_genome_is_found(self):
        mock_genome = f.generate_mock(scale=0.0001, chroms=["chr21", "chr22"], n_guides=30, copies=4, max_mm=3, bulge_copies=0)
        genome = {c: ba.decode() for c, ba in mock_genome["genome"].items()}
        cfg = dict(f.CONFIGS["C2"], mm=3)
        hits = f.gpu_search(genome, mock_genome["guides"], cfg, None, "emulated", scheme="multi", guide_batch=8)
        found = {h.key() for h in hits}
        self.assertGreater(len(mock_genome["truth"]), 100)
        self.assertTrue(all((t[0], t[1], t[2], t[3]) in found for t in mock_genome["truth"]))
        self.assertEqual(hits, f.packed_search(genome, mock_genome["guides"], cfg))


class GuideSummary(unittest.TestCase):
    def hit(self, gid, strand, mm):
        return f.Hit(gid, "c", 1, strand, mm, 0, 0, "A" * 23)

    def test_counts_per_guide_and_strand(self):
        guides = [("g0", "A" * 20), ("g1", "C" * 20), ("g2", "G" * 20)]
        hits = [self.hit("g0", "+", 0), self.hit("g0", "-", 2), self.hit("g0", "-", 1), self.hit("g1", "+", 3)]
        rows = {r["guide"]: r for r in f.guide_summary(hits, guides)}
        self.assertEqual((rows["g0"]["forward"], rows["g0"]["reverse"], rows["g0"]["exact"], rows["g0"]["best"], rows["g0"]["total"]),
                         (1, 2, 1, 0, 3))
        self.assertEqual((rows["g1"]["best"], rows["g1"]["exact"]), (3, 0))
        self.assertEqual((rows["g2"]["total"], rows["g2"]["best"]), (0, None))     # a guide with no hits still has a row

    def test_table_text(self):
        guides = [(f"g{i}", "A" * 20) for i in range(14)]
        text = f.format_guide_summary(f.guide_summary([self.hit("g0", "+", 0), self.hit("g1", "-", 1)], guides), top=10)
        self.assertIn("... and 4 more guides", text)
        self.assertIn("forward", text)
        self.assertTrue(text.strip().split("\n")[-1].lstrip().startswith("all"))
        g2 = [line for line in text.split("\n") if line.strip().startswith("g2 ")][0].split()
        self.assertEqual(g2, ["g2", "0", "0", "0", "-", "0"])                       # no hits: best mismatch shown as '-'


class CommandLine(unittest.TestCase):
    def test_multi_command(self):
        with tempfile.TemporaryDirectory() as td:
            rc, text = capture(f.main, ["multi", "--n-guides", "10", "--runs", "1", "--warmup", "0", "--out-dir", td, "--verify"])
            self.assertEqual(rc, 0, text)
            for part in ("Day 8 - multi-guide search, both strands", "multi_kernel", "per-guide results (both strands)",
                         "identical to the Day 5 pure-Python baseline", "planted sites found"):
                self.assertIn(part, text)
            files = [n for n in os.listdir(td) if n.endswith("_cuda-emulated-multi.json")]
            self.assertEqual(len(files), 1)
            with open(os.path.join(td, files[0])) as fh:
                rec = json.load(fh)
            self.assertEqual((rec["scheme"], rec["n_guides"]), ("multi", 10))

    def test_multi_command_with_a_guide_file_and_small_batches(self):
        rng = random.Random(90)
        with tempfile.TemporaryDirectory() as td:
            guides = [(f"guide{i}", rnd(rng, 20)) for i in range(9)]
            gfa, gtsv = os.path.join(td, "g.fa"), os.path.join(td, "guides.tsv")
            genome = planted(rng, guides, per_chrom=(10, 14))
            f.write_fasta(genome, gfa)
            f.write_guides(guides, gtsv)
            rc, text = capture(f.main, ["multi", "--genome", gfa, "--guides", gtsv, "--guide-batch", "4", "--runs", "1",
                                        "--warmup", "0", "--parser", "builtin", "--out-dir", td, "--verify"])
            self.assertEqual(rc, 0, text)
            self.assertIn("guide8", text.replace("... and", ""))                    # 9 guides: 8 shown plus the summary line
            self.assertIn("identical to the Day 5", text)

    def test_multi_command_errors(self):
        with tempfile.TemporaryDirectory() as td:
            base = ["multi", "--out-dir", td]
            self.assertEqual(capture(f.main, base + ["--config", "C5"])[0], 2)
            with mock.patch.object(f, "HAVE_NUMBA", False):
                self.assertEqual(capture(f.main, base + ["--backend", "gpu"])[0], 2)
            self.assertEqual(capture(f.main, base + ["--guide-batch", "-1", "--runs", "1", "--warmup", "0"])[0], 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)