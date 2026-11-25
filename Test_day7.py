"""Day 7 tests: explicit CUDA host/device memory transfers, transfer accounting and the compact (hits-only) scheme."""
import contextlib, io, json, os, random, tempfile, unittest
from unittest import mock

import numpy as np

import fasta as f


class HostDeviceTransfers(unittest.TestCase):
    def test_host_to_device_makes_noncontiguous_input_contiguous(self):
        source = np.arange(24, dtype=np.uint8).reshape(4, 6)[:, ::2]
        fake_cuda = mock.Mock()
        fake_cuda.to_device.side_effect = lambda array, stream=0: (array, stream)

        with mock.patch.object(f, "HAVE_NUMBA", True), mock.patch.object(f, "cuda", fake_cuda):
            copied, stream = f.host_to_device(source, stream=3)

        np.testing.assert_array_equal(copied, source)
        self.assertTrue(copied.flags.c_contiguous)
        self.assertEqual(stream, 3)
        fake_cuda.to_device.assert_called_once()

    def test_host_to_device_rejects_object_arrays(self):
        with mock.patch.object(f, "HAVE_NUMBA", True):
            with self.assertRaisesRegex(TypeError, "fixed-size dtype"):
                f.host_to_device(np.array([object()], dtype=object))

    def test_host_to_device_requires_numba(self):
        with mock.patch.object(f, "HAVE_NUMBA", False):
            with self.assertRaisesRegex(RuntimeError, "require Numba CUDA"):
                f.host_to_device(np.arange(4, dtype=np.uint8))

    def test_device_to_host_uses_requested_stream(self):
        expected = np.arange(4, dtype=np.uint8)
        device_array = mock.Mock()
        device_array.copy_to_host.return_value = expected

        with mock.patch.object(f, "HAVE_NUMBA", True):
            result = f.device_to_host(device_array, stream=5)

        np.testing.assert_array_equal(result, expected)
        device_array.copy_to_host.assert_called_once_with(stream=5)

    def test_device_to_host_rejects_non_device_arrays(self):
        with mock.patch.object(f, "HAVE_NUMBA", True):
            with self.assertRaisesRegex(TypeError, "copy_to_host"):
                f.device_to_host(np.arange(4, dtype=np.uint8))

    def test_gpu_scan_uses_explicit_host_device_transfers(self):
        codes = f.encode_bases("ACGTACGTTGG")
        guides = np.array([f.encode_guide("ACGTACGT")], dtype=np.uint8)
        masks = f.pam_masks_for("NGG")
        fake_kernel = mock.MagicMock()
        h2d = mock.Mock(side_effect=lambda array: array)
        d2h = mock.Mock(side_effect=lambda array: array)

        with (mock.patch.object(f, "gpu_backend", return_value="gpu"),
              mock.patch.object(f, "host_to_device", h2d),
              mock.patch.object(f, "device_to_host", d2h),
              mock.patch.object(f, "mismatch_kernel", fake_kernel)):
            result = f.scan_codes(codes, guides, 8, masks, 0, backend="gpu")

        self.assertIs(result, h2d.call_args_list[-1].args[0])
        self.assertEqual(h2d.call_count, 4)
        self.assertTrue(np.all(h2d.call_args_list[-1].args[0] == 255))
        d2h.assert_called_once_with(result)
        fake_kernel.__getitem__.return_value.assert_called_once()


# ---- everything below was added with the transfer layer ----------------------------------------
SPACER = "GACGTCTAGCATCGATCGAA"


def capture(fn, *a, **k):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        return fn(*a, **k), out.getvalue()


def rnd(rng, n, alphabet="ACGT"):
    return "".join(rng.choice(alphabet) for _ in range(n))


def planted(rng, guides, n_chroms=2, lower=False):
    genome = {}
    for c in range(n_chroms):
        seq = rnd(rng, rng.randint(150, 400))
        for _ in range(rng.randint(2, 5)):
            gid, g = rng.choice(guides)
            site = list(g)
            for p in rng.sample(range(len(g)), rng.randint(0, 3)):
                site[p] = rng.choice([b for b in "ACGT" if b != site[p]])
            site = "".join(site) + rng.choice(["TGG", "AGG", "CGG", "GGG"])
            site = site if rng.random() < 0.5 else f.revcomp(site)
            p = rng.randrange(0, len(seq) - len(site))
            seq = seq[:p] + site + seq[p + len(site):]
        if lower:                                           # soft-mask some stretches
            for _ in range(3):
                a = rng.randrange(0, len(seq) - 40)
                seq = seq[:a] + seq[a:a + rng.randint(5, 40)].lower() + seq[a + 40:]
        genome[f"c{c}"] = seq
    return genome


class TransferStatsAndSession(unittest.TestCase):
    def test_stats_counters(self):
        st = f.TransferStats()
        st.up(100, 0.5, calls=2)
        st.down(30, 0.25)
        st.alloc(64, calls=3)
        d = st.as_dict()
        self.assertEqual((d["h2d_bytes"], d["h2d_calls"], d["h2d_s"]), (100, 2, 0.5))
        self.assertEqual((d["d2h_bytes"], d["d2h_calls"], d["d2h_s"]), (30, 1, 0.25))
        self.assertEqual((d["allocs"], d["alloc_bytes"], d["launches"]), (3, 64, 0))
        self.assertEqual(set(d), set(f.TransferStats.FIELDS))
        json.dumps(d)

    def test_session_moves_and_counts_bytes(self):
        sess = f.DeviceSession("emulated")
        host = np.arange(10, dtype=np.uint8)
        dev = sess.put(host)
        self.assertIsNot(dev, host)                                        # a copy, like a device buffer
        self.assertEqual((sess.stats.allocs, sess.stats.alloc_bytes, sess.stats.h2d_bytes, sess.stats.h2d_calls), (1, 10, 10, 1))
        buf = sess.alloc((16,), np.uint8)
        self.assertEqual((sess.stats.allocs, sess.stats.h2d_bytes), (2, 10))      # allocating moves nothing
        sess.put_into(buf, host[:6])
        self.assertEqual(buf[:6].tolist(), [0, 1, 2, 3, 4, 5])
        self.assertEqual(buf[6:].tolist(), [0] * 10)
        self.assertEqual((sess.stats.allocs, sess.stats.h2d_bytes, sess.stats.h2d_calls), (2, 16, 2))   # no new allocation
        back = sess.get(buf, 4)
        self.assertEqual(back.tolist(), [0, 1, 2, 3])
        back[0] = 99
        self.assertEqual(buf[0], 0)                                         # the host copy is independent of the device
        self.assertEqual((sess.stats.d2h_bytes, sess.stats.d2h_calls), (4, 1))
        self.assertEqual(sess.get(buf).shape, (16,))
        self.assertEqual(sess.stats.d2h_bytes, 20)

    def test_session_counts_launches_and_runs_kernels(self):
        sess = f.DeviceSession("emulated")
        out = np.zeros(8, np.int32)
        sess.launch(f.mismatch_kernel if False else _fill_kernel, 2, 4, out)
        self.assertEqual(out.tolist(), list(range(8)))
        self.assertEqual(sess.stats.launches, 1)
        self.assertGreaterEqual(sess.stats.kernel_s, 0.0)

    def test_pinned_staging_is_used_only_on_a_real_gpu_and_reused(self):
        class FakeDev:
            def __init__(self, n):
                self.data = np.zeros(n, np.uint8)
            def __getitem__(self, s):
                v = FakeDev(0); v.data = self.data[s]; return v
            def copy_to_device(self, src):
                self.data[:len(src)] = src
        fake = mock.Mock()
        fake.device_array.side_effect = lambda shape, dtype: FakeDev(shape[0])
        fake.pinned_array.side_effect = lambda n, dtype: np.zeros(n, dtype)
        fake.synchronize.return_value = None
        sess = f.DeviceSession("gpu", pinned=True)
        with mock.patch.object(f, "cuda", fake):
            dev = sess.alloc((16,), np.uint8)
            sess.put_into(dev, np.arange(8, dtype=np.uint8))
            sess.put_into(dev, np.arange(4, dtype=np.uint8) + 100)
        self.assertEqual(fake.pinned_array.call_count, 1)                    # grown once, then reused
        self.assertEqual(dev.data[:4].tolist(), [100, 101, 102, 103])
        self.assertEqual(sess.stats.h2d_bytes, 12)
        self.assertFalse(f.DeviceSession("emulated", pinned=True).pinned)    # pinned memory needs a real GPU


def _fill_kernel(out):
    """Module-level so emulate_launch can swap its `cuda` (a stand-in kernel written like a real one)."""
    i = cuda.grid(1)
    if i < out.shape[0]:
        out[i] = i


cuda = None
_fill_kernel = type("K", (), {"py_func": staticmethod(_fill_kernel)})()


class CompactKernel(unittest.TestCase):
    def run_compact(self, codes, guides_codes, glen, mm, strand, threads=128, capacity=64):
        sess = f.DeviceSession("emulated")
        masks = f.pam_masks_for("NGG")
        d_codes, d_masks, d_guides = sess.put(codes), sess.put(masks), sess.put(guides_codes)
        d_hits, d_count = sess.alloc((capacity, 3), np.int32), sess.alloc((1,), np.int32)
        n_pos = len(codes) - glen - 3 + 1
        blocks = (n_pos + threads - 1) // threads
        sess.launch(f.compact_kernel, blocks, threads, d_codes, len(codes), strand, n_pos, d_guides, glen, d_masks, 3, mm,
                    d_hits, d_count, capacity)
        count = int(d_count[0])
        return count, {tuple(r) for r in d_hits[:min(count, capacity)].tolist()}

    def test_both_strands_agree_with_the_day_6_kernel(self):
        rng = random.Random(21)
        masks = f.pam_masks_for("NGG")
        seen = 0
        for _ in range(25):
            glen = rng.choice([8, 12, 20])
            guides = np.array([f.encode_guide(rnd(rng, glen)) for _ in range(rng.randint(1, 4))], np.uint8)
            codes = f.encode_bases(rnd(rng, rng.randint(50, 200), "ACGGGTN"))
            for mm in (1, 4):
                forward = f.scan_codes(codes, guides, glen, masks, mm, "emulated")
                reverse = f.scan_codes(f.revcomp_codes(codes), guides, glen, masks, mm, "emulated")
                for strand, ref in ((0, forward), (1, reverse)):
                    count, rows = self.run_compact(codes, guides, glen, mm, strand, threads=rng.choice([1, 9, 128]))
                    want = {(int(i), int(g), int(ref[i, g])) for i, g in zip(*np.nonzero(ref != 255))}
                    self.assertEqual(rows, want)
                    self.assertEqual(count, len(want))
                    seen += len(want)
        self.assertGreater(seen, 10)

    def test_counter_keeps_counting_past_the_buffer_capacity(self):
        rng = random.Random(22)
        guides = np.array([f.encode_guide("A" * 6)], np.uint8)
        codes = f.encode_bases(("AAAAAATGG" * 12))
        count, rows = self.run_compact(codes, guides, 6, 6, 0, capacity=3)    # every window with a PAM matches
        full_count, full_rows = self.run_compact(codes, guides, 6, 6, 0, capacity=500)
        self.assertGreater(full_count, 3)
        self.assertEqual(count, full_count)                                   # counted all, stored only `capacity`
        self.assertEqual(len(rows), 3)
        self.assertTrue(rows <= full_rows)

    def test_kernel_declaration(self):
        with open(f.__file__) as fh:
            src = fh.read()
        self.assertIn("@cuda.jit\ndef compact_kernel(", src)
        self.assertIn("cuda.atomic.add(counter, 0, 1)", src)


class SchemesAgree(unittest.TestCase):
    def test_matrix_compact_and_multi_equal_the_oracle(self):
        rng = random.Random(23)
        total = 0
        for trial in range(8):
            guides = [(f"g{i}", rnd(rng, 20)) for i in range(rng.randint(1, 3))]
            genome = planted(rng, guides)
            for cfg in (f.CONFIGS["C1"], f.CONFIGS["C2"], f.CONFIGS["C6"]):
                want = f.oracle_search(genome, guides, cfg)
                for scheme in f.SCHEMES:
                    for size in (55, 1_000_000):
                        got = f.gpu_search(genome, guides, cfg, size, "emulated", scheme=scheme)
                        self.assertEqual(got, want, (trial, cfg["pam"], scheme, size))
                total += len(want)
        self.assertGreater(total, 30)

    def test_a_tiny_hit_buffer_is_grown_and_the_result_is_unchanged(self):
        rng = random.Random(24)
        guides = [("g0", SPACER)]
        genome = planted(rng, guides)
        cfg = dict(f.CONFIGS["C2"], mm=6)
        want = f.oracle_search(genome, guides, cfg)
        for scheme in ("compact", "multi"):
            roomy, tiny = f.TransferStats(), f.TransferStats()
            self.assertEqual(f.gpu_search(genome, guides, cfg, 1_000_000, "emulated", scheme=scheme, stats=roomy), want)
            self.assertEqual(f.gpu_search(genome, guides, cfg, 1_000_000, "emulated", scheme=scheme, stats=tiny,
                                          hit_capacity=1), want)
            self.assertGreater(tiny.allocs, roomy.allocs, scheme)              # the buffer was re-allocated
            self.assertGreater(tiny.launches, roomy.launches, scheme)          # ... and the launch repeated

    def test_scheme_and_thread_arguments_are_validated(self):
        genome, guides = {"c": "ACGT" * 40}, [("g", SPACER)]
        with self.assertRaises(ValueError):
            f.gpu_search(genome, guides, f.CONFIGS["C2"], 50, "emulated", scheme="fast")
        with self.assertRaises(ValueError):
            f.gpu_search(genome, guides, f.CONFIGS["C2"], 50, "emulated", threads=0)
        for scheme in f.SCHEMES:
            with self.assertRaises(ValueError):
                f.gpu_search(genome, guides, f.CONFIGS["C4"], 50, "emulated", scheme=scheme)
            self.assertEqual(f.gpu_search({"empty": "", "tiny": "ACGT"}, guides, f.CONFIGS["C2"], 10, "emulated", scheme=scheme), [])


class SoftMaskedInput(unittest.TestCase):
    """Rules s4.1: lowercase (soft-masked) bases are sequence. Found while checking the schemes against each other."""
    def test_every_search_treats_lowercase_as_sequence(self):
        rng = random.Random(25)
        for trial in range(6):
            guides = [(f"g{i}", rnd(rng, 20)) for i in range(2)]
            genome = planted(rng, guides, lower=True)
            upper = {c: s.upper() for c, s in genome.items()}
            self.assertNotEqual(genome, upper)
            cfg = f.CONFIGS["C2"]
            want = f.oracle_search(upper, guides, cfg)
            self.assertEqual(f.oracle_search(genome, guides, cfg), want)
            self.assertEqual(f.packed_search(genome, guides, cfg), want)
            for scheme in f.SCHEMES:
                self.assertEqual(f.gpu_search(genome, guides, cfg, 70, "emulated", scheme=scheme), want, (trial, scheme))
            for h in want:
                self.assertEqual(h.target, h.target.upper())

    def test_reverse_strand_sites_inside_lowercase_are_found(self):
        site = f.revcomp(SPACER + "TGG")
        genome = {"c": "A" * 30 + site.lower() + "A" * 30}
        hits = f.oracle_search(genome, [("g", SPACER)], f.CONFIGS["C2"])
        self.assertEqual([(h.strand, h.start, h.mm) for h in hits], [("-", 30, 0)])


class TransferSavings(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        mock_genome = f.generate_mock(scale=0.0001, chroms=["chr21", "chr22"], n_guides=12)
        cls.genome = {c: ba.decode() for c, ba in mock_genome["genome"].items()}
        cls.guides = mock_genome["guides"]
        cls.cmp = f.compare_schemes(cls.genome, cls.guides, f.CONFIGS["C2"], "emulated")

    def test_all_schemes_return_the_same_hits(self):
        self.assertTrue(self.cmp["identical"])
        self.assertEqual(len({r["hits"] for r in self.cmp["schemes"].values()}), 1)

    def test_compact_moves_far_less_data(self):
        m, c = (self.cmp["schemes"][k]["stats"] for k in ("matrix", "compact"))
        self.assertGreater(m["d2h_bytes"], 50 * c["d2h_bytes"])
        self.assertGreater(m["h2d_bytes"], 5 * c["h2d_bytes"])
        self.assertLess(c["allocs"], m["allocs"])

    def test_multi_launches_once_for_both_strands(self):
        c, mu = (self.cmp["schemes"][k]["stats"] for k in ("compact", "multi"))
        self.assertEqual(c["launches"], 2 * mu["launches"])

    def test_compact_traffic_follows_an_exact_formula(self):
        rng = random.Random(26)
        guides = [("g0", rnd(rng, 20)), ("g1", rnd(rng, 20))]
        genome = planted(rng, guides, n_chroms=1)
        cfg = f.CONFIGS["C2"]
        stats = f.TransferStats()
        hits = f.gpu_search(genome, guides, cfg, 1_000_000, "emulated", scheme="compact", stats=stats)
        n = len(next(iter(genome.values())))
        static = f.pam_masks_for("NGG").nbytes + 2 * 20                       # masks + two 20-base guides, uploaded once
        self.assertEqual(stats.launches, 2)                                   # forward + reverse, no overflow
        self.assertEqual(stats.h2d_bytes, static + n + 2 * 4)                 # static + one codes upload + two counter resets
        self.assertEqual(stats.d2h_bytes, 2 * 4 + 3 * 4 * len(hits))          # two counters + 12 bytes per hit
        self.assertEqual(stats.allocs, 2 + 1 + 1 + 1 + 1)                     # masks, guides, codes, hits, counter

    def test_comparison_table_text(self):
        text = f.format_comparison(self.cmp)
        for part in ("Day 6 matrix", "Day 7 compact", "Day 8 multi", "host -> device", "device -> host", "kernel launches",
                     "download vs Day 6", "all three schemes return identical hits", "emulated"):
            self.assertIn(part, text)
        gpu_cmp = dict(self.cmp, backend="gpu")
        self.assertIn("transfer time", f.format_comparison(gpu_cmp))
        bad = dict(self.cmp, identical=False)
        self.assertIn("DIFFERENT", f.format_comparison(bad))


class CommandLine(unittest.TestCase):
    def test_transfers_command(self):
        with tempfile.TemporaryDirectory() as td:
            rc, text = capture(f.main, ["transfers", "--n-guides", "4", "--out-dir", td])
            self.assertEqual(rc, 0, text)
            for part in ("Day 7 - host/device memory transfers", "Day 8 multi", "identical hits"):
                self.assertIn(part, text)

    def test_transfers_errors(self):
        self.assertEqual(capture(f.main, ["transfers", "--config", "C4"])[0], 2)
        with mock.patch.object(f, "HAVE_NUMBA", False):
            self.assertEqual(capture(f.main, ["transfers", "--backend", "gpu"])[0], 2)

    def test_gpu_command_with_the_compact_scheme_records_its_traffic(self):
        with tempfile.TemporaryDirectory() as td:
            rc, text = capture(f.main, ["gpu", "--scheme", "compact", "--n-guides", "3", "--runs", "1", "--warmup", "0",
                                        "--out-dir", td, "--verify"])
            self.assertEqual(rc, 0, text)
            self.assertIn("transfers   : compact scheme", text)
            self.assertIn("compact_kernel", text)
            files = [n for n in os.listdir(td) if n.endswith("_cuda-emulated-compact.json")]
            self.assertEqual(len(files), 1)
            with open(os.path.join(td, files[0])) as fh:
                rec = json.load(fh)
            self.assertEqual(rec["scheme"], "compact")
            self.assertGreater(rec["transfers"]["h2d_bytes"], 0)
            self.assertEqual(rec["implementation"], "cuda-emulated-compact")

    def test_records_of_different_schemes_do_not_overwrite_each_other(self):
        with tempfile.TemporaryDirectory() as td:
            for scheme in f.SCHEMES:
                self.assertEqual(capture(f.main, ["gpu", "--scheme", scheme, "--n-guides", "2", "--runs", "1",
                                                  "--warmup", "0", "--out-dir", td])[0], 0)
            self.assertEqual(len([n for n in os.listdir(td) if n.endswith(".json")]), 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)