"""Day 6 tests: the first CUDA (@cuda.jit) kernel, its CPU emulation, the chunked GPU search and the terminal view.

Run from the folder that contains fasta.py:
    python -m unittest Test_day6 -v

Without a GPU the kernel runs through fasta.py's CPU emulator (the same kernel code, one thread at a time).
With Numba and an NVIDIA GPU (or NUMBA_ENABLE_CUDASIM=1) the tests that use backend "auto" run the real kernel.
"""
import contextlib, io, json, os, random, tempfile, unittest
from unittest import mock

import numpy as np

import fasta as f

SPACER = "GACGTCTAGCATCGATCGAA"
CONFIGS = [f.CONFIGS["C1"], f.CONFIGS["C2"], f.CONFIGS["C3"], f.CONFIGS["C6"]]


def capture(fn, *a, **k):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        return fn(*a, **k), out.getvalue()


def rnd(rng, n, alphabet="ACGT"):
    return "".join(rng.choice(alphabet) for _ in range(n))


def reference_scan(codes, guides, glen, masks, mm):
    """Plain, separately written version of what the kernel computes."""
    n_pos = len(codes) - glen - len(masks) + 1
    out = np.full((max(n_pos, 0), len(guides)), 255, np.uint8)
    for i in range(max(n_pos, 0)):
        if any(codes[i + glen + k] > 3 or not (int(masks[k]) >> int(codes[i + glen + k])) & 1 for k in range(len(masks))):
            continue
        for g in range(len(guides)):
            d = int(np.sum(codes[i:i + glen] != guides[g]))
            if d <= mm:
                out[i, g] = d
    return out


def planted_genome(rng, guides, pams=("TGG", "AGG", "CGG", "GGG", "TAG", "AAG"), n_chroms=2, with_n=False):
    genome = {}
    for c in range(n_chroms):
        seq = rnd(rng, rng.randint(150, 450))
        for _ in range(rng.randint(2, 5)):
            gid, g = rng.choice(guides)
            site = list(g)
            for p in rng.sample(range(len(g)), rng.randint(0, 3)):
                site[p] = rng.choice([b for b in "ACGT" if b != site[p]])
            site = "".join(site) + rng.choice(pams)
            site = site if rng.random() < 0.5 else f.revcomp(site)
            p = rng.randrange(0, len(seq) - len(site))
            seq = seq[:p] + site + seq[p + len(site):]
        if with_n:
            for _ in range(3):
                p = rng.randrange(0, len(seq) - 4)
                seq = seq[:p] + "N" * rng.randint(1, 3) + seq[p + 3:]
        genome[f"c{c}"] = seq
    return genome


class Encoding(unittest.TestCase):
    def test_bases_become_codes(self):
        self.assertEqual(f.encode_bases("ACGTacgtNRYn").tolist(), [0, 1, 2, 3, 0, 1, 2, 3, 4, 4, 4, 4])
        self.assertEqual(f.encode_bases("").tolist(), [])
        self.assertEqual(f.encode_bases("A\u00e9").tolist(), [0, 4])               # a stray non-ASCII character is just N

    def test_guides_must_be_plain_acgt(self):
        self.assertEqual(f.encode_guide("acgt").tolist(), [0, 1, 2, 3])
        for bad in ("ACGN", "ACGR", "AC-T"):
            with self.assertRaises(ValueError, msg=bad):
                f.encode_guide(bad)

    def test_pam_masks(self):
        self.assertEqual(f.pam_masks_for("NGG").tolist(), [15, 4, 4])              # N = any of the four bases, G = bit 2
        self.assertEqual(f.pam_masks_for("NRG").tolist(), [15, 5, 4])              # R = A or G = bits 0 and 2

    def test_reverse_complement_of_codes_matches_the_string_version(self):
        rng = random.Random(4)
        for _ in range(40):
            s = rnd(rng, rng.randint(0, 80), "ACGTN")
            self.assertEqual(f.revcomp_codes(f.encode_bases(s)).tolist(), f.encode_bases(f.revcomp(s)).tolist())


class KernelMechanics(unittest.TestCase):
    def test_kernel_matches_a_plain_reference_on_random_input(self):
        rng = random.Random(5)
        masks = f.pam_masks_for("NGG")
        compared = 0
        for _ in range(30):
            glen = rng.choice([8, 12, 20])
            guides = np.array([f.encode_guide(rnd(rng, glen)) for _ in range(rng.randint(1, 4))], np.uint8)
            seq = rnd(rng, rng.randint(40, 200), "ACGGGT" + "N" * rng.randint(0, 1))      # G-rich so PAMs occur
            codes = f.encode_bases(seq)
            for mm in (0, 2, 5):
                got = f.scan_codes(codes, guides, glen, masks, mm, "emulated", threads=rng.choice([1, 7, 32, 128]))
                want = reference_scan(codes, guides, glen, masks, mm)
                np.testing.assert_array_equal(got, want)
                compared += int((want != 255).sum())
        self.assertGreater(compared, 20)                                            # the comparison was not vacuous

    def test_pam_failures_and_n_bases(self):
        masks = f.pam_masks_for("NGG")
        g = np.array([f.encode_guide("ACGTACGT")], np.uint8)
        site = f.encode_bases("ACGTACGT" + "TGG")
        self.assertEqual(f.scan_codes(site, g, 8, masks, 0, "emulated").tolist(), [[0]])
        self.assertEqual(f.scan_codes(f.encode_bases("ACGTACGT" + "TAG"), g, 8, masks, 0, "emulated").tolist(), [[255]])
        self.assertEqual(f.scan_codes(f.encode_bases("ACGTACGT" + "TNG"), g, 8, masks, 8, "emulated").tolist(), [[255]])
        self.assertEqual(f.scan_codes(f.encode_bases("ACGNACGT" + "TGG"), g, 8, masks, 0, "emulated").tolist(), [[255]])
        self.assertEqual(f.scan_codes(f.encode_bases("ACGNACGT" + "TGG"), g, 8, masks, 1, "emulated").tolist(), [[1]])

    def test_result_does_not_depend_on_the_thread_layout(self):
        rng = random.Random(6)
        codes = f.encode_bases(rnd(rng, 301, "ACGGGT"))
        guides = np.array([f.encode_guide(rnd(rng, 10)), f.encode_guide(rnd(rng, 10))], np.uint8)
        masks = f.pam_masks_for("NGG")
        base = f.scan_codes(codes, guides, 10, masks, 4, "emulated", threads=1)
        for threads in (2, 7, 32, 100, 128, 301, 1024):
            np.testing.assert_array_equal(f.scan_codes(codes, guides, 10, masks, 4, "emulated", threads), base, threads)

    def test_every_thread_runs_once_with_its_own_index(self):
        seen = []

        def body(n):
            seen.append(f.cuda.grid(1))
        f.emulate_launch(body, 3, 4, (None,))
        self.assertEqual(seen, list(range(12)))

    def test_limits_and_degenerate_inputs(self):
        masks = f.pam_masks_for("NGG")
        g = np.array([f.encode_guide("ACGTACGT")], np.uint8)
        codes = f.encode_bases("ACGTACGTTGGACGTACGTAGG")
        self.assertEqual(f.scan_codes(codes, g, 8, masks, 254, "emulated").shape, (len(codes) - 10, 1))
        for bad in (-1, 255):
            with self.assertRaises(ValueError):
                f.scan_codes(codes, g, 8, masks, bad, "emulated")
        with self.assertRaises(ValueError):
            f.scan_codes(codes, g, 8, masks, 2, "emulated", threads=0)
        self.assertEqual(f.scan_codes(f.encode_bases("ACGT"), g, 8, masks, 2, "emulated").shape, (0, 1))   # too short
        self.assertEqual(f.scan_codes(codes, np.zeros((0, 8), np.uint8), 8, masks, 2, "emulated").shape, (len(codes) - 10, 0))

    def test_the_kernel_is_declared_with_cuda_jit(self):
        src = open(f.__file__).read()
        self.assertIn("@cuda.jit\ndef mismatch_kernel(", src)
        self.assertTrue(callable(f.mismatch_kernel.py_func))
        self.assertIn("cuda.grid(1)", src)

    def test_kernel_on_the_best_available_backend(self):
        backend = f.gpu_backend("auto")                                            # real GPU / simulator if present, else CPU emulation
        rng = random.Random(7)
        masks = f.pam_masks_for("NGG")
        guides = np.array([f.encode_guide(rnd(rng, 12))], np.uint8)
        codes = f.encode_bases(rnd(rng, 120, "ACGGGT"))
        np.testing.assert_array_equal(f.scan_codes(codes, guides, 12, masks, 5, backend),
                                      reference_scan(codes, guides, 12, masks, 5), f"backend {backend}")


class Backends(unittest.TestCase):
    def test_choosing_a_backend(self):
        self.assertEqual(f.gpu_backend("emulated"), "emulated")
        with mock.patch.object(f, "HAVE_NUMBA", False):
            self.assertEqual(f.gpu_backend("auto"), "emulated")
            with self.assertRaises(RuntimeError) as cm:
                f.gpu_backend("gpu")
        self.assertIn("no usable CUDA GPU", str(cm.exception))
        with self.assertRaises(ValueError):
            f.gpu_backend("tpu")

    def test_a_detected_gpu_is_chosen_for_auto_and_gpu(self):
        fake = mock.Mock()
        fake.is_available.return_value = True
        with mock.patch.object(f, "HAVE_NUMBA", True), mock.patch.object(f, "cuda", fake):
            self.assertEqual(f.gpu_backend("auto"), "gpu")
            self.assertEqual(f.gpu_backend("gpu"), "gpu")
        fake.is_available.side_effect = RuntimeError("driver problem")
        with mock.patch.object(f, "HAVE_NUMBA", True), mock.patch.object(f, "cuda", fake):
            self.assertEqual(f.gpu_backend("auto"), "emulated")                  # a broken driver falls back, not crashes

    def test_without_numba_the_decorator_still_works_but_cannot_launch(self):
        stub = f._NoNumbaCuda()

        def kernel(x):
            return x
        for held in (stub.jit(kernel), stub.jit()(kernel)):
            self.assertIs(held.py_func, kernel)
            with self.assertRaises(RuntimeError):
                held[1, 1]
        self.assertFalse(stub.is_available())

    def test_gpu_info_for_each_backend(self):
        info = f.gpu_info("emulated")
        self.assertEqual((info["backend"], info["device"]), ("emulated", None))
        fake = mock.Mock()
        fake.get_current_device.return_value = mock.Mock(compute_capability=(8, 6))
        fake.get_current_device.return_value.name = b"Fake RTX"
        with mock.patch.object(f, "cuda", fake):
            info = f.gpu_info("gpu")
        self.assertEqual((info["device"], info["compute_capability"]), ("Fake RTX", "8.6"))


class SearchMatchesTheReferences(unittest.TestCase):
    def test_equals_oracle_and_pure_python_baseline_on_random_genomes(self):
        rng = random.Random(2026)
        total = 0
        for trial in range(12):
            guides = [(f"g{i}", rnd(rng, 20)) for i in range(rng.randint(1, 3))]
            genome = planted_genome(rng, guides)
            for cfg in CONFIGS:
                want = f.oracle_search(genome, guides, cfg)
                self.assertEqual(f.packed_search(genome, guides, cfg), want)
                for size in (60, 1_000_000):
                    self.assertEqual(f.gpu_search(genome, guides, cfg, size, "emulated"), want, (trial, cfg["pam"], size))
                total += len(want)
        self.assertGreater(total, 60)

    def test_sites_across_chunk_borders_are_found_once(self):
        rng = random.Random(8)
        site = SPACER + "TGG"
        genome = {"c": rnd(rng, 49) + site + rnd(rng, 77) + f.revcomp(site) + rnd(rng, 100)}
        guides = [("g0", SPACER)]
        cfg = dict(f.CONFIGS["C2"], mm=2)
        full = f.oracle_search(genome, guides, cfg)
        self.assertTrue({("+", 49), ("-", 149)} <= {(h.strand, h.start) for h in full})
        for size in (7, 23, 50, 97, 150, 1000):
            got = f.gpu_search(genome, guides, cfg, size, "emulated")
            self.assertEqual(got, full, size)
            self.assertEqual(len(got), len({h.key() for h in got}))

    def test_guides_of_different_lengths_and_n_bases(self):
        rng = random.Random(9)
        guides = [("a", rnd(rng, 20)), ("b", rnd(rng, 18)), ("c", rnd(rng, 23))]
        genome = planted_genome(rng, guides, with_n=True)
        cfg = dict(f.CONFIGS["C2"], mm=3)
        self.assertEqual(f.gpu_search(genome, guides, cfg, 90, "emulated"), f.oracle_search(genome, guides, cfg))

    def test_refusals_and_empty_cases(self):
        genome, guides = {"c": "ACGT" * 30}, [("g", SPACER)]
        for name in ("C4", "C5"):
            with self.assertRaises(ValueError):
                f.gpu_search(genome, guides, f.CONFIGS[name], 50, "emulated")
        with self.assertRaises(ValueError):
            f.gpu_search(genome, [("g", "ACGTACGTACGTACGTACRN")], f.CONFIGS["C2"], 50, "emulated")
        self.assertEqual(f.gpu_search(genome, [], f.CONFIGS["C2"], 50, "emulated"), [])
        self.assertEqual(f.gpu_search({"empty": "", "tiny": "ACGT"}, guides, f.CONFIGS["C2"], 10, "emulated"), [])

    def test_progress_is_reported_once_per_chunk(self):
        rng = random.Random(10)
        genome = {"a": rnd(rng, 250), "b": rnd(rng, 90)}
        calls = []
        hits = f.gpu_search(genome, [("g", rnd(rng, 20))], f.CONFIGS["C2"], 100, "emulated",
                            progress=lambda *a: calls.append(a))
        self.assertEqual([c[0] for c in calls], list(range(1, 5)))                  # 3 chunks of 'a' + 1 of 'b'
        self.assertTrue(all(c[1] == 4 for c in calls))
        self.assertEqual(calls[-1][2], len(hits))
        self.assertEqual([c[2] for c in calls], sorted(c[2] for c in calls))
        self.assertTrue(all(b[3] >= a[3] for a, b in zip(calls, calls[1:])))

    def test_automatic_chunk_size_stays_within_bounds(self):
        self.assertEqual(f.auto_chunk_size(1), 1_000_000)
        self.assertEqual(f.auto_chunk_size(1000), 100_000)
        self.assertEqual(f.auto_chunk_size(10**9), 10_000)
        self.assertEqual(f.auto_chunk_size(0), 1_000_000)


class TerminalView(unittest.TestCase):
    def test_progress_line(self):
        self.assertTrue(f.format_progress(0, 10, 0, 0.0).startswith("[" + " " * 30 + "]   0%"))
        half = f.format_progress(5, 10, 7, 1.5)
        self.assertIn("50%", half)
        self.assertIn("chunk 5/10", half)
        self.assertIn("hits 7", half)
        self.assertIn("1.5s", half)
        done = f.format_progress(10, 10, 12, 2.0)
        self.assertTrue(done.startswith("[" + "=" * 30 + "] 100%"))
        self.assertEqual({len(l.split("]")[0]) for l in (f.format_progress(i, 10, 0, 0) for i in range(11))}, {31})
        self.assertIn("100%", f.format_progress(0, 0, 0, 0))                     # nothing to do counts as done

    def test_a_terminal_redraws_one_line_and_a_pipe_prints_five_steps(self):
        tty, pipe = io.StringIO(), io.StringIO()
        a, b = f.ProgressView(tty, tty=True), f.ProgressView(pipe, tty=False)
        for i in range(1, 11):
            a(i, 10, i, 0.1 * i)
            b(i, 10, i, 0.1 * i)
        self.assertEqual(tty.getvalue().count("\r"), 10)
        self.assertTrue(tty.getvalue().endswith("\n"))
        self.assertEqual(tty.getvalue().count("\n"), 1)                           # only the final newline
        lines = pipe.getvalue().strip().split("\n")
        self.assertLessEqual(len(lines), 6)
        self.assertIn("100%", lines[-1])


class RecordsAndCommand(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        mock_genome = f.generate_mock(scale=0.0001, chroms=["chr21", "chr22"], n_guides=3, copies=4, max_mm=4, bulge_copies=2)
        cls.genome = {c: ba.decode() for c, ba in mock_genome["genome"].items()}
        cls.mock = mock_genome

    def test_record_fields_and_correctness(self):
        rec, hits = f.run_gpu(self.genome, self.mock["guides"], "C2", "emulated", runs=2, warmup=0,
                              truth=self.mock["truth"])
        self.assertEqual((rec["tool"], rec["implementation"], rec["backend"]), ("geneweaver-cuda", "cuda-emulated", "emulated"))
        for key in ("runs_s", "median_search_s", "iqr_s", "gps", "total_hits", "gpu", "threads_per_block", "chunk_size", "hardware"):
            self.assertIn(key, rec)
        self.assertNotIn("peak_python_mem_mb", rec)                                # meaningless for a GPU run
        json.dumps(rec)
        self.assertEqual(rec["correctness"]["planted_found"], rec["correctness"]["planted_total"])
        self.assertEqual(hits, f.packed_search(self.genome, self.mock["guides"], f.CONFIGS["C2"]))
        self.assertEqual(rec["total_hits"], len(hits))

    def test_report_text(self):
        rec, _ = f.run_gpu(self.genome, self.mock["guides"], "C2", "emulated", runs=1, warmup=0, truth=self.mock["truth"])
        text = f.format_gpu(rec, saved="x.json")
        for part in ("emulated", "mismatch_kernel", "threads/block", "throughput", "planted sites found", "saved       : x.json"):
            self.assertIn(part, text)
        rec["backend"], rec["gpu"] = "gpu", dict(device="Fake RTX", simulator=False)
        self.assertIn("NVIDIA GPU: Fake RTX", f.format_gpu(rec))
        rec["gpu"]["simulator"] = True
        self.assertIn("Numba simulator", f.format_gpu(rec))
        rec["correctness"]["matches_baseline"] = False
        self.assertIn("DIFFERS", f.format_gpu(rec))

    def test_command_runs_records_and_shares_the_log_with_the_baseline(self):
        with tempfile.TemporaryDirectory() as td:
            gpu = ["gpu", "--n-guides", "3", "--runs", "2", "--warmup", "0", "--out-dir", td, "--verify"]
            rc, text = capture(f.main, gpu)
            self.assertEqual(rc, 0, text)
            for part in ("Day 6 - first CUDA kernel", "100%", "planted sites found", "identical to the Day 5 pure-Python baseline"):
                self.assertIn(part, text)
            files = [n for n in os.listdir(td) if n.endswith("_cuda-emulated.json")]
            self.assertEqual(len(files), 1)
            with open(os.path.join(td, files[0])) as fh:
                self.assertEqual(json.load(fh)["tool"], "geneweaver-cuda")
            rc, _ = capture(f.main, ["baseline", "--scale", "0.0001", "--chroms", "chr21,chr22", "--n-guides", "3",
                                     "--runs", "1", "--warmup", "0", "--out-dir", td])
            self.assertEqual(rc, 0)
            with open(os.path.join(td, "baseline_log.jsonl")) as fh:
                impls = [json.loads(l)["implementation"] for l in fh if l.strip()]
            self.assertEqual(impls, ["cuda-emulated", "packed"])                   # one log, two implementations to compare

    def test_command_errors_are_clean(self):
        with tempfile.TemporaryDirectory() as td:
            base = ["gpu", "--out-dir", td]
            rc, text = capture(f.main, base + ["--config", "C4"])
            self.assertEqual(rc, 2)
            self.assertIn("mismatch-only", text)
            with mock.patch.object(f, "HAVE_NUMBA", False):
                rc, text = capture(f.main, base + ["--backend", "gpu"])
            self.assertEqual(rc, 2)
            self.assertIn("no usable CUDA GPU", text)
            self.assertEqual(capture(f.main, base + ["--threads", "0", "--runs", "1", "--warmup", "0"])[0], 2)
            self.assertEqual(capture(f.main, base + ["--genome", os.path.join(td, "missing.fa"), "--parser", "builtin"])[0], 2)

    def test_the_gpu_command_runs_on_a_fasta_file(self):
        rng = random.Random(11)
        with tempfile.TemporaryDirectory() as td:
            fa = os.path.join(td, "g.fa")
            f.write_fasta({"c1": rnd(rng, 3000), "c2": rnd(rng, 2000)}, fa)
            rc, text = capture(f.main, ["gpu", "--genome", fa, "--n-guides", "2", "--runs", "1", "--warmup", "0",
                                        "--parser", "builtin", "--out-dir", td, "--verify", "--chunk-size", "1500"])
            self.assertEqual(rc, 0, text)
            self.assertIn("chunk 4/4", text)                                         # 3000/1500 + 2000/1500
            self.assertIn("identical to the Day 5", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)