"""Day 3 tests: Biopython parsing of genomes and guides in fasta.py.

Run from the folder that contains fasta.py:
    python -m unittest test_day3 -v

Tests marked 'Biopython' are skipped (not failed) when Biopython is not installed:
    pip install biopython
"""
import contextlib, gzip, io, os, tempfile, unittest
from unittest.mock import patch

import fasta as f

GENOME = ">chr1 test genome\nacgtNNRYACGT\n>chr2\nACGTACGT\n"
CLEAN = {"chr1": "ACGTNNNNACGT", "chr2": "ACGTACGT"}


def _write(path, text):
    with open(path, "w") as fh:
        fh.write(text)


def quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)


class TmpCase(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.addCleanup(self._td.cleanup)
        self.dir = self._td.name

    def path(self, name, text=None):
        p = os.path.join(self.dir, name)
        if text is not None:
            _write(p, text)
        return p


class Day3FormatAndGenome(TmpCase):
    def test_format_detection(self):
        for name, fmt in (("a.fa", "fasta"), ("a.FASTA", "fasta"), ("a.fna.gz", "fasta"),
                          ("a.fastq", "fastq"), ("a.fq.gz", "fastq"), ("a.gbk", "genbank")):
            self.assertEqual(f.detect_format(name), fmt, name)
        with self.assertRaises(ValueError):
            f.detect_format("genome.2bit")

    def test_names_softmask_and_iupac(self):
        g = f.read_genome(self.path("g.fa", GENOME), parser="builtin")
        self.assertEqual(g, CLEAN)

    def test_gzip_reads_identically(self):
        p = self.path("g.fa.gz")
        with gzip.open(p, "wt") as fh:
            fh.write(GENOME)
        self.assertEqual(f.read_genome(p, parser="builtin"), CLEAN)

    def test_agrees_with_day1_reader(self):
        p = self.path("g.fa", GENOME)
        self.assertEqual(f.read_genome(p, parser="builtin"), f.read_fasta(p))

    def test_duplicate_names_are_an_error(self):
        with self.assertRaises(ValueError):
            f.read_genome(self.path("d.fa", ">a\nACGT\n>a\nACGT\n"), parser="builtin")

    def test_empty_file_is_an_error(self):
        with self.assertRaises(ValueError):
            f.read_genome(self.path("e.fa", ""), parser="builtin")

    def test_builtin_parser_refuses_fastq(self):
        with self.assertRaises(ValueError):
            f.read_genome(self.path("r.fastq", "@r1\nACGT\n+\nIIII\n"), parser="builtin")

    def test_bad_parser_name(self):
        with self.assertRaises(ValueError):
            f.read_genome(self.path("g.fa", GENOME), parser="nope")

    def test_bio_parser_without_biopython_says_how_to_install(self):
        with patch.object(f, "HAVE_BIOPYTHON", False):
            with self.assertRaises(RuntimeError) as cm:
                f.read_genome(self.path("g.fa", GENOME), parser="bio")
        self.assertIn("pip install biopython", str(cm.exception))


class Day3GuidesAndReport(TmpCase):
    def test_guides_from_fasta(self):
        p = self.path("g.fa", ">g1\nGACGTCTAGCATCGATCGAA\n>g2 note\nacgtacgtacgtacgtacgt\n")
        self.assertEqual(f.read_guides_any(p, parser="builtin"),
                         [("g1", "GACGTCTAGCATCGATCGAA"), ("g2", "ACGTACGTACGTACGTACGT")])

    def test_guides_from_day1_tsv(self):
        p = self.path("g.tsv", "g0\tgacgtctagcatcgatcgaa\n")
        self.assertEqual(f.read_guides_any(p), [("g0", "GACGTCTAGCATCGATCGAA")])

    def test_guide_validation(self):
        bad = {"short": ">g\nACGT\n", "long": ">g\n" + "A" * 30 + "\n", "badchar": ">g\nACGTACGTACGTACGTAC1T\n",
               "dupid": ">g\nACGTACGTACGTACGTACGT\n>g\nACGTACGTACGTACGTACGA\n"}
        for name, text in bad.items():
            with self.assertRaises(ValueError, msg=name):
                f.read_guides_any(self.path(name + ".fa", text), parser="builtin")

    def test_iupac_guides_are_allowed(self):
        p = self.path("g.fa", ">g\nACGTACGTACGTACGTACRN\n")
        self.assertEqual(f.read_guides_any(p, parser="builtin")[0][1], "ACGTACGTACGTACGTACRN")

    def test_nonprimary_contigs_flagged(self):
        names = ["chr1", "chr1_KI270706v1_random", "chrUn_KI270302v1", "chr6_GL000250v2_alt", "HLA-A*01:01", "chrX"]
        self.assertEqual(f.flag_nonprimary(names), names[1:5])

    def test_genome_report_numbers(self):
        rep = f.genome_report(self.path("g.fa", GENOME), parser="builtin")
        self.assertEqual((rep["n_records"], rep["total_bp"]), (2, 20))
        self.assertEqual((rep["n_fraction"], rep["masked_fraction"]), (0.2, 0.2))
        self.assertEqual(len(rep["sha256"]), 64)
        self.assertEqual(rep["nonprimary"], [])

    def test_report_warns_about_alt_contigs(self):
        rep = f.genome_report(self.path("g.fa", ">chr1\nACGT\n>chr1_KI270706v1_random\nACGT\n"), parser="builtin")
        self.assertIn("WARNING", f.format_report(rep))


class Day3CommandLine(TmpCase):
    def test_inspect_command(self):
        g = self.path("g.fa", GENOME)
        gd = self.path("guides.fa", ">g1\nGACGTCTAGCATCGATCGAA\n")
        self.assertEqual(quiet(f.main, ["inspect", g, "--guides", gd, "--parser", "builtin"]), 0)

    def test_inspect_reports_errors_with_exit_code_2(self):
        self.assertEqual(quiet(f.main, ["inspect", self.path("missing.fa"), "--parser", "builtin"]), 2)
        self.assertEqual(quiet(f.main, ["inspect", self.path("e.fa", ""), "--parser", "builtin"]), 2)

    def test_inspect_works_on_the_mock_genome(self):
        out = os.path.join(self.dir, "mock")
        quiet(f.main, ["mockgenome", "--scale", "0.0001", "--chroms", "chr21,chr22", "--n-guides", "2", "--out-dir", out])
        gfa, guides = os.path.join(out, "genome.fa"), os.path.join(out, "guides.tsv")
        self.assertEqual(quiet(f.main, ["inspect", gfa, "--guides", guides, "--parser", "builtin"]), 0)


@unittest.skipUnless(f.HAVE_BIOPYTHON, "Biopython not installed (pip install biopython)")
class Day3Biopython(TmpCase):
    def test_biopython_reads_the_same_genome(self):
        p = self.path("g.fa", GENOME)
        self.assertEqual(f.read_genome(p, parser="bio"), CLEAN)
        self.assertEqual(f.read_genome(p, parser="bio"), f.read_genome(p, parser="builtin"))

    def test_biopython_reads_gzip(self):
        p = self.path("g.fa.gz")
        with gzip.open(p, "wt") as fh:
            fh.write(GENOME)
        self.assertEqual(f.read_genome(p, parser="bio"), CLEAN)

    def test_biopython_reads_fastq(self):
        p = self.path("r.fastq", "@r1 first read\nACGTN\n+\nIIIII\n@r2\nacgt\n+\nIIII\n")
        self.assertEqual(f.read_genome(p, parser="bio"), {"r1": "ACGTN", "r2": "ACGT"})

    def test_biopython_reads_guides(self):
        p = self.path("g.fa", ">g1\nGACGTCTAGCATCGATCGAA\n")
        self.assertEqual(f.read_guides_any(p, parser="bio"), [("g1", "GACGTCTAGCATCGATCGAA")])

    def test_crosscheck_command(self):
        self.assertEqual(quiet(f.main, ["inspect", self.path("g.fa", GENOME), "--crosscheck"]), 0)

    def test_report_names_biopython_as_parser(self):
        self.assertEqual(f.genome_report(self.path("g.fa", GENOME), parser="bio")["parser"], "biopython")

    def test_duplicate_names_are_an_error(self):
        with self.assertRaises(ValueError):
            f.read_genome(self.path("d.fa", ">a\nACGT\n>a\nACGT\n"), parser="bio")


if __name__ == "__main__":
    unittest.main(verbosity=2)
