"""Day 8 tests for multi-guide searches on both genome strands."""
import unittest

import fasta as f


class MultiGuideBothStrands(unittest.TestCase):
    def setUp(self):
        self.guides = [
            ("g20", "GACGTCTAGCATCGATCGAA"),
            ("g18", "TACGATCGGACTAACGTA"),
        ]
        sites = [
            ("g20", self.guides[0][1], "TGG", "+"),
            ("g20", self.guides[0][1], "AGG", "-"),
            ("g18", self.guides[1][1], "CGG", "+"),
            ("g18", self.guides[1][1], "TGG", "-"),
        ]
        parts = ["C" * 25]
        self.expected = set()
        start = len(parts[0])
        for guide_id, spacer, pam, strand in sites:
            target = spacer + pam
            parts.append(target if strand == "+" else f.revcomp(target))
            self.expected.add((guide_id, "chr1", start, strand))
            parts.append("C" * 31)
            start += len(target) + 31
        self.genome = {"chr1": "".join(parts)}
        self.config = dict(f.CONFIGS["C1"], mm=0)

    def test_multiple_guides_are_found_on_both_strands(self):
        hits = f.gpu_search(self.genome, self.guides, self.config, chunk_size=37, backend="emulated")
        found = {(hit.guide_id, hit.chrom, hit.start, hit.strand) for hit in hits}
        self.assertEqual(found, self.expected)
        self.assertEqual(hits, f.oracle_search(self.genome, self.guides, self.config))

    def test_chunk_size_does_not_change_multiguide_results(self):
        results = [f.gpu_search(self.genome, self.guides, self.config, size, "emulated")
                   for size in (37, 61, 1000)]
        self.assertEqual(results[0], results[1])
        self.assertEqual(results[0], results[2])


if __name__ == "__main__":
    unittest.main(verbosity=2)
