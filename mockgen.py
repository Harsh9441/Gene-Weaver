"""Mock human-like genome generator (Day 2).

Makes a FASTA file that looks like a real human genome in the ways that matter for GeneWeaver, with a
fixed seed so the same command always gives the same files:

  * human chromosome names and relative lengths (chr1, chr2, ... chrX, chrY)
  * about 41 % G+C, with the G+C level drifting along the chromosome (isochores) and few CG pairs (CpG depletion)
  * about 45 % lower-case (soft-masked repeats), like a real assembly
  * runs of N: telomeres at both ends, a centromere gap, and a few other gaps
  * hidden target sites at known positions (truth.json), and decoys that must NOT be found

It is made-up DNA. It is good for testing speed and correctness, not for biology.

    python -m geneweaver.mockgen --outdir data --size small
    python -m geneweaver.mockgen --outdir data --size all --chroms 24      # small, medium and large
"""
import argparse
import json
import math
import os
import random

import numpy as np

SIZES = {"small": 1_000_000, "medium": 50_000_000, "large": 500_000_000}

# Approximate hg38 chromosome lengths. Only their proportions are used.
HUMAN_LENGTHS = (
    ("chr1", 248_956_422), ("chr2", 242_193_529), ("chr3", 198_295_559), ("chr4", 190_214_555),
    ("chr5", 181_538_259), ("chr6", 170_805_979), ("chr7", 159_345_973), ("chr8", 145_138_636),
    ("chr9", 138_394_717), ("chr10", 133_797_422), ("chr11", 135_086_622), ("chr12", 133_275_309),
    ("chr13", 114_364_328), ("chr14", 107_043_718), ("chr15", 101_991_189), ("chr16", 90_338_345),
    ("chr17", 83_257_441), ("chr18", 80_373_285), ("chr19", 58_617_616), ("chr20", 64_444_167),
    ("chr21", 46_709_983), ("chr22", 50_818_468), ("chrX", 156_040_895), ("chrY", 57_227_415),
)
_BASES = np.frombuffer(b"ACGT", dtype=np.uint8)
_COMP = str.maketrans("ACGTN", "TGCAN")
_N, _C, _G, _T = ord("N"), ord("C"), ord("G"), ord("T")


def revcomp(seq):
    return seq.upper().translate(_COMP)[::-1]


def mutate(seq, k, rng):
    """Change k letters of seq (to make mismatches)."""
    s = list(seq)
    for p in rng.sample(range(len(s)), k):
        s[p] = rng.choice([c for c in "ACGT" if c != s[p]])
    return "".join(s)


def chromosome_plan(total, n_chroms, layout="human"):
    """[(name, length)]. 'human' uses hg38 proportions, 'equal' makes equal chromosomes."""
    if layout == "human":
        base = [HUMAN_LENGTHS[i] if i < len(HUMAN_LENGTHS) else (f"chr{i + 1}", 50_000_000) for i in range(n_chroms)]
    elif layout == "equal":
        base = [(f"chr{i + 1}", 1) for i in range(n_chroms)]
    else:
        raise ValueError("layout must be human or equal")
    s = sum(w for _, w in base)
    return [(name, max(2000, int(total * w / s))) for name, w in base]


def random_dna(n, nrng, gc_mean=0.44, cpg_keep=0.25, isochore=100_000, spread=0.05):
    """n random letters (as uint8 codes of A, C, G, T) with a drifting G+C level and depleted CpG."""
    isochore = min(isochore, max(n // 20, 1000))        # at least about 20 blocks per chromosome
    gcs = np.clip(gc_mean + spread * nrng.standard_normal(n // isochore + 1), 0.30, 0.62).astype(np.float32)
    out = np.empty(n, dtype=np.uint8)
    step = 1 << 20
    for pos in range(0, n, step):
        m = min(step, n - pos)
        gc = gcs[np.arange(pos, pos + m) // isochore]
        u = nrng.random(m, dtype=np.float32)
        code = np.where(u < gc / 2, 1, np.where(u < gc, 2, np.where(u < gc + (1 - gc) / 2, 0, 3)))
        out[pos:pos + m] = _BASES[code]
    if cpg_keep < 1.0:                                  # real DNA loses most CG pairs (C turns into T)
        cg = np.flatnonzero((out[:-1] == _C) & (out[1:] == _G))
        out[cg[nrng.random(len(cg)) > cpg_keep]] = _T
    return out


def soft_mask(arr, fraction, rng):
    """Make about `fraction` of the letters lower case, in repeat-like pieces (short, medium, long)."""
    n = len(arr)
    target = n * -math.log(1 - fraction)                # pieces overlap, so ask for a bit more than the fraction
    covered = 0
    while covered < target:
        r = rng.random()
        ln = rng.randint(20, 150) if r < 0.3 else rng.randint(280, 320) if r < 0.8 else rng.randint(1000, 6000)
        ln = min(ln, n)
        a = rng.randrange(0, n - ln + 1)
        arr[a:a + ln] |= 0x20
        covered += ln


def add_gaps(arr, rng):
    """Runs of N: telomeres at both ends, a centromere gap, and two smaller gaps."""
    n = len(arr)
    tel = min(max(n // 100, 20), 10_000)
    arr[:tel] = _N
    arr[n - tel:] = _N
    cen = min(max(n // 30, 30), 3_000_000)
    c = int(n * rng.uniform(0.35, 0.45))
    arr[c:c + cen] = _N
    for _ in range(2):
        ln = min(max(n // 200, 20), 50_000)
        a = rng.randrange(tel, n - tel - ln)
        arr[a:a + ln] = _N


def _stats(arr):
    counts = np.bincount(arr, minlength=256)
    n_n = int(counts[_N] + counts[_N | 0x20])
    lower = int(sum(counts[b | 0x20] for b in (65, 67, 71, 84)))
    gc = int(sum(counts[b] + counts[b | 0x20] for b in (_C, _G)))
    real = len(arr) - n_n
    return {"length": len(arr), "gc_percent": round(100 * gc / real, 1), "n_percent": round(100 * n_n / len(arr), 1),
            "soft_masked_percent": round(100 * lower / real, 1)}


def make_mock_genome(outdir, size="small", n_chroms=3, n_guides=5, sites_per_guide=3, max_planted_mm=3,
                     seed=1234, line_width=60, layout="human", gc=0.44, repeat_fraction=0.45, cpg_keep=0.25):
    """Write demo.fa, guides.txt, truth.json and summary.json into outdir. Returns (fasta, guides, truth) paths.

    size: "small" (1 million letters), "medium" (50 million), "large" (500 million) or a number of letters.
    Planted sites have 0 to max_planted_mm mismatches; truth.json lists guide, chromosome, start (0-based, forward
    strand, first letter of the 20-letter match), strand and mismatches of each one.
    """
    total = SIZES[size] if isinstance(size, str) else int(size)
    os.makedirs(outdir, exist_ok=True)
    rng = random.Random(seed)
    nrng = np.random.default_rng(seed)
    guides = ["".join(rng.choice("ACGT") for _ in range(20)) for _ in range(n_guides)]

    truth, summary = [], []
    fa_path = os.path.join(outdir, "demo.fa")
    with open(fa_path, "wb") as fa:
        for name, n in chromosome_plan(total, n_chroms, layout):
            arr = random_dna(n, nrng, gc, cpg_keep)
            soft_mask(arr, repeat_fraction, rng)
            add_gaps(arr, rng)                          # after masking, so N stays upper case
            used = []

            def free_spot():
                for _ in range(10_000):
                    p = rng.randrange(30, n - 60)
                    if all(abs(p - q) > 60 for q in used) and not (arr[p - 1:p + 24] == _N).any():
                        used.append(p)
                        return p
                raise ValueError(f"chromosome of {n} letters is too small for {n_guides} guides; use a bigger size")

            planted = 0
            for gi, g in enumerate(guides):
                for _ in range(sites_per_guide):
                    mm = rng.randint(0, max_planted_mm)
                    site = mutate(g, mm, rng) + rng.choice("ACGT") + "GG"
                    strand = rng.choice("+-")
                    p = free_spot()
                    arr[p:p + 23] = np.frombuffer((site if strand == "+" else revcomp(site)).encode(), dtype=np.uint8)
                    truth.append({"guide": f"g{gi + 1}", "chrom": name, "start": p if strand == "+" else p + 3,
                                  "strand": strand, "mismatches": mm})
                    planted += 1
                p = free_spot()                         # decoy: wrong PAM, must not be found
                arr[p:p + 23] = np.frombuffer((g + "TAG").encode(), dtype=np.uint8)
            summary.append({"name": name, **_stats(arr), "planted_sites": planted})
            fa.write(f">{name} mock chromosome\n".encode())
            raw = arr.tobytes()
            for i in range(0, n, line_width):
                fa.write(raw[i:i + line_width] + b"\n")

    guides_path = os.path.join(outdir, "guides.txt")
    with open(guides_path, "w") as f:
        for i, g in enumerate(guides, 1):
            f.write(f"g{i}\t{g}\n")
    truth_path = os.path.join(outdir, "truth.json")
    with open(truth_path, "w") as f:
        json.dump(sorted(truth, key=lambda t: (t["chrom"], t["start"])), f, indent=1)
    with open(os.path.join(outdir, "summary.json"), "w") as f:
        json.dump({"seed": seed, "letters": sum(c["length"] for c in summary), "chromosomes": summary}, f, indent=1)
    return fa_path, guides_path, truth_path


def print_summary(outdir):
    info = json.load(open(os.path.join(outdir, "summary.json")))
    print(f"{'chromosome':<12}{'length':>14}{'G+C %':>8}{'N %':>7}{'lower %':>9}{'sites':>7}")
    for c in info["chromosomes"]:
        print(f"{c['name']:<12}{c['length']:>14,}{c['gc_percent']:>8}{c['n_percent']:>7}{c['soft_masked_percent']:>9}{c['planted_sites']:>7}")
    print(f"{'total':<12}{info['letters']:>14,}")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Make a seeded mock human-like genome.")
    ap.add_argument("--outdir", default="data")
    ap.add_argument("--size", default="small", help="small, medium, large, all, or a number of letters")
    ap.add_argument("--chroms", type=int, default=3, help="how many chromosomes (up to 24 use human names and lengths)")
    ap.add_argument("--guides", type=int, default=5)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--layout", choices=["human", "equal"], default="human")
    ap.add_argument("--gc", type=float, default=0.44, help="G+C level before CpG loss (final is about 41 %%)")
    ap.add_argument("--repeat-fraction", type=float, default=0.45, help="share of lower-case letters")
    a = ap.parse_args(argv)
    sizes = list(SIZES) if a.size == "all" else [a.size if a.size in SIZES else int(a.size)]
    for size in sizes:
        out = os.path.join(a.outdir, size) if a.size == "all" else a.outdir
        paths = make_mock_genome(out, size, a.chroms, a.guides, seed=a.seed, layout=a.layout, gc=a.gc,
                                 repeat_fraction=a.repeat_fraction)
        print("\n".join(paths))
        print_summary(out)


if __name__ == "__main__":
    main()