"""GeneWeaver Week 1 - Day 1: setup + benchmark rules.  Day 2: mock human genome generator.  Day 3: Biopython parsing.  Day 4: chunking with border overlap.

  python fasta.py selftest                                   # all tests
  python fasta.py rules [--out BENCHMARK_RULES.md]           # Day 1: the benchmark rules
  python fasta.py setup --dir geneweaver_bench               # Day 1: project folders + rules + manifest
  python fasta.py mockgenome --scale 0.0005 --out-dir mock_genome --verify   # Day 2
  python fasta.py inspect genome.fa [--guides g.fa] [--parser auto|bio|builtin] [--crosscheck]   # Day 3
  python fasta.py chunks genome.fa --chunk-size 1000000 --config C4 [--guides g.fa --verify]   # Day 4
"""
from __future__ import annotations


# ======================================================================
# DAY 1 - Project setup and the benchmark rules (Rules v1.0), as data + a generated rules document.
# ======================================================================
import json, os

RULES_VERSION = "1.0"

GENOMES = {
    "G-small": "E. coli K-12 MG1655 (about 4.6 Mb): fast iteration and oracle comparison",
    "G-chr": "Human GRCh38 chr21 (about 46 Mb): mid-scale correctness",
    "G-human": "Human GRCh38 primary assembly (about 3.1 Gb): headline benchmark",
    "G-t2t": "T2T-CHM13 v2.0: repeat-heavy robustness (optional)",
    "G-mouse": "GRCm39: cross-species check (optional)",
    "G-mock": "Mock human genome from Day 2 (synthetic): harness testing only, not for headline numbers",
}
GUIDE_SETS = {
    "S-1": "1 guide, fixed published guide",
    "S-100": "100 random guides, fixed seed",
    "S-1k": "1,000 random guides, fixed seed",
    "S-10k": "10,000 random guides, fixed seed",
    "S-real": "at least 50 published guides with GUIDE-seq or CHANGE-seq validated off-targets",
    "S-mock": "guides planted into the mock genome (Day 2)",
}
REFERENCE_TOOLS = [
    ("Mismatch-only reference", "Cas-OFFinder (GPU, OpenCL)", "correctness and speed"),
    ("Mismatch and bulge reference", "CRISPRitz", "correctness and speed"),
    ("Ground-truth oracle", "single-threaded CPU brute force (Day 2)", "correctness only, small inputs"),
]
METRICS = [
    ("End-to-end time", "process start to output fully written, including genome load and index build"),
    ("Search time", "genome resident and guides loaded, until all hits are in host memory"),
    ("Kernel time", "sum of GPU kernel execution time from profiler events"),
    ("Throughput (GPS)", "guide-positions per second = G x 2L / t (both strands), using search time"),
    ("Hits per second", "total hits divided by search time"),
    ("Peak GPU memory", "maximum device memory used"),
    ("Peak host RSS", "maximum resident memory of the process"),
    ("Energy", "joules from GPU power sampling at 10 Hz or faster (optional)"),
]
PROTOCOL = dict(warmup_runs=1, timed_runs=5, max_deviation=0.20)
CORRECTNESS = dict(bulge_recall_small=1.0, bulge_recall_human=0.9999, spot_check_hits=1000)
ACCEPTANCE = [  # (target, threshold)
    ("Mismatch-only correctness", "100% identical to oracle and Cas-OFFinder"),
    ("Bulge correctness", "100% recall on G-chr, at least 99.99% on G-human"),
    ("C2 speed, S-1k, G-human", "at least as fast as Cas-OFFinder on the same GPU"),
    ("C4 speed, S-1k, G-human", "at least 5x faster than CRISPRitz on the same machine"),
    ("Memory", "peak GPU memory fits in 16 GB for S-10k on G-human"),
]


def rules_markdown() -> str:
    L = [f"# GeneWeaver Benchmark Rules v{RULES_VERSION}", "",
         "Rules for measuring the correctness and performance of GeneWeaver, a GPU-accelerated CRISPR "
         "off-target search engine, against reference tools.", "",
         "## 1. Purpose and scope", "",
         "These rules define how GeneWeaver and any comparison tool are tested so results are reproducible "
         "and comparable. They cover exhaustive off-target site enumeration (mismatches and bulges, both "
         "strands). Scoring (CFD, MIT, learned models) is a separate optional track.", "",
         "The benchmark answers three questions: is it correct, is it fast, and does it scale.", "",
         "## 2. Definitions", "",
         "- **Guide:** spacer of 17 to 25 nt (default 20) plus a PAM pattern (default NGG).",
         "- **Edit budget:** maximum mismatches, DNA bulges and RNA bulges allowed.",
         "- **Hit:** `(guide_id, chrom, start, strand, mismatches, dna_bulges, rna_bulges, target)`.",
         "- **Guide-position:** one guide against one genomic position on one strand.",
         "- **Throughput (GPS):** `G x 2L / t` with G guides, L genome length in bp, t seconds.", "",
         "## 3. Reference tools", "", "| Role | Tool | Used for |", "|---|---|---|"]
    L += [f"| {a} | {b} | {c} |" for a, b, c in REFERENCE_TOOLS]
    L += ["", "Record version, commit hash and command line of every tool. When tools disagree, the CPU oracle decides.", "",
          "## 4. Datasets", "", "### 4.1 Genomes", "", "| ID | Description |", "|---|---|"]
    L += [f"| {k} | {v} |" for k, v in GENOMES.items()]
    L += ["", "Use the primary assembly, record the FASTA SHA-256, treat lowercase soft-masking as sequence, "
          "and treat N as a base that never matches. Index or 2-bit conversion time is reported separately.", "",
          "### 4.2 Guide sets", "", "| ID | Description |", "|---|---|"]
    L += [f"| {k} | {v} |" for k, v in GUIDE_SETS.items()]
    L += ["", "Random guides come from the target genome (each has at least one perfect site), seed 20260101, "
          "with no homopolymer run of 6 or more. Guide files are published and hashed.", "",
          "## 5. Search configurations", "", "| ID | PAM | Mismatches | DNA bulges | RNA bulges |", "|---|---|---|---|---|"]
    L += [f"| {k} | {c['pam']} | up to {c['mm']} | {'up to ' + str(c['db']) if c['db'] else '0'} | "
          f"{'up to ' + str(c['rb']) if c['rb'] else '0'} |" for k, c in CONFIGS.items()]
    L += ["", "All configurations search both strands. Headline results use C2 and C4.", "",
          "## 6. Correctness rules", "",
          "- **Normalization:** 0-based half-open forward coordinates; strand `+` or `-`; for `-` the target is "
          "reverse-complemented into guide orientation; chromosome name is the FASTA header up to the first whitespace; duplicates removed.",
          "- **Mismatch-only (C1, C2, C3, C6):** hit sets must be identical to the CPU oracle on small genomes "
          "and to Cas-OFFinder on G-human. Any false negative or false positive is a failure.",
          f"- **Bulges (C4, C5):** recall of {CORRECTNESS['bulge_recall_small']:.0%} against the oracle on small genomes, "
          f"at least {CORRECTNESS['bulge_recall_human']:.2%} against CRISPRitz on G-human, plus a spot check of "
          f"{CORRECTNESS['spot_check_hits']} randomly sampled hits.",
          "- **Canonical alignment:** fewest total edits, then fewest bulges. Bulges are not allowed at either end of an alignment.",
          "- **Required edge cases:** sites at chromosome start and end, tile or batch boundaries, N runs, overlapping "
          "adjacent hits, IUPAC guides, palindromic guides, and a guide with zero hits.", "",
          "## 7. Performance rules", "", "| Metric | Definition |", "|---|---|"]
    L += [f"| {a} | {b} |" for a, b in METRICS]
    L += ["", "### Run protocol", "",
          "1. Run on an otherwise idle machine with no other GPU processes.",
          "2. Fix GPU clocks if possible, or record the observed range.",
          f"3. Do {PROTOCOL['warmup_runs']} untimed warm-up run.",
          f"4. Do at least {PROTOCOL['timed_runs']} timed runs. Report the median and interquartile range, and publish every raw timing.",
          "5. Report warm cache as the headline and cold cache as a separate row.",
          f"6. Re-run if any run differs from the median by more than {PROTOCOL['max_deviation']:.0%}, and note the cause.", "",
          "### Scaling studies", "",
          "- Guide scaling: S-1, S-100, S-1k, S-10k on G-human with C2.",
          "- Genome scaling: G-small, G-chr, G-human with S-1k and C2.",
          "- Budget scaling: C1, C2, C3 with S-1k on G-human.",
          "- Bulge cost: C2, C4, C5 with S-100 on G-human.", "",
          "## 8. Hardware and software reporting", "",
          "Report GPU model, VRAM, driver and CUDA or ROCm version; CPU, cores and RAM; storage type; OS and kernel; "
          "compiler and flags; GeneWeaver commit; and the command line of every run. Reference tools run on the same machine.", "",
          "## 9. Fairness rules", "",
          "- Every tool gets the same genome, guide file, edit budget, GPU and CPU thread count.",
          "- Reference tools use documented release flags and are not handicapped.",
          "- Output goes to local NVMe or tmpfs for all tools; output time counts toward end-to-end time.",
          "- No precomputed answers or genome-specific tuning.",
          "- Silent caps on hits per guide must be disclosed and the run marked as capped.", "",
          "## 10. Reporting format", "",
          "One JSON record per run with: tool, version, genome and its SHA-256, guide set, configuration, all raw timings, "
          "median search time, GPS, total hits, peak GPU and host memory, correctness result, and hardware.", "",
          "## 11. Validity and reproducibility", "",
          "- A performance number is valid only if the same run passes the correctness checks.",
          "- A run is disqualified if it breaks output normalization, has fewer than 5 timed runs, omits hardware reporting, or exceeds a resource cap without disclosure.",
          "- Publish guide files and hashes, build recipe, exact commands, seeds and raw outputs.",
          "- Any change to these rules increments the version; results from different versions are not compared.", "",
          "## 12. Acceptance targets (proposed)", "", "| Target | Threshold |", "|---|---|"]
    L += [f"| {a} | {b} |" for a, b in ACCEPTANCE]
    return "\n".join(L) + "\n"


def manifest_template() -> dict:
    return dict(
        genomes={"G-small": "data/ecoli.fa", "G-human": "data/grch38.fa"}, guide_source="G-human",
        workdir="results", version="<geneweaver commit hash>",
        tool_cmd=("geneweaver search --genome {genome} --guides {guides} --pam {pam} --mm {mm} "
                  "--dna-bulge {db} --rna-bulge {rb} --out {out}"),
        sizes={"S-1": 1, "S-100": 100, "S-1k": 1000, "S-10k": 10000},
        scale={"G-small": "small", "G-chr": "chr", "G-human": "human"})


def setup_project(root: str) -> list:
    """Create the project layout, the rules document and a manifest template. Returns created paths."""
    made = []
    for d in ("data", "results", "mock", "logs"):
        p = os.path.join(root, d)
        os.makedirs(p, exist_ok=True)
        made.append(p)
    files = {
        "BENCHMARK_RULES.md": rules_markdown(),
        "m.json": json.dumps(manifest_template(), indent=2) + "\n",
        os.path.join("data", "README.txt"): "Put genome FASTA files here (see m.json), e.g. ecoli.fa and grch38.fa.\n",
    }
    for name, text in files.items():
        p = os.path.join(root, name)
        if name == "m.json" and os.path.exists(p):         # never overwrite a manifest the user has edited
            continue
        with open(p, "w") as fh:
            fh.write(text)
        made.append(p)
    return made


# ======================================================================
# DAY 1 - Core types, configurations, FASTA I/O, canonical hit format (Rules s2, s5, s6.1).
# ======================================================================
import hashlib, re
from dataclasses import dataclass

# Rules s5: named search configurations
CONFIGS = {
    "C1": dict(pam="NGG", mm=3, db=0, rb=0),
    "C2": dict(pam="NGG", mm=4, db=0, rb=0),
    "C3": dict(pam="NGG", mm=6, db=0, rb=0),
    "C4": dict(pam="NGG", mm=4, db=1, rb=1),
    "C5": dict(pam="NGG", mm=5, db=2, rb=2),
    "C6": dict(pam="NRG", mm=4, db=0, rb=0),
}

IUPAC = {"A": "A", "C": "C", "G": "G", "T": "T", "R": "AG", "Y": "CT", "S": "CG",
         "W": "AT", "K": "GT", "M": "AC", "B": "CGT", "D": "AGT", "H": "ACT",
         "V": "ACG", "N": "ACGT"}
_COMP = str.maketrans("ACGTRYSWKMBDHVN", "TGCAYRSWMKVHDBN")


def revcomp(s: str) -> str:
    return s.translate(_COMP)[::-1]


def pam_matches(window: str, pam: str) -> bool:
    """Genome N (or any non-ACGT) never matches, even against pattern N."""
    return len(window) == len(pam) and all(b in "ACGT" and b in IUPAC[p] for b, p in zip(window, pam))


def read_fasta(path: str) -> dict[str, str]:
    """Upper-cases (soft-mask is sequence), maps non-ACGT to N, name = header up to first space."""
    genome, name, parts = {}, None, []
    with open(path) as fh:
        for line in fh:
            if line.startswith(">"):
                if name is not None:
                    genome[name] = re.sub("[^ACGT]", "N", "".join(parts).upper())
                name, parts = line[1:].split()[0], []
            else:
                parts.append(line.strip())
    if name is not None:
        genome[name] = re.sub("[^ACGT]", "N", "".join(parts).upper())
    return genome


def write_fasta(genome: dict[str, str], path: str) -> None:
    with open(path, "w") as fh:
        for name, seq in genome.items():
            fh.write(f">{name}\n")
            for i in range(0, len(seq), 80):
                fh.write(seq[i:i + 80] + "\n")


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass(frozen=True)
class Hit:
    """Canonical hit (Rules s2). start = 0-based forward-strand start of spacer+PAM window.
    target is in guide orientation (reverse-complemented for '-' hits), includes the PAM."""
    guide_id: str
    chrom: str
    start: int
    strand: str
    mm: int
    db: int
    rb: int
    target: str

    def key(self):
        return (self.guide_id, self.chrom, self.start, self.strand)


def normalize(hits) -> list[Hit]:
    """Rules s6.1: dedupe + stable order."""
    # full tie-breaker: several hits can share (guide, chrom, start, strand) with different edits/lengths
    return sorted(set(hits), key=lambda h: (h.guide_id, h.chrom, h.start, h.strand, h.mm, h.db, h.rb, h.target))


def write_hits(hits, path: str) -> None:
    with open(path, "w") as fh:
        for h in hits:
            fh.write(f"{h.guide_id}\t{h.chrom}\t{h.start}\t{h.strand}\t{h.mm}\t{h.db}\t{h.rb}\t{h.target}\n")


def read_hits(path: str) -> list[Hit]:
    out = []
    with open(path) as fh:
        for line in fh:
            if line.strip() and not line.startswith("#"):
                g, c, s, st, mm, db, rb, t = line.rstrip("\n").split("\t")
                out.append(Hit(g, c, int(s), st, int(mm), int(db), int(rb), t))
    return normalize(out)


def write_guides(guides, path: str) -> None:
    with open(path, "w") as fh:
        for gid, seq in guides:
            fh.write(f"{gid}\t{seq}\n")


def read_guides(path: str):
    with open(path) as fh:
        return [tuple(l.split()) for l in fh if l.strip()]


# ======================================================================
# DAY 2 - Mock human genome generator.
# ======================================================================
import json, os, random, re

GRCH38_MB = {"chr1": 248.96, "chr2": 242.19, "chr3": 198.30, "chr4": 190.21, "chr5": 181.54,
             "chr6": 170.81, "chr7": 159.35, "chr8": 145.14, "chr9": 138.39, "chr10": 133.80,
             "chr11": 135.09, "chr12": 133.28, "chr13": 114.36, "chr14": 107.04, "chr15": 101.99,
             "chr16": 90.34, "chr17": 83.26, "chr18": 80.37, "chr19": 58.62, "chr20": 64.44,
             "chr21": 46.71, "chr22": 50.82, "chrX": 156.04, "chrY": 57.23}
_MIN_CHROM = 20_000
_IUPAC_MOCK = {"A": "A", "C": "C", "G": "G", "T": "T", "R": "AG", "Y": "CT", "N": "ACGT"}
_RC_TABLE = str.maketrans("ACGTacgtN", "TGCAtgcaN")


def _rc(s):
    return s.translate(_RC_TABLE)[::-1]


def _random_seq(rng, n, gc):
    w = [(1 - gc) / 2, gc / 2, gc / 2, (1 - gc) / 2]
    return "".join(rng.choices("ACGT", w, k=n))


def _mutate(rng, s, div):
    """Substitutions at rate div, small indels at div/10."""
    out = []
    for b in s:
        r = rng.random()
        if r < div:
            out.append(rng.choice([x for x in "ACGT" if x != b]))
        elif r < div * 1.05:
            continue                                   # deletion
        elif r < div * 1.10:
            out.append(b + rng.choice("ACGT"))         # insertion
        else:
            out.append(b)
    return "".join(out)


def _background(rng, n):
    """Isochore-structured, CpG-depleted unique sequence."""
    parts, total = [], 0
    while total < n:
        seg = min(n - total, rng.randint(50_000, 300_000))
        gc = rng.choices([0.40, 0.44, 0.49, 0.56], [0.30, 0.35, 0.22, 0.13])[0]
        parts.append(_random_seq(rng, seg, gc))
        total += seg
    return "".join(parts)[:n]


def _families(rng):
    alu = _random_seq(rng, 300, 0.55)
    l1 = _random_seq(rng, 6000, 0.40)
    other = [_random_seq(rng, rng.randint(400, 1500), 0.42) for _ in range(8)]
    return dict(alu=[_mutate(rng, alu, 0.03) for _ in range(3)], l1=_mutate(rng, l1, 0.02), other=other)


def _insert_repeats(rng, ba, fam):
    L = len(ba)

    def alu():
        e = _mutate(rng, rng.choice(fam["alu"]), rng.uniform(0.04, 0.15)) + "A" * rng.randint(10, 30)
        return e

    def l1():
        n = min(len(fam["l1"]), rng.randint(500, 6000))
        return _mutate(rng, fam["l1"][-n:], rng.uniform(0.03, 0.20))      # 5'-truncated

    def other():
        return _mutate(rng, rng.choice(fam["other"]), rng.uniform(0.08, 0.25))

    def micro():
        unit = _random_seq(rng, rng.randint(1, 6), 0.4)
        return unit * rng.randint(10, 40)

    stats = {}
    for name, frac, build in (("alu", 0.10, alu), ("l1", 0.17, l1), ("other", 0.08, other), ("micro", 0.003, micro)):
        target, done = int(frac * L), 0
        while done < target:
            e = build()
            if rng.random() < 0.5:
                e = _rc(e)
            if len(e) >= L:
                break
            pos = rng.randrange(0, L - len(e))
            ba[pos:pos + len(e)] = e.lower().encode()
            done += len(e)
        stats[name] = done
    return stats


def _centromere_and_gaps(rng, ba):
    L = len(ba)
    cen_len = max(2000, int(0.03 * L))
    cen_start = int(L * rng.uniform(0.35, 0.55))
    cen_start = min(cen_start, L - cen_len - 1)
    mono = _random_seq(rng, 171, 0.38)
    hor = "".join(_mutate(rng, mono, 0.20) for _ in range(8))              # 8-mer higher-order repeat
    arr = []
    while sum(map(len, arr)) < cen_len:
        arr.append(_mutate(rng, hor, 0.02))
    ba[cen_start:cen_start + cen_len] = "".join(arr)[:cen_len].lower().encode()
    gaps = []
    tel = max(50, min(10_000, L // 200))
    gaps += [(0, tel), (L - tel, L)]
    gaps.append((cen_start, cen_start + max(100, L // 500)))
    for _ in range(rng.randint(2, 5)):
        g = max(50, L // 5000)
        p = rng.randrange(tel + 1, L - tel - g - 1)
        gaps.append((p, p + g))
    for a, b in gaps:
        ba[a:b] = b"N" * (b - a)


def _deplete_cpg_bytes(rng, ba, keep=0.22):
    """CpG depletion over the finished chromosome (repeats included), preserving case."""
    def f(m):
        if rng.random() < keep:
            return m.group(0)
        low = m.group(0) == b"cg"
        r = rng.choice((b"tg", b"ca") if low else (b"TG", b"CA"))
        return r
    return bytearray(re.sub(rb"CG|cg", f, bytes(ba)))


def _gen_chrom(rng, n, fam):
    ba = bytearray(_background(rng, n).encode())
    rep = _insert_repeats(rng, ba, fam)
    _centromere_and_gaps(rng, ba)
    return _deplete_cpg_bytes(rng, ba), rep


# ------------------------------------------------------------------ planting known sites
def _resolve_pam(rng, pam):
    return "".join(rng.choice(_IUPAC_MOCK[c]) for c in pam)


def _random_guide(rng, length=20):
    while True:
        g = _random_seq(rng, length, 0.5)
        if not re.search(r"(.)\1{5,}", g):
            return g


def _variant(rng, spacer, mm, kind):
    """Return (variant_spacer, mm, db, rb). kind: None | 'dna' | 'rna'."""
    s = list(spacer)
    for p in rng.sample(range(len(s)), mm):
        s[p] = rng.choice([x for x in "ACGT" if x != s[p]])
    if kind == "dna":                                   # extra base in target
        i = rng.randint(3, len(s) - 4)
        s.insert(i, rng.choice("ACGT")); return "".join(s), mm, 1, 0
    if kind == "rna":                                   # base missing in target
        i = rng.randint(3, len(s) - 4)
        del s[i]; return "".join(s), mm, 0, 1
    return "".join(s), mm, 0, 0


def plant_sites(rng, genome, guides, copies, max_mm, bulge_copies, pam="NGG"):
    """Overwrite random N-free regions with known off-target sites. Returns truth rows
    (guide_id, chrom, start, strand, mm, db, rb, target) in the canonical hit format
    (start = forward start of spacer+PAM window; target in guide orientation, incl. PAM)."""
    names = list(genome)
    weights = [len(genome[c]) for c in names]
    used, truth = {c: [] for c in names}, []

    def place(window):
        for _ in range(2000):
            c = rng.choices(names, weights)[0]
            ba = genome[c]
            s = rng.randrange(40, len(ba) - len(window) - 40)
            if any(not (s + len(window) + 30 < a or s > b + 30) for a, b in used[c]):
                continue
            if b"N" in ba[s - 5:s + len(window) + 5]:
                continue
            return c, s
        raise RuntimeError("could not find a free site for all planted sequences; use a larger --scale or fewer guides/copies")

    for gid, spacer in guides:
        jobs = [(0 if k == 0 else rng.randint(0, max_mm), None) for k in range(copies)]
        jobs += [(rng.randint(0, 1), "dna" if b % 2 == 0 else "rna") for b in range(bulge_copies)]
        for mm, kind in jobs:
            var, m, d, r = _variant(rng, spacer, mm, kind)
            target = var + _resolve_pam(rng, pam)
            strand = rng.choice("+-")
            window = target if strand == "+" else _rc(target)
            c, s = place(window)
            genome[c][s:s + len(window)] = window.encode()
            used[c].append((s, s + len(window)))
            truth.append((gid, c, s, strand, m, d, r, target))
    return sorted(truth, key=lambda t: (t[0], t[1], t[2]))


# ------------------------------------------------------------------ top level
def _stats(genome):
    tot = sum(len(s) for s in genome.values())
    n = sum(s.count(b"N") for s in genome.values())
    low = sum(sum(1 for b in s if 97 <= b <= 122) for s in genome.values())
    up = b"".join(s for s in genome.values()).upper()
    a, c, g, t = (up.count(x) for x in b"ACGT")
    acgt = a + c + g + t
    cg = up.count(b"CG")
    return dict(total_bp=tot, n_fraction=round(n / tot, 4), masked_fraction=round(low / tot, 4),
                gc_percent=round(100 * (c + g) / acgt, 2),
                cpg_obs_exp=round(cg / (c * g / acgt), 3) if c and g else None)


def generate_mock(scale=0.0005, chroms=None, seed=20260101, n_guides=10, copies=5,
                  max_mm=4, bulge_copies=2, guide_len=20, pam="NGG"):
    if scale <= 0:
        raise ValueError("scale must be greater than zero")
    bad = [c for c in (chroms or []) if c not in GRCH38_MB]
    if bad:
        raise ValueError(f"unknown chromosome(s): {', '.join(bad)} (use chr1-chr22, chrX, chrY)")
    if n_guides < 1 or copies < 1:
        raise ValueError("need at least 1 guide and 1 copy per guide")
    if bulge_copies < 0:
        raise ValueError("bulge copies cannot be negative")
    if not 0 <= max_mm <= min(10, guide_len):
        raise ValueError("max mismatches for planted sites must be between 0 and 10")
    names = chroms or list(GRCH38_MB)
    est_mb = sum(max(_MIN_CHROM / 1e6, GRCH38_MB[c] * scale) for c in names)
    if est_mb > 200:
        raise ValueError(f"mock genome would be about {est_mb:.0f} Mb; the pure-Python generator is meant for small "
                         "scales (try --scale 0.001)")
    rng = random.Random(seed)
    fam = _families(rng)
    genome = {}
    for name in names:
        n = max(_MIN_CHROM, int(GRCH38_MB[name] * 1e6 * scale))
        genome[name], _ = _gen_chrom(rng, n, fam)
    guides = [(f"g{i:05d}", _random_guide(rng, guide_len)) for i in range(n_guides)]
    truth = plant_sites(rng, genome, guides, copies, max_mm, bulge_copies, pam)
    return dict(genome=genome, guides=guides, truth=truth, stats=_stats(genome))


def _write_fasta(genome, path):
    with open(path, "w") as fh:
        for name, ba in genome.items():
            s = ba.decode()
            fh.write(f">{name}\n")
            for i in range(0, len(s), 80):
                fh.write(s[i:i + 80] + "\n")


def write_mock(mock, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    fa = os.path.join(out_dir, "genome.fa")
    _write_fasta(mock["genome"], fa)
    with open(os.path.join(out_dir, "guides.tsv"), "w") as fh:
        fh.writelines(f"{g}\t{s}\n" for g, s in mock["guides"])
    with open(os.path.join(out_dir, "planted_truth.tsv"), "w") as fh:
        fh.writelines("\t".join(map(str, row)) + "\n" for row in mock["truth"])
    work = os.path.join(out_dir, "bench_out")
    os.makedirs(work, exist_ok=True)
    with open(os.path.join(work, "S-mock.tsv"), "w") as fh:        # pre-made guide set for the benchmark
        fh.writelines(f"{g}\t{s}\n" for g, s in mock["guides"])
    manifest = dict(genomes={"G-mock": fa.replace("\\", "/")}, workdir=work.replace("\\", "/"),
                    version="mock", sizes={"S-mock": len(mock["guides"])}, scale={"G-mock": "small"})
    with open(os.path.join(out_dir, "m.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    return fa


def verify_planted(genome, guides, truth, search_fn, cfg):
    """Every planted site must be found by search_fn(genome_str_dict, guides, cfg) -> hits with .key().
    Planted-only check: extra hits from chance matches are expected and fine."""
    strs = {c: ba.decode() for c, ba in genome.items()}
    found = {h.key() for h in search_fn(strs, guides, cfg)}
    missing = [t for t in truth if (t[0], t[1], t[2], t[3]) not in found]
    return dict(planted=len(truth), found=len(truth) - len(missing), missing=missing[:5])


# ======================================================================
# DAY 2 (verification) - Guide-set generation (Rules s4.2) and the CPU brute-force oracle (Rules s3, s6.3).
# ======================================================================
import random, re

SEED = 20260101


def make_guides(genome, n, seed=SEED, length=20, pam="NGG"):
    """Random guides taken from the genome itself (>=1 perfect site), no homopolymer run >=6."""
    rng = random.Random(seed)
    chroms = [c for c in genome if len(genome[c]) > length + len(pam)]
    weights = [len(genome[c]) for c in chroms]
    guides, seen, tries = [], set(), 0
    while len(guides) < n and tries < n * 20000:
        tries += 1
        c = rng.choices(chroms, weights)[0]
        seq = genome[c]
        s = rng.randrange(len(seq) - length - len(pam) + 1)
        spacer, p = seq[s:s + length], seq[s + length:s + length + len(pam)]
        if (not pam_matches(p, pam) or set(spacer) - set("ACGT")
                or re.search(r"(.)\1{5,}", spacer) or spacer in seen):
            continue
        seen.add(spacer)
        guides.append((f"g{len(guides):05d}", spacer))
    if len(guides) < n:
        raise RuntimeError(f"only found {len(guides)}/{n} valid guides")
    return guides


def count_mm(g: str, t: str, limit: int):
    m = 0
    for a, b in zip(g, t):
        if b not in IUPAC[a]:           # genome N never matches
            m += 1
            if m > limit:
                return None
    return m


def align(g: str, t: str, cfg):
    """Edit alignment of guide g to target spacer t within budget (mm, db, rb).
    DNA bulge = extra base in target; RNA bulge = extra base in guide.
    Bulges are not allowed at either end of the alignment. Returns best (mm, db, rb) or None.
    Canonical tie-break (Rules s6.3): fewest total edits, then fewest bulges."""
    L, n = len(g), len(t)
    M, D, R = cfg["mm"], cfg["db"], cfg["rb"]
    dp = [[set() for _ in range(n + 1)] for _ in range(L + 1)]
    dp[0][0].add((0, 0, 0))
    for i in range(L + 1):
        for j in range(n + 1):
            for m, d, r in dp[i][j]:
                if i < L and j < n:
                    nm = m + (t[j] not in IUPAC[g[i]])
                    if nm <= M:
                        dp[i + 1][j + 1].add((nm, d, r))
                if d < D and 0 < i < L and 0 < j < n - 1:
                    dp[i][j + 1].add((m, d + 1, r))
                if r < R and 0 < j < n and 0 < i < L - 1:
                    dp[i + 1][j].add((m, d, r + 1))
    states = dp[L][n]
    return min(states, key=lambda x: (sum(x), x[1] + x[2])) if states else None


def _scan(chrom, strand, seq, total_len, guides, cfg, out, force_dp):
    pam, pl = cfg["pam"], len(cfg["pam"])
    use_dp = force_dp or cfg["db"] > 0 or cfg["rb"] > 0
    for gid, g in guides:
        L = len(g)
        lens = range(L - cfg["rb"], L + cfg["db"] + 1) if use_dp else (L,)
        for s in range(len(seq)):
            best = None
            for n in lens:
                end = s + n
                if end + pl > len(seq) or not pam_matches(seq[end:end + pl], pam):
                    continue
                if use_dp:
                    res = align(g, seq[s:end], cfg)
                else:
                    m = count_mm(g, seq[s:end], cfg["mm"])
                    res = None if m is None else (m, 0, 0)
                if res and (best is None or (sum(res), res[1] + res[2]) < (sum(best[0]), best[0][1] + best[0][2])):
                    best = (res, n)
            if best:
                (m, d, r), n = best
                target = seq[s:s + n + pl]
                start = s if strand == "+" else total_len - (s + n + pl)
                out.append(Hit(gid, chrom, start, strand, m, d, r, target))


def oracle_search(genome, guides, cfg, force_dp=False):
    hits = []
    for chrom, seq in genome.items():
        _scan(chrom, "+", seq, len(seq), guides, cfg, hits, force_dp)
        _scan(chrom, "-", revcomp(seq), len(seq), guides, cfg, hits, force_dp)
    return normalize(hits)



# ======================================================================
# DAY 3 - Biopython parsing: genomes and guides via Bio.SeqIO (built-in fallback), QC report, primary-assembly check.
# ======================================================================
import gzip, os, re

try:                                    # Biopython is optional: pip install biopython
    from Bio import SeqIO
    HAVE_BIOPYTHON = True
except ImportError:
    SeqIO = None
    HAVE_BIOPYTHON = False

_EXT_FMT = {".fa": "fasta", ".fasta": "fasta", ".fna": "fasta", ".fas": "fasta",
            ".fq": "fastq", ".fastq": "fastq", ".gb": "genbank", ".gbk": "genbank", ".genbank": "genbank"}
# Rules s4.1: primary assembly only (no alt / random / decoy / unplaced contigs unless stated)
_NONPRIMARY = re.compile(r"(_alt$|_random$|_decoy$|_fix$|^chrUn|^HLA-|^chrEBV$)", re.I)


def detect_format(path):
    p = path.lower()
    if p.endswith(".gz"):
        p = p[:-3]
    ext = os.path.splitext(p)[1]
    if ext not in _EXT_FMT:
        raise ValueError(f"unknown sequence file type: {path} (use .fa/.fasta/.fna, .fastq, .gb/.gbk; .gz allowed)")
    return _EXT_FMT[ext]


def _open_text(path):
    return gzip.open(path, "rt") if path.lower().endswith(".gz") else open(path)


def _use_bio(parser):
    if parser not in ("auto", "bio", "builtin"):
        raise ValueError("parser must be auto, bio or builtin")
    if parser == "bio" and not HAVE_BIOPYTHON:
        raise RuntimeError("Biopython is not installed. Run: pip install biopython")
    return HAVE_BIOPYTHON if parser == "auto" else parser == "bio"


def _check_parser(fmt, use_bio):
    if fmt != "fasta" and not use_bio:
        raise ValueError(f"the built-in parser reads FASTA only; install Biopython to read {fmt} (pip install biopython)")


def _iter_records(path, fmt, use_bio):
    """Yield (id, raw_sequence). id = first whitespace-delimited token of the header, as Biopython does."""
    if use_bio:
        with _open_text(path) as fh:
            for rec in SeqIO.parse(fh, fmt):
                yield rec.id, str(rec.seq)
        return
    name, parts = None, []
    with _open_text(path) as fh:
        for line in fh:
            line = line.strip()
            if line.startswith(">"):
                if name is not None:
                    yield name, "".join(parts)
                head = line[1:].split()
                name, parts = (head[0] if head else ""), []
            elif line and name is not None:
                parts.append(line)
    if name is not None:
        yield name, "".join(parts)


def _clean(seq):
    """Day 1 rule: soft-masked bases count as sequence (upper-cased); anything that is not ACGT becomes N."""
    return re.sub("[^ACGT]", "N", seq.upper())


def read_genome(path, parser="auto", fmt=None):
    """Genome as {name: ACGTN string}. parser: auto (Biopython if installed), bio, or builtin."""
    use_bio = _use_bio(parser)
    fmt = fmt or detect_format(path)
    _check_parser(fmt, use_bio)
    genome = {}
    for name, raw in _iter_records(path, fmt, use_bio):
        if name in genome:
            raise ValueError(f"duplicate sequence name {name!r} in {path}")
        genome[name] = _clean(raw)
    if not genome:
        raise ValueError(f"no sequences found in {path}")
    return genome


def read_guides_any(path, parser="auto", length_range=(17, 25)):
    """Guides from a FASTA file (via Biopython when available) or the Day 1 two-column TSV (id<TAB>spacer)."""
    if path.lower().endswith((".tsv", ".txt", ".tab")):
        raw = read_guides(path)
    else:
        use_bio = _use_bio(parser)
        fmt = detect_format(path)
        _check_parser(fmt, use_bio)
        raw = list(_iter_records(path, fmt, use_bio))
    out, seen = [], set()
    lo, hi = length_range
    for gid, seq in raw:
        seq = seq.upper()
        if gid in seen:
            raise ValueError(f"duplicate guide id {gid!r}")
        seen.add(gid)
        bad = set(seq) - set(IUPAC)
        if bad:
            raise ValueError(f"guide {gid}: invalid characters {sorted(bad)}")
        if not lo <= len(seq) <= hi:
            raise ValueError(f"guide {gid}: length {len(seq)} is outside {lo}-{hi}")
        out.append((gid, seq))
    if not out:
        raise ValueError(f"no guides found in {path}")
    return out


def flag_nonprimary(names):
    return [n for n in names if _NONPRIMARY.search(n)]


def genome_report(path, parser="auto", fmt=None):
    """QC summary of a genome file; flags non-primary contigs (Rules s4.1) and records the SHA-256 (Rules s4.1)."""
    use_bio = _use_bio(parser)
    fmt = fmt or detect_format(path)
    _check_parser(fmt, use_bio)
    recs, seen = [], set()
    total = nn = gc = acgt = low = 0
    for name, raw in _iter_records(path, fmt, use_bio):
        if name in seen:
            raise ValueError(f"duplicate sequence name {name!r} in {path}")
        seen.add(name)
        clean = _clean(raw)
        a, c, g, t = (clean.count(x) for x in "ACGT")
        recs.append((name, len(raw)))
        total += len(raw); nn += clean.count("N"); gc += c + g; acgt += a + c + g + t
        low += sum(1 for ch in raw if ch.islower())
    if not recs:
        raise ValueError(f"no sequences found in {path}")
    return dict(parser="biopython" if use_bio else "builtin", format=fmt, n_records=len(recs), total_bp=total,
                n_fraction=round(nn / total, 4), masked_fraction=round(low / total, 4),
                gc_percent=round(100 * gc / acgt, 2) if acgt else None, records=recs,
                nonprimary=flag_nonprimary([r[0] for r in recs]), sha256=sha256(path))


def format_report(rep, top=8):
    L = [f"parser      : {rep['parser']} ({rep['format']})",
         f"sequences   : {rep['n_records']}   total {rep['total_bp']:,} bp",
         f"N fraction  : {rep['n_fraction']}   soft-masked: {rep['masked_fraction']}   GC: "
         + (f"{rep['gc_percent']}%" if rep["gc_percent"] is not None else "n/a (no A/C/G/T)"),
         f"sha256      : {rep['sha256']}"]
    for name, n in rep["records"][:top]:
        L.append(f"  {name:<24}{n:>14,} bp")
    if rep["n_records"] > top:
        L.append(f"  ... and {rep['n_records'] - top} more")
    if rep["nonprimary"]:
        L.append(f"WARNING: {len(rep['nonprimary'])} non-primary contigs (alt/random/decoy/unplaced): "
                 f"{', '.join(rep['nonprimary'][:4])}{' ...' if len(rep['nonprimary']) > 4 else ''}")
        L.append("         Rules s4.1 says to use the primary assembly only unless stated.")
    return "\n".join(L)


def crosscheck(path):
    """True/False if Biopython and the built-in parser agree on a FASTA file; None if Biopython is missing."""
    if not HAVE_BIOPYTHON:
        return None
    return read_genome(path, "bio") == read_genome(path, "builtin")


# ======================================================================
# DAY 4 - Chunking with border overlap: split a chromosome into tiles that each read an overlap halo, so no site is lost at a border.
# ======================================================================
def required_overlap(cfg, guide_len=20):
    """Bases of halo needed on each side of a chunk: the longest possible site window minus 1.
    Longest window = guide + DNA bulges + PAM. (RNA bulges only shorten a site.)"""
    return guide_len + cfg["db"] + len(cfg["pam"]) - 1


def plan_chunks(length, chunk_size, overlap):
    """Chunks as (own_start, own_end, read_start, read_end), half-open, in forward coordinates.
    The owned ranges tile [0, length) exactly once. Each chunk READS `overlap` extra bases on both sides."""
    if length < 0 or chunk_size < 1 or overlap < 0:
        raise ValueError("need length >= 0, chunk_size >= 1, overlap >= 0")
    plan = []
    for own_s in range(0, length, chunk_size):
        own_e = min(length, own_s + chunk_size)
        plan.append((own_s, own_e, max(0, own_s - overlap), min(length, own_e + overlap)))
    return plan


def chunk_stats(lengths, chunk_size, overlap):
    """Summary over chromosomes: number of chunks, bases owned, bases read, and the overlap overhead."""
    n = owned = read = 0
    for L in lengths:
        for o_s, o_e, r_s, r_e in plan_chunks(L, chunk_size, overlap):
            n += 1; owned += o_e - o_s; read += r_e - r_s
    return dict(chunks=n, owned_bp=owned, read_bp=read, overhead=(read / owned - 1) if owned else 0.0)


def chunked_search(genome, guides, cfg, chunk_size, overlap=None, search_fn=None):
    """Search chunk by chunk and merge. A site is reported only by the chunk that owns its forward start,
    so the result equals the unchunked search (given overlap >= required_overlap), with no duplicates.
    search_fn(genome_dict, guides, cfg) -> hits; defaults to the CPU oracle (stand-in for a GPU tile kernel)."""
    search_fn = search_fn or oracle_search
    if not guides:
        return []
    if overlap is None:
        overlap = required_overlap(cfg, max(len(g) for _, g in guides))
    hits = []
    for chrom, seq in genome.items():
        for own_s, own_e, read_s, read_e in plan_chunks(len(seq), chunk_size, overlap):
            for h in search_fn({chrom: seq[read_s:read_e]}, guides, cfg):
                start = read_s + h.start                       # back to chromosome coordinates
                if own_s <= start < own_e:
                    hits.append(Hit(h.guide_id, chrom, start, h.strand, h.mm, h.db, h.rb, h.target))
    return normalize(hits)


def format_chunk_plan(genome_lengths, chunk_size, cfg, guide_len, cfg_name=""):
    ov = required_overlap(cfg, guide_len)
    st = chunk_stats(genome_lengths, chunk_size, ov)
    return "\n".join([
        f"genome      : {len(genome_lengths)} sequences, {sum(genome_lengths):,} bp",
        f"config      : {cfg_name} (PAM {cfg['pam']}, mismatches {cfg['mm']}, DNA bulges {cfg['db']}, RNA bulges {cfg['rb']})",
        f"overlap     : {ov} bp on each side  (= guide {guide_len} + DNA bulges {cfg['db']} + PAM {len(cfg['pam'])} - 1)",
        f"chunk size  : {chunk_size:,} bp",
        f"chunks      : {st['chunks']:,}",
        f"bases read  : {st['read_bp']:,} for {st['owned_bp']:,} owned  (overlap overhead {st['overhead']:.4%})"])


# ======================================================================
# WEEK 1 TESTS AND COMMAND LINE (Day 1 rules/setup + Day 2 mock genome)
# ======================================================================
import argparse, gzip, sys, tempfile


def _read(path):
    with open(path) as fh:
        return fh.read()


def _write(path, text):
    with open(path, "w") as fh:
        fh.write(text)


def verify_sites(genome, guides, truth, max_mm, pam="NGG"):
    """Fast check of planted sites (no genome-wide search): every truth row must sit exactly where the
    truth table says, with a valid PAM, and align to its guide within the planted edit budget."""
    seqs, cfg, bad = dict(guides), dict(pam=pam, mm=max_mm, db=1, rb=1), []
    for gid, c, s, strand, mm, db, rb, target in truth:
        win = genome[c][s:s + len(target)].decode()
        if strand == "-":
            win = revcomp(win)
        spacer, p = target[:-len(pam)], target[-len(pam):]
        best = align(seqs[gid], spacer, cfg)
        if win != target or not pam_matches(p, pam) or best is None or sum(best) > mm + db + rb:
            bad.append((gid, c, s, strand))
    return dict(planted=len(truth), found=len(truth) - len(bad), missing=bad[:5])


def selftest():
    ok = True

    def check(name, cond):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'ok' if cond else 'FAIL'}] {name}")

    print("Day 1 - setup and benchmark rules")
    md = rules_markdown()
    check("rules document is versioned and has 12 sections",
          f"v{RULES_VERSION}" in md.split("\n")[0] and all(f"## {i}. " in md for i in range(1, 13)))
    check("rules table lists every configuration with its exact budget",
          all(f"| {k} | {c['pam']} | up to {c['mm']} |" in md for k, c in CONFIGS.items()))
    check("rules state the protocol numbers (5 timed runs, 20% deviation, median + IQR)",
          "at least 5 timed runs" in md and "20%" in md and "median and interquartile range" in md)
    check("rules list all genomes, guide sets and acceptance targets",
          all(f"| {k} |" in md for k in list(GENOMES) + list(GUIDE_SETS)) and all(a in md for a, _ in ACCEPTANCE))
    check("rules fix the bulge convention (no bulges at alignment ends)", "not allowed at either end" in md)
    with tempfile.TemporaryDirectory() as td:
        made = setup_project(os.path.join(td, "proj"))
        mf = json.loads(_read(os.path.join(td, "proj", "m.json")))
        check("setup creates data/results/mock/logs, rules file and manifest",
              all(os.path.exists(p) for p in made) and len(made) == 7
              and _read(os.path.join(td, "proj", "BENCHMARK_RULES.md")) == md)
        check("manifest template is valid and names genomes, workdir, tool_cmd, sizes",
              all(k in mf for k in ("genomes", "workdir", "tool_cmd", "sizes")) and "{genome}" in mf["tool_cmd"])
    print("Day 1 - core")
    check("six configurations C1-C6", sorted(CONFIGS) == ["C1", "C2", "C3", "C4", "C5", "C6"])
    check("revcomp handles N", revcomp("ACGTN") == "NACGT")
    check("PAM NGG matches TGG, genome N never matches", pam_matches("TGG", "NGG") and not pam_matches("NGG", "NGG"))
    check("PAM NRG accepts AAG", pam_matches("AAG", "NRG"))
    with tempfile.TemporaryDirectory() as td:
        fa = os.path.join(td, "t.fa")
        _write(fa, ">x some description\nacgtnRY\nAC\n")
        check("FASTA: soft-mask upper-cased, non-ACGT -> N, header trimmed", read_fasta(fa) == {"x": "ACGTNNNAC"})
        h = [Hit("g", "c", 5, "+", 1, 0, 0, "ACGTAGG"), Hit("g", "c", 5, "+", 1, 0, 0, "ACGTAGG")]
        hp = os.path.join(td, "h.tsv")
        write_hits(normalize(h), hp)
        check("hit file round-trip and de-duplication", read_hits(hp) == normalize(h) and len(read_hits(hp)) == 1)
        gp = os.path.join(td, "g.tsv")
        write_guides([("g0", "ACGT")], gp)
        check("guide file round-trip", read_guides(gp) == [("g0", "ACGT")])

    print("Day 2 - verification oracle (used to check the mock genome)")
    rng = random.Random(SEED)
    rand = lambda n: "".join(rng.choice("ACGT") for _ in range(n))
    spacer = "GACGTCTAGCATCGATCGAA"
    mm1 = spacer[:5] + ("A" if spacer[5] != "A" else "C") + spacer[6:]
    genome = {"c1": rand(300) + mm1 + "AGG" + rand(300) + revcomp(spacer + "TGG") + rand(200),
              "c2": rand(200) + spacer[:10] + "T" + spacer[10:] + "CGG" + rand(200),
              "c3": rand(200) + spacer[:10] + spacer[11:] + "CGG" + rand(200)}
    guides = [("g0", spacer)]
    c2 = dict(CONFIGS["C2"], mm=2)
    c4 = dict(CONFIGS["C4"], mm=2)
    fast = oracle_search(genome, guides, c2)
    dp = oracle_search(genome, guides, c2, force_dp=True)
    check("fast mismatch path == alignment path (no bulges)", {h.key() for h in fast} == {h.key() for h in dp})
    check("finds 1-mismatch site (+) and exact reverse-strand site (-)", {(h.strand, h.mm) for h in fast} >= {("+", 1), ("-", 0)})
    bh = oracle_search(genome, guides, c4)
    check("finds DNA-bulge site", any(h.chrom == "c2" and h.db == 1 for h in bh))
    check("finds RNA-bulge site", any(h.chrom == "c3" and h.rb == 1 for h in bh))
    check("bulge-free search misses bulge sites", not any(h.chrom in ("c2", "c3") for h in fast))
    nref = {"c": "A" * 10 + spacer[:8] + "N" + spacer[9:] + "TGG" + "A" * 10}
    check("genome N never matches (needs 1 mismatch)",
          not oracle_search(nref, guides, dict(pam="NGG", mm=0, db=0, rb=0))
          and oracle_search(nref, guides, dict(pam="NGG", mm=1, db=0, rb=0)))
    edge = {"c": spacer + "TGG" + "A" * 30, "d": "A" * 30 + spacer + "TGG"}
    check("sites at chromosome start and end", {h.chrom for h in oracle_search(edge, guides, c2)} == {"c", "d"})
    src = {"c": rand(5000)}
    g1, g2 = make_guides(src, 5), make_guides(src, 5)
    check("make_guides: seeded, correct count", g1 == g2 and len(g1) == 5)
    check("make_guides: each guide occurs next to a PAM",
          all(any(h.mm == 0 for h in oracle_search(src, [g], dict(pam="NGG", mm=0, db=0, rb=0))) for g in g1))

    print("Day 2 - mock human genome generator")
    a = generate_mock(scale=0.0002, chroms=["chr21", "chr22"], n_guides=3, copies=4, max_mm=3, bulge_copies=2)
    b = generate_mock(scale=0.0002, chroms=["chr21", "chr22"], n_guides=3, copies=4, max_mm=3, bulge_copies=2)
    st = a["stats"]
    check("deterministic for the same seed", a["genome"] == b["genome"] and a["truth"] == b["truth"])
    check(f"human-like composition (GC {st['gc_percent']}%, masked {st['masked_fraction']}, CpG o/e {st['cpg_obs_exp']})",
          36 <= st["gc_percent"] <= 46 and 0.2 <= st["masked_fraction"] <= 0.6 and st["cpg_obs_exp"] < 0.5)
    r = verify_sites(a["genome"], a["guides"], a["truth"], 3)
    check(f"planted sites sit where the truth table says ({r['found']}/{r['planted']})", r["found"] == r["planted"])
    tiny = generate_mock(scale=0.0001, chroms=["chr21", "chr22"], n_guides=2, copies=3, max_mm=3, bulge_copies=2)
    r = verify_planted(tiny["genome"], tiny["guides"], tiny["truth"], oracle_search, dict(CONFIGS["C4"], mm=3))
    check(f"oracle search finds every planted site ({r['found']}/{r['planted']})", r["found"] == r["planted"])
    print("Day 3 - Biopython parsing (built-in fallback always tested)")
    with tempfile.TemporaryDirectory() as td:
        fa = os.path.join(td, "g.fa")
        _write(fa, ">chr1 test\nacgtNNRYACGT\n>chr2\nACGTACGT\n")
        g = read_genome(fa, parser="builtin")
        check("parse names, soft-mask upper-cased, IUPAC -> N", g == {"chr1": "ACGTNNNNACGT", "chr2": "ACGTACGT"})
        gz = os.path.join(td, "g.fa.gz")
        with gzip.open(gz, "wt") as fh:
            fh.write(_read(fa))
        check("gzip FASTA reads identically", read_genome(gz, parser="builtin") == g)
        rep = genome_report(fa, parser="builtin")
        check("QC report: length, N fraction, soft-masked fraction",
              rep["total_bp"] == 20 and rep["n_records"] == 2 and rep["n_fraction"] == 0.2 and rep["masked_fraction"] == 0.2)
        check("non-primary contigs are flagged",
              flag_nonprimary(["chr1", "chr1_KI270706v1_random", "chrUn_KI270302v1", "chr6_GL000250v2_alt"]) ==
              ["chr1_KI270706v1_random", "chrUn_KI270302v1", "chr6_GL000250v2_alt"])
        gf = os.path.join(td, "guides.fa")
        _write(gf, ">g1\nGACGTCTAGCATCGATCGAA\n>g2 note\nacgtacgtacgtacgtacgt\n")
        check("guides FASTA: ids, upper-case, length checked",
              read_guides_any(gf, parser="builtin") == [("g1", "GACGTCTAGCATCGATCGAA"), ("g2", "ACGTACGTACGTACGTACGT")])
        _write(gf, ">g1\nACGT\n")
        try:
            read_guides_any(gf, parser="builtin"); bad = False
        except ValueError:
            bad = True
        check("too-short guide is rejected", bad)
    if HAVE_BIOPYTHON:
        with tempfile.TemporaryDirectory() as td:
            fa = os.path.join(td, "g.fa")
            _write(fa, ">chr1 test\nacgtNNRYACGT\n>chr2\nACGTACGT\n")
            check("Biopython and built-in parsers agree", crosscheck(fa) is True)
    else:
        print("  [skip] Biopython not installed (pip install biopython): cross-check skipped")
    print("Day 4 - chunking with border overlap")
    check("required overlap = guide + DNA bulges + PAM - 1 (C2: 22, C4: 23, 23-nt guide: 25)",
          required_overlap(CONFIGS["C2"], 20) == 22 and required_overlap(CONFIGS["C4"], 20) == 23
          and required_overlap(CONFIGS["C2"], 23) == 25)
    plan = plan_chunks(1000, 300, 22)
    check("owned ranges tile the chromosome exactly once",
          [(p[0], p[1]) for p in plan] == [(0, 300), (300, 600), (600, 900), (900, 1000)])
    check("each chunk reads an overlap halo on both sides, clipped at the ends",
          plan[0][2:] == (0, 322) and plan[1][2:] == (278, 622) and plan[-1][2:] == (878, 1000))
    r4 = random.Random(SEED)
    rnd = lambda n: "".join(r4.choice("ACGT") for _ in range(n))
    sp = "GACGTCTAGCATCGATCGAA"
    site = sp + "TGG"
    g4 = {"c": rnd(49) + site + rnd(77) + revcomp(site) + rnd(100) + sp[:10] + "T" + sp[10:] + "CGG" + rnd(60)}
    gd4 = [("g0", sp)]
    c2 = dict(CONFIGS["C2"], mm=2)
    c4 = dict(CONFIGS["C4"], mm=2)
    keys = lambda hs: {h.key() for h in hs}
    full2, full4 = oracle_search(g4, gd4, c2), oracle_search(g4, gd4, c4)
    check("sites planted across chunk borders are found by the unchunked search", len(full2) >= 2 and len(full4) >= 3)
    check("chunked == unchunked, mismatch config, chunk sizes 7/50/97/1000",
          all(chunked_search(g4, gd4, c2, n) == full2 for n in (7, 50, 97, 1000)))
    check("chunked == unchunked, with DNA/RNA bulges, chunk sizes 50/97",
          all(chunked_search(g4, gd4, c4, n) == full4 for n in (50, 97)))
    check("one base too little overlap loses a border site",
          keys(chunked_search(g4, gd4, c2, 50, overlap=required_overlap(c2) - 1)) != keys(full2))
    print("SELFTEST", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


def cmd_mockgenome(a):
    chroms = [c.strip() for c in a.chroms.split(",")] if a.chroms else None
    mock = generate_mock(scale=a.scale, chroms=chroms, seed=a.seed, n_guides=a.n_guides,
                         copies=a.copies, max_mm=a.max_mm, bulge_copies=a.bulge_copies)
    fa = write_mock(mock, a.out_dir)
    print(f"genome : {fa}")
    print("stats  :", mock["stats"])
    print(f"guides : {len(mock['guides'])}   planted sites: {len(mock['truth'])}")
    print(f"files  : genome.fa, guides.tsv, planted_truth.tsv, m.json  (in {a.out_dir})")
    rc = 0
    if a.verify:                                              # fast: checks each planted site directly
        r = verify_sites(mock["genome"], mock["guides"], mock["truth"], a.max_mm)
        print(f"verify : {r['found']}/{r['planted']} planted sites correct", "" if r["found"] == r["planted"] else r["missing"])
        rc = 0 if r["found"] == r["planted"] else 1
    if a.verify_oracle:                                       # slow: full oracle search, small genomes only
        total = sum(len(s) for s in mock["genome"].values())
        if total > 400_000:
            print(f"verify-oracle skipped: genome is {total} bp; use --scale 0.0001 or --chroms chr21,chr22")
        else:
            r = verify_planted(mock["genome"], mock["guides"], mock["truth"], oracle_search, dict(CONFIGS["C4"], mm=a.max_mm))
            print(f"oracle : {r['found']}/{r['planted']} planted sites found by CPU search", "" if r["found"] == r["planted"] else r["missing"])
            rc = rc or (0 if r["found"] == r["planted"] else 1)
    return rc


def main(argv=None):
    ap = argparse.ArgumentParser(description="GeneWeaver Week 1: Day 1 setup + benchmark rules, Day 2 mock human genome generator")
    sub = ap.add_subparsers(dest="command", required=True)
    sub.add_parser("selftest", aliases=["test"], help="run all Day 1 and Day 2 tests")
    r = sub.add_parser("rules", help="Day 1: print the benchmark rules (or write them to a file)")
    r.add_argument("--out", help="write the rules to this Markdown file instead of printing")
    s = sub.add_parser("setup", help="Day 1: create the project folders, rules document and manifest template")
    s.add_argument("--dir", default="geneweaver_bench")
    i = sub.add_parser("inspect", help="Day 3: parse a genome (and guides) with Biopython and print a QC report")
    i.add_argument("path", help="genome file: .fa/.fasta/.fna (.gz ok); .fastq/.gb need Biopython")
    i.add_argument("--guides", help="guide FASTA or TSV to validate")
    i.add_argument("--parser", default="auto", choices=["auto", "bio", "builtin"])
    i.add_argument("--crosscheck", action="store_true", help="check Biopython and the built-in parser agree")
    c = sub.add_parser("chunks", help="Day 4: show the chunk plan (and optionally verify chunked == unchunked)")
    c.add_argument("path", help="genome FASTA (.gz ok)")
    c.add_argument("--chunk-size", type=int, default=1_000_000)
    c.add_argument("--config", default="C4", choices=sorted(CONFIGS))
    c.add_argument("--guide-len", type=int, default=20)
    c.add_argument("--guides", help="guide FASTA/TSV, used by --verify")
    c.add_argument("--verify", action="store_true", help="check chunked search == unchunked search (small genomes only)")
    c.add_argument("--parser", default="auto", choices=["auto", "bio", "builtin"])
    m = sub.add_parser("mockgenome", help="generate a mock human genome with planted off-target sites")
    m.add_argument("--scale", type=float, default=0.0005, help="fraction of GRCh38 size (default 0.0005 = 1.5 Mb)")
    m.add_argument("--chroms", default="", help="e.g. chr21,chr22 (default: all 24)")
    m.add_argument("--out-dir", default="mock_genome"); m.add_argument("--seed", type=int, default=SEED)
    m.add_argument("--n-guides", type=int, default=10); m.add_argument("--copies", type=int, default=5)
    m.add_argument("--max-mm", type=int, default=4); m.add_argument("--bulge-copies", type=int, default=2)
    m.add_argument("--verify", action="store_true", help="fast check of every planted site")
    m.add_argument("--verify-oracle", action="store_true", help="full CPU search (small genomes only)")
    a = ap.parse_args(argv)
    if a.command in ("selftest", "test"):
        return selftest()
    if a.command == "inspect":
        try:
            print(format_report(genome_report(a.path, a.parser)))
            rc = 0
            if a.guides:
                gs = read_guides_any(a.guides, a.parser)
                print(f"guides      : {len(gs)} read from {a.guides}, lengths {sorted({len(s) for _, s in gs})}")
            if a.crosscheck:
                same = crosscheck(a.path)
                if same is None:
                    print("crosscheck  : skipped (Biopython is not installed: pip install biopython)")
                else:
                    print("crosscheck  : Biopython and the built-in parser agree" if same else "crosscheck  : PARSERS DISAGREE")
                    rc = 0 if same else 1
            return rc
        except (ValueError, RuntimeError, OSError) as e:
            print("ERROR:", e)
            return 2
    if a.command == "chunks":
        try:
            genome = read_genome(a.path, a.parser)
            cfg = CONFIGS[a.config]
            print(format_chunk_plan([len(s) for s in genome.values()], a.chunk_size, cfg, a.guide_len, a.config))
            if not a.verify:
                return 0
            if not a.guides:
                print("ERROR: --verify needs --guides")
                return 2
            if sum(len(s) for s in genome.values()) > 400_000:
                print("verify skipped: genome is over 400,000 bp (the CPU oracle is slow); use a smaller genome")
                return 0
            guides = read_guides_any(a.guides, a.parser)
            same = chunked_search(genome, guides, cfg, a.chunk_size) == oracle_search(genome, guides, cfg)
            print("verify      : chunked search == unchunked search" if same else "verify      : MISMATCH between chunked and unchunked")
            return 0 if same else 1
        except (ValueError, RuntimeError, OSError) as e:
            print("ERROR:", e)
            return 2
    if a.command == "rules":
        if a.out:
            _write(a.out, rules_markdown())
            print(f"rules written to {a.out}")
        else:
            print(rules_markdown())
        return 0
    if a.command == "setup":
        kept = os.path.exists(os.path.join(a.dir, "m.json"))
        for p in setup_project(a.dir):
            print("created", p)
        if kept:
            print("kept    ", os.path.join(a.dir, "m.json"), "(already exists; not overwritten)")
        print(f"next: put FASTA files in {os.path.join(a.dir, 'data')} and edit {os.path.join(a.dir, 'm.json')}")
        return 0
    if a.scale <= 0:
        ap.error("--scale must be greater than zero")
    try:
        return cmd_mockgenome(a)
    except (ValueError, RuntimeError, OSError) as e:
        print("ERROR:", e)
        return 2


if __name__ == "__main__":
    sys.exit(main())