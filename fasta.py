"""GeneWeaver Week 1 - Day 1: setup + benchmark rules.  Day 2: mock human genome generator.  Day 3: Biopython parsing.  Day 4: chunking with border overlap.  Day 5: pure-Python baseline + timing.

  python fasta.py selftest                                   # all tests
  python fasta.py rules [--out BENCHMARK_RULES.md]           # Day 1: the benchmark rules
  python fasta.py setup --dir geneweaver_bench               # Day 1: project folders + rules + manifest
  python fasta.py mockgenome --scale 0.0005 --out-dir mock_genome --verify   # Day 2
  python fasta.py inspect genome.fa [--guides g.fa] [--parser auto|bio|builtin] [--crosscheck]   # Day 3
  python fasta.py chunks genome.fa --chunk-size 1000000 --config C4 [--guides g.fa --verify]   # Day 4
  python fasta.py animate [--delay 0.45] [--plain]                                           # Day 4 animation
  python fasta.py baseline [--config C2] [--runs 5]                                          # Day 5: timed pure-Python baseline
  python fasta.py gpu [--backend auto|gpu|emulated] [--config C2] [--verify]                # Day 6: CUDA kernel
  python fasta.py --debug baseline [--runs 5]                                                # verbose diagnostics
  python fasta.py baseline --debug [--runs 5]                                                # same, flag after command
  GENEWEAVER_DEBUG=1 python fasta.py selftest                                                # enable diagnostics via environment
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
    """Upper-cases (soft-mask is sequence), maps non-ACGT to N, name = header up to first space.
    Same rules as the Day 3 built-in parser: a bare '>' header gives the name "", text before the first
    header is ignored, and a repeated name is an error (it used to silently overwrite the first record)."""
    genome, name, parts = {}, None, []

    def finish():
        if name in genome:
            raise ValueError(f"duplicate sequence name {name!r} in {path}")
        genome[name] = re.sub("[^ACGT]", "N", "".join(parts).upper())

    with open(path) as fh:
        for line in fh:
            if line.startswith(">"):
                if name is not None:
                    finish()
                head = line[1:].split()
                name, parts = (head[0] if head else ""), []
            elif name is not None:
                parts.append(line.strip())
    if name is not None:
        finish()
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
# DAY 4 (animation) - Terminal animation of the chunked scan with border overlap
# ======================================================================
import time


def demo_genome(length=420, chunk_size=60, seed=SEED):
    """Small synthetic genome with five planted sites; three of them cross a chunk border."""
    rng = random.Random(seed)
    sp = "GACGTCTAGCATCGATCGAA"
    seq = [rng.choice("ACGT") for _ in range(length)]

    def put(start, spacer, pam, strand):
        site = spacer + pam
        w = site if strand == "+" else revcomp(site)
        if 0 <= start and start + len(w) <= length:
            seq[start:start + len(w)] = list(w)

    other = lambda b: "A" if b != "A" else "C"
    mm1 = sp[:6] + other(sp[6]) + sp[7:]
    mm2 = sp[:3] + other(sp[3]) + sp[4:12] + other(sp[12]) + sp[13:]
    c = chunk_size
    put(c - 10, sp, "TGG", "+")            # crosses border 1
    put(2 * c + 8, sp, "AGG", "-")         # inside one chunk
    put(3 * c - 14, mm1, "CGG", "-")       # crosses border 3
    put(4 * c + 20, mm2, "TGG", "+")       # inside one chunk
    put(5 * c - 5, sp, "GGG", "+")         # crosses border 5
    return "".join(seq), [("g0", sp)]


def animation_frames(length=420, chunk_size=60, width=64):
    """Frames (strings of 6 lines each) showing each chunk being read with its halo and the hits it reports."""
    cfg = dict(CONFIGS["C2"], mm=2)
    seq, guides = demo_genome(length, chunk_size)
    ov = required_overlap(cfg, len(guides[0][1]))
    plan = plan_chunks(len(seq), chunk_size, ov)
    per_chunk = []
    for own_s, own_e, rs, re_ in plan:
        hs = [Hit(h.guide_id, "c", rs + h.start, h.strand, h.mm, h.db, h.rb, h.target)
              for h in oracle_search({"c": seq[rs:re_]}, guides, cfg) if own_s <= rs + h.start < own_e]
        per_chunk.append(hs)
    full = oracle_search({"c": seq}, guides, cfg)
    no_overlap = chunked_search({"c": seq}, guides, cfg, chunk_size, overlap=0)
    borders = list(range(chunk_size, length, chunk_size))
    col = lambda p: min(width - 1, p * width // length)
    crosses = lambda h: any(h.start < b < h.start + len(h.target) for b in borders)
    ruler = ["-"] * width
    for b in borders:
        ruler[col(b)] = "|"
    ruler = "".join(ruler)
    legend = "legend:  + forward hit   - reverse hit   * hit crossing a chunk border   = owned   . overlap halo"
    head = "GeneWeaver Day 4 - chunked scan with border overlap"
    info = f"{length} bp, chunk {chunk_size} bp, overlap {ov} bp each side"

    def make(scan_row, status, found, summary=""):
        hit_row = [" "] * width
        for h in found:
            hit_row[col(h.start)] = "*" if crosses(h) else h.strand
        n_cross = sum(crosses(h) for h in found)
        return "\n".join([head,
                          f"genome |{ruler}|  {info}",
                          f"scan   |{scan_row}|  {status}",
                          f"hits   |{''.join(hit_row)}|  found {len(found)} so far, {n_cross} cross a border",
                          summary, legend])

    blank = " " * width
    found = []
    frames = [make(blank, "ready", found)]
    for i, ((own_s, own_e, rs, re_), hs) in enumerate(zip(plan, per_chunk), 1):
        row = [" "] * width
        for p in range(rs, re_):
            row[col(p)] = "."
        for p in range(own_s, own_e):
            row[col(p)] = "="
        frames.append(make("".join(row), f"chunk {i}/{len(plan)}: reading {rs}-{re_} (owns {own_s}-{own_e})", found))
        found = found + hs
        frames.append(make("".join(row), f"chunk {i}/{len(plan)}: reported {len(hs)} hit(s)", found))
    same = len(found) == len(full) and {h.key() for h in found} == {h.key() for h in full}
    frames.append(make(blank, "done", found,
                       f"with overlap: {len(found)} hits | without overlap: {len(no_overlap)} | unchunked: {len(full)}"
                       f"  ->  {'identical to the unchunked search' if same else 'MISMATCH'}"))
    return frames


def play_frames(frames, delay=0.4, ansi=True, out=None):
    """Play frames in place (ANSI cursor moves) or, with ansi=False, print them one after another."""
    out = out or sys.stdout
    if ansi and os.name == "nt":
        os.system("")                                    # turns on ANSI escape codes in the Windows console
    prev = 0
    for k, frame in enumerate(frames):
        lines = frame.split("\n")
        if ansi and k:
            out.write(f"\033[{prev}A")
        for ln in lines:
            out.write(("\033[K" if ansi else "") + ln + "\n")
        out.flush()
        prev = len(lines)
        if k < len(frames) - 1:
            if not ansi:
                out.write("\n")
            if delay:
                time.sleep(delay)


# ======================================================================
# DAY 5 - Pure-Python baseline: 2-bit packed XOR + popcount search, with its time recorded (Rules s2, s7)
# ======================================================================
import datetime, json, platform, statistics, tracemalloc

_TO_DIGIT = str.maketrans("ACGT", "0123")          # A=00 C=01 G=10 T=11, read as a base-4 number
_NON_ACGT = re.compile("[^ACGT]")
try:
    _popcount = int.bit_count                      # Python 3.10+
except AttributeError:
    _popcount = lambda x: bin(x).count("1")


def _pam_regex(pam):
    """Zero-width regex that matches every position where the PAM pattern fits (IUPAC; genome N never matches)."""
    return re.compile("(?=(" + "".join("[" + IUPAC[c] + "]" for c in pam) + "))")


def packed_search(genome, guides, cfg):
    """Mismatch-only search, both strands. Same hits as oracle_search for configs without bulges, much faster:
    find PAM sites with a regex first, pack each candidate spacer into an integer (2 bits per base), then count
    mismatches per guide with XOR and a popcount. This is the same idea the GPU kernel will use."""
    if cfg["db"] or cfg["rb"]:
        raise ValueError("the packed baseline handles mismatch-only configs (C1, C2, C3, C6); "
                         "use the CPU oracle for configs with bulges")
    pam, pl, mm = cfg["pam"], len(cfg["pam"]), cfg["mm"]
    rx = _pam_regex(pam)
    by_len = {}
    for gid, g in guides:
        by_len.setdefault(len(g), []).append((gid, g, int(g.translate(_TO_DIGIT), 4) if not _NON_ACGT.search(g) else None))
    hits = []
    for chrom, seq in genome.items():
        for strand, s in (("+", seq), ("-", revcomp(seq))):
            n = len(s)
            for L, group in by_len.items():
                mask = int("01" * L, 2)                       # low bit of every 2-bit pair
                for m in rx.finditer(s):
                    p = m.start()                              # PAM starts here; spacer is the L bases before it
                    if p < L:
                        continue
                    spacer = s[p - L:p]
                    start = (p - L) if strand == "+" else n - (p + pl)
                    clean = not _NON_ACGT.search(spacer)
                    a = int(spacer.translate(_TO_DIGIT), 4) if clean else None
                    for gid, g, gv in group:
                        if clean and gv is not None:
                            x = a ^ gv
                            d = _popcount((x | (x >> 1)) & mask)
                            if d > mm:
                                continue
                        else:                                  # N in the genome, or IUPAC letters in the guide
                            d = count_mm(g, spacer, mm)
                            if d is None:
                                continue
                        hits.append(Hit(gid, chrom, start, strand, d, 0, 0, spacer + s[p:p + pl]))
    return normalize(hits)


def summarize_times(xs):
    """Median and interquartile range; flags a run set as unstable if any run is >20% from the median (Rules s7.2)."""
    med = statistics.median(xs)
    q = statistics.quantiles(xs, n=4) if len(xs) >= 2 else [xs[0]] * 3
    return dict(median=med, q1=q[0], q3=q[2], min=min(xs), max=max(xs),
                unstable=any(abs(x - med) / med > 0.20 for x in xs) if med else False)


def gps(n_guides, genome_len, seconds):
    """Throughput in guide-positions per second = G x 2L / t (both strands), Rules s2."""
    return n_guides * 2 * genome_len / seconds


def time_runs(fn, runs=5, warmup=1):
    """Rules s7.2: untimed warm-up run(s), then `runs` timed runs. Returns (list of seconds, last result)."""
    if runs < 1 or warmup < 0:
        raise ValueError("need runs >= 1 and warmup >= 0")
    for _ in range(warmup):
        fn()
    times, result = [], None
    for _ in range(runs):
        t0 = time.perf_counter()
        result = fn()
        times.append(time.perf_counter() - t0)
    return times, result


def peak_memory_mb(fn):
    """Peak Python-level memory (MB) of one extra, untimed run (tracemalloc slows code, so it is never timed)."""
    tracemalloc.start()
    try:
        fn()
        return tracemalloc.get_traced_memory()[1] / 1e6
    finally:
        tracemalloc.stop()


def hardware_info():
    info = dict(os=platform.platform(), cpu=platform.processor() or platform.machine(), cores=os.cpu_count(),
                python=f"{platform.python_implementation()} {platform.python_version()}")
    try:
        info["ram_gb"] = round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30, 1)
    except (AttributeError, ValueError, OSError):
        pass
    return info


def genome_digest(genome):
    h = hashlib.sha256()
    for name, seq in genome.items():
        h.update(f"{name}\n{seq}\n".encode())
    return h.hexdigest()


def baseline_record(impl, genome_id, genome_sha, genome_bp, guide_set, n_guides, config, times, n_hits,
                    peak_mb, correctness, load_s=None):
    """One JSON-ready record in the Rules s10 shape, for the pure-Python baseline."""
    st = summarize_times(times)
    return dict(tool="pure-python-baseline", implementation=impl, rules_version=RULES_VERSION,
                created=datetime.datetime.now().isoformat(timespec="seconds"),
                genome=genome_id, genome_sha256=genome_sha, genome_bp=genome_bp, guide_set=guide_set, n_guides=n_guides,
                config=config, config_detail=CONFIGS[config], runs_s=[round(t, 6) for t in times],
                median_search_s=st["median"], iqr_s=[st["q1"], st["q3"]], unstable=st["unstable"],
                gps=gps(n_guides, genome_bp, st["median"]) if st["median"] else None, total_hits=n_hits,
                peak_python_mem_mb=round(peak_mb, 2), load_s=load_s, correctness=correctness, hardware=hardware_info())


def format_baseline(rec, saved=None):
    r = rec
    chk = r["correctness"]
    L = [f"pure-Python baseline ({r['implementation']})",
         f"genome      : {r['genome']}  {r['genome_bp']:,} bp   guides: {r['n_guides']} ({r['guide_set']})   config {r['config']}",
         f"runs (s)    : " + "  ".join(f"{t:.3f}" for t in r["runs_s"]),
         f"median      : {r['median_search_s']:.3f} s   IQR {r['iqr_s'][0]:.3f}-{r['iqr_s'][1]:.3f} s   "
         + ("UNSTABLE (a run was >20% from the median; re-run on an idle machine)" if r["unstable"] else "stable"),
         f"throughput  : {r['gps']:.3e} guide-positions/s",
         f"hits        : {r['total_hits']}   peak Python memory {r['peak_python_mem_mb']} MB"]
    if chk.get("planted_total") is not None:
        L.append(f"check       : planted sites found {chk['planted_found']}/{chk['planted_total']}")
    elif chk.get("note"):
        L.append(f"check       : {chk['note']}")
    if saved:
        L.append(f"saved       : {saved}")
    return "\n".join(L)


def run_baseline(genome, guides, config, impl="packed", runs=5, warmup=1, genome_id="G-mock", guide_set="S-mock",
                 truth=None, load_s=None, genome_sha=None):
    """Time the baseline and return its record. `truth` = planted rows, used for a correctness check."""
    cfg = CONFIGS[config]
    search = packed_search if impl == "packed" else oracle_search
    fn = lambda: search(genome, guides, cfg)
    times, hits = time_runs(fn, runs, warmup)
    peak = peak_memory_mb(fn)
    correctness = dict(note="not checked (no planted truth for this genome)")
    if truth is not None:
        found = {h.key() for h in hits}
        want = [t for t in truth if t[5] == 0 and t[6] == 0 and t[4] <= cfg["mm"]]     # planted, within this budget
        correctness = dict(planted_total=len(want), planted_found=sum((t[0], t[1], t[2], t[3]) in found for t in want))
    return baseline_record(impl, genome_id, genome_sha or genome_digest(genome), sum(len(x) for x in genome.values()),
                           guide_set, len(guides), config, times, len(hits), peak, correctness, load_s)


def save_baseline(rec, out_dir):
    """Write the record as JSON and append one summary line to baseline_log.jsonl, so timings accumulate."""
    os.makedirs(out_dir, exist_ok=True)
    # the genome fingerprint keeps records of different genomes (or sizes) from overwriting each other
    name = (f"baseline_{rec['genome']}_{rec['genome_sha256'][:8]}_{rec['guide_set']}_{rec['config']}_"
            f"{rec['implementation']}.json").replace(os.sep, "_")
    path = os.path.join(out_dir, name)
    with open(path, "w") as fh:
        json.dump(rec, fh, indent=2)
    line = {k: rec[k] for k in ("created", "implementation", "genome", "genome_bp", "guide_set", "n_guides", "config",
                                "median_search_s", "gps", "total_hits")}
    with open(os.path.join(out_dir, "baseline_log.jsonl"), "a") as fh:
        fh.write(json.dumps(line) + "\n")
    return path


# ======================================================================
# DAY 6 - First CUDA mismatch kernel and chunked search
# ======================================================================
try:
    import numpy as np
except ImportError:
    np = None

_emulated_thread_index = 0

try:
    from numba import cuda
    HAVE_NUMBA = True
except ImportError:
    HAVE_NUMBA = False

    _emulated_thread_index = 0

    class _NoNumbaKernel:
        def __init__(self, fn):
            self.py_func = fn

        def __getitem__(self, _):
            raise RuntimeError("Numba is required to launch a CUDA kernel")

    class _NoNumbaCuda:
        def jit(self, fn=None, **kwargs):
            def decorate(func):
                return _NoNumbaKernel(func)
            return decorate(fn) if fn is not None else decorate

        def grid(self, ndim):
            if ndim != 1:
                raise ValueError("the emulator supports one-dimensional grids")
            return _emulated_thread_index

        def is_available(self):
            return False

    cuda = _NoNumbaCuda()


class _EmulatedCuda:
    def grid(self, ndim):
        if ndim != 1:
            raise ValueError("the emulator supports one-dimensional grids")
        return _emulated_thread_index


@cuda.jit
def mismatch_kernel(codes, guides, glen, masks, mm, out):
    i = cuda.grid(1)
    n_pos = len(codes) - glen - len(masks) + 1
    if i < n_pos:
        valid = True
        for k in range(len(masks)):
            base = codes[i + glen + k]
            if base > 3 or (masks[k] & (1 << base)) == 0:
                valid = False
                break
        if valid:
            for g in range(len(guides)):
                distance = 0
                for j in range(glen):
                    if codes[i + j] != guides[g, j]:
                        distance += 1
                if distance <= mm:
                    out[i, g] = distance


def _require_numpy():
    if np is None:
        raise RuntimeError("Day 6 requires NumPy; install it with: python -m pip install numpy")


def encode_bases(sequence):
    _require_numpy()
    return np.fromiter(({"A": 0, "C": 1, "G": 2, "T": 3}.get(b.upper(), 4) for b in sequence),
                       dtype=np.uint8, count=len(sequence))


def encode_guide(sequence):
    _require_numpy()
    sequence = sequence.upper()
    if not sequence or any(base not in "ACGT" for base in sequence):
        raise ValueError("guides must contain only A, C, G, and T")
    return np.fromiter(({"A": 0, "C": 1, "G": 2, "T": 3}[b] for b in sequence),
                       dtype=np.uint8, count=len(sequence))


def pam_masks_for(pam):
    _require_numpy()
    try:
        return np.asarray([sum(1 << "ACGT".index(base) for base in IUPAC[code.upper()])
                           for code in pam], dtype=np.uint8)
    except KeyError as e:
        raise ValueError(f"unsupported IUPAC PAM symbol: {e.args[0]}") from e


def revcomp_codes(codes):
    _require_numpy()
    table = np.asarray([3, 2, 1, 0, 4], dtype=np.uint8)
    return table[np.asarray(codes, dtype=np.uint8)[::-1]]


def emulate_launch(kernel, blocks, threads, args):
    if blocks < 0 or threads < 1:
        raise ValueError("blocks must be non-negative and threads must be positive")
    global _emulated_thread_index, cuda
    held_cuda = cuda
    cuda = _EmulatedCuda()
    try:
        body = getattr(kernel, "py_func", kernel)
        for block in range(blocks):
            for thread in range(threads):
                _emulated_thread_index = block * threads + thread
                body(*args)
    finally:
        cuda = held_cuda
        _emulated_thread_index = 0


def gpu_backend(requested="auto"):
    if requested not in ("auto", "gpu", "emulated"):
        raise ValueError("backend must be auto, gpu, or emulated")
    if requested == "emulated":
        return "emulated"
    available = False
    if HAVE_NUMBA:
        try:
            available = bool(cuda.is_available())
        except (AttributeError, OSError, RuntimeError):
            available = False
    if available:
        return "gpu"
    if requested == "gpu":
        raise RuntimeError("no usable CUDA GPU is available (install Numba and check the CUDA driver)")
    return "emulated"


def gpu_info(backend):
    info = {"backend": backend, "device": None, "compute_capability": None,
            "simulator": os.environ.get("NUMBA_ENABLE_CUDASIM", "").strip() == "1"}
    if backend == "gpu":
        if info["simulator"]:
            info["device"] = "Numba CUDA simulator"
        else:
            device = cuda.get_current_device()
            name = device.name
            info["device"] = name.decode() if isinstance(name, bytes) else str(name)
            capability = getattr(device, "compute_capability", None)
            if capability:
                info["compute_capability"] = f"{capability[0]}.{capability[1]}"
    return info


def scan_codes(codes, guides, glen, masks, mm, backend="auto", threads=128):
    _require_numpy()
    if not isinstance(mm, int) or not 0 <= mm <= 254:
        raise ValueError("mismatch budget must be between 0 and 254")
    if threads < 1:
        raise ValueError("threads must be positive")
    codes = np.asarray(codes, dtype=np.uint8)
    guides = np.asarray(guides, dtype=np.uint8)
    masks = np.asarray(masks, dtype=np.uint8)
    if codes.ndim != 1 or guides.ndim != 2 or masks.ndim != 1:
        raise ValueError("codes and masks must be 1-D and guides must be 2-D")
    if guides.shape[1] != glen or glen < 1 or len(masks) < 1:
        raise ValueError("guide length and PAM masks must be non-empty and match")
    selected = gpu_backend(backend)
    n_pos = max(0, len(codes) - glen - len(masks) + 1)
    out = np.full((n_pos, len(guides)), 255, dtype=np.uint8)
    if n_pos == 0 or len(guides) == 0:
        return out
    blocks = (n_pos + threads - 1) // threads
    args = (codes, guides, glen, masks, mm, out)
    if selected == "gpu":
        mismatch_kernel[blocks, threads](*args)
    else:
        emulate_launch(mismatch_kernel, blocks, threads, args)
    return out


def auto_chunk_size(genome_bp):
    if genome_bp <= 0:
        return 1_000_000
    return max(10_000, min(1_000_000, 100_000_000 // genome_bp))


def _decode_codes(codes):
    alphabet = "ACGTN"
    return "".join(alphabet[int(base)] for base in codes)


def gpu_search(genome, guides, cfg, chunk_size=None, backend="auto", threads=128, progress=None):
    _require_numpy()
    if isinstance(cfg, str):
        cfg = CONFIGS[cfg]
    if cfg["db"] or cfg["rb"]:
        raise ValueError("the CUDA kernel supports mismatch-only configurations")
    if not guides:
        return []
    selected = gpu_backend(backend)
    if chunk_size is None:
        chunk_size = auto_chunk_size(sum(len(seq) for seq in genome.values()))
    if chunk_size < 1:
        raise ValueError("chunk size must be positive")
    if threads < 1:
        raise ValueError("threads must be positive")

    guides_by_len = {}
    for guide_id, sequence in guides:
        encoded = encode_guide(sequence)
        guides_by_len.setdefault(len(sequence), []).append((guide_id, encoded))
    encoded_guides = {
        length: np.stack([entry[1] for entry in entries]).astype(np.uint8, copy=False)
        for length, entries in guides_by_len.items()
    }
    masks = pam_masks_for(cfg["pam"])
    overlap = required_overlap(cfg, max(guides_by_len))
    chunk_plans = [(chrom, seq, plan_chunks(len(seq), chunk_size, overlap))
                   for chrom, seq in genome.items()]
    total_chunks = sum(len(plan) for _, _, plan in chunk_plans)
    completed = 0
    started = time.perf_counter()
    hits = []

    for chrom, seq, plan in chunk_plans:
        codes = encode_bases(seq)
        for own_start, own_end, read_start, read_end in plan:
            forward = codes[read_start:read_end]
            for strand, oriented in (("+", forward), ("-", revcomp_codes(forward))):
                for glen, entries in guides_by_len.items():
                    result = scan_codes(oriented, encoded_guides[glen], glen, masks, cfg["mm"],
                                        selected, threads)
                    for pos in range(result.shape[0]):
                        forward_start = (read_start + pos if strand == "+"
                                         else read_end - (pos + glen + len(masks)))
                        if not own_start <= forward_start < own_end:
                            continue
                        target = _decode_codes(oriented[pos:pos + glen + len(masks)])
                        for guide_index, (guide_id, _) in enumerate(entries):
                            distance = int(result[pos, guide_index])
                            if distance != 255:
                                hits.append(Hit(guide_id, chrom, forward_start, strand, distance, 0, 0, target))
            completed += 1
            if progress is not None:
                progress(completed, total_chunks, len(hits), time.perf_counter() - started)
    return normalize(hits)


def run_gpu(genome, guides, config, backend="auto", runs=5, warmup=1, truth=None,
            chunk_size=None, threads=128, genome_id="G-mock", guide_set="S-mock",
            genome_sha=None, progress=None):
    if runs < 1 or warmup < 0:
        raise ValueError("need runs >= 1 and warmup >= 0")
    cfg = CONFIGS[config] if isinstance(config, str) else config
    selected = gpu_backend(backend)
    n_bp = sum(len(seq) for seq in genome.values())
    if chunk_size is None:
        chunk_size = auto_chunk_size(n_bp)
    search = lambda: gpu_search(genome, guides, cfg, chunk_size, selected, threads)
    for _ in range(warmup):
        search()
    times, hits = [], None
    for index in range(runs):
        t0 = time.perf_counter()
        hits = gpu_search(genome, guides, cfg, chunk_size, selected, threads,
                          progress if index == runs - 1 else None)
        times.append(time.perf_counter() - t0)

    baseline_hits = packed_search(genome, guides, cfg)
    correctness = {
        "matches_baseline": hits == baseline_hits,
        "note": "identical to the Day 5 pure-Python baseline" if hits == baseline_hits
                else "DIFFERS from the Day 5 pure-Python baseline",
        "planted_total": None,
        "planted_found": None,
    }
    if truth is not None:
        found = {hit.key() for hit in hits}
        planted = [row for row in truth if row[5] == 0 and row[6] == 0 and row[4] <= cfg["mm"]]
        correctness.update(planted_total=len(planted),
                           planted_found=sum((row[0], row[1], row[2], row[3]) in found for row in planted))
    stats = summarize_times(times)
    rec = dict(
        tool="geneweaver-cuda",
        implementation=f"cuda-{selected}",
        backend=selected,
        rules_version=RULES_VERSION,
        created=datetime.datetime.now().isoformat(timespec="seconds"),
        genome=genome_id,
        genome_sha256=genome_sha or genome_digest(genome),
        genome_bp=n_bp,
        guide_set=guide_set,
        n_guides=len(guides),
        config=config if isinstance(config, str) else "custom",
        config_detail=cfg,
        runs_s=[round(seconds, 6) for seconds in times],
        median_search_s=stats["median"],
        iqr_s=[stats["q1"], stats["q3"]],
        unstable=stats["unstable"],
        gps=gps(len(guides), n_bp, stats["median"]) if stats["median"] else None,
        total_hits=len(hits),
        gpu=gpu_info(selected),
        threads_per_block=threads,
        chunk_size=chunk_size,
        hardware=hardware_info(),
        correctness=correctness,
    )
    return rec, hits


def save_gpu(rec, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    name = (f"baseline_{rec['genome']}_{rec['genome_sha256'][:8]}_{rec['guide_set']}_{rec['config']}_"
            f"{rec['implementation']}.json").replace(os.sep, "_")
    path = os.path.join(out_dir, name)
    with open(path, "w") as fh:
        json.dump(rec, fh, indent=2)
    line = {key: rec[key] for key in ("created", "implementation", "genome", "genome_bp", "guide_set",
                                      "n_guides", "config", "median_search_s", "gps", "total_hits")}
    with open(os.path.join(out_dir, "baseline_log.jsonl"), "a") as fh:
        fh.write(json.dumps(line) + "\n")
    return path


def format_progress(completed, total, hits, elapsed):
    pct = 100 if total == 0 else min(100, int(completed * 100 / total))
    filled = 30 if total == 0 else min(30, int(completed * 30 / total))
    bar = "=" * filled + " " * (30 - filled)
    return (f"[{bar}] {pct:3d}%  chunk {completed}/{total}  hits {hits}  "
            f"{elapsed:.1f}s")


class ProgressView:
    def __init__(self, out=None, tty=None):
        self.out = out or sys.stdout
        self.tty = self.out.isatty() if tty is None else tty
        self._last_step = 0

    def __call__(self, completed, total, hits, elapsed):
        line = format_progress(completed, total, hits, elapsed)
        if self.tty:
            self.out.write("\r" + line)
            if completed >= total:
                self.out.write("\n")
            self.out.flush()
            return
        step = max(1, total // 5)
        if completed >= total or completed // step > self._last_step:
            self.out.write(line + "\n")
            self.out.flush()
            self._last_step = completed // step


def format_gpu(rec, saved=None):
    gpu = rec["gpu"]
    backend = rec.get("backend", gpu.get("backend"))
    if backend == "emulated":
        device = "CPU emulation"
    elif gpu.get("simulator"):
        device = "Numba simulator"
    else:
        device = f"NVIDIA GPU: {gpu.get('device')}"
    chk = rec["correctness"]
    baseline = ("identical to the Day 5 pure-Python baseline"
                if chk.get("matches_baseline") else "DIFFERS from the Day 5 pure-Python baseline")
    lines = [
        f"backend     : {backend} ({device})",
        f"kernel      : mismatch_kernel, {rec['threads_per_block']} threads/block",
        f"genome      : {rec['genome']}  {rec['genome_bp']:,} bp   guides: {rec['n_guides']}   config {rec['config']}",
        f"runs (s)    : " + "  ".join(f"{seconds:.3f}" for seconds in rec["runs_s"]),
        f"median      : {rec['median_search_s']:.3f} s",
        f"throughput  : {rec['gps']:.3e} guide-positions/s" if rec["gps"] is not None else "throughput  : n/a",
        f"hits        : {rec['total_hits']}",
        f"baseline    : {baseline}",
    ]
    if chk.get("planted_total") is not None:
        lines.append(f"check       : planted sites found {chk['planted_found']}/{chk['planted_total']}")
    if saved:
        lines.append(f"saved       : {saved}")
    return "\n".join(lines)


def cmd_gpu(a):
    try:
        cfg = CONFIGS[a.config]
        if cfg["db"] or cfg["rb"]:
            print(f"ERROR: {a.config} has bulges; the CUDA kernel is mismatch-only.")
            return 2
        selected = gpu_backend(a.backend)
        truth, load_s, genome_sha = None, None, None
        if a.genome:
            t0 = time.perf_counter()
            genome = read_genome(a.genome, a.parser)
            load_s = round(time.perf_counter() - t0, 4)
            genome_sha = sha256(a.genome)
            genome_id = os.path.basename(a.genome)
            guides = read_guides_any(a.guides, a.parser) if a.guides else make_guides(genome, a.n_guides, a.seed)
            guide_set = os.path.basename(a.guides) if a.guides else f"random-{len(guides)}"
        else:
            chroms = [name.strip() for name in a.chroms.split(",") if name.strip()] or None
            mock = generate_mock(scale=a.scale, chroms=chroms, seed=a.seed, n_guides=a.n_guides)
            genome = {name: seq.decode() for name, seq in mock["genome"].items()}
            guides, truth = mock["guides"], mock["truth"]
            genome_id, guide_set = "G-mock", "S-mock"
        view = ProgressView()
        print("Day 6 - first CUDA kernel")
        rec, _ = run_gpu(genome, guides, a.config, selected, a.runs, a.warmup, truth,
                         a.chunk_size, a.threads, genome_id, guide_set, genome_sha, view)
        saved = save_gpu(rec, a.out_dir)
        print(format_gpu(rec, saved))
        if a.verify and (not rec["correctness"]["matches_baseline"]
                         or (truth is not None and rec["correctness"]["planted_found"] != rec["correctness"]["planted_total"])):
            return 1
        return 0
    except (ValueError, RuntimeError, OSError) as e:
        print("ERROR:", e)
        return 2


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


def _raises(exc, fn, *args):
    try:
        fn(*args)
    except exc:
        return True
    return False


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
    print("Day 5 - pure-Python baseline and recorded timing")
    check("packed XOR+popcount baseline == CPU oracle (mismatch config, both strands, border sites)",
          packed_search(g4, gd4, c2) == oracle_search(g4, gd4, c2))
    check("baseline refuses configs with bulges", all(
        _raises(ValueError, packed_search, g4, gd4, CONFIGS[c]) for c in ("C4", "C5")))
    st = summarize_times([1.0, 1.1, 0.9, 1.0, 1.0])
    check("timing summary: median 1.0, stable; a 2x outlier is flagged unstable",
          st["median"] == 1.0 and not st["unstable"] and summarize_times([1, 1, 1, 1, 2])["unstable"])
    check("throughput = guides x 2 x length / seconds", gps(10, 1_000_000, 2.0) == 1e7)
    calls = []
    ts, _ = time_runs(lambda: calls.append(1), runs=5, warmup=1)
    check("1 warm-up + 5 timed runs", len(calls) == 6 and len(ts) == 5)
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
    ap.add_argument("-d", "--debug", action="store_true", help="print diagnostic details to stderr")
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
    an = sub.add_parser("animate", help="Day 4: play a terminal animation of the chunked scan")
    an.add_argument("--length", type=int, default=420, help="demo genome length in bp")
    an.add_argument("--chunk-size", type=int, default=60)
    an.add_argument("--delay", type=float, default=0.45, help="seconds per frame")
    an.add_argument("--plain", action="store_true", help="print frames one after another (no in-place redraw)")
    b = sub.add_parser("baseline", help="Day 5: time the pure-Python baseline and record the result")
    b.add_argument("-d", "--debug", action="store_true", default=argparse.SUPPRESS,
                   help="print diagnostic details to stderr")
    b.add_argument("--genome", help="genome FASTA; if omitted a mock human genome is generated")
    b.add_argument("--guides", help="guide FASTA/TSV; if omitted guides are generated")
    b.add_argument("--config", default="C2", choices=sorted(CONFIGS))
    b.add_argument("--impl", default="packed", choices=["packed", "oracle"],
                   help="packed = 2-bit XOR+popcount baseline; oracle = naive reference (small genomes)")
    b.add_argument("--runs", type=int, default=5); b.add_argument("--warmup", type=int, default=1)
    b.add_argument("--scale", type=float, default=0.0005, help="mock genome size, fraction of GRCh38")
    b.add_argument("--chroms", default="", help="mock genome chromosomes, e.g. chr21,chr22")
    b.add_argument("--n-guides", type=int, default=10); b.add_argument("--seed", type=int, default=SEED)
    b.add_argument("--out-dir", default="results"); b.add_argument("--parser", default="auto", choices=["auto", "bio", "builtin"])
    b.add_argument("--force", action="store_true", help="allow the slow oracle on a large genome")
    g = sub.add_parser("gpu", help="Day 6: run the CUDA mismatch kernel (or its CPU emulator)")
    g.add_argument("-d", "--debug", action="store_true", default=argparse.SUPPRESS,
                   help="print diagnostic details to stderr")
    g.add_argument("--genome", help="genome FASTA; if omitted a mock genome is generated")
    g.add_argument("--guides", help="guide FASTA/TSV; if omitted guides are generated")
    g.add_argument("--config", default="C2", choices=sorted(CONFIGS))
    g.add_argument("--backend", default="auto", choices=["auto", "gpu", "emulated"])
    g.add_argument("--threads", type=int, default=128, help="CUDA threads per block")
    g.add_argument("--chunk-size", type=int, help="bases per chunk (chosen automatically by default)")
    g.add_argument("--runs", type=int, default=5); g.add_argument("--warmup", type=int, default=1)
    g.add_argument("--scale", type=float, default=0.0001, help="mock genome size as a fraction of GRCh38")
    g.add_argument("--chroms", default="chr21,chr22")
    g.add_argument("--n-guides", type=int, default=10); g.add_argument("--seed", type=int, default=SEED)
    g.add_argument("--out-dir", default="results")
    g.add_argument("--parser", default="auto", choices=["auto", "bio", "builtin"])
    g.add_argument("--verify", action="store_true", help="check results against the Day 5 baseline and planted sites")
    m = sub.add_parser("mockgenome", help="generate a mock human genome with planted off-target sites")
    m.add_argument("--scale", type=float, default=0.0005, help="fraction of GRCh38 size (default 0.0005 = 1.5 Mb)")
    m.add_argument("--chroms", default="", help="e.g. chr21,chr22 (default: all 24)")
    m.add_argument("--out-dir", default="mock_genome"); m.add_argument("--seed", type=int, default=SEED)
    m.add_argument("--n-guides", type=int, default=10); m.add_argument("--copies", type=int, default=5)
    m.add_argument("--max-mm", type=int, default=4); m.add_argument("--bulge-copies", type=int, default=2)
    m.add_argument("--verify", action="store_true", help="fast check of every planted site")
    m.add_argument("--verify-oracle", action="store_true", help="full CPU search (small genomes only)")
    a = ap.parse_args(argv)
    debug_env = os.environ.get("GENEWEAVER_DEBUG", "").strip().lower() in ("1", "true", "yes", "on")
    debug = a.debug or debug_env

    def debug_log(message):
        if debug:
            print(f"[DEBUG] {message}", file=sys.stderr)

    if debug:
        source = "command-line flag" if a.debug else "GENEWEAVER_DEBUG"
        debug_log(f"enabled by {source}; command={a.command}")
    if a.command in ("selftest", "test"):
        debug_log("starting built-in self-test")
        result = selftest()
        debug_log(f"built-in self-test exit code={result}")
        return result
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
    if a.command == "animate":
        if a.chunk_size < 20 or a.length < 6 * a.chunk_size + 25 or a.delay < 0:
            print("ERROR: need --chunk-size >= 20, --length >= 6 x chunk size + 25, and --delay >= 0")
            return 2
        try:
            play_frames(animation_frames(a.length, a.chunk_size), a.delay, ansi=sys.stdout.isatty() and not a.plain)
        except KeyboardInterrupt:
            print("\nstopped")
        return 0
    if a.command == "baseline":
        try:
            cfg = CONFIGS[a.config]
            debug_log(f"baseline config={a.config} implementation={a.impl} runs={a.runs} warmup={a.warmup}")
            if a.impl == "packed" and (cfg["db"] or cfg["rb"]):
                print(f"ERROR: {a.config} has bulges; the packed baseline is mismatch-only. Use --impl oracle (slow) or C1/C2/C3/C6.")
                return 2
            truth, load_s, sha = None, None, None
            if a.genome:
                debug_log(f"reading genome from {a.genome}")
                t0 = time.perf_counter()
                genome = read_genome(a.genome, a.parser)
                load_s = round(time.perf_counter() - t0, 4)
                sha, gid = sha256(a.genome), os.path.basename(a.genome)
                guides = read_guides_any(a.guides, a.parser) if a.guides else make_guides(genome, a.n_guides, a.seed)
                gset = os.path.basename(a.guides) if a.guides else f"random-{len(guides)}"
            else:
                chroms = [c.strip() for c in a.chroms.split(",")] if a.chroms else None
                debug_log(f"generating mock genome scale={a.scale} chromosomes={chroms or 'default'}")
                mock = generate_mock(scale=a.scale, chroms=chroms, seed=a.seed, n_guides=a.n_guides)
                genome = {c: ba.decode() for c, ba in mock["genome"].items()}
                guides, truth, gid, gset = mock["guides"], mock["truth"], "G-mock", "S-mock"
            total = sum(len(x) for x in genome.values())
            debug_log(f"loaded {len(genome)} contigs, {total:,} bp, {len(guides)} guides"
                      + (f" in {load_s:.4f}s" if load_s is not None else ""))
            if a.impl == "oracle" and total > 400_000 and not a.force:
                print(f"ERROR: the naive oracle on {total:,} bp is very slow; use a smaller genome or add --force.")
                return 2
            rec = run_baseline(genome, guides, a.config, a.impl, a.runs, a.warmup, gid, gset, truth, load_s, sha)
            debug_log(f"search complete: {rec['total_hits']} hits; saving record under {a.out_dir}")
            saved = save_baseline(rec, a.out_dir)
            print(format_baseline(rec, saved))
            return 0
        except (ValueError, RuntimeError, OSError) as e:
            print("ERROR:", e)
            return 2
    if a.command == "gpu":
        debug_log(f"GPU config={a.config} backend={a.backend} runs={a.runs} warmup={a.warmup}")
        return cmd_gpu(a)
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
    print("Loading configuration....")
    print("Done loading configuration")
    sys.exit(main())