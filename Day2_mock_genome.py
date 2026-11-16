"""DAY 2 - Mock human genome generator.

Builds a scaled-down, human-like genome so the benchmark can be exercised without GRCh38:
  * GRCh38 chromosome length proportions (chr1-22, X, Y) times --scale
  * isochores (GC 36-52%) and CpG depletion
  * repeat families, soft-masked (lowercase): Alu-like, L1-like fragments, other interspersed
    families, microsatellites, and an alpha-satellite-like centromere array
  * N gaps: telomeres, centromere edge, a few internal gaps
  * planted off-target sites (mismatch and optional DNA/RNA-bulge variants) with a truth table,
    so correctness can be checked against known answers.
Outputs: genome.fa, guides.tsv, planted_truth.tsv, m.json (manifest ready for the benchmark).
"""
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


def _deplete_cpg(rng, s, keep=0.25):
    return re.sub("CG", lambda m: m.group(0) if rng.random() < keep else rng.choice(("TG", "CA")), s)


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
        raise RuntimeError("could not find a free site; use a larger --scale")

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
    rng = random.Random(seed)
    names = chroms or list(GRCH38_MB)
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