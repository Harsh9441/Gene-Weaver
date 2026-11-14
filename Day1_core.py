"""DAY 1 - Core types, configurations, FASTA I/O, canonical hit format (Rules s2, s5, s6.1)."""
from __future__ import annotations
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
    return sorted(set(hits), key=lambda h: (h.guide_id, h.chrom, h.start, h.strand))


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
        items = guides.items() if isinstance(guides, dict) else guides
        for gid, seq in items:
            fh.write(f"{gid}\t{seq}\n")


def read_guides(path: str):
    out = []
    with open(path) as fh:
        for line in fh:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                out.append((parts[0], parts[1]))
    return out