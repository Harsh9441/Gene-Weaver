
"""OVERALL DRIVER - runs the whole GeneWeaver benchmark (Days 1-6) end to end.

  python run_all.py selftest                       # no GPU, no tools needed
  python run_all.py correctness --manifest m.json --tool geneweaver --config C2 --guide-set S-100 --genome G-small
  python run_all.py bench       --manifest m.json --tool geneweaver --ref-tool casoffinder
  python run_all.py all         --manifest m.json --tool geneweaver --ref-tool casoffinder

Manifest (JSON):
  Optional: "commands": {"crispritz": "<template using {genome} {guides} {out} {pam} {mm} {db} {rb}>"}
  {"genomes": {"G-small": "ecoli.fa", "G-human": "grch38.fa"},
   "guide_source": "G-human", "workdir": "bench_out", "version": "<geneweaver commit>",
   "sizes": {"S-1": 1, "S-100": 100, "S-1k": 1000, "S-10k": 10000},
   "scale": {"G-small": "small", "G-chr": "chr", "G-human": "human"}}
"""
from __future__ import annotations
import argparse, json, os, random, sys
from day1_core import CONFIGS, read_fasta, read_guides, revcomp, sha256, write_guides
from day2_oracle import SEED, make_guides, oracle_search
from day3_runners import prepare
from day4_correctness import compare, edge_case_suite, spot_check, verdict
from day5_perf import build_record, save_record, timed_runs
from day6_scaling import STUDIES, markdown_report, run_studies, save_results

SIZES = {"S-1": 1, "S-100": 100, "S-1k": 1000, "S-10k": 10000}


# ---------------------------------------------------------------- selftest
def selftest():
    rng = random.Random(SEED)
    rand = lambda n: "".join(rng.choice("ACGT") for _ in range(n))
    spacer = "GACGTCTAGCATCGATCGAA"
    # forward site with 1 mismatch, reverse site exact, a DNA-bulge site
    mm = spacer[:5] + ("A" if spacer[5] != "A" else "C") + spacer[6:]
    bulged = spacer[:10] + "T" + spacer[10:]
    genome = {"c1": rand(300) + mm + "AGG" + rand(300) + revcomp(spacer + "TGG") + rand(200),
              "c2": rand(200) + bulged + "CGG" + rand(200)}
    guides = [("g0", spacer)]
    ok = True

    def check(name, cond):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'ok' if cond else 'FAIL'}] {name}")

    c = dict(CONFIGS["C2"], mm=2)
    fast = oracle_search(genome, guides, c)
    dp = oracle_search(genome, guides, c, force_dp=True)
    check("fast mismatch path == DP path (no bulges)", compare(fast, dp)["identical"])
    check("finds 1-mismatch forward site and exact reverse site",
          {(h.strand, h.mm) for h in fast} >= {("+", 1), ("-", 0)})
    cb = dict(CONFIGS["C4"], mm=2)
    bh = oracle_search(genome, guides, cb)
    check("finds DNA-bulge site on c2", any(h.chrom == "c2" and h.db == 1 for h in bh))
    check("bulge-free search misses the bulge site", not any(h.chrom == "c2" for h in fast))
    for name, passed in edge_case_suite(lambda g, gu, cf: oracle_search(g, gu, cf, force_dp=True)).items():
        check(f"edge case: {name}", passed)
    res = spot_check(bh, genome, guides, cb, k=50)
    check("spot-check: every hit re-verifies", res["invalid"] == 0)
    print("SELFTEST", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


# ---------------------------------------------------------------- helpers
def load_manifest(path):
    m = json.load(open(path))
    m.setdefault("workdir", "bench_out"); m.setdefault("version", "unknown")
    m.setdefault("sizes", SIZES); m.setdefault("scale", {})
    os.makedirs(m["workdir"], exist_ok=True)
    return m


def tool_cmd_for(m, tool):
    """Per-tool command template: manifest "commands": {tool: template}, falling back to "tool_cmd"."""
    return m.get("commands", {}).get(tool, m.get("tool_cmd"))


def get_guides(m, set_id):
    path = os.path.join(m["workdir"], f"{set_id}.tsv")
    if not os.path.exists(path):
        src = read_fasta(m["genomes"][m.get("guide_source", next(iter(m["genomes"])))])
        write_guides(make_guides(src, m["sizes"][set_id]), path)
    return read_guides(path), path


def run_point(m, tool, ref_tool, genome_id, set_id, cfg_id, runs=5, cold=False, use_oracle=False):
    gpath = m["genomes"][genome_id]                      # KeyError -> skipped by run_studies
    guides, gfile = get_guides(m, set_id)
    cfg = CONFIGS[cfg_id]
    genome_len = sum(len(s) for s in read_fasta(gpath).values())
    cmd, parse = prepare(tool, gpath, guides, gfile, cfg_id, m["workdir"], tool_cmd_for(m, tool))
    perf = timed_runs(cmd, runs=runs, cold=cold)
    hits = parse()
    correctness = {"verdict": "unchecked"}
    ref = None
    if use_oracle:
        ref = oracle_search(read_fasta(gpath), guides, cfg)
    elif ref_tool:
        rcmd, rparse = prepare(ref_tool, gpath, guides, gfile, cfg_id, m["workdir"], tool_cmd_for(m, ref_tool))
        timed_runs(rcmd, runs=1, warmup=0)
        ref = rparse()
    if ref is not None:
        res = compare(hits, ref)
        correctness = dict(res, verdict=verdict(res, cfg, m["scale"].get(genome_id, "small")))
        correctness["spot_check"] = spot_check(hits, read_fasta(gpath), guides, cfg, k=1000)["invalid"]
    rec = build_record(tool, m["version"], genome_id, sha256(gpath), genome_len, set_id,
                       len(guides), cfg_id, perf, len(hits), correctness)
    save_record(rec, os.path.join(m["workdir"], f"{tool}_{genome_id}_{set_id}_{cfg_id}.json"))
    return rec


def cmd_correctness(a):
    m = load_manifest(a.manifest)
    rec = run_point(m, a.tool, a.ref_tool, a.genome, a.guide_set, a.config, runs=1,
                    use_oracle=a.oracle)
    print(json.dumps(rec["correctness"], indent=2, default=str))
    return 0 if rec["correctness"].get("verdict") == "pass" else 1


def cmd_bench(a):
    m = load_manifest(a.manifest)
    mk = lambda tool, ref: (lambda g, s, c: run_point(m, tool, ref, g, s, c, runs=a.runs, cold=a.cold))
    results = run_studies(mk(a.tool, a.ref_tool))
    ref_results = run_studies(mk(a.ref_tool, None)) if a.ref_tool else None
    save_results(results, os.path.join(m["workdir"], "results.json"))
    report = markdown_report(results, ref_results)
    open(os.path.join(m["workdir"], "report.md"), "w").write(report)
    print(report)
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["selftest", "correctness", "bench", "all"])
    ap.add_argument("--manifest"); ap.add_argument("--tool", default="geneweaver")
    ap.add_argument("--ref-tool"); ap.add_argument("--genome", default="G-small")
    ap.add_argument("--guide-set", default="S-100"); ap.add_argument("--config", default="C2")
    ap.add_argument("--runs", type=int, default=5); ap.add_argument("--cold", action="store_true")
    ap.add_argument("--oracle", action="store_true", help="use CPU oracle as reference (small genomes)")
    a = ap.parse_args()
    if a.command == "selftest":
        return selftest()
    if not a.manifest:
        ap.error("--manifest is required")
    if a.command == "correctness":
        return cmd_correctness(a)
    if a.command == "bench":
        return cmd_bench(a)
    rc = cmd_correctness(a)          # all: correctness gate first (Rules s12), then performance
    return rc if rc else cmd_bench(a)


if __name__ == "__main__":
    sys.exit(main())