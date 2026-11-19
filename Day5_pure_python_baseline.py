"""Day 5: benchmark the pure-Python GeneWeaver baseline."""

import argparse
import json
import platform
import sys
from datetime import datetime, timezone

import fasta


def main(argv=None):
    ap = argparse.ArgumentParser(description="Run and record the pure-Python baseline")
    ap.add_argument("--genome", default="mock_genome/genome.fa")
    ap.add_argument("--guides", default="mock_genome/guides.tsv")
    ap.add_argument("--config", choices=sorted(fasta.CONFIGS), default="C2")
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--out", default="Day5_pure_python_baseline.json")
    args = ap.parse_args(argv)

    genome = fasta.read_genome(args.genome, parser="builtin")
    guides = fasta.read_guides(args.guides)
    record = fasta.pure_python_baseline(
        genome,
        guides,
        fasta.CONFIGS[args.config],
        warmup_runs=args.warmup,
        timed_runs=args.runs,
    )
    record.update({
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "config_name": args.config,
        "genome_path": args.genome,
        "guides_path": args.guides,
        "platform": platform.platform(),
        "python": sys.version.split()[0],
    })

    with open(args.out, "w") as fh:
        json.dump(record, fh, indent=2)
        fh.write("\n")

    print(f"tool        : {record['tool']}")
    print(f"config      : {args.config}")
    print(f"genome      : {record['genome_bp']:,} bp")
    print(f"guides      : {record['guide_count']:,}")
    print(f"timings     : {[round(t, 6) for t in record['raw_timing_seconds']]}")
    print(f"median      : {record['median_search_seconds']:.6f} seconds")
    print(f"throughput  : {record['guide_positions_per_second']:.2f} guide-positions/second")
    print(f"hits        : {record['hits']:,}")
    print(f"record      : {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
