import os, sys, csv, glob, math, argparse
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"xtick.labelsize": 12, "ytick.labelsize": 12})
import numpy as np

COLLECTIVES = {"MPI_Allreduce", "MPI_Bcast", "MPI_Barrier", "MPI_Reduce",
               "MPI_Scatter", "MPI_Gather", "MPI_Allgather", "MPI_Alltoall"}
C_BAR, C_HOT, C_BASE = "#4C72B0", "#C44E52", "#888888"


def load_labels(path):
    raw2label = {}
    canon2raw = {}
    if not path or not os.path.exists(path):
        return canon2raw, raw2label
    with open(path) as f:
        for row in csv.DictReader(f):
            try:
                raw = int(row["callsite_id_raw"])
            except (KeyError, ValueError):
                continue
            func = row.get("func_name", "")
            src  = row.get("source_function", "").strip().split("(")[0].strip()
            loc  = row.get("source_location", "").strip()
            fn   = os.path.basename(loc.split(":")[0]) if loc else ""
            ln   = loc.split(":")[-1] if ":" in loc else ""
            raw2label[raw] = (f"{func} @ {src} ({fn}:{ln})" if src
                              else f"{func} (cs {raw})")
            canon = row.get("canonical_callsite_id", "")
            if canon != "":
                try:
                    canon2raw[int(canon)] = raw
                except ValueError:
                    pass
    return canon2raw, raw2label


def node_of(rank, nnodes, np_total, mapby):
    if mapby == "block":
        per = max(1, math.ceil(np_total / nnodes))
        return rank // per
    return rank % nnodes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("result_dir")
    ap.add_argument("--resolved-csv", default=None)
    ap.add_argument("--callsite", type=int, default=None)
    ap.add_argument("--callsite-raw", type=int, default=None)
    ap.add_argument("--nodes", type=int, default=16)
    ap.add_argument("--map-by", choices=["node", "block"], default="node")
    ap.add_argument("--pct", type=float, default=5.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    result_dir = args.result_dir.rstrip("/")
    run_name = os.path.basename(result_dir)
    canon2raw, raw2label = load_labels(args.resolved_csv)

    target_raw = args.callsite_raw
    if target_raw is None and args.callsite is not None:
        target_raw = canon2raw.get(args.callsite)
    if target_raw is None:
        sys.exit("ERROR: pass --callsite (canonical) or --callsite-raw")
    label = raw2label.get(target_raw, f"callsite {target_raw}")

    print(f"Reading traces from {run_name} ...", flush=True)
    tag_entries = defaultdict(dict)
    files = sorted(glob.glob(os.path.join(result_dir, "comm_trace-*.txt")),
                   key=lambda p: int(os.path.basename(p)
                                     .replace("comm_trace-", "").replace(".txt", "")))
    if not files:
        sys.exit(f"ERROR: no comm_trace-*.txt in {result_dir}")
    for tf in files:
        rank = int(os.path.basename(tf).replace("comm_trace-", "").replace(".txt", ""))
        with open(tf) as f:
            for row in csv.DictReader(f):
                if int(row["callsite_id"]) == target_raw and row["func_name"] in COLLECTIVES:
                    tag_entries[int(row["tag"])][rank] = int(row["start"])

    all_ranks = sorted({r for d in tag_entries.values() for r in d})
    np_total = (max(all_ranks) + 1) if all_ranks else 0

    node_count = defaultdict(int)
    n_valid = 0
    for tag, d in tag_entries.items():
        if len(d) < 2:
            continue
        n_valid += 1
        k = max(1, int(math.ceil(len(d) * args.pct / 100.0)))
        for r in sorted(d, key=lambda r: d[r], reverse=True)[:k]:
            node_count[node_of(r, args.nodes, np_total, args.map_by)] += 1
    if n_valid == 0:
        sys.exit("ERROR: no multi-rank invocations for this callsite")

    tot = sum(node_count.values())
    baseline = 100.0 / args.nodes
    pct = {n: 100.0 * node_count.get(n, 0) / tot for n in range(args.nodes)}

    print(f"Target: {label}   map-by={args.map_by}   nodes={args.nodes}")
    print(f"  invocations={n_valid}  uniform baseline={baseline:.1f}%/node")
    for n in sorted(pct, key=lambda n: -pct[n]):
        print(f"    node{n:2d}: {pct[n]:5.1f}%")

    x = np.arange(args.nodes)
    y = np.array([pct[n] for n in range(args.nodes)])
    hot = y > 2 * baseline
    colors = [C_HOT if h else C_BAR for h in hot]

    fig, ax = plt.subplots(figsize=(11, 5.2))
    ax.bar(x, y, width=0.72, color=colors)
    ax.axhline(baseline, ls="--", color=C_BASE, lw=1.2)

    for n in range(args.nodes):
        if y[n] > 2 * baseline:
            ax.annotate(f"{y[n]:.0f}%", xy=(n, y[n]),
                        xytext=(n, y[n] + y.max() * 0.02), ha="center",
                        fontsize=11.5, color=C_HOT, fontweight="bold")

    ax.set_xlabel(f"physical node id  (rank mod {args.nodes}, --map-by {args.map_by})", fontsize=14)
    ax.set_ylabel(f"% of straggler-slots (slowest {args.pct:.0f}%)", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([f"N{n}" for n in range(args.nodes)], fontsize=11)
    ax.set_ylim(0, max(y.max() * 1.15, 2 * baseline))
    ax.grid(axis="y", ls="--", alpha=0.3, zorder=0)
    ax.text(args.nodes - 0.5, baseline, f"uniform baseline {baseline:.1f}%",
            ha="right", va="bottom", fontsize=12, color=C_BASE)
    ax.set_title(
        f"{run_name}   |   straggler share by physical node\n"
        f"{label}   |   {n_valid} invocations   |   "
        f"tall bar = persistently slow host",
        fontsize=13, loc="left", pad=8)

    plt.tight_layout()
    out = args.out or f"{run_name}_bynode_cs{target_raw}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
