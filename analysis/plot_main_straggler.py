import os, sys, csv, glob, argparse
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"xtick.labelsize": 12, "ytick.labelsize": 12})
import numpy as np

COLLECTIVES = {"MPI_Allreduce", "MPI_Bcast", "MPI_Barrier", "MPI_Reduce",
               "MPI_Scatter", "MPI_Gather", "MPI_Allgather", "MPI_Alltoall"}

C_BAR   = "#4C72B0"
C_HOT   = "#C44E52"
C_BASE  = "#888888"


def load_labels(path):
    canon2raw, raw2label = {}, {}
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("result_dir")
    ap.add_argument("--resolved-csv", default=None)
    ap.add_argument("--callsite", type=int, default=None)
    ap.add_argument("--callsite-raw", type=int, default=None)
    ap.add_argument("--ranks-per-node", type=int, default=8)
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

    last_count = defaultdict(int)
    n_valid = 0
    for tag, d in tag_entries.items():
        if len(d) < 2:
            continue
        n_valid += 1
        last_rank = max(d, key=lambda r: d[r])
        last_count[last_rank] += 1

    if n_valid == 0:
        sys.exit("ERROR: no multi-rank invocations for this callsite")

    counts = {r: last_count.get(r, 0) for r in all_ranks}
    baseline = n_valid / max(np_total, 1)
    print(f"Target: {label}")
    print(f"  invocations={n_valid}  (bars sum to {n_valid})  "
          f"random-baseline count ~{baseline:.1f}")
    n_culprits = sum(1 for c in counts.values() if c > 0)
    print(f"  distinct ranks that were ever THE slowest: {n_culprits} of {np_total}")
    top = sorted(counts, key=lambda r: -counts[r])[:12]
    print("  WHICH rank was THE slowest, and how many times:")
    for r in top:
        if counts[r] == 0:
            break
        print(f"    R{r:<4} {counts[r]:>6}  ({100.0*counts[r]/n_valid:5.1f}% of invocations)")

    x = np.array(all_ranks)
    y = np.array([counts[r] for r in all_ranks], dtype=float)
    hot = y > (5 * baseline)
    colors = [C_HOT if h else C_BAR for h in hot]

    fig, ax = plt.subplots(figsize=(16, 5.2))
    ax.bar(x, y, width=0.9, color=colors)

    ax.axhline(baseline, ls="--", color=C_BASE, lw=1.1)

    rpn = args.ranks_per_node
    for xb in []:
        ax.axvline(xb - 0.5, color="#bbbbbb", ls="--", lw=0.6, alpha=0.5, zorder=0)
    ymax = max(y.max() * 1.18, baseline * 4)
    for nidx, xb in enumerate([]):
        ax.text(xb + rpn / 2 - 0.5, ymax * 0.97, f"N{nidx}", fontsize=9,
                ha="center", va="top", color="#999999")

    for r in top[:5]:
        if counts[r] > 5 * baseline:
            ax.annotate(f"R{r}\n{counts[r]}\n({100.0*counts[r]/n_valid:.0f}%)",
                        xy=(r, counts[r]),
                        xytext=(r, counts[r] + ymax * 0.03), ha="center",
                        fontsize=10.5, color=C_HOT, fontweight="bold")

    ax.set_xlabel("MPI Rank", fontsize=14)
    ax.set_ylabel(f"times this rank was THE slowest (of {n_valid})", fontsize=14)
    ax.set_xlim(-1, np_total)
    ax.set_ylim(0, ymax)
    ax.grid(axis="y", ls="--", alpha=0.3, zorder=0)
    ax.text(np_total * 0.5, baseline, f"random baseline {baseline:.1f} of {n_valid}",
            ha="center", va="bottom", fontsize=12, color=C_BASE)
    ax.set_title(
        f"{run_name}   |   main-straggler count (single last-arriver per invocation)\n"
        f"{label}   |   {n_valid} invocations   |   "
        f"one dominant bar = a single bottleneck rank gating the barrier",
        fontsize=13, loc="left", pad=8)

    plt.tight_layout()
    out = args.out or f"{run_name}_mainstraggler_cs{target_raw}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
