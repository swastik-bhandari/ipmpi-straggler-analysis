import os, csv, argparse
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"xtick.labelsize": 12, "ytick.labelsize": 12})
import numpy as np

COLOR_DEFAULT   = "#4C72B0"
COLOR_STRAGGLER = "#C44E52"
COLOR_MAXWAIT   = "#DD8452"
LINE_MIN        = "#C44E52"
LINE_MAX        = "#DD8452"

def load_labels(resolved_csv):
    labels = {}
    if not resolved_csv or not os.path.exists(resolved_csv):
        return labels
    with open(resolved_csv) as f:
        for row in csv.DictReader(f):
            cs   = int(row["callsite_id_raw"])
            func = row["func_name"]
            src  = row.get("source_function", "").strip().split("(")[0].strip()
            loc  = row.get("source_location", "").strip()
            if loc:
                fname  = os.path.basename(loc.split(":")[0])
                lineno = loc.split(":")[-1] if ":" in loc else ""
                labels[cs] = f"{func} @ {src}\n({fname}:{lineno})"
            else:
                labels[cs] = f"{func} @ {src}" if src else f"{func} (cs {cs})"
    return labels

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("agg_csv")
    parser.add_argument("--resolved-csv", default=None)
    parser.add_argument("--ranks-per-node", type=int, default=8)
    parser.add_argument("--top-n", type=int, default=4)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    run_name = os.path.basename(args.agg_csv).replace("_rank_delay_agg.csv", "")
    labels   = load_labels(args.resolved_csv)

    rows = []
    with open(args.agg_csv) as f:
        for row in csv.DictReader(f):
            rows.append({
                "rank": int(row["rank"]),
                "cs":   int(row["callsite_id"]),
                "func": row["func_name"],
                "nc":   int(row["num_calls"]),
                "mean": float(row["mean_delay_us"]),
                "sum":  float(row["sum_delay_us"]),
            })

    all_ranks = sorted(set(r["rank"] for r in rows))
    np_val    = max(all_ranks) + 1

    by_cs   = defaultdict(dict)
    cs_meta = {}
    for r in rows:
        by_cs[r["cs"]][r["rank"]] = r["mean"]
        cs_meta.setdefault(r["cs"], {"func": r["func"], "nc": r["nc"], "tot": 0.0})
        cs_meta[r["cs"]]["tot"] += r["sum"]

    top_cs = sorted(cs_meta, key=lambda c: -cs_meta[c]["tot"])[:args.top_n]
    print(f"Plotting top {len(top_cs)} callsites | {len(all_ranks)} ranks")

    rank_arr = np.array(all_ranks)
    mean_mat = np.zeros((len(top_cs), len(all_ranks)))
    for i, cs in enumerate(top_cs):
        for j, rk in enumerate(all_ranks):
            mean_mat[i, j] = by_cs[cs].get(rk, 0.0)

    rpn = args.ranks_per_node

    fig, axes = plt.subplots(len(top_cs), 1,
                             figsize=(16, 4.5 * len(top_cs)), sharex=True)
    if len(top_cs) == 1:
        axes = [axes]

    for i, (ax, cs) in enumerate(zip(axes, top_cs)):
        nc        = cs_meta[cs]["nc"]
        tot       = cs_meta[cs]["tot"]
        mean_vals = mean_mat[i]

        idx_min  = int(np.argmin(mean_vals))
        idx_max  = int(np.argmax(mean_vals))
        rank_min = rank_arr[idx_min]
        rank_max = rank_arr[idx_max]
        val_min  = mean_vals[idx_min]
        val_max  = mean_vals[idx_max]

        bar_colors = [COLOR_DEFAULT] * len(rank_arr)
        bar_colors[idx_max] = COLOR_MAXWAIT
        bar_colors[idx_min] = COLOR_STRAGGLER

        ax.bar(rank_arr, mean_vals, width=0.85, color=bar_colors, alpha=0.88)

        ax.axhline(y=val_max, color=LINE_MAX, linestyle="--",
                   linewidth=1.2, alpha=0.85, zorder=3,
                   label=f"max wait  R{rank_max} ({val_max:,.0f} ns)")
        ax.axhline(y=val_min, color=LINE_MIN,
                   linestyle="--", linewidth=1.2, alpha=0.85, zorder=3,
                   label=f"straggler R{rank_min} ({val_min:,.0f} ns)")

        ax.annotate(
            f"R{rank_min}",
            xy=(rank_min, val_min),
            xytext=(rank_min, val_max * 0.35),
            fontsize=9.5, ha="center", color=LINE_MIN, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=LINE_MIN, lw=1.0),
        )
        ax.annotate(
            f"R{rank_max}\n{val_max/1e6:,.1f} ms",
            xy=(rank_max, val_max),
            xytext=(rank_max, val_max * 1.10),
            fontsize=9.5, ha="center", color=LINE_MAX, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=LINE_MAX, lw=1.0),
        )

        ax.set_ylabel("Arrival delay / invocation (ns)", fontsize=11)
        ax.tick_params(axis="y", labelsize=7)
        ymax_lim = val_max * 1.35 if val_max > 0 else 1
        ax.set_ylim(0, ymax_lim)

        ax.legend(fontsize=9.5, loc="upper right", framealpha=0.7)

        # Node boundaries
        for xb in range(rpn, np_val, rpn):
            ax.axvline(x=xb - 0.5, color="#888888",
                       linestyle="--", linewidth=0.7, alpha=0.5, zorder=0)

        ymax = ax.get_ylim()[1]
        for nidx, xb in enumerate(range(0, np_val, rpn)):
            ax.text(xb + rpn / 2 - 0.5, ymax * 0.96,
                    f"N{nidx}", fontsize=8.5, ha="center",
                    va="top", color="#555555")

        lbl = labels.get(cs, f"{cs_meta[cs]['func']} (cs {cs})")
        ax.set_title(
            f"{lbl}  |  {nc} call(s)/rank  |  total arrival delay = {tot/1e9:.3f} s",
            fontsize=11.5, loc="left", pad=4)
        ax.grid(axis="y", linestyle="--", alpha=0.3, zorder=0)
        ax.set_xlim(-1, np_val)

    axes[-1].set_xlabel("MPI Rank", fontsize=14)
    fig.suptitle(
        f"{run_name}\n"
        f"Per-Rank arrival delay / invocation (ns)  |  "
        f"Red = straggler (arrived last)  |  Orange = longest waiter",
        fontsize=13, fontweight="bold")
    plt.tight_layout()

    out = args.out or args.agg_csv.replace("_rank_delay_agg.csv", "_rank_delay_plot.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")

if __name__ == "__main__":
    main()
