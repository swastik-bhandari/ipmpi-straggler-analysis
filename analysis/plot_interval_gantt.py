import os, sys, csv, glob, argparse
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"xtick.labelsize": 12, "ytick.labelsize": 12})
import matplotlib.patches as mpatches
import numpy as np

COLLECTIVES = {"MPI_Allreduce", "MPI_Bcast", "MPI_Barrier", "MPI_Reduce",
               "MPI_Scatter", "MPI_Gather", "MPI_Allgather", "MPI_Alltoall"}
WAIT_FUNCS = {"MPI_Wait", "MPI_Waitany", "MPI_Waitall", "MPI_Waitsome"}
POST_FUNCS = {"MPI_Isend", "MPI_Irecv", "MPI_Send", "MPI_Recv"}

COL = {
    "MPI_Allreduce": "#C44E52", "MPI_Bcast": "#8C1FAB", "MPI_Barrier": "#C44E52",
    "MPI_Reduce": "#C44E52",
    "MPI_Send": "#2E7D46", "MPI_Isend": "#2E7D46",
    "MPI_Recv": "#6FBF8B", "MPI_Irecv": "#6FBF8B",
    "MPI_Wait": "#8899AA", "MPI_Waitall": "#8899AA",
    "MPI_Waitany": "#8899AA", "MPI_Waitsome": "#8899AA",
}
COMPUTE_COL = "#4C72B0"
WAIT_COL    = "#8899AA"
POST_COL    = "#2E7D46"
TICK_COL    = "#123D66"
DEFAULT_COL = "#BBBBBB"

S, E, F, TAG, CS, SRC, DST, SZ = range(8)


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


def parse_traces(result_dir):
    files = sorted(
        glob.glob(os.path.join(result_dir, "comm_trace-*.txt")),
        key=lambda p: int(os.path.basename(p)
                          .replace("comm_trace-", "").replace(".txt", "")))
    if not files:
        sys.exit(f"ERROR: no comm_trace-*.txt in {result_dir}")
    rank_events = {}
    for tf in files:
        rank = int(os.path.basename(tf)
                   .replace("comm_trace-", "").replace(".txt", ""))
        evs = []
        with open(tf) as f:
            for row in csv.DictReader(f):
                evs.append((
                    int(row["start"]), int(row["end"]), row["func_name"],
                    int(row["tag"]), int(row["callsite_id"]),
                    int(row["src"]), int(row["dest"]), int(row["msg_size"]),
                ))
        evs.sort(key=lambda e: e[S])
        rank_events[rank] = evs
    return rank_events


def target_invocations(rank_events, target_raw):
    tag_entries = defaultdict(dict)
    for rank, evs in rank_events.items():
        for e in evs:
            if e[CS] == target_raw and e[F] in COLLECTIVES:
                tag_entries[e[TAG]][rank] = e[S]
    return tag_entries


def interval_slice(evs, target_raw, tag):
    ti = next((i for i, e in enumerate(evs)
               if e[TAG] == tag and e[CS] == target_raw), None)
    if ti is None:
        return None
    pj = next((j for j in range(ti - 1, -1, -1)
               if evs[j][F] in COLLECTIVES), None)
    if pj is None:
        return None
    return (evs[pj][E], evs[ti][S], evs[ti][E], evs[pj + 1:ti])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("result_dir")
    ap.add_argument("--resolved-csv", default=None)
    ap.add_argument("--callsite", type=int, default=None)
    ap.add_argument("--callsite-raw", type=int, default=None)
    ap.add_argument("--tag", type=int, default=None)
    ap.add_argument("--pick", type=int, nargs=3, default=None,
                    help="override the 3 ranks: straggler waiter median")
    ap.add_argument("--zoom", type=float, default=None,
                    help="clip x to the last N ms before the barrier")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    result_dir = args.result_dir.rstrip("/")
    run_name = os.path.basename(result_dir)
    canon2raw, raw2label = load_labels(args.resolved_csv)

    print(f"Reading traces from {run_name} ...", flush=True)
    rank_events = parse_traces(result_dir)

    target_raw = args.callsite_raw
    if target_raw is None and args.callsite is not None:
        target_raw = canon2raw.get(args.callsite)
    if target_raw is None:
        sys.exit("ERROR: pass --callsite (canonical) or --callsite-raw")
    label = raw2label.get(target_raw, f"callsite {target_raw}")
    print(f"Target: raw={target_raw}  |  {label}")

    tag_entries = target_invocations(rank_events, target_raw)
    tags = [t for t, d in tag_entries.items() if len(d) >= 2]
    if not tags:
        sys.exit("ERROR: no multi-rank invocations for this callsite")

    if args.tag is not None:
        tag = args.tag
    else:
        tag = max(tags, key=lambda t: (max(tag_entries[t].values())
                                       - min(tag_entries[t].values())))
    entries = tag_entries[tag]
    spread = max(entries.values()) - min(entries.values())

    exit_all = {}
    for rank, evs in rank_events.items():
        for e in evs:
            if e[TAG] == tag and e[CS] == target_raw and e[F] in COLLECTIVES:
                exit_all[rank] = e[E]
                break
    exit_spread = (max(exit_all.values()) - min(exit_all.values())
                   if len(exit_all) >= 2 else 0)
    print(f"Invocation tag={tag}  arrival spread = {spread/1e3:.2f} ms  "
          f"exit spread = {exit_spread/1e3:.2f} ms")

    if args.pick:
        strag, waiter, median = args.pick
    else:
        strag  = max(entries, key=lambda r: entries[r])
        waiter = min(entries, key=lambda r: entries[r])
        srt = sorted(entries, key=lambda r: entries[r])
        median = srt[len(srt) // 2]
    rows = [(strag, "straggler (arrived last)", "#C44E52"),
            (median, "median rank", "#555555"),
            (waiter, "longest waiter (arrived first)", "#DD8452")]
    print(f"Rows: straggler R{strag} | median R{median} | waiter R{waiter}")

    slices = {}
    for r, _, _ in rows:
        sl = interval_slice(rank_events[r], target_raw, tag)
        if sl is None:
            sys.exit(f"ERROR: could not slice interval for R{r}")
        slices[r] = sl
    t_ref = min(sl[0] for sl in slices.values())
    def ms(us): return (us - t_ref) / 1e3

    fig, ax = plt.subplots(figsize=(16, 5.0))
    bar_h = 0.58
    straggler_entry_ms = ms(slices[strag][1])
    global_end_ms = max(ms(sl[2]) for sl in slices.values())
    seen = set()

    for yi, (r, tag_txt, ccol) in enumerate(rows):
        y = len(rows) - 1 - yi
        prev_end, entry, end, between = slices[r]

        cursor = prev_end
        segs = []
        budget = {"compute": 0.0, "wait": 0.0, "post": 0.0, "coll": 0.0}
        wait_starts = []
        for ev in between:
            if ev[S] > cursor:
                segs.append((cursor, ev[S], "compute", None))
                budget["compute"] += ev[S] - cursor
            segs.append((ev[S], ev[E], "mpi", ev[F]))
            if ev[F] in WAIT_FUNCS:
                budget["wait"] += ev[E] - ev[S]
                wait_starts.append(ev[S])
            elif ev[F] in POST_FUNCS:
                budget["post"] += ev[E] - ev[S]
            cursor = max(cursor, ev[E])
        if entry > cursor:
            segs.append((cursor, entry, "compute", None))
            budget["compute"] += entry - cursor
        segs.append((entry, end, "mpi", "MPI_Allreduce"))
        budget["coll"] += end - entry

        for s_us, e_us, kind, func in segs:
            x0 = ms(s_us); w = max(ms(e_us) - x0, 0.0005)
            color = COMPUTE_COL if kind == "compute" else COL.get(func, DEFAULT_COL)
            seen.add("compute" if kind == "compute" else func)
            ax.barh(y, w, left=x0, height=bar_h, color=color,
                    edgecolor="none", zorder=3)

        if wait_starts:
            xs = [ms(s) for s in wait_starts]
            ax.vlines(xs, y - bar_h/2 - 0.14, y - bar_h/2 - 0.02,
                      color=TICK_COL, lw=0.5, alpha=0.55, zorder=4)
            seen.add("_ticks")

        lab = (f"R{r}\n{tag_txt}\n"
               f"compute {budget['compute']/1e3:.0f} ms  |  "
               f"halo-wait {budget['wait']/1e3:.0f} ms")
        ax.text(-0.012, y, lab, transform=ax.get_yaxis_transform(),
                ha="right", va="center", fontsize=11, color=ccol,
                fontweight="bold")

    ax.axvline(straggler_entry_ms, ls="--", color="#C44E52", lw=1.4, zorder=5)
    ax.axvline(0, ls=":", color="#888888", lw=1.0, zorder=1)

    if args.zoom:
        ax.set_xlim(straggler_entry_ms - args.zoom, global_end_ms + args.zoom * 0.04)
        zoom_txt = f"  |  zoom: last {args.zoom:.0f} ms"
    else:
        ax.set_xlim(-spread/1e3*0.02, global_end_ms + spread/1e3*0.02)
        zoom_txt = ""

    ax.set_yticks([])
    ax.set_ylim(-0.8, len(rows) - 0.4)
    ax.set_xlabel("time since previous collective (ms)", fontsize=14)
    ax.grid(axis="x", ls="--", alpha=0.3, zorder=0)
    ax.set_title(
        f"{run_name}   |   interval timeline (3 ranks){zoom_txt}\n"
        f"{label}   |   tag {tag}   |   arrival spread {spread/1e3:.1f} ms   "
        f"vs   exit spread {exit_spread/1e3:.1f} ms",
        fontsize=13, loc="left", pad=8)

    handles = [mpatches.Patch(color=COMPUTE_COL, label="compute (local)")]
    if any(f in seen for f in WAIT_FUNCS):
        handles.append(mpatches.Patch(color=WAIT_COL, label="MPI_Waitany (halo wait)"))
    if any(f in seen for f in POST_FUNCS):
        handles.append(mpatches.Patch(color=POST_COL, label="Isend/Irecv (posts, sub-pixel)"))
    if "_ticks" in seen:
        handles.append(plt.Line2D([0], [0], color=TICK_COL, lw=1.0,
                                  label="halo-wait start (tick)"))
    if "MPI_Allreduce" in seen:
        handles.append(mpatches.Patch(color=COL["MPI_Allreduce"], label="MPI_Allreduce"))
    handles += [
        plt.Line2D([0], [0], color="#C44E52", ls="--", lw=1.4, label="straggler arrival"),
        plt.Line2D([0], [0], color="#888888", ls=":", lw=1.0, label="prev barrier (t=0)"),
    ]
    ax.legend(handles=handles, fontsize=11, loc="upper left",
              bbox_to_anchor=(1.005, 1.0), borderaxespad=0, framealpha=0.95)

    plt.tight_layout()
    zt = f"_zoom{int(args.zoom)}" if args.zoom else ""
    out = args.out or f"{run_name}_gantt_cs{target_raw}_tag{tag}{zt}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
