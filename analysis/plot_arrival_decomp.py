import os, sys, csv, glob, bisect, argparse
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"xtick.labelsize": 12, "ytick.labelsize": 12})
import matplotlib.patches as mpatches
import numpy as np

COLLECTIVES = {"MPI_Allreduce", "MPI_Bcast", "MPI_Barrier", "MPI_Reduce",
               "MPI_Scatter", "MPI_Gather", "MPI_Allgather", "MPI_Alltoall"}

C_COMPUTE   = "#4C72B0"
C_P2P       = "#55A868"
C_IDLE      = "#F4A58A"
E_STRAG     = "#C44E52"
E_WAIT      = "#DD8452"

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


def build_index(rank_events):
    """
    Precompute, per rank:
      tagidx : {collective_tag -> event_index}   (collectives only; tags unique)
      collidx: sorted list of collective event indices
    Enables O(log) previous-collective lookup instead of a per-invocation rescan.
    """
    rank_tagidx, rank_collidx = {}, {}
    for rank, evs in rank_events.items():
        tmap, cidx = {}, []
        for i, e in enumerate(evs):
            if e[F] in COLLECTIVES:
                tmap[e[TAG]] = i
                cidx.append(i)
        rank_tagidx[rank] = tmap
        rank_collidx[rank] = cidx
    return rank_tagidx, rank_collidx


def target_invocations(rank_events, target_raw):
    tag_entries = defaultdict(dict)
    for rank, evs in rank_events.items():
        for e in evs:
            if e[CS] == target_raw and e[F] in COLLECTIVES:
                tag_entries[e[TAG]][rank] = e[S]
    return tag_entries


def decomp_one(rank_events, rank_tagidx, rank_collidx, tag):
    """rank -> dict(compute, p2p, entry, interval) for invocation `tag`."""
    out = {}
    for rank, evs in rank_events.items():
        ti = rank_tagidx[rank].get(tag)
        if ti is None:
            continue
        cidx = rank_collidx[rank]
        pos = bisect.bisect_left(cidx, ti)
        if pos == 0:
            continue
        pj = cidx[pos - 1]
        entry, prev_end = evs[ti][S], evs[pj][E]
        interval = max(entry - prev_end, 0)
        p2p = sum(evs[k][E] - evs[k][S] for k in range(pj + 1, ti))
        p2p = min(max(p2p, 0), interval)
        out[rank] = {"compute": interval - p2p, "p2p": p2p,
                     "entry": entry, "interval": interval}
    return out


def add_idle(per):
    if not per:
        return per
    max_entry = max(d["entry"] for d in per.values())
    for d in per.values():
        d["idle"] = max_entry - d["entry"]
    return per


def neighbor_info(rank_events):
    """rank -> (num_send_neighbors, total_send_bytes). Blocking + nonblocking."""
    info = {}
    for rank, evs in rank_events.items():
        dests, hb = set(), 0
        for e in evs:
            if e[F] in ("MPI_Send", "MPI_Isend") and e[DST] >= 0:
                dests.add(e[DST]); hb += e[SZ]
        info[rank] = (len(dests), hb)
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("result_dir")
    ap.add_argument("--resolved-csv", default=None)
    ap.add_argument("--callsite",      type=int, default=None)
    ap.add_argument("--callsite-raw",  type=int, default=None)
    ap.add_argument("--mode", choices=["worst", "mean"], default="worst")
    ap.add_argument("--ranks-per-node", type=int, default=8)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    result_dir = args.result_dir.rstrip("/")
    run_name   = os.path.basename(result_dir)
    canon2raw, raw2label = load_labels(args.resolved_csv)

    print(f"Reading traces from {run_name} ...", flush=True)
    rank_events = parse_traces(result_dir)
    np_total = len(rank_events)
    print(f"  np = {np_total}")

    target_raw = args.callsite_raw
    if target_raw is None and args.callsite is not None:
        if args.callsite not in canon2raw:
            sys.exit(f"ERROR: canonical id {args.callsite} not in resolved csv")
        target_raw = canon2raw[args.callsite]

    if target_raw is None:
        print("\nAvailable collective callsites (pass one via --callsite / --callsite-raw):")
        seen = defaultdict(int)
        for evs in rank_events.values():
            for e in evs:
                if e[F] in COLLECTIVES:
                    seen[e[CS]] += 1
        print(f"  {'canonical':>9}  {'raw_id':>10}  {'records':>8}  label")
        raw2canon = {v: k for k, v in canon2raw.items()}
        for raw in sorted(seen, key=lambda r: -seen[r]):
            c = raw2canon.get(raw, "")
            print(f"  {str(c):>9}  {raw:>10}  {seen[raw]:>8}  "
                  f"{raw2label.get(raw, '(unresolved)')}")
        sys.exit(0)

    label = raw2label.get(target_raw, f"callsite {target_raw}")
    print(f"\nTarget: raw={target_raw}  |  {label}")

    rank_tagidx, rank_collidx = build_index(rank_events)
    tag_entries = target_invocations(rank_events, target_raw)
    tags = [t for t, d in tag_entries.items() if len(d) >= 2]
    if not tags:
        sys.exit(f"ERROR: no multi-rank invocations for callsite {target_raw}")
    print(f"  invocations found: {len(tags)}")

    ranks = sorted(rank_events.keys())
    comp = {r: 0.0 for r in ranks}
    p2p  = {r: 0.0 for r in ranks}
    idle = {r: 0.0 for r in ranks}
    cnt  = {r: 0   for r in ranks}

    if args.mode == "worst":
        worst_tag = max(tags, key=lambda t: (max(tag_entries[t].values())
                                             - min(tag_entries[t].values())))
        spread = (max(tag_entries[worst_tag].values())
                  - min(tag_entries[worst_tag].values()))
        print(f"  worst invocation: tag={worst_tag}  "
              f"arrival spread = {spread/1e3:.2f} ms")
        per = add_idle(decomp_one(rank_events, rank_tagidx, rank_collidx, worst_tag))
        for r, d in per.items():
            comp[r], p2p[r], idle[r], cnt[r] = d["compute"], d["p2p"], d["idle"], 1
        subtitle = f"worst invocation (tag {worst_tag}), spread {spread/1e3:.2f} ms"
    else:
        for t in tags:
            per = add_idle(decomp_one(rank_events, rank_tagidx, rank_collidx, t))
            for r, d in per.items():
                comp[r] += d["compute"]; p2p[r] += d["p2p"]
                idle[r] += d["idle"];    cnt[r] += 1
        for r in ranks:
            if cnt[r]:
                comp[r] /= cnt[r]; p2p[r] /= cnt[r]; idle[r] /= cnt[r]
        subtitle = f"mean over {len(tags)} invocations"

    active = [r for r in ranks if cnt[r] > 0]
    if not active:
        sys.exit("ERROR: no ranks decomposed (no previous collective found?)")

    totals   = np.array([comp[r] + p2p[r] + idle[r] for r in active])
    idle_arr = np.array([idle[r] for r in active])
    print("\n-- validation --")
    print(f"  stacked total  : mean={totals.mean()/1e3:.2f} ms  "
          f"std={totals.std()/1e3:.2f} ms  "
          f"(should be ~constant: all ranks leave the previous barrier together)")
    print(f"  idle (T_delay)  : max={idle_arr.max()/1e3:.2f} ms  "
          f"min={idle_arr.min()/1e3:.3f} ms  (straggler idle should be ~0)")
    strag = active[int(np.argmin([idle[r] for r in active]))]
    waiter = active[int(np.argmax([idle[r] for r in active]))]
    print(f"  straggler (min idle) : R{strag}  "
          f"compute={comp[strag]/1e3:.2f} p2p={p2p[strag]/1e3:.2f} "
          f"idle={idle[strag]/1e3:.2f} ms")
    print(f"  longest waiter (max idle): R{waiter}  "
          f"compute={comp[waiter]/1e3:.2f} p2p={p2p[waiter]/1e3:.2f} "
          f"idle={idle[waiter]/1e3:.2f} ms")

    nb = neighbor_info(rank_events)
    csv_out = (args.out or f"{run_name}_decomp_cs{target_raw}_{args.mode}.png"
               ).replace(".png", ".csv")
    with open(csv_out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "compute_us", "p2p_us", "idle_us", "total_us",
                    "send_neighbors", "send_bytes"])
        for r in ranks:
            n, hb = nb.get(r, (0, 0))
            w.writerow([r, f"{comp[r]:.1f}", f"{p2p[r]:.1f}", f"{idle[r]:.1f}",
                        f"{comp[r]+p2p[r]+idle[r]:.1f}", n, hb])
    print(f"\nWrote: {csv_out}")

    x = np.array(ranks)
    cvals = np.array([comp[r] for r in ranks]) / 1e3
    pvals = np.array([p2p[r]  for r in ranks]) / 1e3
    ivals = np.array([idle[r] for r in ranks]) / 1e3

    fig, ax = plt.subplots(figsize=(16, 6))
    ax.bar(x, cvals, width=0.9, color=C_COMPUTE, label="compute (local)")
    ax.bar(x, pvals, width=0.9, bottom=cvals, color=C_P2P,
           label="p2p (halo Send/Recv/Wait)")
    ax.bar(x, ivals, width=0.9, bottom=cvals + pvals, color=C_IDLE,
           label="idle (T_delay: wait at collective)")

    tots = cvals + pvals + ivals
    ax.set_ylim(0, float(tots.max()) * 1.18)
    for r, ec, tag_txt in [(strag, E_STRAG, "straggler"), (waiter, E_WAIT, "waiter")]:
        j = ranks.index(r)
        tot = float(tots[j])
        ax.bar(x[j], tot, width=0.9, facecolor="none",
               edgecolor=ec, linewidth=1.8, zorder=5)
        ax.annotate(f"R{r} {tag_txt}", xy=(r, tot),
                    xytext=(r, float(tots.max()) * 1.05), ha="center", va="bottom",
                    fontsize=11, color=ec, fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color=ec, lw=1.0),
                    annotation_clip=False)

    rpn = args.ranks_per_node
    for xb in []:
        ax.axvline(xb - 0.5, color="#888888", ls="--", lw=0.6, alpha=0.5, zorder=0)
    ymax = ax.get_ylim()[1]
    for nidx, xb in enumerate([]):
        ax.text(xb + rpn / 2 - 0.5, ymax * 0.97, f"N{nidx}", fontsize=9,
                ha="center", va="top", color="#666666")

    ax.set_xlabel("MPI Rank", fontsize=14)
    ax.set_ylabel("time per phase (ms, summed over interval)", fontsize=14)
    ax.set_xlim(-1, np_total)
    ax.grid(axis="y", ls="--", alpha=0.3, zorder=0)
    ax.legend(fontsize=12, loc="lower right", framealpha=0.95)
    ax.set_title(
        f"{run_name}  |  arrival decomposition\n"
        f"{label}  |  {subtitle}",
        fontsize=13, loc="left", pad=6)

    plt.tight_layout()
    out = args.out or f"{run_name}_decomp_cs{target_raw}_{args.mode}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
