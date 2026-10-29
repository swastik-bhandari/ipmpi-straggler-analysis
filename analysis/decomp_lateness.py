#!/usr/bin/env python3
"""Split each rank's lateness at a collective into: exit time from the previous collective (MPI
algorithm order), computation since then, and point-to-point MPI time since then.
  start = prev_collective_end + p2p_time + compute   (per rank, per invocation; clock-corrected)
After centering each term per invocation, lateness = exit + p2p + compute exactly.
Usage: decomp_lateness.py <run folder> <callsite_id> [<callsite_id> ...]
       decomp_lateness.py <run folder> <call-site text> ... --summary <csv with call_site,callsite_id>
"""
import argparse, glob, re
import numpy as np, pandas as pd

COLL = {"MPI_Allreduce", "MPI_Bcast", "MPI_Barrier", "MPI_Reduce", "MPI_Scatter", "MPI_Gather",
        "MPI_Allgather", "MPI_Alltoall", "MPI_Alltoallv", "MPI_Allgatherv", "MPI_Scan", "MPI_Reduce_scatter"}


def load(run, ids):
    off = {}
    try:
        o = pd.read_csv(run + "/clock_offsets.csv"); off = dict(zip(o["rank"].astype(int), o["offset_us"].astype(float)))
    except FileNotFoundError:
        pass
    recs = []
    files = glob.glob(run + "/comm_trace-*.txt.gz") + glob.glob(run + "/comm_trace-*.txt")
    for f in files:
        r = int(re.search(r"comm_trace-(\d+)", f).group(1))
        t = pd.read_csv(f, usecols=["func_name", "start", "end", "tag", "callsite_id"])
        t = t.sort_values("start", kind="stable").reset_index(drop=True)
        st = t["start"].to_numpy(float) - off.get(r, 0.0); en = t["end"].to_numpy(float) - off.get(r, 0.0)
        isc = t["func_name"].isin(COLL).to_numpy() & (t["tag"].to_numpy() > 0)
        p2p = np.where(t["func_name"].isin(COLL).to_numpy(), 0.0, en - st).cumsum()
        ci = np.flatnonzero(isc)
        cs = t["callsite_id"].to_numpy()
        for i_d in ids:
            tgt = ci[cs[ci] == i_d]
            pos = np.searchsorted(ci, tgt)
            ok = pos > 0
            tgt, prev = tgt[ok], ci[pos[ok] - 1]
            recs.append(pd.DataFrame(dict(cs=i_d, rank=r, tag=t["tag"].to_numpy()[tgt], start=st[tgt],
                                          prev_end=en[prev], p2p=p2p[tgt - 1] - p2p[prev])))
    d = pd.concat(recs, ignore_index=True)
    d["compute"] = d["start"] - d["prev_end"] - d["p2p"]
    n = d["rank"].nunique()
    d = d[d.groupby(["cs", "tag"])["rank"].transform("size") == n].copy()
    g = d.groupby(["cs", "tag"])
    for c in ("start", "prev_end", "p2p", "compute"):
        d[c + "_c"] = d[c] - g[c].transform("mean")
    return d, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run"); ap.add_argument("sites", nargs="+")
    ap.add_argument("--summary", default=None)
    ap.add_argument("--top", type=int, default=6)
    a = ap.parse_args()
    pick = {}
    if a.summary:
        s = pd.read_csv(a.summary)
        for txt in a.sites:
            m = s[s["call_site"].str.contains(txt, regex=False)]
            for _, row in m.iterrows():
                pick[int(row["callsite_id"])] = row["call_site"]
    else:
        pick = {int(x): f"callsite {x}" for x in a.sites}
    if not pick:
        raise SystemExit("no call site matched")
    d, n = load(a.run, list(pick))
    print(f"run {a.run}  ({n} ranks)")
    for cid, name in pick.items():
        x = d[d["cs"] == cid]
        if x.empty:
            print(f"\n== {name}: no complete invocations"); continue
        r = x.groupby("rank")[["start_c", "prev_end_c", "p2p_c", "compute_c", "p2p", "compute"]].mean()
        v = r["start_c"].var()
        share = {c: (r[c].cov(r["start_c"]) / v if v > 0 else float("nan")) for c in ("prev_end_c", "p2p_c", "compute_c")}
        ninv = x["tag"].nunique()
        print(f"\n== {name}   invocations {ninv}   rank-pattern spread (sd) {np.sqrt(v):.1f} us")
        print(f"   share of the rank pattern:  previous-collective exit {share['prev_end_c']:+.2f} | "
              f"point-to-point MPI {share['p2p_c']:+.2f} | compute {share['compute_c']:+.2f}")
        print(f"   per invocation (median rank): compute {r['compute'].median():.1f} us, p2p {r['p2p'].median():.1f} us")
        top = r.sort_values("start_c", ascending=False).head(a.top)
        print("   latest ranks: lateness = exit + p2p + compute (us)")
        for k, q in top.iterrows():
            print(f"     rank {k:4d}: {q.start_c:+8.1f} = {q.prev_end_c:+8.1f} {q.p2p_c:+8.1f} {q.compute_c:+8.1f}")


if __name__ == "__main__":
    main()
