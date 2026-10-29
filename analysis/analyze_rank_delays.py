import os, sys, csv, glob, argparse
from collections import defaultdict

COLLECTIVES = {"MPI_Allreduce", "MPI_Bcast", "MPI_Barrier",
               "MPI_Reduce", "MPI_Scatter", "MPI_Gather",
               "MPI_Allgather", "MPI_Alltoall"}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir")
    parser.add_argument("--out-dir", default=".")
    args = parser.parse_args()

    result_dir = args.result_dir.rstrip("/")
    run_name   = os.path.basename(result_dir)
    os.makedirs(args.out_dir, exist_ok=True)

    trace_files = sorted(glob.glob(os.path.join(result_dir, "comm_trace-*.txt")))
    if not trace_files:
        sys.exit(f"ERROR: no comm_trace-*.txt in {result_dir}")

    print(f"Run : {run_name}  |  np = {len(trace_files)}")

    call_groups = defaultdict(list)

    print("Reading...", end="", flush=True)
    total = 0
    for tf in trace_files:
        rank = int(os.path.basename(tf).replace("comm_trace-","").replace(".txt",""))
        with open(tf) as f:
            for row in csv.DictReader(f):
                if row["func_name"] not in COLLECTIVES:
                    continue
                call_groups[(int(row["tag"]), row["func_name"])].append((
                    rank,
                    int(row["start"]),
                    int(row["end"]),
                    int(row["callsite_id"]),
                ))
                total += 1
    print(f" {total} collective records | {len(call_groups)} unique invocations")

    # ── Compute arrival delay per rank per invocation ─────────────────────────
    # t_delay(r) = max_start_across_ranks - start(r)
    # Rank that arrives last has t_delay = 0 (it IS the straggler).
    # All other ranks have t_delay > 0 (they waited for the straggler).
    agg = defaultdict(lambda: {"func": "", "sum_us": 0.0, "max_us": 0.0, "count": 0})

    skipped = 0
    for (tag, func_name), entries in call_groups.items():
        if len(entries) < 2:
            skipped += 1
            continue

        starts      = [e[1] for e in entries]
        max_start   = max(starts)
        cs_ids      = [e[3] for e in entries]
        callsite_id = max(set(cs_ids), key=cs_ids.count)

        for (rank, start, end, _) in entries:
            t_delay_us = max_start - start      # nanoseconds (same unit as timestamps)
            k = (rank, callsite_id)
            agg[k]["func"]    = func_name
            agg[k]["sum_us"] += t_delay_us
            agg[k]["count"]  += 1
            if t_delay_us > agg[k]["max_us"]:
                agg[k]["max_us"] = t_delay_us

    if skipped:
        print(f"  (skipped {skipped} single-rank invocations)")

    agg_rows = []
    for (rank, cs), v in agg.items():
        mean_us = v["sum_us"] / v["count"]
        agg_rows.append({
            "rank":           rank,
            "callsite_id":    cs,
            "func_name":      v["func"],
            "num_calls":      v["count"],
            "mean_delay_us":  mean_us,
            "max_delay_us":   v["max_us"],
            "sum_delay_us":   v["sum_us"],
        })
    agg_rows.sort(key=lambda r: (r["callsite_id"], r["rank"]))

    agg_csv = os.path.join(args.out_dir, f"{run_name}_rank_delay_agg.csv")
    with open(agg_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["rank","callsite_id","func_name","num_calls",
                                          "mean_delay_us","max_delay_us","sum_delay_us"])
        w.writeheader()
        w.writerows(agg_rows)
    print(f"Wrote: {agg_csv}")

    by_cs = defaultdict(lambda: {"func": "", "nc": 0, "tot_us": 0.0, "means": []})
    for r in agg_rows:
        by_cs[r["callsite_id"]]["func"]    = r["func_name"]
        by_cs[r["callsite_id"]]["nc"]      = r["num_calls"]
        by_cs[r["callsite_id"]]["tot_us"] += r["sum_delay_us"]
        by_cs[r["callsite_id"]]["means"].append(r["mean_delay_us"])

    print(f"\n── Summary ──")
    print(f"  {'callsite_id':>12}  {'func':>22}  {'calls/rank':>10}  "
          f"{'mean_delay/call(µs)':>20}  {'total(s)':>10}")
    for cs, v in sorted(by_cs.items(), key=lambda x: -x[1]["tot_us"]):
        avg_us = sum(v["means"]) / len(v["means"])
        print(f"  {cs:>12}  {v['func']:>22}  {v['nc']:>10}  "
              f"{avg_us:>20.3f}  {v['tot_us']/1e9:>10.4f}")

    # ── Cross-check vs per_callsite_resolved.csv ──────────────────────────────
    # Note: per_callsite_resolved.csv uses the column name "t_sync_mean" (IPMPI
    # internal naming, not ours). We read it here only for cross-checking.
    # Our output uses "mean_delay_us" / "sum_delay_us" / "max_delay_us".
    # Ratio ≈ 1/np is expected: IPMPI averages across ALL ranks;
    # our value is per single rank.
    resolved = os.path.join(result_dir, "per_callsite_resolved.csv")
    if os.path.exists(resolved):
        print(f"\n── Cross-check vs per_callsite_resolved.csv ──")
        print(f"  All values in ns. Ratio ≈ 1/np is expected:")
        print(f"  arrival_delay in CSV = average over ALL ranks (last-to-arrive = 0).")
        print(f"  Our mean             = average per single rank.")
        print(f"  {'cs_raw':>12}  {'source_func':<30}  "
              f"{'arrival_delay_csv(ns)':>22}  {'our_mean(ns)':>14}  {'ratio':>6}")
        with open(resolved) as f:
            for row in csv.DictReader(f):
                cs          = int(row["callsite_id_raw"])
                delay_csv   = float(row["t_sync_mean"]) if row["t_sync_mean"] else 0.0
                src         = row.get("source_function", "").split("(")[0][:29]
                our         = [r["mean_delay_us"] for r in agg_rows
                               if r["callsite_id"] == cs]
                our_m       = sum(our) / len(our) if our else 0.0
                ratio       = our_m / delay_csv if delay_csv > 0 else float("nan")
                print(f"  {cs:>12}  {src:<30}  {delay_csv:>22.3f}  "
                      f"{our_m:>14.3f}  {ratio:>6.3f}")

if __name__ == "__main__":
    main()
