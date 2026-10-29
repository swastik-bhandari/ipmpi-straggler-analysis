"""Figures for the 8-node AWS campaign (values from the campaign's analysis outputs)."""
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
BLUE, RED, GREY = "#3b6fb6", "#c8504f", "#9aa3ad"

# 1. NPB IS class B: the default Alltoallv algorithm choice creates a jam at 128 ranks
labels = ["64 ranks\ndefault\n(pairwise)", "64 ranks\nforced linear", "128 ranks\ndefault\n(linear)", "128 ranks\nforced pairwise"]
secs = [10.3, 112.0, 110.8, 10.6]
cols = [BLUE, RED, RED, BLUE]
fig, ax = plt.subplots(figsize=(7.5, 4))
b = ax.bar(labels, secs, color=cols, width=0.6)
for r, s in zip(b, secs):
    ax.text(r.get_x() + r.get_width() / 2, s + 2, f"{s:g} s", ha="center", va="bottom")
ax.set_ylabel("run time (s)")
ax.set_title("NPB IS (class B): Open MPI's Alltoallv algorithm, not the code, sets the run time", fontsize=11)
ax.set_ylim(0, 130)
fig.tight_layout(); fig.savefig("aws_is_alltoallv.png", dpi=160)

# 2. Exit spread of the previous collective vs entry spread of this one (HPCG, miniFE, miniAMR; 128 ranks, large)
rows = [  # (label, exit_us, entry_us, what creates the per-rank pattern)
    ("HPCG  ReportResults", 494.5, 493.5, "carried over"),
    ("miniAMR  profile.c:1309", 438.0, 446.0, "carried over"),
    ("miniFE  CG dot product (cg_solve:158)", 451.5, 472.2, "carried over"),
    ("miniAMR  refine.c:422", 418.5, 529.0, "point-to-point"),
    ("miniFE  CG dot product (cg_solve:179)", 419.0, 995.2, "point-to-point"),
    ("HPCG  ComputeResidual", 513.5, 8001.4, "compute"),
    ("HPCG  CG dot product", 492.5, 9456.0, "compute + p2p"),
    ("miniFE  matrix setup", 450.2, 10094.8, "compute"),
    ("miniAMR  final Barrier", 362.5, 136504.0, "compute"),
]
y = np.arange(len(rows))
fig, ax = plt.subplots(figsize=(9, 4.8))
ax.barh(y + 0.2, [r[1] for r in rows], height=0.38, color=GREY, label="exit spread of previous collective")
ax.barh(y - 0.2, [r[2] for r in rows], height=0.38, color=BLUE, label="entry spread at this collective")
ax.set_xscale("log"); ax.set_xlim(100, 3e6)
ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows]); ax.invert_yaxis()
for i, r in enumerate(rows):
    ax.text(max(r[1], r[2]) * 1.4, i, r[3], va="center", fontsize=9, color="#333")
ax.set_xlabel("spread across 128 ranks, median per call (µs, log scale)")
ax.set_title("Ranks leave each collective ~0.45 ms apart; large delays are created in between", fontsize=11)
ax.legend(loc="upper right", frameon=False, fontsize=9)
fig.tight_layout(); fig.savefig("aws_exit_vs_entry.png", dpi=160)

# 3. HPCG large, 128 ranks: the main CG Allreduce splits into rank, node and core effects
node = {"A": 1536.7, "B": 1056.1, "C": 828.0, "D": 277.6, "H": -1.2, "E": -185.6, "F": -881.2, "G": -2630.4}
core = {"0": 733.6, "1": 651.0, "2": 499.5, "4": 894.2, "9": 340.9, "14": 219.4}
rank = {"84": 1568.0, "81": 1503.2, "82": 1412.7, "80": 1381.7, "87": 1292.5, "77": 1175.7}
fig, axs = plt.subplots(1, 3, figsize=(11, 3.6), gridspec_kw={"width_ratios": [8, 6, 6]})
for ax, (t, d, c) in zip(axs, [("node effect (machine)", node, RED), ("core effect (largest)", core, GREY),
                               ("rank effect (largest)", rank, BLUE)]):
    k = list(d); v = [d[x] / 1000 for x in k]
    ax.bar(k, v, color=c); ax.axhline(0, color="#555", lw=0.8); ax.set_title(t, fontsize=11)
axs[0].set_ylabel("lateness (ms)")
axs[2].set_xlabel("ranks 80–87 = one x-row of the 8×4×4 process grid", fontsize=9)
fig.suptitle("HPCG (large, 128 ranks), main CG Allreduce: late because of machine, core and rank, measured separately",
             fontsize=11)
fig.tight_layout(); fig.savefig("aws_hpcg_rank_node_core.png", dpi=160)
print("ok")
