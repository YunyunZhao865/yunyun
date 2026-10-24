"""Regenerate Figs. 2-4 from the real analysis outputs (results/*.csv)."""
import os, json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES, FIG = os.path.join(ROOT, "results"), os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.family": "Times New Roman", "font.size": 10, "mathtext.fontset": "stix"})

# ---------------------------------------------------------------- Fig. 2 frontier
fr = pd.read_csv(os.path.join(RES, "frontier.csv"))
fig, ax = plt.subplots(figsize=(6.2, 4.6))
ax.plot(fr["risk"], fr["planned"], "-", color="#333a44", lw=1.8)
ax.plot(fr["risk"], fr["planned"], "o", color="#333a44", ms=4)
for _, r in fr.iterrows():
    if r["ratio"] in (5, 10, 20, 40):
        ax.annotate(f"$C_f/C_p$ = {r['ratio']:g}\n($p^*$ = {r['threshold']:.3f})", (r["risk"], r["planned"]),
                    textcoords="offset points", xytext=(8, 4) if r["ratio"] != 40 else (6, -26), fontsize=8.5)
base = fr[fr["ratio"] == 20].iloc[0]
ax.plot(base["risk"], base["planned"], "o", ms=8, mfc="#b0283f", mec="k", zorder=5)
ax.set_xlabel("Residual risk exposure (normalized, run-to-failure total = 100)")
ax.set_ylabel("Planned maintenance cost (normalized)")
ax.text(0.98, 0.97, r"$\rho$ = 0.9", transform=ax.transAxes, ha="right", va="top")
ax.set_xlim(fr["risk"].min() - 1, fr["risk"].max() + 3)
ax.set_ylim(0, fr["planned"].max() * 1.2)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "Fig2_cost_risk_frontier.png"), dpi=300)
plt.close(fig)

# ---------------------------------------------------------------- Fig. 3 partition
t5 = pd.read_csv(os.path.join(RES, "table5_economic_comparison.csv"), index_col=0)
labels = ["Run-to-\nfailure", "Periodic\nPM", "Fixed-thr.\nPdM", "Cost-\nsensitive", "Cost-adaptive\n(proposed)"]
fig, ax = plt.subplots(figsize=(6.2, 4.6))
x = np.arange(len(t5))
b1 = ax.bar(x, t5["planned"], 0.62, color="#3b5ca8", edgecolor="k", lw=0.6, label="Planned maintenance cost")
b2 = ax.bar(x, t5["risk"], 0.62, bottom=t5["planned"], color="#a8293f", edgecolor="k", lw=0.6,
            label="Residual risk exposure")
for i, (pl, rk, tt) in enumerate(zip(t5["planned"], t5["risk"], t5["total"])):
    ax.text(i, tt + 3, f"{tt:.1f}", ha="center", fontweight="bold" if i == 4 else None)
    if rk > 8:
        ax.text(i, pl + rk / 2, f"{rk:.1f}", ha="center", va="center", color="w")
    if pl > 8:
        ax.text(i, pl / 2, f"{pl:.1f}", ha="center", va="center", color="w")
    elif pl > 0:
        ax.text(i + 0.36, pl / 2 + 1, f"{pl:.1f}", ha="left", va="center", fontsize=8.5)
ax.set_xticks(x, labels)
ax.set_ylabel("Cost (normalized, run-to-failure total = 100)")
ax.set_ylim(0, 170)
ax.legend(frameon=False, loc="upper right")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "Fig3_cost_partition.png"), dpi=300)
plt.close(fig)

# ---------------------------------------------------------------- Fig. 4 sensitivity
se = pd.read_csv(os.path.join(RES, "sensitivity.csv"))
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6), sharey=True)
vmax = np.nanmax(np.abs(se["adv_vs_cs"].clip(-30, 30)))
for ax, kind, xl in zip(axes, ["additive", "logit"],
                        ["(a) Additive noise SD on calibrated probability", "(b) Noise SD on logit scale"]):
    d = se[se["noise"] == kind].pivot(index="ratio", columns="sigma", values="adv_vs_cs")
    im = ax.imshow(d.values.clip(-30, 30), aspect="auto", origin="lower", cmap="RdBu", vmin=-30, vmax=30)
    ax.set_xticks(range(d.shape[1]), [f"{c:g}" for c in d.columns])
    ax.set_yticks(range(d.shape[0]), [f"{r:g}" for r in d.index])
    for i in range(d.shape[0]):
        for j in range(d.shape[1]):
            v = d.values[i, j]
            ax.text(j, i, f"{v:.0f}".replace("-0", "0") if abs(v) < 0.5 else f"{v:.0f}", ha="center", va="center", fontsize=7.5,
                    color="w" if abs(v) > 18 else "k")
    ax.set_xlabel(xl, fontsize=9)
axes[0].set_ylabel("Loss-to-cost ratio $C_f/C_p$")
cb = fig.colorbar(im, ax=axes, fraction=0.04, pad=0.02)
cb.set_label("Relative cost advantage over cost-sensitive baseline (%)", fontsize=8.5)
fig.savefig(os.path.join(FIG, "Fig4_sensitivity.png"), dpi=300, bbox_inches="tight")
plt.close(fig)
print("figures written to", FIG)
