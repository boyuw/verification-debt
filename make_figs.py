#!/usr/bin/env python3
"""results/summary.json -> figures/*.pdf

Every plotted number comes from summary.json, which is itself a pure function of
results/raw/*.jsonl.  Nothing here is typed by hand.
"""

from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
FIG = os.path.join(HERE, "figures")

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 160, "savefig.bbox": "tight", "pdf.fonttype": 42,
})

LABEL = {"invoices": "Invoice\nexceptions", "cleansing": "Data\ncleansing",
         "emails": "Email\nresolutions", "reports": "Report\nsummaries"}
C_V, C_R = "#2f6f8f", "#c46a3a"


def load():
    with open(os.path.join(RES, "summary.json")) as fh:
        return json.load(fh)


def fig_decomposition(s, model, path):
    rows = s["by_model"].get(model, [])
    if not rows:
        return
    rows = sorted(rows, key=lambda r: -r["D"])
    fig, ax = plt.subplots(figsize=(4.6, 2.9))
    x = np.arange(len(rows))
    v = [r["V_over_S"] for r in rows]
    r_ = [r["R_over_S"] for r in rows]
    ax.bar(x, v, color=C_V, label="verification $V/S$", width=0.62)
    ax.bar(x, r_, bottom=v, color=C_R, label="rework $R/S$", width=0.62)
    for i, row in enumerate(rows):
        ax.text(i, row["D"] + 0.06, f"{row['D']:.2f}", ha="center", va="bottom", fontsize=8)
    ax.axhline(1.0, color="0.25", lw=1.0, ls="--")
    ax.text(len(rows) - 0.45, 1.02, "$D=1$", fontsize=7.5, color="0.25", ha="right")
    ax.set_xticks(x)
    ax.set_xticklabels([LABEL.get(r["family"], r["family"]) for r in rows], fontsize=8)
    ax.set_ylabel("cost per unit of work displaced  ($D$)")
    ax.set_title(f"Debt-ratio decomposition, {model}")
    ax.legend(frameon=False, fontsize=7.5, loc="upper right")
    ax.set_ylim(0, max(max(r["D"] for r in rows) * 1.22, 1.25))
    fig.savefig(path)
    plt.close(fig)


def fig_monotone(s, path):
    rows = s["pooled"]
    vis = [s["verifiability"][r["family"]] for r in rows]
    D = [r["D"] for r in rows]
    lo = [r["D"] - r["D_lo"] for r in rows]
    hi = [r["D_hi"] - r["D"] for r in rows]
    fig, ax = plt.subplots(figsize=(4.3, 2.9))
    ax.errorbar(vis, D, yerr=[lo, hi], fmt="o", ms=6, color="#333333",
                ecolor="0.6", capsize=3, zorder=3)
    # per-family offsets keep the two near-tied low-D labels (emails/cleansing) apart
    OFF = {"invoices": (7, 6), "cleansing": (7, 6), "emails": (7, -13), "reports": (7, 6)}
    for r, x_, y_ in zip(rows, vis, D):
        dx, dy = OFF.get(r["family"], (7, 6))
        ax.annotate(LABEL.get(r["family"], r["family"]).replace("\n", " "), (x_, y_),
                    textcoords="offset points", xytext=(dx, dy), fontsize=7.5)
    if len(rows) > 1:
        m, b = np.polyfit(vis, D, 1)
        xs = np.linspace(min(vis) - 0.04, max(vis) + 0.04, 20)
        ax.plot(xs, m * xs + b, color=C_V, lw=1.2, ls=":", zorder=2)
    rk = s["pooled_rank"]
    ax.set_xlabel("verifiability score (rated before any run)")
    ax.set_ylabel("debt ratio $D$")
    ax.set_title(f"$D$ vs verifiability   "
                 f"($\\rho_s={rk['spearman']:.2f}$, $p_{{\\rm perm}}={rk['p_perm']:.2f}$)")
    ax.axhline(1.0, color="0.6", lw=0.9, ls="--")
    fig.savefig(path)
    plt.close(fig)


def fig_guardrail(s, path):
    rows = s["pooled_guardrail"]
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(4.3, 2.9))
    # cleansing and emails sit almost on top of each other; separate their labels
    OFF = {"invoices": (6, 5), "cleansing": (6, 10), "emails": (6, -14), "reports": (6, 5)}
    for r in rows:
        inc = -r["cut_pct"]                      # positive = debt rose
        ax.scatter(r["throughput_cost_pct"], inc, s=42, color="#333333", zorder=3)
        dx, dy = OFF.get(r["family"], (6, 5))
        ax.annotate(LABEL.get(r["family"], r["family"]).replace("\n", " "),
                    (r["throughput_cost_pct"], inc),
                    textcoords="offset points", xytext=(dx, dy), fontsize=7.5)
    ax.axhline(-30, color=C_R, lw=1.0, ls="--")   # B3: a >=30% CUT = a <=-30% increase
    ax.axvline(15, color=C_V, lw=1.0, ls="--")
    ax.set_ylim(bottom=-45)
    ax.text(17.5, -22, "B3 requires $\\geq 30\\%$ cut", fontsize=7, color=C_R)
    ax.text(15.4, ax.get_ylim()[0] + 1, "and $\\leq 15\\%$ throughput cost", fontsize=7,
            color=C_V, rotation=90, va="bottom")
    ax.set_xlabel("throughput cost (extra effort per product, %)")
    ax.set_ylabel("increase in $D$ (%)")
    ax.set_title("Guardrails: debt change vs throughput spent")
    fig.savefig(path)
    plt.close(fig)


def fig_sensitivity(s, path):
    rows = s.get("sensitivity") or []
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(4.3, 2.7))
    fams = sorted({r["family"] for r in rows})
    lams = [0.5, 1.0, 2.0, 4.0]
    for i, f in enumerate(fams):
        pts = [r for r in rows if r["family"] == f and r["model"] == "qwen2.5:14b-instruct"]
        if not pts:
            continue
        by_lam = {r["lambda"]: r["D"] for r in pts}
        ys = [by_lam.get(l) for l in lams]
        ax.plot(range(len(lams)), ys, marker="o", ms=3, lw=1.2,
                label=LABEL.get(f, f).replace("\n", " "))
    ax.set_xticks(range(len(lams)))
    ax.set_xticklabels([f"{l:g}" for l in lams])
    ax.axhline(1.0, color="0.6", lw=0.9, ls="--")
    ax.set_xlabel("human/agent effort conversion factor $\\lambda$")
    ax.set_ylabel("debt ratio $D(\\lambda)$")
    ax.set_title("Proxy sensitivity: level scales, ordering is $\\lambda$-invariant")
    ax.legend(frameon=False, fontsize=7)
    fig.savefig(path)
    plt.close(fig)


def main():
    os.makedirs(FIG, exist_ok=True)
    s = load()
    fig_decomposition(s, "qwen2.5:14b-instruct", os.path.join(FIG, "fig1_decomposition.pdf"))
    fig_monotone(s, os.path.join(FIG, "fig2_monotone.pdf"))
    fig_guardrail(s, os.path.join(FIG, "fig3_guardrail.pdf"))
    fig_sensitivity(s, os.path.join(FIG, "fig4_sensitivity.pdf"))
    print("wrote", ", ".join(sorted(os.listdir(FIG))))


if __name__ == "__main__":
    main()
