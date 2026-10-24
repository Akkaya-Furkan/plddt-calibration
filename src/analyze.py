"""Calibration analysis and figures from results/per_protein.csv and results/per_residue.csv.gz.
Every number quoted in the README comes from results/summary.json written here."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr
from metrics import expected_calibration_error, auroc

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
N_BOOT = 2000
BINS = np.linspace(0, 100, 11)
COL = {"esmfold": "#d95f02", "afdb": "#1b9e77"}
NAME = {"esmfold": "ESMFold", "afdb": "AlphaFold DB"}
rng = np.random.default_rng(13)


def novelty_col(pred):
    return "novel_vs_esmfold" if pred == "esmfold" else "novel_vs_af2"


def bin_stats(df):
    """Per-protein, per-bin counts and sums of confidence and accuracy (for cluster bootstrap)."""
    b = np.clip(np.floor(df.plddt / 10 + 1e-9).astype(int), 0, 9)
    g = df.assign(bin=b).groupby(["entity_id", "bin"]).agg(n=("lddt", "size"), c=("plddt", "sum"), a=("lddt", "sum"))
    return g


def calib_from_stats(stats, ents):
    s = stats.reindex(pd.MultiIndex.from_product([ents, range(10)]), fill_value=0)
    n = s.n.values.reshape(len(ents), 10).sum(0)
    c = s.c.values.reshape(len(ents), 10).sum(0)
    a = s.a.values.reshape(len(ents), 10).sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mc, ma = c / n / 100, a / n
    ece = np.nansum(n / n.sum() * np.abs(mc - ma))
    return n, mc, ma, ece


def calibration(df):
    """Binned calibration of pLDDT/100 against residue lDDT-Ca with 95% protein-level bootstrap CIs."""
    df = df.dropna(subset=["lddt"])
    stats = bin_stats(df)
    ents = np.array(sorted(df.entity_id.unique()))
    n, mc, ma, ece = calib_from_stats(stats, ents)
    boots_a, boots_e = [], []
    for _ in range(N_BOOT):
        e = rng.choice(ents, len(ents))
        _, _, a_b, ece_b = _boot(stats, e)
        boots_a.append(a_b); boots_e.append(ece_b)
    boots_a = np.array(boots_a)
    return {"n": n, "conf": mc, "acc": ma, "acc_lo": np.nanpercentile(boots_a, 2.5, 0), "acc_hi": np.nanpercentile(boots_a, 97.5, 0),
            "ece": float(ece), "ece_lo": float(np.percentile(boots_e, 2.5)), "ece_hi": float(np.percentile(boots_e, 97.5)),
            "n_res": int(len(df)), "n_prot": int(len(ents))}


def _boot(stats, ents):
    """Bootstrap resample with repeats: weight each protein by its draw count."""
    uniq, w = np.unique(ents, return_counts=True)
    s = stats.reindex(pd.MultiIndex.from_product([uniq, range(10)]), fill_value=0)
    W = np.repeat(w, 10)
    n = (s.n.values * W).reshape(len(uniq), 10).sum(0)
    c = (s.c.values * W).reshape(len(uniq), 10).sum(0)
    a = (s.a.values * W).reshape(len(uniq), 10).sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mc, ma = c / n / 100, a / n
    return n, mc, ma, float(np.nansum(n / n.sum() * np.abs(mc - ma)))


def boot_stat(df, fn, key="entity_id"):
    ents = np.array(sorted(df[key].unique()))
    groups = {e: g for e, g in df.groupby(key)}
    pt = fn(df)
    bs = []
    for _ in range(N_BOOT // 4):
        d = pd.concat([groups[e] for e in rng.choice(ents, len(ents))])
        bs.append(fn(d))
    return float(pt), float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))


def main():
    prot = pd.read_csv(RES / "per_protein.csv")
    res = pd.read_csv(RES / "per_residue.csv.gz")
    out = {"n_proteins": {}, "global": {}, "residue": {}, "disorder": {}, "leakage": {}}
    tables = []
    calibs = {}
    for pred in ("esmfold", "afdb"):
        P, R = prot[prot.predictor == pred], res[res.predictor == pred]
        out["n_proteins"][pred] = int(len(P))
        # global (per-protein) relations
        g = {}
        for tgt in ("lddt_ca", "tm_score"):
            rho = spearmanr(P.mean_plddt, P[tgt])[0]
            g[f"spearman_meanplddt_{tgt}"] = boot_stat(P, lambda d: spearmanr(d.mean_plddt, d[tgt])[0])
            g[f"pearson_meanplddt_{tgt}"] = float(pearsonr(P.mean_plddt, P[tgt])[0])
        g["mae_meanplddt_vs_100lddt"] = float((P.mean_plddt - 100 * P.lddt_ca).abs().mean())
        g["bias_meanplddt_minus_100lddt"] = float((P.mean_plddt - 100 * P.lddt_ca).mean())
        g["median_tm"], g["median_lddt"], g["median_rmsd"] = float(P.tm_score.median()), float(P.lddt_ca.median()), float(P.rmsd.median())
        hi = P[P.mean_plddt >= 70]; lo = P[P.mean_plddt < 70]
        g["n_mean_plddt_ge70"], g["frac_tm_ge_0.5_given_ge70"] = int(len(hi)), float((hi.tm_score >= 0.5).mean()) if len(hi) else None
        g["n_mean_plddt_lt70"], g["frac_tm_ge_0.5_given_lt70"] = int(len(lo)), float((lo.tm_score >= 0.5).mean()) if len(lo) else None
        out["global"][pred] = g
        # residue-level calibration
        c = calibration(R)
        calibs[pred] = c
        Rv = R.dropna(subset=["lddt"])
        out["residue"][pred] = {
            "n_residues": c["n_res"], "n_proteins": c["n_prot"], "ece": c["ece"], "ece_ci": [c["ece_lo"], c["ece_hi"]],
            "spearman": boot_stat(Rv, lambda d: spearmanr(d.plddt, d.lddt)[0]),
            "mean_conf": float(Rv.plddt.mean() / 100), "mean_acc": float(Rv.lddt.mean()),
            "terminal": {k: {"n": int(len(d)), "mean_plddt": float(d.plddt.mean()), "mean_lddt": float(d.lddt.mean()),
                              "ece": expected_calibration_error(d.plddt / 100, d.lddt)[0]} for k, d in Rv.groupby("terminal")},
        }
        tables.append(pd.DataFrame({"predictor": pred, "plddt_bin_lo": BINS[:-1], "plddt_bin_hi": BINS[1:], "n_residues": c["n"],
                                    "mean_plddt_over_100": c["conf"], "mean_lddt_ca": c["acc"], "lddt_ci_lo": c["acc_lo"], "lddt_ci_hi": c["acc_hi"]}))
        # disorder proxy: residues absent from the experimental model
        Rm = R.assign(unres=~R.resolved)
        out["disorder"][pred] = {
            "n_unresolved": int(Rm.unres.sum()), "n_total": int(len(Rm)),
            "mean_plddt_unresolved": float(Rm[Rm.unres].plddt.mean()), "mean_plddt_resolved": float(Rm[~Rm.unres].plddt.mean()),
            "auroc_low_plddt_predicts_unresolved": boot_stat(Rm, lambda d: auroc(-d.plddt, d.unres)),
            "auroc_interior_only": boot_stat(Rm[~Rm.terminal], lambda d: auroc(-d.plddt, d.unres)),
            "frac_unresolved_below_50": float((Rm[Rm.unres].plddt < 50).mean()),
            "frac_resolved_below_50": float((Rm[~Rm.unres].plddt < 50).mean()),
        }
        # leakage: sequence clusters with no PDB member before the predictor's training cut-off vs the rest
        nc = novelty_col(pred)
        lk = {}
        for flag, name in ((True, "novel_cluster"), (False, "seen_cluster")):
            ids = P[P[nc] == flag].entity_id
            d = Rv[Rv.entity_id.isin(ids)]
            if len(ids) >= 5:
                lk[name] = {"n_proteins": int(len(ids)), "n_residues": int(len(d)), "median_tm": float(P[P[nc] == flag].tm_score.median()),
                            "median_lddt": float(P[P[nc] == flag].lddt_ca.median()),
                            "mean_plddt": float(P[P[nc] == flag].mean_plddt.mean()),
                            "ece": expected_calibration_error(d.plddt / 100, d.lddt)[0],
                            "ece_ci": boot_stat(d, lambda x: expected_calibration_error(x.plddt / 100, x.lddt)[0])[1:],
                            "bias_conf_minus_acc": float(d.plddt.mean() / 100 - d.lddt.mean())}
        out["leakage"][pred] = lk
        if lk.get("novel_cluster"):
            ids = P[P[nc]].entity_id
            cn = calibration(R[R.entity_id.isin(ids)])
            calibs[pred + "_novel"] = cn
    pd.concat(tables).to_csv(RES / "calibration_table.csv", index=False)
    # protein-level comparison table
    comp = prot.groupby("predictor").agg(n=("entity_id", "size"), median_length=("length", "median"), mean_plddt=("mean_plddt", "median"),
                                          lddt_ca=("lddt_ca", "median"), tm_score=("tm_score", "median"), rmsd=("rmsd", "median")).round(3)
    comp.to_csv(RES / "comparison_table.csv")
    # head-to-head on proteins both predictors cover
    w = prot.pivot(index="entity_id", columns="predictor", values=["mean_plddt", "lddt_ca", "tm_score"]).dropna()
    out["head_to_head"] = {"n_common": int(len(w)),
                           "median_tm": {p: float(w["tm_score"][p].median()) for p in ("esmfold", "afdb")},
                           "median_lddt": {p: float(w["lddt_ca"][p].median()) for p in ("esmfold", "afdb")}}
    json.dump(out, open(RES / "summary.json", "w"), indent=1)
    figures(prot, res, calibs)
    print(json.dumps(out, indent=1)[:6000])


def figures(prot, res, calibs):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.4))
    ax[0].plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")
    for pred in ("esmfold", "afdb"):
        c = calibs[pred]; ok = c["n"] > 50
        ax[0].errorbar(c["conf"][ok], c["acc"][ok], yerr=[(c["acc"] - c["acc_lo"])[ok], (c["acc_hi"] - c["acc"])[ok]], marker="o", ms=4, capsize=2,
                       color=COL[pred], label=f"{NAME[pred]} (ECE {c['ece']:.3f})")
    ax[0].set(xlabel="mean pLDDT / 100 (bin)", ylabel="mean residue lDDT-Cα", xlim=(0, 1), ylim=(0, 1), title="Residue-level calibration\n(95% CI: bootstrap over proteins)")
    ax[0].legend(frameon=False, loc="upper left")
    for pred in ("esmfold", "afdb"):
        c = calibs[pred]
        ax[1].bar(np.arange(10) + (0.4 if pred == "afdb" else 0) - 0.2, c["n"], 0.4, color=COL[pred], label=NAME[pred])
    ax[1].set(xlabel="pLDDT bin (10-point width)", ylabel="residues", title="Residues per confidence bin", xticks=range(10), xticklabels=[f"{int(b)}" for b in BINS[:-1]])
    ax[1].set_yscale("log"); ax[1].legend(frameon=False)
    fig.tight_layout(); fig.savefig(RES / "fig1_calibration.png", dpi=200); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(10, 4.4))
    for pred in ("esmfold", "afdb"):
        P = prot[prot.predictor == pred]
        ax[0].scatter(P.mean_plddt, P.tm_score, s=14, alpha=.7, color=COL[pred], label=NAME[pred])
        ax[1].scatter(P.mean_plddt, 100 * P.lddt_ca, s=14, alpha=.7, color=COL[pred], label=NAME[pred])
    ax[0].axhline(0.5, color="gray", lw=.8, ls=":"); ax[0].set(xlabel="mean pLDDT", ylabel="TM-score (TM-align)", title="Global: mean pLDDT vs TM-score", ylim=(0, 1.02))
    ax[1].plot([0, 100], [0, 100], "k--", lw=1); ax[1].set(xlabel="mean pLDDT", ylabel="100 × lDDT-Cα", title="Global: mean pLDDT vs lDDT-Cα", xlim=(0, 100), ylim=(0, 100))
    ax[0].legend(frameon=False); fig.tight_layout(); fig.savefig(RES / "fig2_global.png", dpi=200); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(10, 4.4))
    ax[0].plot([0, 1], [0, 1], "k--", lw=1)
    for key, ls in (("", "-"), ("_novel", ":")):
        for pred in ("esmfold", "afdb"):
            c = calibs.get(pred + key)
            if c is None: continue
            ok = c["n"] > 30
            ax[0].plot(c["conf"][ok], c["acc"][ok], ls, marker="o", ms=3, color=COL[pred],
                       label=f"{NAME[pred]}, {'no earlier homologue' if key else 'all'} (n={c['n_prot']})")
    ax[0].set(xlabel="mean pLDDT / 100", ylabel="mean residue lDDT-Cα", title="Effect of training-set overlap", xlim=(0, 1), ylim=(0, 1)); ax[0].legend(frameon=False, fontsize=7.5, loc="upper left")
    for pred, off in (("esmfold", -0.2), ("afdb", 0.2)):
        R = res[res.predictor == pred]
        for k, (flag, c) in enumerate(((False, "#444"), (True, "#c33"))):
            d = R[R.resolved == (not flag)].plddt
            ax[1].boxplot(d, positions=[k + off + (0 if True else 0)], widths=.3, showfliers=False, patch_artist=True,
                          boxprops=dict(facecolor=COL[pred], alpha=.6 if not flag else .3), medianprops=dict(color="k"))
    ax[1].set(xticks=[0, 1], xticklabels=["resolved in PDB", "unresolved in PDB\n(disorder proxy)"], ylabel="pLDDT", title="pLDDT of resolved vs unresolved residues\n(orange: ESMFold, green: AFDB)")
    fig.tight_layout(); fig.savefig(RES / "fig3_leakage_disorder.png", dpi=200); plt.close(fig)


if __name__ == "__main__":
    main()
