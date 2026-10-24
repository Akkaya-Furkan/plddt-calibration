"""Structure-comparison and calibration metrics (numpy only, except TM-score via tmtools)."""
import numpy as np

LDDT_THRESHOLDS = (0.5, 1.0, 2.0, 4.0)


def lddt_ca(ref, mod, inclusion_radius=15.0, thresholds=LDDT_THRESHOLDS):
    """Per-residue CA-only lDDT of `mod` against `ref` (both (N,3), residues in 1:1 correspondence).

    For residue i, all other residues j with ref distance < inclusion_radius form the
    neighbourhood; the score is the mean fraction of those distances preserved within each
    threshold. Superposition-free. Residues with no neighbours get NaN.
    """
    ref, mod = np.asarray(ref, float), np.asarray(mod, float)
    d_ref = np.linalg.norm(ref[:, None] - ref[None], axis=-1)
    d_mod = np.linalg.norm(mod[:, None] - mod[None], axis=-1)
    mask = (d_ref < inclusion_radius) & ~np.eye(len(ref), dtype=bool)
    diff = np.abs(d_ref - d_mod)
    kept = np.mean([(diff < t) for t in thresholds], axis=0)  # (N,N) fraction of thresholds passed
    n = mask.sum(1)
    out = np.where(n > 0, (kept * mask).sum(1) / np.maximum(n, 1), np.nan)
    return out


def kabsch_rmsd(a, b):
    """RMSD after optimal rigid superposition; also returns the rotation/translation applied to a."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ca, cb = a.mean(0), b.mean(0)
    H = (a - ca).T @ (b - cb)
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1, 1, d])
    R = Vt.T @ D @ U.T
    a2 = (a - ca) @ R.T + cb
    return float(np.sqrt(((a2 - b) ** 2).sum(1).mean())), a2


def tm_score(mod, ref, seq_mod, seq_ref):
    """TM-score of `mod` against `ref`, normalised by the reference length, from TM-align
    (structure-based alignment; inputs are the CA coordinates of the residues common to both)."""
    from tmtools import tm_align
    r = tm_align(np.asarray(mod, float), np.asarray(ref, float), seq_mod, seq_ref)
    return float(r.tm_norm_chain2)


def expected_calibration_error(conf, acc, n_bins=10):
    """ECE with equal-width bins on [0,1]; returns (ece, table) where table rows are
    (bin_lo, bin_hi, n, mean_conf, mean_acc)."""
    conf, acc = np.asarray(conf, float), np.asarray(acc, float)
    ok = ~(np.isnan(conf) | np.isnan(acc))
    conf, acc = conf[ok], acc[ok]
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.floor(conf * n_bins + 1e-9).astype(int), 0, n_bins - 1)  # same rule as analyze.py
    rows, ece = [], 0.0
    for b in range(n_bins):
        m = idx == b
        if m.sum() == 0:
            rows.append((edges[b], edges[b + 1], 0, np.nan, np.nan))
            continue
        mc, ma = conf[m].mean(), acc[m].mean()
        ece += m.mean() * abs(mc - ma)
        rows.append((edges[b], edges[b + 1], int(m.sum()), mc, ma))
    return float(ece), rows


def auroc(score, label):
    """AUROC for `score` predicting binary `label` (rank-based, ties averaged)."""
    from scipy.stats import rankdata
    score, label = np.asarray(score, float), np.asarray(label, bool)
    n1, n0 = label.sum(), (~label).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(score)
    return float((r[label].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def bootstrap_ci(values_by_group, stat, n_boot=2000, seed=0):
    """Cluster bootstrap: resample *groups* (proteins) with replacement; `stat` takes a list
    of per-group arrays. Returns (point, lo, hi) for a 95% percentile interval."""
    rng = np.random.default_rng(seed)
    g = len(values_by_group)
    point = stat(values_by_group)
    bs = [stat([values_by_group[i] for i in rng.integers(0, g, g)]) for _ in range(n_boot)]
    return point, float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))
