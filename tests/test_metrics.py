import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from metrics import lddt_ca, kabsch_rmsd, tm_score, expected_calibration_error, auroc, bootstrap_ci
from structures import map_by_alignment


def helix(n=40):
    t = np.arange(n) * 100 / 180 * np.pi
    return np.c_[2.3 * np.cos(t), 2.3 * np.sin(t), 1.5 * np.arange(n)]


def test_lddt_identity_and_rigid_motion_invariance():
    x = helix()
    assert np.allclose(lddt_ca(x, x), 1.0)
    th = 0.7
    R = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
    assert np.allclose(lddt_ca(x, x @ R.T + 5), 1.0)


def test_lddt_drops_for_displaced_residue():
    x = helix(); y = x.copy(); y[20] += 8.0
    s = lddt_ca(x, y)
    assert s[20] < 0.3 and s[0] > 0.9 and s.min() == s[20]


def test_rmsd_and_tm():
    x = helix(60)
    r, _ = kabsch_rmsd(x + 3.0, x)
    assert r < 1e-6
    seq = "A" * 60
    assert tm_score(x, x, seq, seq) > 0.99
    rng = np.random.default_rng(0)
    assert tm_score(rng.normal(size=(60, 3)) * 20, x, seq, seq) < 0.5


def test_ece_perfect_and_overconfident():
    rng = np.random.default_rng(1)
    c = rng.uniform(size=20000)
    acc = c + rng.normal(0, 0.0, size=c.size)
    assert expected_calibration_error(c, acc)[0] < 0.03
    assert abs(expected_calibration_error(np.full(100, 0.9), np.full(100, 0.5))[0] - 0.4) < 1e-9


def test_ece_ignores_nan():
    assert expected_calibration_error([0.5, np.nan], [0.5, 0.1])[0] < 1e-9


def test_auroc():
    assert auroc([1, 2, 3, 4], [0, 0, 1, 1]) == 1.0
    assert auroc([4, 3, 2, 1], [0, 0, 1, 1]) == 0.0
    assert auroc([1, 1, 1, 1], [0, 1, 0, 1]) == 0.5
    assert np.isnan(auroc([1, 2], [1, 1]))


def test_cluster_bootstrap_contains_point():
    rng = np.random.default_rng(2)
    groups = [rng.normal(0.5, 0.1, 30) for _ in range(40)]
    p, lo, hi = bootstrap_ci(groups, lambda g: np.mean(np.concatenate(g)), n_boot=300)
    assert lo < p < hi and hi - lo < 0.1


def test_alignment_maps_only_identical_positions():
    m = map_by_alignment("MKTAYIAKQR", "GGMKTAYIAKQRGG")
    assert list(m) == list(range(2, 12))
    m = map_by_alignment("MKTAYIAKQR", "MKTAWIAKQR")
    assert m[4] == -1 and m[0] == 0
