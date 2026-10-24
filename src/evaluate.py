"""Compute per-protein and per-residue accuracy of ESMFold / AlphaFold DB models against PDB structures."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from structures import read_experimental_ca, read_predicted_ca, map_by_alignment
from metrics import lddt_ca, kabsch_rmsd, tm_score

ROOT = Path(__file__).resolve().parents[1]
MIN_COMMON = 40       # minimum number of residues common to model and experiment
MIN_COVERAGE = 0.8    # AFDB: fraction of resolved residues that must be matched in the model
TERM = 10             # residues within this distance of a chain end are flagged as terminal


def evaluate_one(entry, predictor):
    seq = entry["sequence"]
    n = len(seq)
    if not (ROOT / f"data/exp/{entry['pdb_id']}.cif").exists():
        return None
    exp_xyz, resolved = read_experimental_ca(ROOT / f"data/exp/{entry['pdb_id']}.cif", int(entry["entity_id"].split("_")[1]), n)
    if predictor == "esmfold":
        pdb = ROOT / f"data/esmfold/{entry['entity_id']}.pdb"
        if not pdb.exists():
            return None
        pseq, pxyz, plddt = read_predicted_ca(pdb)
        if pseq != seq:
            return None
        mapping = np.arange(n)
    else:
        if not entry["uniprot"] or not (ROOT / f"data/afdb/{entry['uniprot']}.pdb").exists():
            return None
        pseq, pxyz, plddt = read_predicted_ca(ROOT / f"data/afdb/{entry['uniprot']}.pdb")
        mapping = map_by_alignment(seq, pseq)
    valid = resolved & (mapping >= 0)
    if resolved.sum() == 0 or valid.sum() / resolved.sum() < MIN_COVERAGE or valid.sum() < MIN_COMMON:
        return None
    idx = np.where(valid)[0]
    ref, mod = exp_xyz[idx], pxyz[mapping[idx]]
    s = "".join(seq[i] for i in idx)
    lddt = lddt_ca(ref, mod)
    rmsd, _ = kabsch_rmsd(mod, ref)
    tm = tm_score(mod, ref, s, s)
    pl_res = plddt[mapping[idx]]          # resolved residues only
    pl_all = plddt[mapping[mapping >= 0]]  # every residue of the model that maps to the entity: independent of what the experiment resolved
    # residue table over the sequence positions mapped in the model (resolved or not)
    mapped = np.where(mapping >= 0)[0]
    res = pd.DataFrame({"entity_id": entry["entity_id"], "predictor": predictor, "pos": mapped + 1,
                        "plddt": plddt[mapping[mapped]], "resolved": resolved[mapped]})
    lmap = dict(zip(idx + 1, lddt))
    res["lddt"] = res.pos.map(lmap)
    res["terminal"] = (res.pos <= TERM) | (res.pos > n - TERM)
    prot = {"entity_id": entry["entity_id"], "pdb_id": entry["pdb_id"], "predictor": predictor, "length": n,
            "n_resolved": int(resolved.sum()), "n_evaluated": int(valid.sum()),
            "mean_plddt": float(pl_all.mean()), "mean_plddt_resolved": float(pl_res.mean()), "lddt_ca": float(np.nanmean(lddt)), "tm_score": tm, "rmsd": rmsd,
            "release_date": entry["release_date"], "novel_vs_af2": entry["novel_vs_af2"],
            "novel_vs_esmfold": entry["novel_vs_esmfold"], "cluster30": entry["cluster30"]}
    return prot, res


def main():
    sel = json.load(open(ROOT / "data/selected.json"))
    prots, ress = [], []
    for e in sel:
        for p in ("esmfold", "afdb"):
            out = evaluate_one(e, p)
            if out:
                prots.append(out[0]); ress.append(out[1])
    pd.DataFrame(prots).to_csv(ROOT / "results/per_protein.csv", index=False)
    pd.concat(ress).to_csv(ROOT / "results/per_residue.csv.gz", index=False)
    print(pd.DataFrame(prots).groupby("predictor").size())


if __name__ == "__main__":
    main()
