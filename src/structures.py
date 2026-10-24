"""Parsing of experimental mmCIF files and predicted PDB files into per-residue CA tables."""
import numpy as np
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from Bio.Align import PairwiseAligner

THREE2ONE = {"ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F", "GLY": "G", "HIS": "H", "ILE": "I",
             "LYS": "K", "LEU": "L", "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q", "ARG": "R", "SER": "S",
             "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y", "MSE": "M"}


def read_experimental_ca(cif_path, entity_num, seq_len):
    """CA coordinates of the first model / first chain of polymer entity `entity_num`, indexed by
    1-based label_seq_id (the position in the deposited sequence). Returns (xyz[seq_len,3] with NaN
    for unresolved residues, resolved_mask[seq_len])."""
    d = MMCIF2Dict(str(cif_path))
    col = lambda k: d[k]
    xyz = np.full((seq_len, 3), np.nan)
    chain = None
    for i, atom in enumerate(col("_atom_site.label_atom_id")):
        if atom != "CA" or col("_atom_site.label_entity_id")[i] != str(entity_num):
            continue
        if col("_atom_site.pdbx_PDB_model_num")[i] != "1" or col("_atom_site.label_alt_id")[i] not in (".", "A"):
            continue
        asym = col("_atom_site.label_asym_id")[i]
        chain = chain or asym
        if asym != chain:
            continue
        sid = int(col("_atom_site.label_seq_id")[i])
        xyz[sid - 1] = [float(col("_atom_site.Cartn_x")[i]), float(col("_atom_site.Cartn_y")[i]), float(col("_atom_site.Cartn_z")[i])]
    return xyz, ~np.isnan(xyz[:, 0])


def read_predicted_ca(pdb_path):
    """(sequence, xyz[N,3], plddt[N] on 0-100 scale) from a predicted PDB file with pLDDT in the B-factor.
    ESMFold (ESM Atlas) writes pLDDT on a 0-1 scale; it is rescaled to 0-100."""
    seq, xyz, b = [], [], []
    for line in open(pdb_path):
        if line.startswith("ATOM") and line[12:16].strip() == "CA":
            seq.append(THREE2ONE.get(line[17:20], "X"))
            xyz.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
            b.append(float(line[60:66]))
    b = np.array(b)
    if b.max() <= 1.0:
        b = b * 100
    return "".join(seq), np.array(xyz), b


def map_by_alignment(query_seq, target_seq):
    """Map positions of `query_seq` onto `target_seq` by local alignment; returns an int array of
    target indices (-1 where unaligned or non-identical)."""
    al = PairwiseAligner(mode="local", match_score=2, mismatch_score=-1, open_gap_score=-5, extend_gap_score=-0.5)
    best = al.align(query_seq, target_seq)[0]
    m = np.full(len(query_seq), -1)
    for (qs, qe), (ts, te) in zip(*best.aligned):
        for k in range(qe - qs):
            if query_seq[qs + k] == target_seq[ts + k]:
                m[qs + k] = ts + k
    return m
