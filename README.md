# How well does pLDDT predict structural accuracy?

**Question.** pLDDT is the per-residue confidence reported by AlphaFold and ESMFold. Does it track the *actual* accuracy
of a model against an experimental structure, i.e. is it **calibrated** (a residue with pLDDT 80 is ~0.80 correct)?
Does that hold for structures the predictors could not have seen, and for regions that are disordered?

## Data
All inputs are public and fetched by script (`make all`).

| Item | Source | Use |
|---|---|---|
| 181 single-chain X-ray structures, 60–400 residues, resolution ≤ 2.5 Å, **released 2022-06-01 – 2025-06-30** | RCSB PDB (search + GraphQL APIs) | ground truth |
| ESMFold v1 models (181) | ESM Atlas public API (`api.esmatlas.com`) | predictor 1 (pLDDT in B-factor) |
| AlphaFold DB models (149; 32 entries have no UniProt link, no AFDB model, or < 80 % matching residues) | AlphaFold Protein Structure Database | predictor 2 |

*Date filter.* Every test entry was released after the AlphaFold2 (2021-09-30) and ESMFold (~2020-05) structure-training
cut-offs. Because a new entry can still be a close homologue of an old one, each entry is additionally labelled by its
RCSB 30 %-sequence-identity cluster: **"no earlier homologue"** means no member of the cluster was released before the
predictor's cut-off (60/181 clusters for ESMFold, 41/149 for AlphaFold DB). One entry per cluster was kept, so the set is
fold-diverse at the 30 % level. Selection is seeded (`seed=13`); the list is in `data/selected.json`.

## Approach
1. `src/select_set.py` – query RCSB, deduplicate by 30 % cluster, label novelty, write `data/selected.json`.
2. `scripts/download_data.py` – download mmCIF, AFDB models, run ESMFold (no GPU was used; the public API ran the model).
3. `src/evaluate.py` – map model residues to the experimental chain (deposited-sequence index for ESMFold; local alignment,
   identical residues only, for AFDB) and compute, on the resolved Cα atoms:
   per-residue **lDDT-Cα** (own implementation, 15 Å radius, thresholds 0.5/1/2/4 Å, superposition-free),
   **TM-score** (TM-align via `tmtools`), **RMSD** (Kabsch). Per-protein "lDDT-Cα" is the mean of residue scores.
4. `src/analyze.py` – calibration of pLDDT/100 against residue lDDT-Cα (10 equal-width bins, expected calibration error,
   Spearman ρ), global relations of mean pLDDT with TM-score / lDDT, disorder analysis, and the novelty comparison.
   All confidence intervals are 95 % percentile **bootstrap over proteins** (not residues).

Reproduce: `make env && make all && make test`. Every number below is in `results/summary.json`.

## Results
**Model quality** (median over proteins; `results/comparison_table.csv`)

| Predictor | n | mean pLDDT | lDDT-Cα | TM-score | RMSD (Å) |
|---|---|---|---|---|---|
| ESMFold | 181 | 88.2 | 0.920 | 0.953 | 1.44 |
| AlphaFold DB | 149 | 94.1 | 0.949 | 0.973 | 0.93 |

**Calibration** (residue level; `results/calibration_table.csv`, `results/fig1_calibration.png`)
- Both predictors are close to the diagonal: ECE = **0.016** (95 % CI 0.013–0.022; 36,467 residues) for ESMFold and **0.012** (0.004–0.024; 30,196 residues) for AlphaFold DB.
- Residue-level rank correlation of pLDDT with lDDT-Cα is moderate: Spearman ρ = **0.71** (0.66–0.75) for ESMFold and **0.60** (0.54–0.66) for AlphaFold DB. Calibrated on average does not mean sharp per residue.
- Most residues are in the top bin (pLDDT ≥ 90: 22,868 ESMFold, 25,484 AFDB residues), so the low-confidence bins have wide intervals; AFDB has only 3 residues at pLDDT 20–30.
- Residues within 10 positions of a chain end are less accurate and slightly less well calibrated (ESMFold ECE 0.026 vs 0.015 interior).

**Global accuracy** (per protein; `results/fig2_global.png`)
- Mean pLDDT vs lDDT-Cα: Spearman ρ = 0.75 (0.66–0.82) ESMFold, 0.60 (0.48–0.71) AFDB; vs TM-score: 0.70 (0.59–0.79) and 0.61 (0.48–0.71).
- 165/181 ESMFold models have mean pLDDT ≥ 70; 164 of them (99.4 %) have TM-score ≥ 0.5. Of the 16 below 70, 11 (69 %) still have TM ≥ 0.5. AFDB: 148/149 ≥ 70, all with TM ≥ 0.5, so AFDB offers no low-confidence global cases to test.
- Mean pLDDT is on average 3.3 points below 100 × lDDT-Cα for ESMFold (conservative) and 1.1 below for AFDB.

**Training-set overlap** (`results/fig3_leakage_disorder.png`, left)

| | proteins | ECE (95 % CI) | median TM |
|---|---|---|---|
| ESMFold, no earlier homologue | 60 | 0.009 (0.008–0.021) | 0.920 |
| ESMFold, earlier homologue | 121 | 0.021 (0.016–0.026) | 0.964 |
| AFDB, no earlier homologue | 41 | 0.022 (0.010–0.037) | 0.955 |
| AFDB, earlier homologue | 108 | 0.010 (0.003–0.026) | 0.978 |

Models are somewhat less accurate on proteins without earlier homologues (lower TM/lDDT), as expected, but calibration did
**not** visibly degrade: the ECE differences are within the bootstrap intervals and have opposite sign for the two
predictors. With 41–60 proteins in the novel subsets the test has limited power.

**Disordered / unresolved regions** (`results/fig3_leakage_disorder.png`, right). Residues missing from the experimental
model (proxy for disorder or mobility; 2,661 ESMFold-mapped and 1,805 AFDB-mapped residues) have much lower pLDDT than
resolved ones (ESMFold 61.2 vs 86.2; AFDB 59.8 vs 94.1). pLDDT as a classifier of "unresolved" has AUROC **0.86** (0.83–0.89)
for ESMFold and **0.95** (0.92–0.97) for AFDB; excluding the 10 terminal residues on each side changes little (0.855, 0.942).
But 34 % (ESMFold) and 38 % (AFDB) of unresolved residues have pLDDT ≥ 50, and 5.9 % of ESMFold's resolved residues are below 50.

## Limitations
- **Accuracy reference is a single crystal structure.** lDDT against one X-ray model penalises genuine flexibility and
  crystal-packing differences; lDDT is computed on Cα atoms only (not all-atom lDDT), residues are scored equally
  rather than contact-weighted, and chains were evaluated alone (no ligand, partner or oligomer context).
- **Novelty is approximate.** It relies on RCSB's 30 % identity clusters, whose coverage rules can miss homologous
  domains; no independent search against a frozen pre-cut-off PDB was done. The "no earlier homologue" flag uses the AF2
  cut-off of 2021-09-30 for AFDB models; if the deployed models used an earlier cut-off, a few more entries (3/149 by an external check) would count as novel.
  AFDB models are the versions served at download time (October 2026) and were not pinned or checked individually for provenance.
- **Selection bias.** Only X-ray structures ≤ 2.5 Å, one chain, ≤ 400 residues (the ESM Atlas length limit) and with
  deposited sequences of 20 standard amino acids. This favours well-ordered, easy-to-crystallise proteins; 83 % (ESMFold) and 95 % (AFDB) of scored residues
  have pLDDT ≥ 80, so calibration at low confidence rests on few residues and proteins.
- **Unresolved ≠ disordered.** Missing density also arises from tags, linkers and crystal artefacts; no curated disorder
  database (e.g. DisProt) was used. Disorder AUROCs are therefore a proxy.
- **AFDB mapping.** Only residues identical between the deposited sequence and the UniProt model are scored; TM-score is
  normalised by the number of scored residues, not by the full chain. Mean pLDDT is taken over all mapped model residues.
- **Not done:** ColabFold/AF2 re-prediction (no GPU on the machine used), recalibration (isotonic / temperature) of
  pLDDT, comparison with predicted aligned error, local ESMFold inference. ESMFold predictions come from a remote public API.
- 181 / 149 proteins are a modest sample; intervals are wide for subgroup claims.

## Layout
`src/` pipeline and metrics · `scripts/download_data.py` · `tests/` (metric unit tests) · `results/` tables, figures, `summary.json`, per-protein and per-residue CSVs · `data/selected.json` selected set · `requirements.txt`, `Makefile`.
