# How well does pLDDT predict structural accuracy?

**Question.** How well does pLDDT predict the real accuracy of predicted protein structures?

## Data
Proteins with experimental structures released after the predictors' training cut-off; ESMFold/ColabFold predictions.

## Approach
1. Date-filtered, fold-diverse protein set
2. Structure prediction
3. TM-score / lDDT / RMSD vs pLDDT, global and per residue
4. Calibration curve; disordered regions analysed separately

## Deliverables
Comparison table, calibration figure, write-up.

## Limitations
- Structures seen in training make calibration look optimistic; the date filter is essential.

## Status
Planned — part of a 30-project computational biology and scientific software portfolio.
