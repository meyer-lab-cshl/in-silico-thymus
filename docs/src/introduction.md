# Introduction

Assumptions:
- Constant population of all cell types maintained
- Thymocytes either die by negative selection, or leave the thymus after a certain number of interactions
- mTECs die and are replaced by a new mTEC after a certain number of interactions
- mTECs/DCs can interact with multiple thymocytes at once
- mTECs remain stationary
- DCs are motile
- Thymocytes are motile
- TCR:pMHC binding strength is calculated by comparing their strings through generated binding matrices
- If number of complexes parameter is >1, the reaction strengths for each complex are summed and then compared to negative selection threshold
- mTECs go through a development process as they have more interactions, which changes the genes expressed and peptides presented
- Thymocytes each have a randomly generated TCR
- DCs copy the presented peptides of any mTEC that they interact with

