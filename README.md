# Active learning guided computational discovery of plant-based redoxmers for organic non-aqueous redox flow battery

*Supplementary Information*

*This repo consists of 3 folders: CSME, Active Learning, and Data*

## CSME
- Contains the CSME code and it's input file

## Active Learning
- MBO_Flavone.ipynb has the Multi-Objective Bayesian Optimization based active learning code implemented for the flavone library
- all_mbo_functions.py has utility functions for the MBO_Flavone.ipynb

- Feature Generation: This folder has Jupyter Notebooks to generate rdkit features from the SMILES, and feature reduction. Chemfunctions.py has utility functions for the feature generation and feature reduction. 

## Data
- Unzip Data.zip
- Contains 40K isoflavone and 60K flavone molecule smiles with rdkit features.
- Contains AL Selected 100 flavone molecules with redox potentials and solvation energy: Data/60K Flavone/flavone_ytrain_AL_Selected_110mols.csv
- Contains AL Selected 50 isoflavone molecules with redox potentials and solvation energy: Data/40K Isoflavone/isoflavone_ytrain_AL_selected_60mols.csv
