import pandas as pd
import numpy as np

data = pd.read_csv('DrugLists/FDA-Drugs_All-PDB_2022.txt',
                   sep='\t', header=None)
chains = data[data.columns[1]]
pdbs = [chain.split(':')[0] for chain in chains]
print(len(np.unique(pdbs)))
