import os
import numpy as np
import pandas as pd

DATA_DIR = 'DrugLists'
OUT_DIR = os.path.join('Drugs', 'SMILES')
DRUG_FILE = 'Comprehensive_Drug_List.txt'

ID_COLUMN = 'UniqueID'

drug_data = pd.read_csv(os.path.join(DATA_DIR, DRUG_FILE), sep='\t')

curated_columns = [ID_COLUMN] + \
    [x for x in drug_data.columns if 'smile' in x.lower()]

print(curated_columns)
drug_data = drug_data[curated_columns]
n, m = np.shape(drug_data)
for i in range(n):
    out_file_name = os.path.join(OUT_DIR, f'{drug_data.loc[i, ID_COLUMN]}.smi')
    print(out_file_name)
    out_file = open(out_file_name, 'w')
    for smiles_column in curated_columns[1:]:
        smiles_string = str(drug_data.loc[i, smiles_column])
        if smiles_string == 'nan' or smiles_string == 'None':
            continue
        if len(smiles_string) > 0:
            out_file.write(f'{smiles_string}\n')
    out_file.close()
