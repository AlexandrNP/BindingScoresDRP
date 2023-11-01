from genericpath import isfile
import os
import pandas as pd
import subprocess

BINDING_AFFINITY_DIR = 'BindingAffinity/DATA/BindingAffinity'


def get_number_of_docked_complexes():
    complexes_dir = 'Docking/Complexes'
    complex_counter = 0
    for pdb_dir in os.listdir(complexes_dir):
        pdb_dir = os.path.join(complexes_dir, pdb_dir)
        if not os.path.isdir(pdb_dir):
            continue
        for complex in os.listdir(pdb_dir):
            complex_path = os.path.join(pdb_dir, complex)
            if os.path.isfile(complex_path):
                complex_counter += 1
    return complex_counter


def get_number_of_complexes_with_binding_affinity():
    command = ['ls', '-lh', BINDING_AFFINITY_DIR, '|', 'wc', '-l']
    print(' '.join(command))
    count = subprocess.check_output(command).decode()
    return int(count)


def get_binding_affinity_dataset():
    binding_affinity_dataset_path = 'binding_affinity.tsv'

    if os.path.isfile(binding_affinity_dataset_path):
        return pd.read_csv(binding_affinity_dataset_path, sep='\t')

    pdbs = []
    drug_ids = []
    binding_affinities = []
    for affinity_filename in os.listdir(BINDING_AFFINITY_DIR):
        if 'Drug' not in affinity_filename:
            continue
        affinity_file = os.path.join(BINDING_AFFINITY_DIR, affinity_filename)
        pdb = affinity_filename.split('-')[0]
        drug_id = affinity_filename.split('-')[1].split('.')[0]
        data = pd.read_csv(affinity_file, sep='\t')
        binding_affinity = data.values[0, 1]
        #print(pdb, drug_id, binding_affinity)
        pdbs.append(pdb)
        drug_ids.append(drug_id)
        binding_affinities.append(binding_affinity)

    binding_affinity_df = pd.DataFrame({
        'PDB': pdbs,
        'Drug_UniqueID': drug_ids,
        'BindingAffinity': binding_affinities
    })
    binding_affinity_df = binding_affinity_df.pivot(
        'PDB', 'Drug_UniqueID', 'BindingAffinity')
    binding_affinity_df.fillna(0)
    binding_affinity_df.to_csv(binding_affinity_dataset_path, sep='\t')
    return binding_affinity_df


if __name__ == "__main__":
    #print('Number of docked complexes: ', get_number_of_docked_complexes())
    # print('Number of complexes with binding affinity: ',
    #      get_number_of_complexes_with_binding_affinity())
    binding_affinity_data = get_binding_affinity_dataset()
    print(binding_affinity_data)
