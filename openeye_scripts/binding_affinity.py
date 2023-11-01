import pandas as pd
import subprocess
import string
import os


def get_chains_no_l():
    alphabet = list(string.ascii_uppercase)
    alphabet.remove('L')
    alphabet_no_L = alphabet
    return alphabet_no_L


LIGAND_CHAIN = 'L:LIG'
PROT_CHAINS = ','.join(get_chains_no_l())
COMPLEXES_DIR = os.path.join('Docking', 'Complexes')


def get_binding_affinity(complex_pdb):
    prodigy_lig_command = ['prodigy_lig', '-c',
                           PROT_CHAINS, LIGAND_CHAIN, '-i', complex_pdb]
    print(' '.join(prodigy_lig_command))
    binding_affinity = 0
    try:
        result = str(subprocess.check_output(prodigy_lig_command))
        binding_affinity = float(result.split('\\t')[-1].split('\\n')[0])
    except:
        pass
    return binding_affinity


def get_protein_ligand_complexes():
    complexes_with_name_and_path = []
    for pdbs_dir in os.listdir(COMPLEXES_DIR):
        dir_path = os.path.join(COMPLEXES_DIR, pdbs_dir)
        if not os.path.isdir(dir_path):
            continue
        pdbs = [(item_name.split('.')[0], os.path.join(dir_path, item_name)) for item_name in os.listdir(
            dir_path) if '.pdb' in item_name]

        if len(pdbs) == 0:
            continue

        pdb_name = pdbs[0][0].split('-')[0]
        complexes_with_name_and_path.append((pdb_name, pdbs))
    return complexes_with_name_and_path


if __name__ == "__main__":
    complexes = get_protein_ligand_complexes()
    binding_affinity_series = []
    for pdb_name, complex_list in complexes:
        protein_binding_energy = pd.Series([])
        protein_binding_energy['PDB'] = pdb_name
        for complex_name, complex_path in complex_list:
            print(complex_path)
            drug_name = complex_name.split('-')[1]
            binding_affinity = get_binding_affinity(complex_path)
            protein_binding_energy[drug_name] = binding_affinity
            print(complex_name, binding_affinity)
        binding_affinity_series.append(protein_binding_energy)
    binding_affinity_matrix = pd.concat(
        binding_affinity_series, axis=0, ignore_index=False)
    binding_affinity_matrix.to_csv(
        'binding_affinity.tsv', sep='\t', index=None)
