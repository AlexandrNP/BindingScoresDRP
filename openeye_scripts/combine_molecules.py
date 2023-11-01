# coding=utf8
from ast import excepthandler
from postprocess_fred import get_score_files
from openeye import oechem
from Bio import PDB
from Bio.PDB import PDBParser
import pandas as pd
import subprocess
import string
import sys
import os

import warnings
warnings.filterwarnings("ignore")

PDB_DIR = 'PDB'
OUT_DIR = 'Docking/Complexes'
FRED_PATH = 'FRED'
LIGAND_DIR = 'Docking'


def check_create_dir(dir_path):
    try:
        os.mkdir(dir_path)
    except:
        pass


def get_best_tautomers(score_file):
    data = pd.read_csv(score_file, sep='\t')
    best_scores = {}
    n, m = data.shape
    for i in range(n):
        title = data.loc[data.index[i], 'Title']
        score = data.loc[data.index[i], 'FRED Chemgauss4 score']
        drug_name = title.split('-')[0]
        if drug_name not in best_scores:
            best_scores[drug_name] = (title, score)
        elif score < best_scores[drug_name][1]:
            best_scores[drug_name] = (title, score)
    return best_scores


def get_ligand_dict(ligand_db):
    tmp_dir = 'tmp_ligands'
    check_create_dir(tmp_dir)

    molecules_dict = {}
    dict_ifs = oechem.oemolistream()
    dict_ofs = oechem.oemolostream()
    if dict_ifs.open(ligand_db):
        for molecule in dict_ifs.GetOEGraphMols():
            title = molecule.GetTitle()
            pdb_path = os.path.join(tmp_dir, f"{title}.pdb")
            if dict_ofs.open(pdb_path):
                molecules_dict[title] = pdb_path
                oechem.OEWriteMolecule(dict_ofs, molecule)
    return molecules_dict


def copy(self):
    shallow = copy.copy(self)
    for child in self.child_dict.values():
        shallow.disordered_add(child.copy())
    return shallow


def generate_complexes(score_files_path, ligand_dir, output_dir):
    fred_score_files = get_score_files(score_files_path)
    alphabet = string.ascii_uppercase
    tmp_dir = "tmp_pdb"
    check_create_dir(tmp_dir)

    for pdb_name, fred_score_file in fred_score_files:
        print(pdb_name)
        print(fred_score_file)

        ligand_db = os.path.join(ligand_dir, pdb_name,
                                 f'{pdb_name}_DB_docked.oeb')

        ligand_dict = get_ligand_dict(ligand_db)

        original_pdb = os.path.join(PDB_DIR, f'{pdb_name}.pdb')
        target_pdb = os.path.join(tmp_dir, f'{pdb_name}-protein.pdb')

        subprocess.run(["python3", "splitmolcomplexlowlevel.py",
                       "-in", original_pdb,
                        "-out", target_pdb])

        pdb_out_dir = os.path.join(output_dir, pdb_name)
        try:
            os.mkdir(pdb_out_dir)
        except:
            pass
        print(target_pdb)

        tautomers = get_best_tautomers(fred_score_file)
        print(list(tautomers.items())[0])
        for drug_name, info in tautomers.items():
            ligand_title, score = info[0], info[1]
            ligand_path = ligand_dict[ligand_title]
            structures = [
                (ligand_title, ligand_path),
                (pdb_name, target_pdb)
            ]

            pdb_io = PDB.PDBIO()
            master_pdb = None
            chain_counter = 0

            output_file = os.path.join(
                pdb_out_dir, f'{pdb_name}-{drug_name}.pdb')
            for struct_name, struct_path in structures:
                parser = PDBParser()

                pdb_structure = parser.get_structure(struct_name, struct_path)
                chains = list(pdb_structure.get_chains())

                chains_num = len(chains)
                marked_chains = [False] * chains_num
                for i in range(chains_num):
                    for j in range(chains_num):
                        if chains[j].id == alphabet[chain_counter]:
                            marked_chains[j] = True
                            chain_counter += 1
                            break

                for i in range(chains_num):
                    if not marked_chains[i]:
                        chains[i].id = alphabet[chain_counter]
                        chain_counter += 1

                if master_pdb is None:
                    master_pdb = pdb_structure.copy()
                else:
                    for i in range(len(chains)):
                        chains[i].detach_parent()
                        master_pdb[0].add(chains[i])

            pdb_io.set_structure(master_pdb[0])
            pdb_io.save(output_file)

            break
        break


if __name__ == "__main__":
    generate_complexes(FRED_PATH, LIGAND_DIR, OUT_DIR)
