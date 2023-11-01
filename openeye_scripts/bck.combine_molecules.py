# coding=utf8
from ast import excepthandler
from combine_molecules import copy
from postprocess_fred import get_score_files
from openeye import oechem
from Bio import PDB
from Bio.PDB import PDBParser, Select, is_aa
import pandas as pd
import subprocess
import string
import xpdb
import sys
import os

import warnings
warnings.filterwarnings("ignore")

PDB_DIR = 'PDB'
OUT_DIR = 'Docking/Complexes'
FRED_PATH = 'FRED'
LIGAND_DIR = 'Docking'


class ProtSelect(Select):
    def accept_residue(self, residue):
        return 1 if is_aa(residue) == True else 0


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


def generate_complexes(score_files_path, ligand_dir, output_dir):
    fred_score_files = get_score_files(score_files_path)
    alphabet = list(string.ascii_uppercase)
    alphabet.remove('L')
    alphabet_no_L = alphabet
    print(alphabet_no_L)
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
        # if os.path.exists(target_pdb):
        #    continue

        protein_extraction_command = ["python3", "splitmolcomplexlowlevel.py",
                                      "-in", original_pdb,
                                      "-out", target_pdb]

        # subprocess.run(protein_extraction_command)

        pdb_out_dir = os.path.join(output_dir, pdb_name)
        check_create_dir(pdb_out_dir)

        ##############
        # DEBUGGING!!!
        ##############
        pdb_io = PDB.PDBIO()
        parser = PDBParser(
            PERMISSIVE=True, structure_builder=xpdb.SloppyStructureBuilder()
        )

        pdb_structure = parser.get_structure(pdb_name, original_pdb)
        pdb_io.set_structure(pdb_structure)
        pdb_io.save(target_pdb, ProtSelect())
        ##############

        print(target_pdb)

        tautomers = get_best_tautomers(fred_score_file)
        # print(list(tautomers.items())[0])
        for drug_name, info in tautomers.items():
            ligand_title, score = info[0], info[1]
            if ligand_title not in ligand_dict:
                continue

            ligand_path = ligand_dict[ligand_title]
            structures = [
                (ligand_title, ligand_path),
                (pdb_name, target_pdb)
            ]

            master_pdb = None
            chain_counter = 0
            structure_id = 0
            structure_count = 0

            output_file = os.path.join(
                pdb_out_dir, f'{pdb_name}-{drug_name}.pdb')
            too_long = False

            for struct_name, struct_path in structures:

                pdb_structure = parser.get_structure(struct_name, struct_path)
                chains = list(pdb_structure.get_chains())

                #print('Original chains: ', [chain.id for chain in chains])
                chains_num = len(chains)
                chain_counter = 0

                # print(struct_name)
                if structure_count == 0:
                    longest_id = 0
                    for i in range(1, chains_num):
                        if len(list(chains[i].get_residues())) > len(list(chains[longest_id].get_residues())):
                            longest_id = i
                    if len(chains) <= longest_id:
                        continue
                    chains[longest_id].id = 'L'
                    chains = [chains[longest_id]]

                    #counter = 0
                    for chain in chains:
                        for residue in chain.get_residues():
                            # residue = PDB.Residue.Residue(
                            #    ('LIG', residue.id[1], residue.id[2]), 'LIG', residue.get_segid())
                            # print(residue)
                            # print(residue)
                            #res_id = residue.id
                            # chains[longest_id].detach_child(res_id)
                            #residue.id = ('LIG', counter, res_id[2])
                            residue.resname = 'LIG'
                            # chains[longest_id].add
                            #counter += 1
                            # print(residue)
                            # print(residue.id)
                # else:
                #    marked_chains = [False] * chains_num
                #    for i in range(chains_num):
                #        for j in range(chains_num):
                #            if chains[j].id == alphabet_no_L[chain_counter]:
                #                marked_chains[j] = True
                #                chain_counter += 1
                #                break
                #
                #    for i in range(chains_num):
                #        if not marked_chains[i]:
                #            chains[i].id = alphabet_no_L[chain_counter]
                #            chain_counter += 1

                #print('New chains: ', [chain.id for chain in chains])
                if master_pdb is None:
                    master_pdb = pdb_structure
                else:
                    master_chains_ids = {
                        chain.id for chain in master_pdb[structure_id].get_chains()}
                    for i in range(chains_num):
                        chains[i].detach_parent()
                        if chains[i].id in master_chains_ids:
                            while alphabet_no_L[chain_counter] in master_chains_ids:
                                chain_counter += 1
                                if chain_counter == len(alphabet_no_L):
                                    too_long = True
                                    chain_counter -= 1
                                    break
                            chains[i].id = alphabet_no_L[chain_counter]
                        master_chains_ids.add(chains[i].id)
                        if too_long:
                            break
                        master_pdb[structure_id].add(chains[i])

                structure_count += 1

            if too_long:
                continue

            pdb_io.set_structure(master_pdb[structure_id])
            #structure_id += 1
            pdb_io.save(output_file)


if __name__ == "__main__":
    generate_complexes(FRED_PATH, LIGAND_DIR, OUT_DIR)
