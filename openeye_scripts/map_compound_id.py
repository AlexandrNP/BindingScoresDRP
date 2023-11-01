from openeye import oechem
import pickle
import os

#ifs = oechem.oemolistream(oechem.OEFormat_SDF)
#ofs = oechem.oemolostream(oechem.OEFormat_PDB)

DRUG_DIR = 'Drugs/Lib'
DRUG_DB_OUT = 'Drugs/DB/DB.oeb.gz'
DRUG_TITLE_FILE = 'drug_titles.pickle'
MAX_TAUTOMERS = 100


def count_molecules():
    ifs = oechem.oemolistream()
    num_molecules = 0
    if ifs.open(DRUG_DB_OUT):
        num_molecules = len(list(ifs.GetOEGraphMols()))
    return num_molecules


def preprocess_titles():
    total_molecules = 0
    drug_list = os.listdir(DRUG_DIR)
    drug_titles = {}
    non_zero_drugs = 0
    ifs = oechem.oemolistream()
    ofs = oechem.oemolostream()
    if ofs.open(DRUG_DB_OUT):
        for drug in drug_list:
            if '.oeb.gz' not in drug:
                continue
            drug_name = drug.split('.')[0]
            drug_path = os.path.join(DRUG_DIR, drug)
            print(drug_name)
            drug_titles[drug_name] = []
            if ifs.open(drug_path):

                tautomer_counter = 0
                for molecule in ifs.GetOEGraphMols():
                    total_molecules += 1
                    title = f"{drug_name}-{molecule.GetTitle()}"
                    drug_titles[drug_name].append(title)
                    molecule.SetTitle(title)
                    oechem.OEWriteMolecule(ofs, molecule)
                    tautomer_counter += 1
                    if tautomer_counter >= MAX_TAUTOMERS:
                        break

                if tautomer_counter > 0:
                    non_zero_drugs += 1

    pickle.dump(drug_titles, open(DRUG_TITLE_FILE, 'wb'))
    print(f"Non-zero conformers drug num: {non_zero_drugs}")
    print(f"Total number of molecules: {total_molecules}")


if __name__ == "__main__":
    preprocess_titles()
