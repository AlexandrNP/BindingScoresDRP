from genericpath import isdir
import pandas as pd
import numpy as np
import os

fred_path = 'FRED'


def get_dirs(path):
    dirs = []
    listdir = os.listdir(path)
    for item in listdir:
        if os.path.isdir(os.path.join(path, item)) and item != '.' and item != '..':
            dirs.append(item)
    return dirs


def get_score_files(dir_path):
    file_paths = []
    docked_dirs = get_dirs(dir_path)

    for dir in docked_dirs:
        pdb_name = dir.split(os.path.pathsep)[-1]
        file_path = os.path.join(dir_path, dir, f"{pdb_name}_DB.score")
        if not os.path.exists(file_path):
            continue
        file_paths.append((pdb_name, file_path))
    return file_paths


def get_best_scores(data, pdb_name):
    title_col = data.columns[0]
    score_col = data.columns[1]
    data[title_col] = [x.split('-')[0] for x in data[title_col]]
    data = data.groupby(by=[title_col]).min()
    data.columns = [pdb_name]
    return data


if __name__ == "__main__":

    score_files = get_score_files(fred_path)
    drug_nums = []
    drug_scores = []

    for pdb_name, score_file in score_files:
        data = pd.read_csv(score_file, sep='\t')
        drugs = [x.split('-')[0] for x in data['Title']]
        drug_num = len(np.unique(drugs))
        print(f'Unique drugs num for {pdb_name}: {drug_num}')
        drug_nums.append(drug_num)
        drug_scores.append(get_best_scores(data, pdb_name).transpose())

    drug_scores = pd.concat(drug_scores, axis=0, ignore_index=False)
    drug_scores = drug_scores.dropna(axis=1, how='all')
    print(drug_scores)
    drug_scores.to_csv('gausschem4_scores.tsv', sep='\t')

    print(f"Total PDBs processed: {len(score_files)}")
    print(
        f"Average number of docked drug molecules per pdb: {np.mean(drug_nums)}")
    print(
        f"Median number of docked drug molecules per pdb: {np.median(drug_nums)}")
    print(f"Min number of docked drug molecules per pdb: {np.min(drug_nums)}")
    print(f"Max number of docked drug molecules per pdb: {np.max(drug_nums)}")
