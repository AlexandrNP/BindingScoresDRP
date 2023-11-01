import pandas as pd
import numpy as np


def get_binding_affinity_data():
    data = pd.read_csv('binding_affinity.tsv', sep='\t', index_col=0)
    return data.transpose()


def get_gausschem4_data():
    data = pd.read_csv('gausschem4_scores.tsv', sep='\t', index_col=0)
    return data.transpose()


def get_pdb2gene_mapping():
    data = pd.read_csv(
        'DrugLists/All-Drugs_All-PDB_2022.txt', sep='\t', header=None)
    mapping = {}
    pdbs = [x.split(':')[0].lower() for x in data[data.columns[1]]]
    genes = data[data.columns[2]]

    for pdb, gene in zip(pdbs, genes):
        mapping[pdb] = gene

    return mapping


def split_docking_datasets_on_genes(docking_data, mapping):
    splits = {}
    for column in docking_data.columns[1:]:
        gene = mapping[column.lower()]
        if gene not in splits:
            splits[gene] = []
        splits[gene].append(docking_data[column])
    return splits


def get_max_idx(multi_pockets):
    max_values_idx = 0
    max_values = 0
    for i in range(len(multi_pockets)):
        values = np.count_nonzero(~np.isnan(multi_pockets[i]))
        if values > max_values:
            max_values = values
            max_values_idx = i
    return max_values_idx


def fill_nans_in_max_pocket(multi_pockets, max_idx):
    max_pocket = multi_pockets[max_idx]
    for i in range(len(multi_pockets)):
        nan_idx = np.isnan(max_pocket)
        max_pocket[nan_idx] = multi_pockets[i][nan_idx]
    return max_pocket


def merge_splits_by_most_values(gene_splits):
    merged_dict = {}
    for gene in gene_splits:
        multi_pockets = gene_splits[gene]
        max_values_idx = get_max_idx(multi_pockets)
        #merged_dict[gene] = multi_pockets[max_values_idx]
        merged_dict[gene] = fill_nans_in_max_pocket(
            multi_pockets, max_values_idx)

    dataframe = pd.DataFrame.from_dict(merged_dict, orient='columns')
    return dataframe


def map_to_genes(data):
    mapping = get_pdb2gene_mapping()
    splits = split_docking_datasets_on_genes(data, mapping)
    genes_dataframe = merge_splits_by_most_values(splits)
    genes_dataframe = genes_dataframe.fillna(0)
    return genes_dataframe


def weight_by_gene_expression(docking_data, gene_data):
    docking_data.columns = [x.lower() for x in docking_data.columns]
    gene_data.columns = [x.lower() for x in gene_data.columns]
    print(np.intersect1d(docking_data.columns, gene_data.columns))


if __name__ == "__main__":
    binding_affinity_data = get_binding_affinity_data()
    gausschem4_data = get_gausschem4_data()

    datasets = [binding_affinity_data, gausschem4_data]
    dataset_names = ['genes_binding_affinity.tsv',
                     'genes_gausschem4_scores.tsv']

    for name, dataset in zip(dataset_names, datasets):
        mapped_binding = map_to_genes(dataset)
        print(mapped_binding)
        mapped_binding.to_csv(name, sep='\t')
        weight_by_gene_expression(mapped_binding, )
