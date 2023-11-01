from tkinter import W
import pandas as pd
import numpy as np
import pickle
import os
from copy import deepcopy
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA, SparsePCA


DATA_DIR = '../test_data'
GENE_SET_DIR = DATA_DIR
GENE_EXPRESSION_DIR = DATA_DIR
DRUG_DIR = DATA_DIR
RESPONSE_DIR = DATA_DIR
OUT_DIR = 'Response_Datasets'
SOURCES = ['CCLE', 'CTRP', 'GDSC']


def get_gene_set(gene_set_name: str) -> list:
    gene_set_file = None
    if gene_set_name == 'LINCS':
        gene_set_file = 'lincs1000_list.txt'
    elif gene_set_name == 'ONCOGENES':
        gene_set_file = 'oncogenes_list.txt'
    elif gene_set_name.lower() == 'oncogenes_binding':
        gene_set_file = 'oncogenes_binding.txt'
    elif gene_set_name.lower() == 'oncogenes_gausschem4':
        gene_set_file = 'oncogenes_gausschem4.txt'
    else:
        raise Exception('Unknown gene set')

    gene_set_path = os.path.join(GENE_SET_DIR, gene_set_file)
    gene_set = pd.read_csv(gene_set_path, sep='\t', header=None)

    return list(gene_set[gene_set.columns[0]])


def get_gene_expression_data(expression_file_name: str, source: str, gene_set_name: str) -> pd.DataFrame:

    gene_expression_path = os.path.join(
        GENE_EXPRESSION_DIR, expression_file_name)
    cols_to_save = None
    if gene_set_name == 'ALL':
        cols_to_save = None  # gene_expression_data.columns
    else:
        cols_to_save = get_gene_set(gene_set_name) + ['Sample']
        gene_expression_data_columns = pd.read_csv(
            gene_expression_path, sep='\t', nrows=1, index_col=None).columns.tolist()
        cols_to_save = np.intersect1d(
            cols_to_save, gene_expression_data_columns)

    gene_expression_data = None
    source_filtered_data = None
    if expression_file_name == 'Combat_AllGenes_UniqueSample.txt':
        gene_expression_data = pd.read_csv(
            gene_expression_path, sep='\t', index_col=None)
        gene_expression_data.index = gene_expression_data.geneSymbol
        gene_expression_data = gene_expression_data.iloc[:, 2:]
        gene_expression_data = np.transpose(gene_expression_data)
        gene_expression_data['Sample'] = gene_expression_data.index
        source_filtered_data = gene_expression_data.reset_index()

    else:
        if cols_to_save is not None:
            gene_expression_data = pd.read_csv(
                gene_expression_path, sep='\t', index_col=None, usecols=cols_to_save)
        else:
            gene_expression_data = pd.read_csv(
                gene_expression_path, sep='\t', index_col=None)

        sample_names = gene_expression_data['Sample']
        source_filtered_idx = [sample_names.index[x]
             for x in sample_names.index if source.lower() in sample_names.loc[x].lower()]

        source_filtered_data = gene_expression_data.loc[
            gene_expression_data.index[source_filtered_idx], gene_expression_data.columns]
    source_filtered_data = source_filtered_data.rename(
        {'Sample': 'CELL'}, axis='columns')

    return source_filtered_data


def get_drug_data(drug_features_filename):
    drug_features_path = os.path.join(DRUG_DIR, drug_features_filename)
    drug_data = pd.read_csv(drug_features_path, sep='\t')
    drug_data = drug_data.rename({'UniqueID': 'Drug_UniqueID'}, axis='columns')
    drug_data = drug_data.dropna(axis=1, how='any')

    drug_data_columns = list(drug_data.columns)
    drug_data_columns.remove('ID')
    drug_data_columns.remove('Drug_UniqueID')

    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    drug_data[drug_data_columns] = imp.fit_transform(
        drug_data[drug_data_columns])

    return drug_data


def get_response_data(response_filename, source):
    response_path = os.path.join(RESPONSE_DIR, response_filename)
    response_data = pd.read_csv(response_path, sep='\t')
    response_data = response_data[[
        'SOURCE', 'CCLE_CCL_UniqueID', 'NCI60_CCL_UniqueID', 'CELL', 'Drug_UniqueID', 'AUC']]
    column_to_keep = 'CELL'  # 'CELL'
    response_data['CELL'] = response_data[column_to_keep]

    # Remove samples that have both NCI60 and CCLE mapping

    response_data_source = response_data['SOURCE']
    response_data_source_idx = [source.lower() in x.lower()
                                for x in response_data_source]
    response_data_filtered = response_data.loc[response_data.index[response_data_source_idx],
                                               response_data.columns]
    response_data_filtered = response_data_filtered.drop(
        ['SOURCE', 'CCLE_CCL_UniqueID', 'NCI60_CCL_UniqueID'], axis=1)
    return response_data_filtered


def get_docking_data(docking_data_filename):
    docking_data = pd.read_csv(docking_data_filename, sep='\t',
                               header=None, index_col=0,
                               dtype=str, low_memory=False)

    docking_data = docking_data.replace('NaN', np.nan)

    docking_data = docking_data.transpose()
    data_columns = docking_data.columns[1:]
    docking_data.columns = ['Drug_UniqueID'] + list(data_columns)
    docking_data = docking_data.replace('NaN', np.nan)
    numeric_data = docking_data[data_columns].astype(float)
    docking_data[data_columns] = numeric_data
    max_score = docking_data[data_columns].max().max()
    docking_data = docking_data.fillna(max_score * 10)
    return docking_data


def get_binding_affinity_data(binding_affinity_filename):
    binding_affinity_data = pd.read_csv(binding_affinity_filename, sep='\t',
                                        index_col=0, header=None, dtype=str)

    print(binding_affinity_data.columns)
    binding_affinity_data = binding_affinity_data.replace('NaN', np.nan)
    binding_affinity_data = binding_affinity_data.replace('', np.nan)
    binding_affinity_data = binding_affinity_data.transpose()
    data_columns = binding_affinity_data.columns[1:]
    binding_affinity_data.columns = ['Drug_UniqueID'] + list(data_columns)
    binding_affinity_data = binding_affinity_data.replace('NaN', np.nan)

    numeric_data = binding_affinity_data[data_columns].astype(float)
    binding_affinity_data[data_columns] = numeric_data
    binding_affinity_data = binding_affinity_data.fillna(0)
    print(binding_affinity_data.columns)
    return binding_affinity_data


def get_binding_affinity_data_PCA(binding_affinity_filename, n_components):
    dataframe = get_binding_affinity_data(binding_affinity_filename)
    data_cols = dataframe.columns[1:]
    pca = SparsePCA(n_components)
    X = pca.fit_transform(dataframe[data_cols].values)
    result = pd.DataFrame(X)
    result['Drug_UniqueID'] = dataframe['Drug_UniqueID']

    return result


def weight_by_gene_expression(docking_data, merged_data, name):
    docking_data.columns = [x.lower() for x in docking_data.columns]
    gene_columns = [
        x.lower() if 'Drug_' not in x else x for x in merged_data.columns]
    merged_data.columns = gene_columns
    gene_columns = np.intersect1d(docking_data.columns, gene_columns)

    combined_data = docking_data[gene_columns]
    combined_data = pd.DataFrame(
        combined_data.values, columns=combined_data.columns, index=docking_data[docking_data.columns[0]])
    combined_data.columns = [f'{x}.{name}' for x in combined_data.columns]
    combined_data_columns = combined_data.columns
    combined_data['Drug_UniqueID'] = list(combined_data.index)

    print(type(merged_data['Drug_UniqueID'][0]))
    print(type(combined_data['Drug_UniqueID'][0]))

    new_data = pd.merge(merged_data, combined_data,
                        on='Drug_UniqueID', how='inner')
    print(new_data[combined_data_columns])
    print(new_data[gene_columns])
    new_data[combined_data_columns] = np.multiply(
        new_data[combined_data_columns].values, new_data[gene_columns].values)

    return new_data


def merge_data(gene_data, drug_data, docking_data, binding_affinity_data, response_data, binding_data_by_gene=None, gausschem4_by_gene=None):
    original_response_data = response_data.copy()

    def swap_dataframe(df1, df2):
        df3 = deepcopy(df1)
        df1 = deepcopy(df2)
        df2 = df3
        return df1, df2

    def non_unique_merge(df1, df2, key_column):
        # Only df2 dataframe can contain duplicates
        # Check if df1 contains duplicates
        if len(np.unique(df1[key_column].values)) > len(df1[key_column].values):
            df1, df2 = swap_dataframe(df1, df2)
        if len(np.unique(df1[key_column].values)) != len(df1[key_column].values):
            raise Exception("Both dataframes have duplicate values")
        n, _ = df1.shape
        df1_all_columns = list(df1.columns)
        df1_data_columns = df1_all_columns.remove(key_column)

        df1 = df1.reset_index()
        df2 = df2.reset_index()
        reshaped_data = {}

        for i in range(n):
            key = df1.loc[i, key_column]
            data = df1[df1_data_columns]
            df2_keys = df2[key_column]
            indices = df2_keys.where(df2_keys == key)
            for index in indices:
                reshaped_data[index] = data

        additional_df = pd.DataFrame.from_dict(reshaped_data, orient='index')
        merged = df2.concatenate(additional_df)
        return merged

    gene_response = pd.merge(gene_data, response_data, on='CELL', how='inner')
    gene_response_cols = list(gene_response.columns)
    gene_response_cols.remove('AUC')
    gene_response = gene_response[gene_response_cols]

    drug_data = drug_data.drop_duplicates(
        subset=['Drug_UniqueID'], ignore_index=True)


    if 'gausschem4' not in docking_data.columns[-1]:
        docking_data.columns = [f'{col_name}.gausschem4' if col_name !=
                                'Drug_UniqueID' else col_name for col_name in docking_data.columns]
    if 'binding' not in binding_affinity_data.columns[-1]:
        binding_affinity_data.columns = [f'{col_name}.binding' if col_name !=
                                         'Drug_UniqueID' else col_name for col_name in binding_affinity_data.columns]
    drug_data = pd.merge(drug_data, docking_data,
                         on='Drug_UniqueID', how='inner')
    drug_data = pd.merge(drug_data, binding_affinity_data,
                         on='Drug_UniqueID', how='inner')

    drug_response = pd.merge(
        drug_data, original_response_data, on='Drug_UniqueID', how='inner')
    fully_merged = pd.merge(gene_response, drug_response, on=[
                            'CELL', 'Drug_UniqueID'], how='inner')

    if binding_data_by_gene is not None:
        fully_merged = weight_by_gene_expression(
            binding_data_by_gene, fully_merged, 'binding')
    if gausschem4_by_gene is not None:
        fully_merged = weight_by_gene_expression(
            gausschem4_by_gene, fully_merged, 'gausschem4')

    print('Merging process...')
    print(gene_response.shape, drug_data.shape, fully_merged.shape)

    return fully_merged


def check_file_and_run(filename, function_name, function_params):
    data = None
    if not os.path.isfile(filename):
        data = function_name(*function_params)
        data.to_csv(filename, sep='\t', index=None)
    else:
        data = pd.read_csv(filename, sep='\t')
    return data


def process_data_deepttc(args):
    train_drug = test_drug = train_rna = test_rna = None
    if not os.path.exists(args.train_data_rna) or \
            not os.path.exists(args.test_data_rna) or \
            args.generate_input_data:
        obj = DataEncoding(args.vocab_dir, args.cancer_id,
                           args.sample_id, args.target_id, args.drug_id)
        train_drug, test_drug = obj.Getdata.ByCancer(random_seed=args.rng_seed)

        train_drug, train_rna, test_drug, test_rna = obj.encode(
            traindata=train_drug,
            testdata=test_drug)
        print('Train Drug:')
        print(train_drug)
        print('Train RNA:')
        print(train_rna)

        pickle.dump(train_drug, open(args.train_data_drug, 'wb'), protocol=4)
        pickle.dump(test_drug, open(args.test_data_drug, 'wb'), protocol=4)
        pickle.dump(train_rna, open(args.train_data_rna, 'wb'), protocol=4)
        pickle.dump(test_rna, open(args.test_data_rna, 'wb'), protocol=4)
    else:
        train_drug = pickle.load(open(args.train_data_drug, 'rb'))
        test_drug = pickle.load(open(args.test_data_drug, 'rb'))
        train_rna = pickle.load(open(args.train_data_rna, 'rb'))
        test_rna = pickle.load(open(args.test_data_rna, 'rb'))
    return train_drug, test_drug, train_rna, test_rna



def get_drug_map():
    import os
    drug_map_file = 'drug_map.pickle'
    if os.path.isfile(drug_map_file):
        return pickle.load(open(drug_map_file, 'rb'))
    drug_list = pd.read_csv(os.path.join(DATA_DIR, 'Comprehensive_Drug_List.txt'), sep='\t')
    drug_map = {}
    for col in drug_list.columns[1:6]:
        if col == 'UniqueID':
            continue
        for unique_id, current_id in zip(drug_list['UniqueID'], drug_list[col]):
            drug_map[current_id] = unique_id
    pickle.dump(drug_map, open(drug_map_file, 'wb'))
    return drug_map

if __name__ == "__main__":

    drug_data_filename = 'drug_data_imputed.tsv'
    docking_data_filename = 'docking_data.tsv'
    binding_affinity_filename = 'binding_affinity_postprocessed.tsv'
    binding_affinity_PCA_filename = 'PCA_binding_affinity.tsv'
    gene_set_name = 'ONCOGENES_GAUSSCHEM4'

    try:
        os.mkdir(OUT_DIR)
    except:
        pass

    docking_data = check_file_and_run(
        docking_data_filename, get_docking_data, ['gausschem4_scores.tsv'])
    binding_affinity_data = check_file_and_run(
        binding_affinity_filename, get_binding_affinity_data, ['binding_affinity.tsv'])
    binding_affinity_data_PCA = check_file_and_run(
        binding_affinity_PCA_filename, get_binding_affinity_data_PCA, ['binding_affinity.tsv', 100])
    binding_data_by_gene = pd.read_csv('genes_binding_affinity.tsv', sep='\t')
    gausschem4_by_gene = pd.read_csv('genes_gausschem4_scores.tsv', sep='\t')

    for source in SOURCES:
        gene_data_filename = f'{source}_gene_expression_{gene_set_name.lower()}_binding_affinity.tsv'
        response_data_filename = f'{source}_response.tsv'

        gene_data = check_file_and_run(gene_data_filename, get_gene_expression_data, [
                                       'combined_rnaseq_data', source, gene_set_name])
        drug_data = check_file_and_run(drug_data_filename, get_drug_data, [
                                       'JasonPanDrugsAndNCI60_dragon7_descriptors.tsv'])
        response_data = get_response_data('drug_response_data.txt', source)

        merged_data = merge_data(gene_data, drug_data, docking_data, binding_affinity_data,
                                 response_data, gausschem4_by_gene=None, binding_data_by_gene=None)

        output_file_path = os.path.join(
            OUT_DIR, f'{source}_{gene_set_name.lower()}_no_drug_missing_val_gausschem4_dataset_docking_binding.tsv')
        merged_data.to_csv(output_file_path, sep='\t')
