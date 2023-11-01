import os
import pickle
import numpy as np
import pandas as pd
import improve_utils
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
from improve_utils import improve_globals as ig
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, GroupKFold, train_test_split, GroupShuffleSplit, StratifiedShuffleSplit, KFold, ShuffleSplit
from preprocess import DATA_DIR, get_docking_data, check_file_and_run



def calculate_scores(y_true, y_pred):
    performance = np.empty(7)
    performance.fill(np.nan)
    performance = pd.Series(performance, index=[
                            'R2', 'MSE', 'MAE', 'pearsonCor', 'pearsonCorPvalue', 'spearmanCor', 'spearmanCorPvalue'])
    performance.loc['R2'] = r2_score(y_true, y_pred)
    performance.loc['MSE'] = mean_squared_error(y_true, y_pred)
    performance.loc['MAE'] = mean_absolute_error(y_true, y_pred)
    rho, pval = pearsonr(y_true, y_pred)
    performance.loc['pearsonCor'] = rho
    performance.loc['pearsonCorPvalue'] = pval
    rho, pval = spearmanr(y_true, y_pred)
    performance.loc['spearmanCor'] = rho
    performance.loc['spearmanCorPvalue'] = pval
    return performance


def load_data_deep(source, datadir, gene_set=None):
    bindings = get_docking_data('../test_data/gausschem4_scores.tsv')
    pretty_indent = '#' * 10
    print(f'{pretty_indent} {source.upper()} {pretty_indent}')
    source = source.lower()

    # Load data
    from preprocess import get_gene_expression_data, get_drug_data, get_response_data, check_file_and_run
    gene_set_name = None
    if gene_set == 'ALL':
        gene_set = None
    if gene_set is not None:
        gene_set_name = gene_set.split('/')[-1].split('.')[0]
    else:
        gene_set_name = 'ALL'
    gene_data_filename = f'{source}_gene_expression_{gene_set_name.lower()}_binding_affinity_genes.tsv'
    drug_data_filename = 'drug_data_imputed_1.tsv'

    gene_expression = check_file_and_run(gene_data_filename, get_gene_expression_data, [
                                         'combined_rnaseq_data', source, gene_set_name])

    gene_expression.columns = [
        'CancID' if x == 'CELL' else f'ge_{x.lower()}' for x in gene_expression.columns]

    drug_descriptors = check_file_and_run(drug_data_filename, get_drug_data, [
                                          'JasonPanDrugsAndNCI60_dragon7_descriptors.tsv'])
    responses = get_response_data(os.path.join('drug_response_data.txt'), source)
    responses.columns = ['CancID', 'DrugID', 'AUC']

    # smiles = pd.read_csv(f"{datadir}/smiles_{source}.csv")   # SMILES
    drug_info = pd.read_csv(os.path.join(DATA_DIR, 'Comprehensive_Drug_List.txt'), sep='\t')
    smiles_cols = ['SMILES(drug_info)', 'SMILES(Jason_SMILES)',
                   'SMILES(NCI60_drug)', 'smiles(Broad)']

    smiles = drug_info[smiles_cols].groupby(
        {x: 'SMILES' for x in drug_info[smiles_cols].columns}, axis=1).first().dropna()
    smiles['DrugID'] = drug_info[source.upper()]
    smiles = smiles.dropna()

    # Use landmark genes
    if gene_set is not None:
        gene_set = pd.read_csv(gene_set, sep='\t').transpose().astype(
            str).values.squeeze().tolist()
        genes = gene_set + [str(x).lower() for x in gene_set]
        genes = ["ge_" + str(g) for g in genes]
    else:
        genes = gene_expression.columns[1:]

    # print(len(set(genes).intersection(set(gene_expression.columns[1:]))))
    genes = list(set(genes).intersection(set(gene_expression.columns[1:])))
    cols = ["CancID"] + genes
    gene_expression = gene_expression[cols]

    return gene_expression, drug_descriptors, None, smiles, responses, bindings


def prepare_dataframe(gene_expression, smiles, bindings, responses, model):
    print(f'@@@ ORIGINAL DRUG DATA: {smiles.shape}')
    response_metric = 'AUC'
    gene_expression, drug_data, binding_data = model.preprocess(gene_expression, smiles, bindings, responses, response_metric, use_map=True)
    drug_data = drug_data.drop(['index'], axis=1)
    
    if 'DrugID' in binding_data.columns:
        drug_data = pd.merge(drug_data, binding_data, on='DrugID', how='inner')
        binding_data = binding_data.drop(['DrugID'], axis=1)
    binding_columns = binding_data.columns

    gene_expression = gene_expression.loc[:, ~
                                          gene_expression.columns.duplicated()].copy()
    drug_data = drug_data.loc[:, ~drug_data.columns.duplicated()].copy()
    drug_columns = drug_data.columns
    drug_columns = list(drug_columns) + ['CancID']

    data = pd.merge(gene_expression, drug_data, on='CancID', how='inner')
    print(f'@@@@@ MERGED BINDING DATA SHAPE: {data.shape} @@@@@')

    gene_expression = gene_expression.drop(['CancID'], axis=1)
    gene_expression_columns = gene_expression.columns

    return data, gene_expression_columns, drug_columns, binding_columns


def run_cross_benchmark(model):
    results_dir = f'Results/DeepTTC'
    if not os.path.isdir(results_dir):
        os.mkdir(results_dir)
    sources = ['ccle'] #['ccle', 'ctrp'] # 'gdsc', 'nci60']  # ccle ctrp

    deepttc = False
    ddr = True
    lightgbm = False

    do_not_recompute_cv = False
    if sum([deepttc, ddr, lightgbm]) != 1:
        raise Exception('Exactly one model mode have to be specified: deepttc, ddr, or lightgbm')

    
    drug_list = pd.read_csv('../test_data/Comprehensive_Drug_List.txt', sep='\t')
    drug_list = drug_list[['UniqueID', 'CTRP', 'GDSC', 'CCLE']]


    gene_set_path = os.path.join(DATA_DIR, 'oncogenes_gausschem4.txt')
    gene_set_name = 'oncogenes_gausschem4'

    sets = {}
    all_results = {}
    for source in sources:
        source = source.lower()

        datadir = f"{DATA_DIR}/data.{source}"
        gene_expression, descriptors, morgan, smiles, responses, bindings = load_data_deep(
            source, datadir, gene_set=gene_set_path)

        binding_columns = []
        preprocessor = model
        data = None
        if deepttc:
            data, gene_expression_columns, drug_columns, binding_columns = prepare_dataframe(
                gene_expression, smiles, bindings, responses, preprocessor)
            data = data.drop_duplicates(subset=['DrugID', 'CancID'])
            data = data.reset_index()
        else:
            responses = responses[['CancID', 'DrugID', 'AUC']]
            responses.columns = ['CancID', 'DrugID', 'Label']

            from preprocess import get_drug_data
            drug_data_filename = 'drug_data_imputed_1.tsv'
            descriptors = check_file_and_run(drug_data_filename, get_drug_data, [
                                             'JasonPanDrugsAndNCI60_dragon7_descriptors.tsv'])
            descriptors['DrugID'] = descriptors['Drug_UniqueID']
            responses = responses.merge(
                drug_list, left_on='DrugID', right_on='UniqueID', how='inner')
            drug_data = pd.merge(
                responses, descriptors, left_on='UniqueID', right_on='DrugID', how='inner')
            drug_data = drug_data.drop_duplicates(
                subset=['UniqueID', 'CancID'])
            print('DRUG DATA SIZE')
            print(drug_data.shape)
            gene_expression_columns = gene_expression.columns
            drug_descriptors_columns = drug_data.columns
            data = pd.merge(gene_expression, drug_data, on='CancID')
            data = data.drop(
                ['CancID'], axis=1)
            
        suffix = ''
        if deepttc:
            suffix = 'DeepTTC'
        if lightgbm:
            suffix = 'LightGBM'
        if ddr:
            suffix = 'DDR'
        suffix = f'{suffix}'
        
        

        def encode_columns(data, column):
            if column in data.columns:
                label_encoder = LabelEncoder()
                label_encoder_path = os.path.join(
                    results_dir, f'label_encoder_{source}_{gene_set_name.lower()}_{suffix}_{column}.pickle')
                data[column] = label_encoder.fit_transform(data[column])
                pickle.dump(label_encoder, open(label_encoder_path, 'wb'))
            return data

        data = encode_columns(data, 'CancID')
        data = encode_columns(data, 'DrugID')
        if lightgbm or ddr:
            data = data.drop(['DrugID_x', 'UniqueID', 'CTRP', 'GDSC', 'CCLE', 'ID', 'Drug_UniqueID', 'DrugID_y'], axis=1)
        

        
        def generate_cv_partition(groups, n_splits=10, random_state=1, validation_size=0.1, main_cv_type='shuffle', validation_split_type='stratified', out_dir=None):
            out_filename = 'CV_partitions.pickle'
            cv_path = None
            if out_dir is not None:
                cv_path = os.path.join(out_dir, out_filename)
                print(cv_path)
                if os.path.isfile(cv_path):
                    return pickle.load(open(cv_path, 'rb'))
            X = np.array(range(len(groups)))
            groups = np.array(groups)
            test_size = 1. / n_splits
            train_size = 1 - test_size
            validation_size = validation_size / train_size

            main_cv = None
            if main_cv_type == 'stratified':
                main_cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            elif main_cv_type == 'shuffle':
                main_cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            elif main_cv_type == 'grouped':
                main_cv = GroupKFold(n_splits=n_splits)
            else:
                raise Exception('Unknown Main CV type!')
            
            validation_split = None
            if validation_split_type == 'stratified':
                validation_split = StratifiedShuffleSplit(n_splits=1, test_size=validation_size, random_state=random_state)
            elif validation_split_type == 'shuffle':
                validation_split = ShuffleSplit(n_splits=1, test_size=validation_size, random_state=random_state)
            elif validation_split_type == 'grouped':
                validation_split = GroupShuffleSplit(n_splits=1, test_size=validation_size, random_state=random_state)
            else:
                raise Exception('Unknown Validation CV type!')
            

            cv_idx_splits = []
            for train_index_outer, test_index in main_cv.split(X, groups, groups):
                X_train = X[train_index_outer]
                groups_train = groups[train_index_outer]
                
                for train_index, validation_index in validation_split.split(X_train, groups_train, groups_train):
                    # X contains indices of the original dataset
                    train_index = X_train[train_index]
                    validation_index = X_train[validation_index]
                    cv_idx_splits.append( (train_index, validation_index, test_index) )
                    break

            if out_dir is not None:
                pickle.dump(cv_idx_splits, open(cv_path, 'wb'))
            return cv_idx_splits

        from Milestone_16_Functions import generate_cross_validation_partition
        groups = range(data.shape[0])
        #groups = data['DrugID']
        cv_type = 'grouped'
        cv_partitions = generate_cv_partition(groups, n_splits=10, random_state=1, validation_size=0.1, main_cv_type=cv_type, validation_split_type='stratified', out_dir=results_dir)  
        #generate_cross_validation_partition(groups, n_folds=10, n_repeats=1, portions=[8, 1, 1], random_seed=1)

        set_name = f'{source}'
        print(f'Dataset size: {data.shape}')
        print(f'Dataset size: {data.shape}')
        print(f'Dataset size: {data.shape}')
        for split_idx in range(len(cv_partitions)):
            train_idx, val_idx, test_idx = cv_partitions[split_idx]
            training_results_path = os.path.join(
                results_dir, f'training_results_{set_name}_{suffix}_split_{split_idx}.pickle')
            validation_results_path = os.path.join(
                results_dir, f'validation_results_{set_name}_{suffix}_split_{split_idx}.pickle')

            print(f'CV iteration {split_idx}')
            if os.path.isfile(training_results_path) and do_not_recompute_cv:
                if split_idx < 10:
                    continue

            if deepttc:
                from copy import deepcopy
                print(f'Dataset size: {data.shape[0]}')
                # Removing deepcopy
                train_drug = data.loc[data.index[train_idx], drug_columns] #)
                train_rna = data.loc[data.index[train_idx],
                                     gene_expression_columns]
                train_binding = data.loc[data.index[train_idx],
                                         binding_columns]
                val_drug = data.loc[data.index[val_idx], drug_columns]
                val_rna = data.loc[data.index[val_idx],
                                   gene_expression_columns]
                val_binding = data.loc[data.index[val_idx], binding_columns]
                test_drug = data.loc[data.index[test_idx], drug_columns]
                test_rna = data.loc[data.index[test_idx],
                                    gene_expression_columns]
                test_binding = data.loc[data.index[test_idx], binding_columns]

                rna_scaler = StandardScaler()                  
                train_rna = rna_scaler.fit_transform(train_rna)
                val_rna = rna_scaler.transform(val_rna)
                test_rna = rna_scaler.transform(test_rna)
                binding_scaler = StandardScaler()
                train_binding = binding_scaler.fit_transform(train_binding)
                val_binding = binding_scaler.transform(val_binding)
                test_binding = binding_scaler.transform(test_binding)


                sets[set_name] = (test_drug, test_rna, test_binding)

                training_results, validation_results = model.train(train_drug, train_rna, train_binding,
                            val_drug, val_rna, val_binding)
                pickle.dump(training_results, open(training_results_path, 'wb'))
                pickle.dump(validation_results, open(validation_results_path, 'wb'))


            if lightgbm:
                data_cols = [x for x in data.columns if x != 'Label']
                X = data[data_cols]
                y = data['Label']

                scaler = StandardScaler()
                X_train = X.loc[X.index[train_idx], :]
                X_train = scaler.fit_transform(X_train)
                y_train = y[y.index[train_idx]]
                X_val = X.loc[X.index[val_idx], :]
                X_val = scaler.transform(X_val)
                y_val = y[y.index[val_idx]]
                X_test = X.loc[X.index[test_idx], :]
                X_test = scaler.transform(X_test)
                y_test = y[y.index[test_idx]]

                import lightgbm as lgb
                parameters = {
                    'n_estimators': 1000,
                    'n_jobs': 20
                }
                model = lgb.LGBMRegressor(
                    n_estimators=parameters['n_estimators'],
                    n_jobs=parameters['n_jobs'])

                model.fit(X=X_train, y=y_train.values, eval_set=[(X_val, y_val.values)],
                          early_stopping_rounds=30)
                feature_importances = pd.Series(
                    model.feature_importances_, data_cols).sort_values(ascending=False)[:20]
                print(feature_importances)
                
            if ddr:
                to_drop = [x for x in data.columns if ('drug' in x.lower()) or ('unique' in x.lower()) or ('cancid' in x.lower()) or ('id' in x.lower()) or ('sample' in x.lower())]
                to_drop = to_drop + ['CTRP', 'GDSC', 'CCLE']
                to_drop = np.intersect1d(to_drop, data.columns)
                data = data.drop(to_drop, axis=1)
                data_cols = [x for x in data.columns if x != 'Label']
                X = data[data_cols]
                y = data['Label']

                
                scaler = StandardScaler()
                X.loc[X.index[train_idx], :] = scaler.fit_transform(X.loc[X.index[train_idx], :])
                X_train = X.loc[X.index[train_idx], :]
                y_train = y[y.index[train_idx]]
                
                X.loc[X.index[val_idx], :] = scaler.transform(X.loc[X.index[val_idx], :])
                X_val = X.loc[X.index[val_idx], :]
                y_val = y[y.index[val_idx]]
                
                X.loc[X.index[test_idx], :] = scaler.transform(X.loc[X.index[test_idx], :])
                X_test = X.loc[X.index[test_idx], :]
                y_test = y[y.index[test_idx]]

                from DeepDrugRegressor import DeepDrugRegressor

                ddr_model = DeepDrugRegressor()
                gene_expression_columns = np.intersect1d(X.columns, gene_expression_columns)
                drug_descriptors_columns = np.intersect1d(X.columns, drug_descriptors_columns)
                #gene_expression_idx = np.array([x in gene_expression_columns for x in X.columns])
                #drug_descriptors_idx = np.array([x in drug_descriptors_columns for x in X.columns])
                #breakpoint()
                ddr_model.train(X_train[gene_expression_columns], X_train[drug_descriptors_columns], y_train, X_val[gene_expression_columns], X_val[drug_descriptors_columns], y_val, dropout_rate=0.1)

            print('Finished training')

            print('Predicting...')
            print(f'Predicting {set_name}')
            if set_name not in all_results:
                all_results[set_name] = {}
            
            if deepttc:
                drug_test_set, rna_test_set, binding_test_set = sets[set_name]
                y_test = drug_test_set['Label']
                _, y_pred, _, _, _, _, _, _, _, _ = model.predict(
                    drug_test_set, rna_test_set, binding_test_set)
                res = pd.DataFrame.from_dict({'Test': y_test, 'Pred': y_pred})
                test_results_path = os.path.join(
                    results_dir, f'test_results_{set_name}_{suffix}_split_{split_idx}.pickle')
                test_results = {}
                test_results[0] = (y_test, drug_test_set['DrugID'], drug_test_set['CancID'], y_pred)
                pickle.dump(test_results, open(test_results_path, 'wb'))
                print(res)
            if lightgbm:
                y_pred = None
                y_pred = model.predict(X_test)
            if ddr:
                y_pred = None
                y_pred = ddr_model.predict(X_test[gene_expression_columns], X_test[drug_descriptors_columns])
            cv_scores = calculate_scores(y_test.values, y_pred)
            all_results[set_name][split_idx] = cv_scores

            out_path = os.path.join(
                results_dir, f'{set_name}_{set_name}_{suffix}_split_{split_idx}.tsv')
            cv_scores = pd.DataFrame(cv_scores)
            cv_scores.to_csv(out_path, sep='\t')
            print(cv_scores)

        for set_name in sets:
            results = pd.DataFrame.from_dict(all_results[set_name], orient='index')
            summarized_results = results.mean(axis=0)
            print(results)
            print(summarized_results)

            full_out_file_name = f'test_{source}_{gene_set_name.lower()}_{suffix}_full_cv_scores.tsv'
            full_out_path = os.path.join(results_dir, full_out_file_name)
            summarized_out_file_name = f'test_{source}_{gene_set_name.lower()}_{suffix}_mean_cv_scores.tsv'
            summarized_out_path = os.path.join(
                results_dir, summarized_out_file_name)

            results.to_csv(full_out_path, sep='\t')
            summarized_results.to_csv(summarized_out_path, sep='\t', header=None)
