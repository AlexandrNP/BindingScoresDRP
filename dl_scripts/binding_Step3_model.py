# python3
# -*- coding:utf-8 -*-

"""
@author:野山羊骑士
@e-mail：thankyoulaojiang@163.com
@file：PycharmProject-PyCharm-model.py
@time:2021/9/15 16:33 
"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from lifelines.utils import concordance_index
from scipy.stats import pearsonr, spearmanr
import copy
import time
import pickle

import torch
from torch.utils import data
import torch.nn.functional as F
from torch.autograd import Variable
from torch import dropout, nn

from model_helper import Encoder_MultipleLayers, Embeddings
from Step2_DataEncoding import DataEncoding
from preprocess import get_drug_map

device = torch.device('cpu')


class data_process_loader(data.Dataset):
    def __init__(self, list_IDs, labels, drug_df, rna_df, binding_df, add_noise=False):
        'Initialization'
        drug_df = drug_df.reset_index()
        self.labels = labels
        self.list_IDs = list_IDs
        self.drug_df = pd.DataFrame(drug_df).reset_index(drop=True)
        self.rna_df = pd.DataFrame(rna_df).reset_index(drop=True)
        self.binding_df = pd.DataFrame(binding_df).reset_index(drop=True)
        self.drug_df.loc[drug_df.index, 'DrugID'] = [
            int(x.split('_')[-1]) for x in drug_df['DrugID']]
        self.drug_ids = torch.tensor(self.drug_df['DrugID'])
        self.add_noise = add_noise
        self.rna_diameters = rna_df.apply(np.ptp, axis=0)
        self.binding_diameters = binding_df.apply(np.ptp, axis=0)
        self.rna_size = rna_df.shape[1]
        self.binding_size = binding_df.shape[1]

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'
        index = self.list_IDs[index]
        v_d = self.drug_df.iloc[index]['drug_encoding']
        v_p = np.array(self.rna_df.iloc[index])
        v_b = np.array(self.binding_df.iloc[index])
        d_id = np.array(self.drug_df.iloc[index]['DrugID'])
        if self.add_noise:
            std = 0.05
            v_p = v_p + \
                np.random.normal(0, std, size=len(
                    self.rna_diameters))*self.rna_diameters
            v_p = np.array(v_p)
            v_b = v_b + \
                np.random.normal(0, std, size=len(
                    self.binding_diameters))*self.binding_diameters
            v_b = np.array(v_b)
        y = np.array(self.labels[index])

        return v_d, v_p, v_b, y, d_id


class transformer(nn.Sequential):
    def __init__(self, input_dim_drug,
                 transformer_emb_size_drug, dropout,
                 transformer_n_layer_drug,
                 transformer_intermediate_size_drug,
                 transformer_num_attention_heads_drug,
                 transformer_attention_probs_dropout,
                 transformer_hidden_dropout_rate):
        super(transformer, self).__init__()

        self.emb = Embeddings(input_dim_drug,
                              transformer_emb_size_drug,
                              50,
                              dropout)

        self.encoder = Encoder_MultipleLayers(transformer_n_layer_drug,
                                              transformer_emb_size_drug,
                                              transformer_intermediate_size_drug,
                                              transformer_num_attention_heads_drug,
                                              transformer_attention_probs_dropout,
                                              transformer_hidden_dropout_rate)

    def forward(self, v):
        e = v[0].long().to(device)
        e_mask = v[1].long().to(device)
        ex_e_mask = e_mask.unsqueeze(1).unsqueeze(2)
        ex_e_mask = (1.0 - ex_e_mask) * -10000.0

        emb = self.emb(e)
        encoded_layers = self.encoder(emb.float(), ex_e_mask.float())
        return encoded_layers[:, 0]


class MLP(nn.Sequential):
    def __init__(self, input_dim, mlp_hidden_dims=[1024, 256, 64], hidden_dim_out=256):
        input_dim_gene = input_dim
        hidden_dim_gene = hidden_dim_out
        mlp_hidden_dims_gene = mlp_hidden_dims
        super(MLP, self).__init__()
        layer_size = len(mlp_hidden_dims_gene) + 1
        dims = [input_dim_gene] + mlp_hidden_dims_gene + [hidden_dim_gene]
        self.predictor = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1]) for i in range(layer_size)]) 
        self.dropout = nn.Dropout(0)

    def forward(self, v):
        # predict
        v = v.float().to(device)
        for i, l in enumerate(self.predictor):
            v = self.dropout(v)
            v = F.relu(l(v))
        return v

    def set_dropout(self, dropout_rate):
        self.dropout = nn.Dropout(dropout_rate)


class Classifier(nn.Sequential):
    def __init__(self, args, model_drug, model_gene, model_binding=None):
        super(Classifier, self).__init__()
        self.use_binding = args.use_binding
        self.input_dim_drug = args.input_dim_drug_classifier
        self.input_dim_gene = args.input_dim_gene_classifier
        self.input_dim_binding_classifier = args.input_dim_binding_classifier
        self.model_drug = model_drug
        self.model_gene = model_gene
        self.model_binding = model_binding
        self.dropout = nn.Dropout(args.dropout)
        self.hidden_dims = [2048, 1024, 1024, 512, 256]
        layer_size = len(self.hidden_dims) + 1
        if not self.use_binding:
            self.input_dim_binding_classifier = 0

        dims = [self.input_dim_drug + self.input_dim_gene + self.input_dim_binding_classifier] + \
            self.hidden_dims + [1]
        self.dims = dims
        self.predictor = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1]) for i in range(layer_size)])

    def forward(self, v_D, v_P, v_B):
        # each encoding
        v_D = self.model_drug(v_D)
        v_P = self.model_gene(v_P)

        # concatenate and classify
        v_f = None
        if self.use_binding:
            v_B = self.model_binding(v_B)
            v_f = torch.cat((v_D, v_P, v_B), 1)
        else:
            v_f = torch.cat((v_D, v_P), 1)
        for i, l in enumerate(self.predictor):
            if i == (len(self.predictor) - 1):
                v_f = l(v_f)
            else:
                v_f = F.relu(self.dropout(l(v_f)))
        return v_f


class DeepTTC:
    def __init__(self, modeldir, args):
        self.model_drug = transformer(args.input_dim_drug,
                                      args.transformer_emb_size_drug,
                                      args.dropout,
                                      args.transformer_n_layer_drug,
                                      args.transformer_intermediate_size_drug,
                                      args.transformer_num_attention_heads_drug,
                                      args.transformer_attention_probs_dropout,
                                      args.transformer_hidden_dropout_rate)
        self.device = device
        self.modeldir = modeldir
        self.record_file = os.path.join(
            self.modeldir, "valid_markdowntable.txt")
        self.pkl_file = os.path.join(self.modeldir, "loss_curve_iter.pkl")
        self.args = args
        self.model = None

    def test(self, datagenerator, model):
        y_label = []
        y_pred = []
        best_scores = {}
        model.eval()
        for i, (v_drug, v_gene, v_binding, label, drug_id) in enumerate(datagenerator):
            score = model(v_drug, v_gene, v_binding)
            loss_fct = torch.nn.MSELoss()
            n = torch.squeeze(score, 1)
            dev_label = Variable(torch.from_numpy(
                np.array(label)).float()).to(self.device)
            dev_drug_id = Variable(torch.from_numpy(
                np.array(drug_id)).int()).to(self.device)
            loss = loss_fct(n, dev_label)
            logits = torch.squeeze(score).detach().cpu().numpy()
            label_ids = label.to('cpu').numpy()
            y_label = y_label + label_ids.flatten().tolist()
            y_pred = y_pred + logits.flatten().tolist()

        model.train()
        loss = mean_squared_error(y_label, y_pred)

        return y_label, y_pred, \
            mean_squared_error(y_label, y_pred), \
            np.sqrt(mean_squared_error(y_label, y_pred)), \
            pearsonr(y_label, y_pred)[0], \
            pearsonr(y_label, y_pred)[1], \
            spearmanr(y_label, y_pred)[0], \
            spearmanr(y_label, y_pred)[1], \
            concordance_index(y_label, y_pred), \
            loss

    def train(self, train_drug, train_rna, train_binding, val_drug, val_rna, val_binding):
        model_gene = MLP(input_dim=np.shape(train_rna)[1], mlp_hidden_dims=[
                         1024, 256, 64], hidden_dim_out=self.args.input_dim_gene_classifier)
        #model_gene.set_dropout(0.1)
        model_binding = MLP(input_dim=np.shape(train_binding)[
                            1], mlp_hidden_dims=[1024, 1024, 256, 64, 32], hidden_dim_out=self.args.input_dim_binding_classifier)
        #model_binding.set_dropout(0.1)

        if self.args.use_binding:
            self.model = Classifier(
                self.args, self.model_drug, model_gene, model_binding)
        else:
            self.model = Classifier(self.args, self.model_drug, model_gene)

        lr = self.args.learning_rate
        decay = 0  # 1e-3
        BATCH_SIZE = self.args.batch_size
        train_epoch = self.args.epochs
        earlier_stopping_num = 1500
        self.model = self.model.to(self.device)

        opt = torch.optim.Adam(self.model.parameters(),
                               lr=lr, weight_decay=decay)

        loss_history = []

        def get_drug_weight(dataset):
            weights = torch.empty(dataset.shape[0])
            unique_drugs, counts = np.unique(
                dataset['DrugID'].values, return_counts=True)
            drug_map = {}
            for drug_id, count in zip(unique_drugs, counts):
                drug_map[drug_id] = 1./count

            i = 0
            for drug_id in dataset['DrugID']:
                weights[i] = drug_map[drug_id]
                i += 1

            return weights

        train_weights = get_drug_weight(train_drug)
        train_sampler = torch.utils.data.sampler.WeightedRandomSampler(
            train_weights, 2*int(train_drug.shape[0])) 


        params = {'batch_size': BATCH_SIZE,
                  'shuffle': False,
                  'num_workers': 0,
                  'drop_last': False}

        print(train_drug)
        print('DrugID' in train_drug.columns)
        train_drug = train_drug.reset_index()
        train_rna = pd.DataFrame(train_rna)
        train_binding = pd.DataFrame(train_binding)
        val_drug = val_drug.reset_index()
        val_rna = pd.DataFrame(val_rna)
        val_binding = pd.DataFrame(val_binding)
        training_generator = data.DataLoader(data_process_loader(
            train_drug.index.values, train_drug.Label.values, train_drug, train_rna, train_binding, add_noise=False), sampler=train_sampler, **params)
        validation_generator = data.DataLoader(data_process_loader(
            val_drug.index.values, val_drug.Label.values, val_drug, val_rna, val_binding), **params)

        model_max = copy.deepcopy(self.model)

        valid_metric_record = []
        valid_metric_header = ['# epoch', "MSE", 'RMSE',
                               "Pearson Correlation", "with p-value",
                               'Spearman Correlation', "with p-value2",
                               "Concordance Index"]
        def float2str(x): return '%0.4f' % x
        print('--- Go for Training ---')
        t_start = time.time()
        iteration_loss = 0
        earlier_stopping_counter = 0
        min_loss_val = 1E10

        best_scores = {}
        for epo in range(train_epoch):
            for i, (v_d, v_p, v_b, label, drug_ids) in enumerate(training_generator):
                score = self.model(v_d, v_p, v_b)
                label = Variable(torch.from_numpy(
                    np.array(label))).float().to(self.device)

                loss_fct = torch.nn.MSELoss()
                n = torch.squeeze(score, 1).float()
                loss = loss_fct(n, label)
                loss_history.append(loss.item())
                loss_val = loss.cpu().detach().numpy()
                iteration_loss += 1

                opt.zero_grad()
                loss.backward()
                opt.step()
                if (i % 1000 == 0):
                    t_now = time.time()
                    print('Training at Epoch ' + str(epo + 1) +
                          ' iteration ' + str(i) +
                          ' with loss ' + str(loss.cpu().detach().numpy())[:7] +
                          ". Total time " + str(int(t_now - t_start) / 3600)[:7] + " hours")

            loss_val = None
            with torch.set_grad_enabled(False):
                y_true, y_pred, mse, rmse, \
                    person, p_val, \
                    spearman, s_p_val, CI,\
                    loss_val = self.test(validation_generator, self.model)
                lst = ["epoch " + str(epo)] + list(map(float2str, [mse, rmse, person, p_val, spearman,
                                                                   s_p_val, CI]))
                valid_metric_record.append(lst)
                general_loss = str(loss_val.item())[:7]
                loss_val = float(loss_val.item())

                print(loss_val)
                if loss_val < min_loss_val:
                    model_max = copy.deepcopy(self.model)
                    print('Validation at Epoch ' + str(epo + 1) +
                          ' with loss:' + general_loss +
                          ', MSE: ' + str(mse)[:7] +
                          ' , Pearson Correlation: ' + str(person)[:7] +
                          ' with p-value: ' + str(p_val) +
                          ' Spearman Correlation: ' + str(spearman)[:7] +
                          ' with p_value: ' + str(s_p_val) +
                          ' , Concordance Index: ' + str(CI)[:7])

            earlier_stopping_counter += 1
            if min_loss_val > loss_val:
                print(loss_val)
                min_loss_val = loss_val
                earlier_stopping_counter = 0
            if earlier_stopping_counter >= earlier_stopping_num:
                break

        self.model = model_max

        with open(self.pkl_file, 'wb') as pck:
            pickle.dump(loss_history, pck)

        print('--- Training Finished ---')

    def predict(self, drug_data, rna_data, binding_data):
        drug_data = drug_data.reset_index(drop=True)
        rna_data = pd.DataFrame(rna_data).reset_index(drop=True)
        binding_data = pd.DataFrame(binding_data).reset_index(drop=True)
        print('predicting...')
        self.model.to(device)
        print('\tModel is on the device')
        info = data_process_loader(drug_data.index.values,
                                   drug_data.Label.values,
                                   drug_data, rna_data, binding_data)
        print('\tData process loader is ready')
        params = {'batch_size': 128,
                  'shuffle': False,
                  'num_workers': 8,
                  'drop_last': False
                  }  # ,
        generator = data.DataLoader(info, **params)
        print('\tGenerator constructed')

        print('\tPrediction started')
        y_label, y_pred, mse, rmse, person, p_val, spearman, s_p_val, CI, loss_val = \
            self.test(generator, self.model)
        print('\tReturning results')

        return y_label, y_pred, mse, rmse, person, p_val, spearman, s_p_val, CI

    def save_model(self):
        torch.save(self.model.state_dict(), self.modeldir + '/model.pt')

    def clean_model(self):
        import gc
        del self.model
        torch.cuda.empty_cache()
        gc.collect()

    def load_pretrained(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

        if self.device == 'cuda':
            state_dict = torch.load(path)
        else:
            state_dict = torch.load(path, map_location=torch.device('cpu'))

        if next(iter(state_dict))[:7] == 'module.':
            # the pretrained model is from data-parallel module
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:]  # remove `module.`
                new_state_dict[name] = v
            state_dict = new_state_dict

        self.model.load_state_dict(state_dict)


    def preprocess(self, rna_data, drug_data, binding_data, response_data, response_metric='AUC', use_map=True):
        args = self.args
        obj = DataEncoding(args.vocab_dir, args.cancer_id,
                           args.sample_id, args.target_id, args.drug_id)
        drug_col = 'DrugID'
        if use_map:
            drug_map = get_drug_map()
            to_drop = []
            for i in range(drug_data.shape[0]):
                drug_id = drug_data.loc[drug_data.index[i], drug_col]
                if drug_id in drug_map:
                    drug_data.loc[drug_data.index[i],
                                  drug_col] = drug_map[drug_id]
                else:
                    to_drop.append(i)
            if len(to_drop) > 0:
                drug_data = drug_data.drop(drug_data.index[to_drop], axis=0)

        drug_smiles = drug_data

        drugid2smile = dict(
            zip(drug_smiles[drug_col], drug_smiles['SMILES']))
        smile_encode = pd.Series(drug_smiles['SMILES'].unique()).apply(
            obj._drug2emb_encoder)
        uniq_smile_dict = dict(
            zip(drug_smiles['SMILES'].unique(), smile_encode))

        drug_data.drop(['SMILES'], inplace=True, axis=1)
        drug_data['smiles'] = [drugid2smile[i] for i in drug_data['DrugID']]
        drug_data['drug_encoding'] = [uniq_smile_dict[i]
                                      for i in drug_data['smiles']]
        drug_data = drug_data.reset_index()

        print(response_data.columns)
        response_data = response_data[['CancID', 'DrugID', response_metric]]
        drug_data = pd.merge(response_data, drug_data,
                             on='DrugID', how='inner')
        drug_data['Label'] = drug_data[response_metric]
        drug_data.drop(response_metric, axis=1, inplace=True)

        binding_data.columns = ['DrugID'] + list(binding_data.columns[1:])

        print('@@@@@@@@ BINDING DATA DRUGS @@@@@@@@')
        print(response_data['CancID'])


        binding_data_cols = [
            col for col in binding_data.columns if col not in drug_data.columns or col == 'DrugID']
        binding_data = binding_data[binding_data_cols]
        


        response_data = response_data[['CancID', 'DrugID', response_metric]]
        response_data.columns = ['CancID', 'DrugID', 'Label']
        drug_data.index = range(drug_data.shape[0])
        rna_data.index = range(rna_data.shape[0])
        binding_data.index = range(binding_data.shape[0])

        print('Preprocessing...!!!')
        print(np.shape(rna_data), np.shape(drug_data), np.shape(binding_data))
        return rna_data, drug_data, binding_data