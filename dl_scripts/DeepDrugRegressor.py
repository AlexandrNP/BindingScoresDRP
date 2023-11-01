import os
import copy
import time
import torch
import pickle
import pandas as pd
import numpy as np
import torch.nn.functional as F
from torch.utils import data
from torch import dropout, nn
from prettytable import PrettyTable
from torch.utils.data import SequentialSampler

from sklearn.metrics import mean_squared_error
from lifelines.utils import concordance_index
from scipy.stats import pearsonr, spearmanr

device = torch.device('cpu')
  

class BaseDataLoader(data.Dataset):
    binding_features = None

    def __init__(self, labels, cell_features, drug_features, binding_columns_idx=None):
        labels = pd.DataFrame(labels)
        cell_features = pd.DataFrame(cell_features)
        drug_features = pd.DataFrame(drug_features)
        self.labels = labels.reset_index()

        if binding_columns_idx is not None:
            self.binding_features = drug_features[binding_columns_idx]
            drug_features = np.delete(drug_features, binding_columns_idx)
        self.drug_features = drug_features.reset_index()
        self.cell_features = cell_features.reset_index()
        self.indices = range(len(labels))
        if 'index' in self.cell_features.columns:
            self.cell_features.drop(['index'], axis=1, inplace=True)
        if 'index' in self.drug_features.columns:
            self.drug_features.drop(['index'], axis=1, inplace=True)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        index = self.indices[index]
        sample_features_cell = np.array(
            self.cell_features.loc[index, :].values, dtype=float)
        sample_features_drug = np.array(
            self.drug_features.loc[index, :].values, dtype=float)
        y = self.labels.loc[index].values

        return sample_features_cell, sample_features_drug, y


class Regressor(nn.Sequential):
    def __init__(self, input_size, hidden_layer_sizes, dropout_rate):
        self.input_size = input_size
        self.hidden_layer_sizes = hidden_layer_sizes
        self.output_size = self.hidden_layer_sizes[-1]
        super(Regressor, self).__init__()

        self.dropout = nn.Dropout(dropout_rate)
        self.layers_num = len(self.hidden_layer_sizes)
        dimensions = [self.input_size] + self.hidden_layer_sizes
        self.predictor = nn.ModuleList(
            [nn.Linear(dimensions[i], dimensions[i + 1]) for i in range(self.layers_num)])

    def set_device(self, device):
        self.device = device

    def forward(self, values):
        # predict
        values = values.float().to(self.device)
        for i, layer in enumerate(self.predictor):
            values = self.dropout(values)
            if i < len(self.predictor) -1:
                values = F.relu(layer(values))
            else:
                values = layer(values)
            
        return values


class ConcatenationNetwork(nn.Sequential):
    def __init__(self, subnetworks, hidden_layer_sizes, dropout_rate):
        super(ConcatenationNetwork, self).__init__()
        self.subnetworks = subnetworks
        self.input_size = sum([subnet.output_size for subnet in subnetworks])
        self.dropout = nn.Dropout(dropout_rate)
        self.hidden_layer_sizes = hidden_layer_sizes
        self.layers_num = len(self.hidden_layer_sizes)
        dimensions = [self.input_size] + self.hidden_layer_sizes
        self.predictor = nn.ModuleList(
            [nn.Linear(dimensions[i], dimensions[i + 1]) for i in range(self.layers_num)])

    def set_device(self, device):
        self.device = device
        for i in range(len(self.subnetworks)):
            self.subnetworks[i].set_device(self.device)

    def forward(self, subnetwork_inputs):
        # Run previous steps
        encodings = []
        if len(subnetwork_inputs) != len(self.subnetworks):
            raise Exception(
                'Mismatch between number of subnetworks and number of grouped inputs')
        for subnet, input in zip(self.subnetworks, subnetwork_inputs):
            encodings.append(subnet(input))

        values = torch.cat(encodings, 1)
        for i, layer in enumerate(self.predictor):
            if i < self.layers_num:
                values = F.relu(self.dropout(layer(values)))
            else:
                values = F.relu(layer(values))
        return values


class DeepDrugRegressor:
    def __init__(self):
        self.device =  device
        self.model = None
        self.mode = 'fcnn'

    def score(self, datagenerator, model):
        y_label = []
        y_pred = []
        model.eval()
        for i, (v_gene, v_drug, label) in enumerate(datagenerator):
            if self.mode == 'module':
                scores = model([v_gene, v_drug])
            elif self.mode == 'fcnn':
                scores = model(torch.cat([v_gene, v_drug], 1))
            loss_fct = torch.nn.MSELoss()
            n = torch.squeeze(scores, 1)
            label = label[:, 1]
            loss = loss_fct(n, torch.autograd.Variable(torch.from_numpy(
                np.array(label)).float()).to(self.device))
            logits = torch.squeeze(scores).detach().cpu().numpy()
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

    def train(self, cell_line_features_train, drug_features_train, y_train, cell_line_features_val, drug_features_val, y_val, dropout_rate=0.1):
        if self.mode == 'module':
            model_cell_line = Regressor(input_size=np.shape(cell_line_features_train)[
                1], hidden_layer_sizes=[512, 512, 128, 64], dropout_rate=dropout_rate)  # [1024, 512, 256]
            model_drug = Regressor(input_size=np.shape(drug_features_train)[
                1], hidden_layer_sizes=[512, 512, 128, 64], dropout_rate=dropout_rate)  # [1021, 512, 256]
            # model_drug = Regressor(input_size=np.shape(drug_features_train)[1], hidden_layer_sizes=[1021, 512, 256], dropout_rate=dropout_rate)
            model_cell_line = model_cell_line.to(self.device)
            model_drug = model_drug.to(self.device)
            self.model = ConcatenationNetwork([model_cell_line, model_drug], hidden_layer_sizes=[
                128, 64, 1], dropout_rate=dropout_rate)
        elif self.mode == 'fcnn':
            train = np.concatenate(
                (cell_line_features_train, drug_features_train), axis=1)
            self.model = Regressor(input_size=np.shape(train)[1], hidden_layer_sizes=[
                                   4096, 2048, 1536, 1024, 768, 512, 256, 128, 64, 32, 1], dropout_rate=dropout_rate)  # 4096, 2048, 1536, 1024, 768, 512, 256, 128, 64, 32, 1

        self.model.set_device(self.device)

        learning_rate = 1e-4
        decay = 1e-8  # 1e-3
        batch_size = 1024
        train_epochs = 100
        self.model = self.model.to(self.device)
        opt = torch.optim.Adam(self.model.parameters(),
                               lr=learning_rate, weight_decay=decay)

        params = {'batch_size': batch_size,
                  'shuffle': True,
                  'num_workers': 0,
                  'drop_last': False}

        training_generator = data.DataLoader(BaseDataLoader(
            y_train, cell_line_features_train, drug_features_train), **params)
        validation_generator = data.DataLoader(BaseDataLoader(
            y_val, cell_line_features_val, drug_features_val), **params)

        max_MSE = 10000
        model_max = copy.deepcopy(self.model)

        valid_metric_record = []
        valid_metric_header = ['# epoch', "MSE", 'RMSE',
                               "Pearson Correlation", "with p-value",
                               'Spearman Correlation', "with p-value2",
                               "Concordance Index"]
        table = PrettyTable(valid_metric_header)

        def float2str(x):
            return '%0.4f' % x
        print('--- Go for Training ---')
        t_start = time.time()
        iteration_loss = 0
        loss_history = []

        for epo in range(train_epochs):
            for i, (v_d, v_p, label) in enumerate(training_generator):
                v_p = v_p.float().to(self.device)
                v_d = v_d.float().to(self.device)
                score = None
                if self.mode == 'module':
                    score = self.model([v_d, v_p])
                elif self.mode == 'fcnn':
                    score = self.model(torch.cat([v_d, v_p], 1))

                label = torch.autograd.Variable(torch.from_numpy(
                    np.array(label))).float().to(self.device)
                label = label[:, 1]

                loss_fct = torch.nn.MSELoss()
                n = torch.squeeze(score, 1).float()
                loss = loss_fct(n, label)
                loss_history.append(loss.item())
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

            with torch.set_grad_enabled(False):
                # regression: MSE, Pearson Correlation, with p-value, Concordance Index
                y_true, y_pred, mse, rmse, \
                    person, p_val, \
                    spearman, s_p_val, CI,\
                    loss_val = self.score(validation_generator, self.model)
                lst = ["epoch " + str(epo)] + list(map(float2str, [mse, rmse, person, p_val, spearman,
                                                                   s_p_val, CI]))
                valid_metric_record.append(lst)
                if mse < max_MSE:
                    model_max = copy.deepcopy(self.model)
                    max_MSE = mse
                    print('Validation at Epoch ' + str(epo + 1) +
                          ' with loss:' + str(loss_val.item())[:7] +
                          ', MSE: ' + str(mse)[:7] +
                          ' , Pearson Correlation: ' + str(person)[:7] +
                          ' with p-value: ' + str(p_val)[:7] +
                          ' Spearman Correlation: ' + str(spearman)[:7] +
                          ' with p_value: ' + str(s_p_val)[:7] +
                          ' , Concordance Index: ' + str(CI)[:7])
            table.add_row(lst)

        self.model = model_max
        print('--- Training Finished ---')

    def predict(self, rna_data, drug_data):
        print('predicting...')
        self.model.to(self.device)
        print('\tModel is on the device')
        info = BaseDataLoader(pd.DataFrame(range(np.shape(drug_data)[0])),
                              rna_data,
                              drug_data)
        print('\tData process loader is ready')
        params = {'batch_size': 16,
                  'shuffle': False,
                  'num_workers': 0,
                  'drop_last': False,
                  'sampler': SequentialSampler(info)}
        generator = data.DataLoader(info, **params)
        print('\tGenerator constructed')

        print('\tPrediction started')
        y_pred = []
        self.model.eval()
        for i, (v_cell, v_drug, label) in enumerate(generator):
            v = torch.cat([v_cell, v_drug], 1)
            scores = None
            if self.mode == 'module':
                scores = self.model([v_cell, v_drug])
            elif self.mode == 'fcnn':
                scores = self.model(torch.cat([v_cell, v_drug], 1))
            # scores = self.model(v)
            logits = torch.squeeze(scores).detach().cpu().numpy()
            y_pred = y_pred + logits.flatten().tolist()
        print('\tReturning results')

        return y_pred
