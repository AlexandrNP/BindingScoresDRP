import pandas as pd
import numpy as np
from sklearn.cluster import SpectralCoclustering
from sklearn.cluster import SpectralBiclustering
from sklearn.metrics import consensus_score
from matplotlib import pyplot as plt

SCORES_FILE = 'gausschem4_scores.tsv'
# SCORES_FILE = 'binding_affinity.tsv'


def impute_nan(scores_data, nan_modifier):
    max_val = scores_data.max().max() * nan_modifier
    return scores_data.fillna(max_val)


def biclustering(data, n_clusters=2):
    X = data.values

    model = SpectralBiclustering(
        n_clusters=n_clusters, random_state=42).fit(X)
    # score = consensus_score(
    #    model.biclusters_, (rows[:, row_idx], columns[:, col_idx]))

    fit_data = X[np.argsort(model.row_labels_)]
    fit_data = fit_data[:, np.argsort(model.column_labels_)]

    plt.matshow(fit_data, cmap=plt.cm.Blues)

    # plt.matshow(
    #   np.outer(np.sort(model.row_labels_) + 1,
    #            np.sort(model.column_labels_) + 1),
    #   cmap=plt.cm.Blues,
    # )

    # plt.title("Checkerboard structure of rearranged data")
    cax = plt.axes([0.80, 0.1, 0.075, 0.75])
    plt.colorbar(cax=cax)
    plt.show()


def coclustering(data, n_clusters=2):
    X = data.values
    model = SpectralCoclustering(
        n_clusters=n_clusters, random_state=42).fit(X) #42
    fit_data = X[np.argsort(model.row_labels_)]
    fit_data = fit_data[:, np.argsort(model.column_labels_)]
    plt.matshow(fit_data, cmap=plt.cm.Blues)

    cax = plt.axes([0.80, 0.1, 0.075, 0.75])
    plt.colorbar(cax=cax)
    plt.savefig(f'coclustering_{n_clusters}.pdf')
    plt.show()


def correlation_studies(data, n_clusters):
    correlation_matrix = data.corr()
    biclustering(correlation_matrix, n_clusters)
    # coclustering(correlation_matrix, n_clusters)

def get_moa():
    data = pd.read_csv('DrugLists/Comprehensive_Drug_List.txt', sep='\t')
    moa = data['moa(Broad)'].dropna()
    return moa

data = pd.read_csv(SCORES_FILE, sep='\t', index_col=0)
data = impute_nan(data, 0)
print(np.max(data.values))
data_norm = pd.DataFrame(data.values - np.max(data.values))
data_norm.index = data.index
data_norm.columns = data.columns
data = np.log(np.abs(data) + 1)

# print(data)

# vals = data[data < 0].values.flatten()

# plt.hist(vals, bins=1000)
# plt.show()

moa = get_moa()
#print(moa)
print(len(np.unique(moa)))
#biclustering(data, n_clusters=4)
#coclustering(data, n_clusters=3)
# correlation_studies(data, n_clusters=2)
# print(data)

data = pd.read_csv('binding_affinity.tsv', sep='\t')
data.values.isna()
missing_rate = sum(sum(data.isna().values))/(data.shape[0]*data.shape[1])
