import os
import pickle
import pandas as pd
from sklearn.metrics import r2_score



if __name__ == "__main__":

    data_dir = 'Results/DeepTTC_drug_blind_mixed_sampling_mse_reg_drug'
    for root, dirs, files in os.walk(data_dir):
        current_directory = os.path.split(root)[-1]
        for file_name in files:
            if 'ctrp' not in file_name or 'test' not in file_name or 'pickle' not in file_name:
                continue
            data = pickle.load(open(os.path.join(data_dir, file_name), 'rb'))[0]
            print(data[0])
            break