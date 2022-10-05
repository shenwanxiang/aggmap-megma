import warnings, os
warnings.filterwarnings("ignore")

from copy import copy
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import confusion_matrix, precision_recall_curve, accuracy_score
from sklearn.metrics import roc_auc_score, matthews_corrcoef, precision_score, recall_score, f1_score


import seaborn as sns
import matplotlib.pyplot as plt
from joblib import dump, load

from aggmap import AggMap, loadmap
from aggmap import AggMapNet as AggModel

from aggmap.AggMapNet import load_model, save_model
from aggmap import show


np.random.seed(666) #just for reaptable results

import os


save_dir = './result_Met2Img_AggMapNet'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
gpuid = 0


def score(dfr):
    y_true = dfr.y_true
    y_score = dfr.y_score
    y_pred = dfr.y_pred

    '''
    the metrics are taken from orignal paper:
    Meta-Signer: Metagenomic Signature Identifier based on Rank Aggregation of Features
    https://github.com/YDaiLab/Meta-Signer/blob/bd6a1cd98d1035f848ecb6e53d9ee67a85871db2/src/utils/metasigner_io.py#L34
    '''
    auc = roc_auc_score(y_true, y_score, average='weighted')        
    mcc = matthews_corrcoef(y_true, y_pred)
    pres = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    acc = accuracy_score(y_true, y_pred)
    
    print('roc-auc: %.3f, mcc: %.3f, pres: %.3f, recall: %.3f, f1: %.3f' % (auc, mcc, pres, recall, f1))

    return acc, auc, mcc, pres, recall, f1



items = ["Met2Img_Cirrhosis_fillup-spb-gray",
        "Met2Img_Cirrhosis_fillup-spb-jet",
        "Met2Img_IBD_fillup-spb-gray",
         "Met2Img_IBD_fillup-spb-jet",
         "Met2Img_Obesity_fillup-spb-gray",
         "Met2Img_Obesity_fillup-spb-jet",
         "Met2Img_T2D_fillup-spb-gray",
         "Met2Img_T2D_fillup-spb-jet",
         "Met2Img_CRC_fillup-spb-gray", 
         "Met2Img_CRC_fillup-spb-jet",
        ]

for item in items:
    data_path = './Met2Img_AggMapNet_data/%s.pkl' % item 
    df = pd.read_pickle(data_path)

    X = np.stack(df.X.tolist())
    Y = pd.get_dummies(df.label).values
    dfy = df.label

    info = {'dataset': df.dataset.iloc[0], 'X_size' : X.shape, 'Met2Img': df.Met2Img.iloc[0]}
    pd.Series(info).to_csv(os.path.join(save_dir, 'info_%s.csv' % item))

    outer_fold = 10
    repeat_seeds = [8, 16, 32, 64, 128, 256, 1024, 2048, 4096, 8192] #10 repeats random seeds 8, 16, 32, 64, 128

    each_fold_results = []
    run_all_res = []

    for i, repeat_seed in enumerate(repeat_seeds): 
        outer = StratifiedKFold(n_splits = outer_fold, shuffle = True, random_state = repeat_seed)
        outer_idx = outer.split(range(len(dfy)), dfy)
        run_one_res = []
        for j, idx in enumerate(outer_idx):
            fold_num = "fold_%s" % str(j).zfill(2) 
            print('#'*50 + ' repeat_seed: %s; %s ' % (repeat_seed, fold_num) + '#'*50 )

            train_idx, test_idx = idx

            testY = Y[test_idx]
            testX = X[test_idx]

            trainX = X[train_idx]
            trainY = Y[train_idx]

            print("\n input train and test X shape is %s, %s " % (trainX.shape,  testX.shape))

            clf = AggModel.MultiClassEstimator(epochs = 50, 
                                               verbose = 0, gpuid=gpuid) #
            clf.fit(trainX, trainY)  #, 

            pred_proba = clf.predict_proba(testX)
            y_true = testY[:,1] 
            y_score = pred_proba[:,1]
            y_pred = np.argmax(pred_proba, axis=1)

            dfr = pd.DataFrame([y_true, y_score, y_pred]).T
            dfr.columns = ['y_true', 'y_score', 'y_pred']
            dfr.index = dfy.iloc[test_idx].index

            acc, auc, mcc, pres, recall, f1  = score(dfr)
            run_one_res.append(dfr)
            ts = pd.Series([acc, auc, mcc, pres, recall, f1, i, repeat_seed]).round(3)
            ts.index = ['acc','auc', 'mcc', 'pres', 'recall', 'f1', 'i', 'repeat_seed']

            print(ts.to_dict())
            each_fold_results.append(ts.to_dict())
        run_all_res.append(pd.concat(run_one_res))

    pd.DataFrame(each_fold_results).groupby('repeat_seed').mean().mean()
    pd.DataFrame(each_fold_results).to_csv(os.path.join(save_dir, '%s.csv' % item))

