"""
@author: achm

Finding best ensemble of strong models using grid search, based on evaluation sets

"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
import sys
import cPickle
import time
import copy
from sklearn.metrics import log_loss
import glob
import os
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

sys.path.append('../input')
import evaluation

sys.setrecursionlimit(100000000)

global_filter_out =  ['id', 'min_ANNmuon', 'production', 'mass']

train_table = {}
train_eval_table = {}
train_eval_score = {}

# Load training and train evaluation data and store it into dict
print("Load the training/training_eval data using pandas")
for i in range(0,5):
    print i
    train_table[i] = pd.read_csv("./input/training_%i.csv" %i)
    train_eval_table[i] = pd.read_csv("./input/training_eval_%i.csv" %i)
    train_eval_score[i] = 0

# Indicate which features to use
features = list(train_table[0].columns[1:-4])

# Load test data
print("Load test data using pandas")
test = pd.read_csv('../input/test.csv')

print time.localtime()
ks_cutoff = 0.09

for i in range(0,5):
    best_xs = []
    print "### %i ###" %i
    # read Keras model
    search_dir = "./model/final_keras_%i/new_" %i
    files = filter(os.path.isfile, glob.glob(search_dir + "*"))
    files.sort(key=lambda x: os.path.getmtime(x))

    print files
    fh = open(files[-1],"rb")
    keras_model = cPickle.load(fh)
    fh.close()

    scaler = StandardScaler()
    scaler.fit(np.array(train_table[i][features]))

    keras_pred = keras_model.predict(scaler.transform(np.array(train_eval_table[i][features])), batch_size=256)[:,1]
    keras_test = keras_model.predict(scaler.transform(np.array(test[features])), batch_size=256)[:,1]

    # read xgboost model
    search_dir = "./model/final_gbm_%i/new_" %i
    files = filter(os.path.isfile, glob.glob(search_dir + "*"))
    files.sort(key=lambda x: os.path.getmtime(x))

    print files
    fh = open(files[-1],"rb")
    gbm_model = cPickle.load(fh)
    fh.close()

    gbm_pred = gbm_model.predict(xgb.DMatrix(train_eval_table[i][features]))
    gbm_test = gbm_model.predict(xgb.DMatrix(test[features]))


    # Grid search to compute best score
    def multichoose(n,k):
        if k < 0 or n < 0: return "Error"
        if not k: return [[0]*n]
        if not n: return []
        if n == 1: return [[k]]
        return [[0]+val for val in multichoose(n-1,k)] + \
            [[val[0]+1]+val[1:] for val in multichoose(n,k-1)]

    n = 2
    k = 1000
    for xs in multichoose(n,k):
        #print xs
        preds = (xs[0]*keras_pred + xs[1]*gbm_pred)/float(k)
        score = evaluation.roc_auc_truncated(train_eval_table[i]['signal'], preds)
        if score>=train_eval_score[i]:
            train_eval_score[i] = score
            print score
            best_xs = xs

    print train_eval_score[i]
    print best_xs
    test["prediction_%i" %i] = (best_xs[0]*keras_test + best_xs[1]*gbm_test)/float(k)

    with open('./output/strong_submission_%i.csv' %i, 'w') as f:
        f.write('id,prediction\n')
        for ID, p in zip(test['id'], test["prediction_%i" %i]):
            f.write('%s,%.8f\n' % (ID, p))

    # Save best combination weight
    temp_file_name = "./output/best_xs_%i" %i
    fh = open(temp_file_name, "wb")
    cPickle.dump(best_xs,fh)
    fh.close()

    # Save best score
    temp_file_name = "./output/best_score_%i" %i
    fh = open(temp_file_name, "wb")
    cPickle.dump(train_eval_score[i],fh)
    fh.close()

ensemble_prediction = (test["prediction_0"] + test["prediction_1"] + test["prediction_2"] + test["prediction_3"] + test["prediction_4"])/5.
with open('./output/strong_ensemble_submission.csv', 'w') as f:
    f.write('id,prediction\n')
    for ID, p in zip(test['id'], ensemble_prediction):
        f.write('%s,%.8f\n' % (ID, p))
