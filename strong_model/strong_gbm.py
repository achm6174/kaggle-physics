"""
@author: achm

Strong model based on Gradient Boosting (xgboost) with hyperparameters optimization

"""

import numpy as np
import pandas as pd
import xgboost as xgb
import sys
import cPickle
import time
import copy
from sklearn.metrics import log_loss

sys.path.append('../input')
import evaluation

sys.setrecursionlimit(100000000)

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

print time.localtime()

# Load max_score
try:
    for i in range(0,5):
        fh = open("./model/final_gbm_%i/max_score_%i" %(i,i), "rb")
        train_eval_score[i] = cPickle.load(fh)
        fh.close()
except Exception as e:
    print "pre-build model not exist"

max_score = 0
init_seed = 6174
max_n_tree = 40000

#while True:
print "####"
for temp_max_depth in [4,5,6,7]:
    for temp_child_weight in [1,5,10,50,100]:
        for i in range(0,5):
            print i
            np.random.seed(init_seed)

            params = {"objective": "binary:logistic",
                      "eval_metric": "logloss",
                      "eta": 0.005,
                      "max_depth": temp_max_depth,
                      "min_child_weight": temp_child_weight,
                      "silent": 1,
                      "subsample": 0.7,
                      "colsample_bytree": 0.7,
                      "max_delta_step": 0.7,
                      "seed": 6174}

            dtrain = xgb.DMatrix(train_table[i][features], label = train_table[i]["signal"])

            # Run CV
            gbm = xgb.cv(params, dtrain , max_n_tree, nfold= 5)

            #print gbm
            cv_test_score = [eval(x[(x.find(':')+1):x.find('\t',10)]) for x in gbm]
            #print cv_test_score
            cv_best_ntree = cv_test_score.index(min(cv_test_score))
            #print cv_best_ntree

            # train model
            gbm = xgb.train(params, dtrain, cv_best_ntree+1)

            # Prediction
            preds = gbm.predict(xgb.DMatrix(train_eval_table[i][features]))

            score = evaluation.roc_auc_truncated(train_eval_table[i]['signal'], preds)
            print score

            if score>=train_eval_score[i]:
                train_eval_score[i] = score
                temp_file_name = "./model/final_gbm_%i/new_xgb_s%i_sc%i_tree%i" %(i, init_seed, score*100000, cv_best_ntree+1)

                # Save model
                fh = open(temp_file_name, "wb")
                cPickle.dump(gbm, fh)
                fh.close()

                # Save score
                temp_file_name = "./model/final_gbm_%i/max_score_%i" %(i, i)
                fh = open(temp_file_name, "wb")
                cPickle.dump(train_eval_score[i],fh)
                fh.close()

                # Save params
                temp_file_name = "./model/final_gbm_%i/params_%i" %(i, i)
                fh = open(temp_file_name, "wb")
                cPickle.dump(params,fh)
                fh.close()
