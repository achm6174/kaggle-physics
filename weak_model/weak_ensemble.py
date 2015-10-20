"""
@author: achm

Stacking of two models (1,2).
1 is the ensemble of keras, randomforest, xgboost, hep_ml.
2 is a keras model with more neuron, deeper structure, more epoch.

"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.advanced_activations import PReLU
from keras.layers.convolutional import Convolution1D
from keras.models import Sequential
from keras.utils import np_utils
from keras.layers.embeddings import Embedding
from hep_ml.losses import BinFlatnessLossFunction
from hep_ml.gradientboosting import UGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
import sys
import cPickle
import copy
import sys
sys.setrecursionlimit(100000000)

###################################### 1 ######################################################################
###################################### Build Keras model ######################################################
def get_training_data():
    filter_out = ['id', 'min_ANNmuon', 'production', 'mass', 'signal', 'SPDhits', 'IP', 'IPSig', 'isolationc']
    f = open('../input/training.csv')
    data = []
    y = []
    ids = []
    for i, l in enumerate(f):
        if i == 0:
            labels = l.rstrip().split(',')
            label_indices = dict((l, i) for i, l in enumerate(labels))
            continue

        values = l.rstrip().split(',')
        filtered = []
        for v, l in zip(values, labels):
            if l not in filter_out:
                filtered.append(float(v))

        label = values[label_indices['signal']]
        ID = values[0]

        data.append(filtered)
        y.append(float(label))
        ids.append(ID)
    return ids, np.array(data), np.array(y)


def get_test_data():
    filter_out = ['id', 'min_ANNmuon', 'production', 'mass', 'signal', 'SPDhits', 'IP', 'IPSig', 'isolationc']
    f = open('../input/test.csv')
    data = []
    ids = []
    for i, l in enumerate(f):
        if i == 0:
            labels = l.rstrip().split(',')
            continue

        values = l.rstrip().split(',')
        filtered = []
        for v, l in zip(values, labels):
            if l not in filter_out:
                filtered.append(float(v))

        ID = values[0]
        data.append(filtered)
        ids.append(ID)
    return ids, np.array(data)


def preprocess_data(X, scaler=None):
    if not scaler:
        scaler = StandardScaler()
        scaler.fit(X)
    X = scaler.transform(X)
    return X, scaler

# get training data
print("Load training data using pandas")
ids, X, y = get_training_data()

# shuffle the data
np.random.seed(671)
np.random.shuffle(X)
np.random.seed(671)
np.random.shuffle(y)

# preprocess the data
X, scaler = preprocess_data(X)
y = np_utils.to_categorical(y)

# split into training / evaluation data
nb_train_sample = int(len(y) * 0.78)
X_train = X[:nb_train_sample]
X_eval = X[nb_train_sample:]
y_train = y[:nb_train_sample]
y_eval = y[nb_train_sample:]

print("Keras")
try:
    # Load prebuild model
    temp_file_name = "./model/1/model_keras"
    fh = open(temp_file_name, "rb")
    model = cPickle.load(fh)
    fh.close()
except:
    print "Prebuild model cannot be found"
    # deep pyramidal MLP, narrowing with depth
    model = Sequential()
    model.add(Dropout(0.13))
    model.add(Dense(X_train.shape[1], 75))
    model.add(PReLU((75,)))

    model.add(Dropout(0.11))
    model.add(Dense(75, 50))
    model.add(PReLU((50,)))

    model.add(Dropout(0.09))
    model.add(Dense(50, 30))
    model.add(PReLU((30,)))

    model.add(Dropout(0.07))
    model.add(Dense(30, 25))
    model.add(PReLU((25,)))

    model.add(Dense(25, 2))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    # train model
    model.fit(X_train, y_train, batch_size=128, nb_epoch=150, validation_data=(X_eval, y_eval), verbose=2, show_accuracy=True)
    # Save model
    temp_file_name = "./model/1/model_keras"
    fh = open(temp_file_name, "wb")
    cPickle.dump(model, fh)
    fh.close()

# generate submission
print("Load test data using pandas")
ids, X = get_test_data()
X, scaler = preprocess_data(X, scaler)
predskeras = model.predict(X, batch_size=256)[:, 1]

################ randomforest, xgboost, UGradientBoosting ###################
print("Load training/test data using pandas")
train = pd.read_csv("../input/training.csv")
test  = pd.read_csv("../input/test.csv")
features = list(train.columns[1:-5])

################### UGradientBoosting ##########################
print("UGradientBoostingClassifier")
try:
    # Load prebuild model
    temp_file_name = "./model/1/model_ugb"
    fh = open(temp_file_name, "rb")
    clf = cPickle.load(fh)
    fh.close()
except:
    print "Prebuild model cannot be found"
    loss = BinFlatnessLossFunction(['mass'], n_bins=15, uniform_label=0)
    clf = UGradientBoostingClassifier(loss=loss, n_estimators=150, subsample=0.1,
                                      max_depth=7, min_samples_leaf=10,
                                      learning_rate=0.1, train_features=features, random_state=11)
    clf.fit(train[features + ['mass']], train['signal'])

    # Save model
    temp_file_name = "./model/1/model_ugb"
    fh = open(temp_file_name, "wb")
    cPickle.dump(clf, fh)
    fh.close()
fb_preds = clf.predict_proba(test[features])[:,1]
################## Random Forest ################
print("Random Forest model")
try:
    # Load prebuild model
    temp_file_name = "./model/1/model_rf"
    fh = open(temp_file_name, "rb")
    rf = cPickle.load(fh)
    fh.close()
except:
    print "Prebuild model cannot be found"
    rf = RandomForestClassifier(n_estimators=250, n_jobs=-1, criterion="entropy", random_state=1)
    rf.fit(train[features], train["signal"])

    # Save model
    temp_file_name = "./model/1/model_rf"
    fh = open(temp_file_name, "wb")
    cPickle.dump(rf, fh)
    fh.close()
rf_preds = rf.predict_proba(test[features])[:,1]
################ Xgbosot ###########################
print("XGBoost model")
try:
    # Load prebuild model
    temp_file_name = "./model/1/model_gbm"
    fh = open(temp_file_name, "rb")
    gbm = cPickle.load(fh)
    fh.close()
except:
    print "Prebuild model cannot be found"
    params = {"objective": "binary:logistic",
              "eta": 0.2,
              "max_depth": 7,
              "min_child_weight": 1,
              "silent": 1,
              "colsample_bytree": 0.7,
              "seed": 1}
    num_trees=450
    gbm = xgb.train(params, xgb.DMatrix(train[features], train["signal"]), num_trees)
    # Save model
    temp_file_name = "./model/1/model_gbm"
    fh = open(temp_file_name, "wb")
    cPickle.dump(gbm, fh)
    fh.close()
gbm_preds = gbm.predict(xgb.DMatrix(test[features]))
# Ensemble
ensemble_1_weight = [0.24, 0.3, 0.26, 0.2]

test_probs = ((ensemble_1_weight[0]* rf_preds) +
              (ensemble_1_weight[1]* gbm_preds) +
              (ensemble_1_weight[2]* predskeras) +
              (ensemble_1_weight[3]* fb_preds))

# Save ensemble weight
temp_file_name = "./model/1/ensemble_1_weight"
fh = open(temp_file_name, "wb")
cPickle.dump(ensemble_1_weight, fh)
fh.close()
###################################### 2 ######################################################################
###################################### Build Keras model ######################################################
global_filter_out =  ['id', 'min_ANNmuon', 'production', 'mass', 'signal', 'SPDhits']

def get_training_data():
    filter_out =global_filter_out
    f = open('../input/training.csv')
    data = []
    y = []
    ids = []
    for i, l in enumerate(f):
        if i == 0:
            labels = l.rstrip().split(',')
            label_indices = dict((l, i) for i, l in enumerate(labels))
            continue

        values = l.rstrip().split(',')
        filtered = []
        for v, l in zip(values, labels):
            if l not in filter_out:
                filtered.append(float(v))

        label = values[label_indices['signal']]
        ID = values[0]

        data.append(filtered)
        y.append(float(label))
        ids.append(ID)
    return ids, np.array(data), np.array(y)

def get_test_data():
    filter_out = global_filter_out
    f = open('../input/test.csv')
    data = []
    ids = []
    for i, l in enumerate(f):
        if i == 0:
            labels = l.rstrip().split(',')
            continue

        values = l.rstrip().split(',')
        filtered = []
        for v, l in zip(values, labels):
            if l not in filter_out:
                filtered.append(float(v))

        ID = values[0]
        data.append(filtered)
        ids.append(ID)
    return ids, np.array(data)


def preprocess_data(X, scaler=None):
    if not scaler:
        scaler = StandardScaler()
        scaler.fit(X)
    X = scaler.transform(X)
    return X, scaler

# # get training data
ids, X, y = get_training_data()

# shuffle the data
np.random.seed(6174)
np.random.shuffle(X)
np.random.seed(6174)
np.random.shuffle(y)

# preprocess the data
X, scaler = preprocess_data(X)
y = np_utils.to_categorical(y)

# split into training / evaluation data
nb_train_sample = int(len(y) * 0.97)
X_train = X[:nb_train_sample]
X_eval = X[nb_train_sample:]
y_train = y[:nb_train_sample]
y_eval = y[nb_train_sample:]

init_seed = 6174
init_nb_epoch = 1500
np.random.seed(init_seed)
try:
    # Load prebuild model
    temp_file_name = "./model/2/model_keras_s%i_ep%i" %(init_seed, init_nb_epoch)
    fh = open(temp_file_name, "rb")
    model = cPickle.load(fh)
    fh.close()
except:
    model = Sequential()
    model.add(Dense(X_train.shape[1], 512, init = "glorot_normal"))
    model.add(Activation('tanh'))
    model.add(Dropout(0.5))

    model.add(Dense(512, 256, init = "glorot_normal"))
    model.add(Activation('relu'))
    model.add(Dropout(0.4))

    model.add(Dense(256, 128, init = "glorot_normal"))
    model.add(Activation('tanh'))
    model.add(Dropout(0.3))

    model.add(Dense(128, 64, init = "glorot_normal"))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    model.add(Dense(64, 2, init = "glorot_normal"))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    # train model
    model.fit(X_train, y_train, batch_size=256, nb_epoch=init_nb_epoch, validation_data=(X_eval, y_eval), verbose=2, show_accuracy=True)

    # Save model
    temp_file_name = "./model/2/model_keras_s%i_ep%i" %(init_seed, init_nb_epoch)
    fh = open(temp_file_name, "wb")
    cPickle.dump(model, fh)
    fh.close()

# Prediction
ids, X_test = get_test_data()
X_test, scaler = preprocess_data(X_test, scaler)
preds = model.predict(X_test, batch_size=256)[:, 1]

# 1, 2 Stacking prediction
preds = (preds + test_probs)/2.

# Write output
with open('./output/weak_prediction_ep_%i_a1_5_a2_5.csv' %init_nb_epoch, 'w') as f:
    f.write('id,prediction\n')
    for ID, p in zip(ids, preds):
        f.write('%s,%.8f\n' % (ID, p))

# a1=0
preds = (test_probs)

# Write output
with open('./output/weak_prediction_ep_%i_a1_0_a2_10.csv' %init_nb_epoch, 'w') as f:
    f.write('id,prediction\n')
    for ID, p in zip(ids, preds):
        f.write('%s,%.8f\n' % (ID, p))
