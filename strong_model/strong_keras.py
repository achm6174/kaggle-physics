"""
@author: achm

Strong model based on Deep Learning (Keras)

Takes serval days to optimize the CV score

"""

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.advanced_activations import PReLU
from keras.layers.convolutional import Convolution1D
from keras.models import Sequential
from keras.utils import np_utils
from keras.layers.embeddings import Embedding
from sklearn.preprocessing import StandardScaler
import sys
import cPickle
import time
import copy
import glob
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler
import os

sys.path.append('../input')
import evaluation

sys.setrecursionlimit(100000000)

train_table = {}
train_eval_table = {}
train_eval_score = {}
train_model = {}

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
        fh = open("./model/final_keras_%i/max_score_%i" %(i,i), "rb")
        train_eval_score[i] = cPickle.load(fh)
        fh.close()
except Exception as e:
    print "pre-build model not exist"

max_score = 0
ks_cutoff = 0.09
init_seed = int(time.time())

try:
    for i in range(0,5):
        search_dir = "./model/final_keras_%i/new_" %i
        temp_file = filter(os.path.isfile, glob.glob(search_dir + "*"))
        temp_file.sort(key=lambda x: os.path.getmtime(x))
        print temp_file[-1]
        fh = open(temp_file[-1], "rb")
        train_model[i] = cPickle.load(fh)
        fh.close()
except Exception as e:
    #print e
    print "No prebuild model..."
    deep_model = copy.deepcopy(Sequential())
    deep_model.add(Dense((train_table[i][features]).shape[1], 512, init = "glorot_normal"))
    deep_model.add(Activation('tanh'))
    deep_model.add(Dropout(0.5))

    deep_model.add(Dense(512, 256, init = "glorot_normal"))
    deep_model.add(Activation('relu'))
    deep_model.add(Dropout(0.4))

    deep_model.add(Dense(256, 128, init = "glorot_normal"))
    deep_model.add(Activation('tanh'))
    deep_model.add(Dropout(0.3))

    deep_model.add(Dense(128, 64, init = "glorot_normal"))
    deep_model.add(Activation('relu'))
    deep_model.add(Dropout(0.2))

    deep_model.add(Dense(64, 2, init = "glorot_normal"))
    deep_model.add(Activation('softmax'))
    deep_model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    # Keras model #
    print("Create model...")
    for i in range(0,5):
        print i
        train_model[i] = copy.deepcopy(deep_model)

print "Start training..."
while True:
    print "####"
    for i in range(0,5):
        print i
        np.random.seed(init_seed)
        scaler = StandardScaler()
        # Build model
        temp_model = copy.deepcopy(train_model[i])
        temp_model.fit(scaler.fit_transform(np.array(train_table[i][features])), np_utils.to_categorical(train_table[i]["signal"]),
         batch_size=256, nb_epoch=1, verbose=2, show_accuracy=True)

        # Prediction
        preds = temp_model.predict(scaler.transform(np.array(train_eval_table[i][features])), batch_size=256)[:,1]

        score = evaluation.roc_auc_truncated(train_eval_table[i]['signal'], preds)
        if score>=train_eval_score[i]:
            print score

            # Update score
            train_eval_score[i] = score

            # Save model
            temp_file_name = "./model/final_keras_%i/new_keras_s%i_sc%i" %(i, init_seed, score*100000)
            fh = open(temp_file_name, "wb")
            cPickle.dump(temp_model, fh)
            fh.close()

            # Save score
            temp_file_name = "./model/final_keras_%i/max_score_%i" %(i, i)
            fh = open(temp_file_name, "wb")
            cPickle.dump(train_eval_score[i],fh)
            fh.close()

            #Update model
            train_model[i] = copy.deepcopy(temp_model)

    init_seed = np.random.random_integers(1,2**30)
