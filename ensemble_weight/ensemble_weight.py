"""
@author: achm

Calculate the weight of the Final ensemble of strong and weak model via keras

"""

import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from keras.models import Sequential
from sklearn.preprocessing import StandardScaler
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.advanced_activations import PReLU
from keras.layers.convolutional import Convolution1D
from keras.models import Sequential
from keras.utils import np_utils
from keras.layers.embeddings import Embedding
import xgboost as xgb
import sys
import cPickle
import copy
import glob

# Load data
print("Load the training/test data using pandas")
training = pd.read_csv("../input/training.csv")
training['ensemble_weight'] = 1
training.drop('min_ANNmuon', axis=1, inplace=True)
training.drop('mass', axis=1, inplace=True)
training.drop('production', axis=1, inplace=True)
training.drop('signal', axis=1, inplace=True)

param_epoch = 300

for i in range(0,5):
    print "### %i ###" %i
    try:
        fh = open("./model/keras_%i_epoch_%i" %(i,param_epoch), "rb")
        deep_model = cPickle.load(fh)
        fh.close()
    except:
        scaler = StandardScaler()
        np.random.seed(6174)
        print "No prebuild model..."
        testing  = pd.read_csv("./input/testing_%i.csv" %i)
        testing['ensemble_weight'] = 0

        #scaler = StandardScaler()
        result = pd.concat([training, testing])
        y = result["ensemble_weight"]

        # Drop Unnesscary features
        result.drop('ensemble_weight', axis=1, inplace=True)
        result.drop('id', axis=1, inplace=True)

        deep_model = copy.deepcopy(Sequential())
        deep_model.add(Dense(result.shape[1], 512, init = "glorot_normal"))
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

        deep_model.fit(scaler.fit_transform(np.array(result)), np_utils.to_categorical(y),
         batch_size=256, nb_epoch=param_epoch, verbose=2, show_accuracy=True)

        # save model
        temp_file_name = "./model/keras_%i_epoch_%i" %(i,param_epoch)
        fh = open(temp_file_name, "wb")
        cPickle.dump(deep_model,fh)
        fh.close()

        # save scalar
        temp_file_name = "./model/keras_scalar_%i_epoch_%i" %(i,param_epoch)
        fh = open(temp_file_name, "wb")
        cPickle.dump(scaler,fh)
        fh.close()

    fh = open("./model/keras_scalar_%i_epoch_%i" %(i,param_epoch), "rb")
    scaler = cPickle.load(fh)
    fh.close()

    # Make Prediction
    testing_eval  = pd.read_csv("./input/testing_eval_%i.csv" %i)
    #################### FIX #########################
    ids = testing_eval['id']
    testing_eval.drop('id', axis=1, inplace=True)
    ##################################################
    ensemble_weight = deep_model.predict(scaler.transform(testing_eval), batch_size=256)[:, 1]

    # Generate ensemble weight
    with open('./output/ensemble_weight_%i_epoch_%i.csv' %(i,param_epoch), 'w') as f:
        f.write('id,weight\n')
        for ID, p in zip(ids, ensemble_weight):
            f.write('%s,%.8f\n' % (ID, p))

# Combine
print("Load ensemble weighting")
ensemble_weight_0 = pd.read_csv("./output/ensemble_weight_0_epoch_%i.csv" %param_epoch)
ensemble_weight_1 = pd.read_csv("./output/ensemble_weight_1_epoch_%i.csv" %param_epoch)
ensemble_weight_2 = pd.read_csv("./output/ensemble_weight_2_epoch_%i.csv" %param_epoch)
ensemble_weight_3 = pd.read_csv("./output/ensemble_weight_3_epoch_%i.csv" %param_epoch)
ensemble_weight_4 = pd.read_csv("./output/ensemble_weight_4_epoch_%i.csv" %param_epoch)
ensemble_weight = pd.concat([ensemble_weight_0, ensemble_weight_1, ensemble_weight_2, ensemble_weight_3, ensemble_weight_4])
ensemble_weight.to_csv("./output/ensemble_weight_epoch_%i.csv" %param_epoch, index=False)
