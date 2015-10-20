"""
@author: achm

Final ensemble of strong and weak model, weight based on ../ensemble_weight
"""

import numpy as np
import pandas as pd
import sys
sys.path.append('../input')
import evaluation
import cPickle
import copy
import glob
import os

print("Load test data using pandas")
test  = pd.read_csv("../input/test.csv")

################################################################################
# Load weak model submission
print("Load weak model")
ep = 1500

# Two final submission path
#search_dir = "../weak_model/output/weak_prediction_ep_%i_a1_0_a2_10" %ep
search_dir = "../weak_model/output/weak_prediction_ep_%i_a1_5_a2_5" %ep
temp_file = filter(os.path.isfile, glob.glob(search_dir + "*"))
temp_file.sort(key=lambda x: os.path.getmtime(x))
weak_model_sub = pd.read_csv(temp_file[-1])

# Double Check
if weak_model_sub.shape[0] != test.shape[0]:
    print "ERROR!"
    sys.exit(0)
if not all(weak_model_sub['id'] == test['id']):
    print "ERROR!"
    sys.exit(0)
if not all(weak_model_sub['prediction'] <= 1):
    print "ERROR!"
    sys.exit(0)
if not all(weak_model_sub['prediction'] >= 0):
    print "ERROR!"
    sys.exit(0)

################################################################################
# Load Semi-Strong model submission
print("Load Semi-strong model")
semi_strong_model_sub = pd.read_csv("../semi_strong_model/output/semi_strong_ensemble_submission.csv")

# Double Check
if semi_strong_model_sub.shape[0] != test.shape[0]:
    print "ERROR!"
    sys.exit(0)
if not all(semi_strong_model_sub['id'] == test['id']):
    print "ERROR!"
    sys.exit(0)
if not all(semi_strong_model_sub['prediction'] <= 1):
    print "ERROR!"
    sys.exit(0)
if not all(semi_strong_model_sub['prediction'] >= 0):
    print "ERROR!"
    sys.exit(0)

################################################################################
# Load strong model submission
print("Load strong model")
strong_model_sub = pd.read_csv("../strong_model/output/strong_ensemble_submission.csv")

# Double Check
if strong_model_sub.shape[0] != test.shape[0]:
    print "ERROR!"
    sys.exit(0)
if not all(strong_model_sub['id'] == test['id']):
    print "ERROR!"
    sys.exit(0)
if not all(strong_model_sub['prediction'] <= 1):
    print "ERROR!"
    sys.exit(0)
if not all(strong_model_sub['prediction'] >= 0):
    print "ERROR!"
    sys.exit(0)


################################################################################
# Load ensemble weighting
print("Load ensemble weighting")
temp_epoch=300
ensemble = pd.read_csv("../ensemble_weight/output/ensemble_weight_epoch_%i.csv" %temp_epoch)

# Double Check
if ensemble.shape[0] != test.shape[0]:
    print "ERROR!"
    sys.exit(0)
if not all(ensemble['id'] == test['id']):
    print "ERROR!"
    sys.exit(0)

################################################################################
# Setting
option = 7
rho = 0

def sigmoid_norm(x, norm):
    return 1/(1+np.exp(-(x*(2*norm)-norm)))

def sigmoid_shift_slope(x, target, slope):
    return 1/(1+np.exp(-(x-target)*slope))

# Ensemble Signal processing
if option==0: # simple weighting
    ensemble['strong_pred'] = np.array(strong_model_sub['prediction'])
    ensemble['semi_strong_pred'] = np.array(semi_strong_model_sub['prediction'])
    ensemble['weak_pred'] = np.array(weak_model_sub['prediction'])
    ensemble['prediction'] =  ((1-rho)*ensemble['weight']*ensemble['strong_pred'] +
                                rho*ensemble['weight']*ensemble['semi_strong_pred'] +
                                (1-ensemble['weight'])*ensemble['weak_pred'])
elif option ==1: #absulute cutoff
    threshold = 0.15
    ensemble['strong_pred'] = np.array(strong_model_sub['prediction'])
    ensemble['semi_strong_pred'] = np.array(semi_strong_model_sub['prediction'])
    ensemble['weak_pred'] = np.array(weak_model_sub['prediction'])
    ensemble['prediction'] = ((1.0*(ensemble['weight']>threshold)) * ensemble['strong_pred'] * (1-rho)+
                                (1.0*(ensemble['weight']>threshold)) * ensemble['semi_strong_pred'] * (rho) +
                                (1.0*(ensemble['weight']<=threshold)) * ensemble['weak_pred'])
elif option ==2: #absulute cutoff stacking
    threshold = 0.15
    ensemble['strong_pred'] = np.array(strong_model_sub['prediction'])
    ensemble['semi_strong_pred'] = np.array(semi_strong_model_sub['prediction'])
    ensemble['weak_pred'] = np.array(weak_model_sub['prediction'])
    ensemble['prediction'] = ((1.0*(ensemble['weight']>threshold)) * (ensemble['strong_pred']*(1-rho) + ensemble['semi_strong_pred']*rho + ensemble['weak_pred'])/2. +
                             (1.0*(ensemble['weight']<=threshold)) * ensemble['weak_pred'] )
elif option ==3: # cutoff with weighting
    threshold = 0.15
    ensemble['strong_pred'] = np.array(strong_model_sub['prediction'])
    ensemble['semi_strong_pred'] = np.array(semi_strong_model_sub['prediction'])
    ensemble['weak_pred'] = np.array(weak_model_sub['prediction'])
    ensemble['prediction'] = ( (1.0*(ensemble['weight']>threshold)) * (ensemble['weight']*ensemble['strong_pred']*(1-rho) +
                                ensemble['semi_strong_pred']*rho + (1-ensemble['weight'])*ensemble['weak_pred']) +
                                (1.0*(ensemble['weight']<=threshold)) * ensemble['weak_pred'] )
elif option ==4: # sigmoid norm
    s_norm = 10
    ensemble['strong_pred'] = np.array(strong_model_sub['prediction'])
    ensemble['semi_strong_pred'] = np.array(semi_strong_model_sub['prediction'])
    ensemble['weak_pred'] = np.array(weak_model_sub['prediction'])
    ensemble['prediction'] = ( sigmoid_norm(ensemble['weight'], norm=s_norm)*(ensemble['strong_pred']*(1-rho) + ensemble['semi_strong_pred']*rho) +
                                (1-sigmoid_norm(ensemble['weight'], norm=s_norm))*ensemble['weak_pred'] )
elif option ==5: # Tanh
    ensemble['strong_pred'] = np.array(strong_model_sub['prediction'])
    ensemble['semi_strong_pred'] = np.array(semi_strong_model_sub['prediction'])
    ensemble['weak_pred'] = np.array(weak_model_sub['prediction'])
    ensemble['prediction'] = ( np.tanh(ensemble['weight'])*(ensemble['strong_pred']*(1-rho) + ensemble['semi_strong_pred']*rho) +
                                (1-np.tanh(ensemble['weight']))*ensemble['weak_pred'] )
elif option ==6: # cutoff with Sigmoid
    threshold = 0.15
    s_norm = 10
    ensemble['strong_pred'] = np.array(strong_model_sub['prediction'])
    ensemble['semi_strong_pred'] = np.array(semi_strong_model_sub['prediction'])
    ensemble['weak_pred'] = np.array(weak_model_sub['prediction'])
    ensemble['prediction'] = ( (1.0*(ensemble['weight']>threshold)) *
                                (sigmoid(ensemble['weight'], norm=s_norm)*(ensemble['strong_pred']*(1-rho) + ensemble['semi_strong_pred']*rho) +
                                (1-sigmoid(ensemble['weight'], norm=s_norm))*ensemble['weak_pred']) +
                                (1.0*ensemble['weight']<=threshold) * ensemble['weak_pred'] )
elif option ==7: # Shifted Sigmoid with slope control
    target = 0.071
    slope = 100
    ensemble['strong_pred'] = np.array(strong_model_sub['prediction'])
    ensemble['semi_strong_pred'] = np.array(semi_strong_model_sub['prediction'])
    ensemble['weak_pred'] = np.array(weak_model_sub['prediction'])
    ensemble['prediction'] = ( sigmoid_shift_slope(ensemble['weight'], target, slope) * (ensemble['strong_pred']*(1-rho) + ensemble['semi_strong_pred']*rho) +
                                (1-sigmoid_shift_slope(ensemble['weight'], target, slope)) * ensemble['weak_pred'] )

if not all(ensemble['prediction'] <= 1):
    print "ERROR!"
    sys.exit(0)
if not all(ensemble['prediction'] >= 0):
    print "ERROR!"
    sys.exit(0)

# Write final output
print "Generate submission"
if option ==4:
    with open('./output/ensemblesw_ep_%i_option_%i_norm_%i.csv' %(ep, option, s_norm), 'w') as f:
        f.write('id,prediction\n')
        for ID, p in zip(ensemble['id'], ensemble['prediction']):
            f.write('%s,%.8f\n' % (ID, p))
elif option ==6:
    with open('./output/ensemblesw_ep_%i_option_%i_tshold_%i_norm_%i.csv' %(ep, option, threshold*1000, s_norm), 'w') as f:
        f.write('id,prediction\n')
        for ID, p in zip(ensemble['id'], ensemble['prediction']):
            f.write('%s,%.8f\n' % (ID, p))
elif option ==7:
    with open('./output/ensemblesw_ep_%i_option_%i_ep2_%i_param_%i_%i.csv' %(ep, option, temp_epoch, target*1000, slope), 'w') as f:
        f.write('id,prediction\n')
        for ID, p in zip(ensemble['id'], ensemble['prediction']):
            f.write('%s,%.8f\n' % (ID, p))
else:
    with open('./output/ensemblesw_ep_%i_option_%i_tshold_%i_ep2_%i.csv' %(ep, option, threshold*1000, temp_epoch), 'w') as f:
        f.write('id,prediction\n')
        for ID, p in zip(ensemble['id'], ensemble['prediction']):
            f.write('%s,%.8f\n' % (ID, p))
