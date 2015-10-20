## Summary
This is my model framework in Flavours of Physics: Finding t -> µµµ https://www.kaggle.com/c/flavours-of-physics.

The model gives 16/673 at public and private leaderboard with Weighted AUC = 0.991841. Without any usage of the following, which are prohibited in the competition:
* Use of agreement data, correlation data
* Local evaluation of agreement test and correlation test
* Reconstructuion of mass feature using equation from Special Relativity 

## Instruction

#### Download Data
* download `sample_submission.csv`, `training.csv`, `test.csv`, and put into folder `./input`.

#### Weak model
* run `python ./weak_model/weak_ensemble.py` to generate weak model. This will take a few hours.

#### Semi-Strong model
* run `python ./semi_strong_model/split_data.py` to split data into 5 folds.
* run `python ./semi_strong_model/semi_strong_gbm.py` to generate xgboost model. This will take a few days in optimization.
* run `python ./semi_strong_model/semi_strong_keras.py` to generate keras model. This will take a few days in optimization.
* Finally run `python ./semi_strong_model/semi_strong_ensemble.py` to generate an ensemble model. This will take an hour.

#### Strong model
* run `python ./strong_model/split_data.py` to split data into 5 folds.
* run `python ./strong_model/strong_gbm.py` to generate xgboost model. This will take a few days in optimization.
* run `python ./strong_model/strong_keras.py` to generate keras model. This will take a few days in optimization.
* Finally run `python ./strong_model/strong_ensemble.py` to generate an ensemble model. This will take an hour.

#### Ensemble weight
* run `python ./ensemble_weight/split_data.py` to split data into 5 folds.
* run `python ./ensemble_weight/ensemble_weight.py` to generate ensemble weight. This will take a few hours.

#### Ensemble
* run `python ./ensemble_model/main_ensemble_1.py` to generate first final submission.
* run `python ./ensemble_model/main_ensemble_2.py` to generate second final submission.

## Requirement
* 16GB ram with GPU supported
* Python 2.7
* xgboost: https://github.com/dmlc/xgboost 
* keras: https://github.com/fchollet/keras 
* scikit-learn: https://github.com/scikit-learn/scikit-learn 
* hep_ml: https://github.com/arogozhnikov/hep_ml 
* NumPy: https://github.com/numpy/numpy 
* Pandas: https://github.com/pydata/pandas 
* SciPy: https://github.com/scipy/scipy 


## Document
`./doc/rare_decay_solution.docx`
