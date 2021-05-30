## Summary
This is my solution for Kaggle: Flavours of physics - Identify a rare decay phenomenon.

## Problem
The laws of nature ensure that some physical quantities, such as energy or momentum, are conserved. From Noether’s theorem, we know that each conservation law is associated with a fundamental symmetry. For example, conservation of energy is due to the time-invariance (the outcome of an experiment would be the same today or tomorrow) of physical systems. The fact that physical systems behave the same, regardless of where they are located or how they are oriented, gives rise to the conservation of linear and angular momentum.

Symmetries are also crucial to the structure of the Standard Model of particle physics, our present theory of interactions at microscopic scales. Some are built into the model, while others appear accidentally from it. In the Standard Model, lepton flavour, the number of electrons and electron-neutrinos, muons and muon-neutrinos, and tau and tau-neutrinos, is one such conserved quantity.

Interestingly, in many proposed extensions to the Standard Model, this symmetry doesn’t exist, implying decays that do not conserve lepton flavour are possible. One decay searched for at the LHC is τ → μμμ (or τ → 3μ). Observation of this decay would be a clear indication of the violation of lepton flavour and a sign of long-sought new physics.

## Our solution
The model gives 16/673 at public and private leaderboard with Weighted AUC = 0.991841. Without any usage of the following, which are prohibited in the competition:

* Use of agreement data, correlation data
* Local evaluation of agreement test and correlation test
* Reconstructuion of mass feature using equation from Special Relativity
The summary of the approach is to create three different groups of model from linear (underfitting) to nonlinear (overfitting), and then ensemble it in second layer with convex optimization.

## Instruction

#### Download Data
* download `sample_submission.csv`, `training.csv`, `test.csv`, and put into folder `./input`.

#### Run all model
* run `sh run.sh`

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
[`./doc/rare_decay_solution.pdf`](./doc/rare_decay_solution.pdf)
