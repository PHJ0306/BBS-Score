import optuna
import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneOut
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import random
import scipy.io as sio
import pickle
"""
File: hyperparameter_tuning.py

This script performs hyperparameter tuning for multiple regression models using Optuna.
It reads and preprocesses a CSV dataset by removing missing values and dropping specific columns 
(e.g., 'NUM' and 'Diff.of stance time'). The features (all columns except 'BBS_Score') and 
the target ('BBS_Score') are scaled using StandardScaler.

A custom data splitting strategy is implemented in the 'split_train' function, which 
randomly shuffles the data and creates training and testing sets over multiple iterations.

The script then defines an 'objective' function that sets up the hyperparameter search space 
for each of the following regression models:
    - Ridge
    - Lasso
    - ElasticNet
    - Support Vector Regression (SVR)
    - Decision Tree Regressor
    - Random Forest Regressor
    - Linear Regression

For each model, Optuna optimizes the hyperparameters to maximize the R2 score, and the best 
predictions along with the corresponding true values are stored in a dictionary for further evaluation.
"""

data = pd.read_csv(".../10Feature+BBS_Score(+IDS).csv")
data = data.dropna()
data = data.drop(columns=['NUM', 'Diff.of stance time'])

X = data.drop(columns=['BBS_Score'])
y = data['BBS_Score']

scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1)).flatten()

predictions_dict = {}

def split_train(X, y, model):
    n = len(X)
    predictions = []
    true_values = []

    for _ in range(5):
        indices = list(range(n))
        random.shuffle(indices)
        
        train_indices = indices[:45]
        test_indices = indices[45:50]
        additional_test_indices = random.sample(train_indices, 5)
        test_indices.extend(additional_test_indices)
        
        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        predictions.extend(y_pred)
        true_values.extend(y_test)
            
    predictions = scaler_y.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
    true_values = scaler_y.inverse_transform(np.array(true_values).reshape(-1, 1)).flatten()
    
    return predictions, true_values

predictions_dict = {}

def objective(trial, model_name):
    if model_name == 'Ridge':
        model = Ridge(alpha=trial.suggest_loguniform('alpha', 1e-3, 1e3),
                      solver=trial.suggest_categorical('solver', ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']))
    elif model_name == 'Lasso':
        model = Lasso(alpha=trial.suggest_loguniform('alpha', 1e-3, 1e3),
                      max_iter=trial.suggest_int('max_iter', 100, 10000),
                      selection=trial.suggest_categorical('selection', ['cyclic', 'random']))
    elif model_name == 'ElasticNet':
        model = ElasticNet(alpha=trial.suggest_loguniform('alpha', 1e-3, 1e3),
                            l1_ratio=trial.suggest_uniform('l1_ratio', 0.1, 0.9),
                            max_iter=trial.suggest_int('max_iter', 100, 10000),
                            selection=trial.suggest_categorical('selection', ['cyclic', 'random']))
    elif model_name == 'SVR':
        kernel = trial.suggest_categorical('kernel', ['linear','rbf','sigmoid'])
        shrinking = trial.suggest_categorical('shrinking', [True, False])

        if kernel == 'linear':
            model = SVR(kernel=kernel, C=trial.suggest_loguniform('C', 1, 20),
                        epsilon=trial.suggest_uniform('epsilon', 0.01, 1), 
                        tol=trial.suggest_loguniform('tol', 1e-5, 1e-1),
                        shrinking=shrinking)
        elif kernel == 'rbf':
            model = SVR(kernel=kernel, C=trial.suggest_loguniform('C', 1, 20),
                        gamma=trial.suggest_loguniform('gamma', 1e-4, 4),
                        epsilon=trial.suggest_uniform('epsilon', 0.01, 1),                    
                        tol=trial.suggest_loguniform('tol', 1e-5, 1e-1),
                        shrinking=shrinking)

        elif kernel == 'sigmoid':
            model = SVR(kernel=kernel, C=trial.suggest_loguniform('C', 1, 20),
                        coef0=trial.suggest_uniform('coef0', 0, 1),
                        epsilon=trial.suggest_uniform('epsilon', 0.01, 1),
                        gamma=trial.suggest_loguniform('gamma', 1e-4, 4),
                        tol=trial.suggest_loguniform('tol', 1e-5, 1e-1),
                        shrinking=shrinking)
 
    elif model_name == 'DecisionTree':
        model = DecisionTreeRegressor(max_depth=trial.suggest_int('max_depth', 1, 5),
                                      min_samples_split=trial.suggest_int('min_samples_split', 2, 5),
                                      min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 5),
                                      max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2']),
                                      max_leaf_nodes = trial.suggest_int('max_leaf_nodes', 2, 20),
                                      splitter = trial.suggest_categorical('splitter', ['best', 'random']))
    elif model_name == 'RandomForest':
        model = RandomForestRegressor(n_estimators=trial.suggest_int('n_estimators', 1, 50),
                                      max_depth=trial.suggest_int('max_depth', 1, 10),
                                      min_samples_split=trial.suggest_int('min_samples_split', 2, 5))
    else:
        model = LinearRegression(fit_intercept=trial.suggest_categorical('fit_intercept', [True, False]))
    
    predictions, true_values = split_train(X_scaled, y_scaled, model)    

    if model_name not in predictions_dict:
        predictions_dict[model_name] = {'predictions': [], 'true_values': []}
    
    predictions_dict[model_name]['predictions'] = predictions
    predictions_dict[model_name]['true_values'] = true_values

    r2 = r2_score(true_values, predictions)
    return r2

model_names = ['Ridge', 'Lasso', 'ElasticNet', 'SVR', 'DecisionTree', 'RandomForest', 'LinearRegression']
r2_values = {}

for model_name in model_names:
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, model_name), n_trials=100)
    with open(f"{model_name}_best_params.pkl", "wb") as f:
        pickle.dump(study.best_params, f)

    r2_values[model_name] = study.best_value  

sio.savemat('predictions_dict.mat', predictions_dict)