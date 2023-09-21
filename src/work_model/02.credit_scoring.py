import catboost as ctb
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.metrics import confusion_matrix

from work_model.dic_models import comparador_modelos

# LOAD DATA
x_train = pd.read_csv('data/preprocessed/train_preprocessed.csv', sep=';', decimal=',')
print(x_train.shape)
x_test = pd.read_csv('data/preprocessed/test_preprocessed.csv', sep=';', decimal=',')
print(x_test.shape)
y_train = pd.read_csv('data/preprocessed/y_train_preprocessed.csv', sep=';', decimal=',')
y_train = y_train.squeeze()
y_test = pd.read_csv('data/preprocessed/y_test_preprocessed.csv', sep=';', decimal=',')
y_test = y_test.squeeze()

path = 'models/'
nombre_modelo = 'tfm'
num_ejecucion = 3

# 'emailage_EAAdvice'
# GENERACIÓN Y COMPARACIÓN DE MODELOS
# GridSearchCV
# RandomizedSearchCV


#################################
# Parámetros de Logit
#################################
C_range = np.linspace(0.1, 0.3, 10)
penalty_range = ['l1', 'l2', 'elasticnet']  # los penalty se pueden aplicar según el solver
class_weight_range = ['balanced']
solver_range = ['liblinear', 'newton-cholesky', 'saga']
max_iter_range = range(10, 200, 10)

C_range = np.linspace(0.1, 0.3, 10)
penalty_range = [None, 'l2']
class_weight_range = ['balanced']
solver_range = ['newton-cholesky']
max_iter_range = range(300, 600, 100)

# ‘liblinear’ - [‘l1’, ‘l2’]
# ‘newton-cholesky’ - [‘l2’, None]
# ‘sag’ - [‘l2’, None]
# ‘saga’ - [‘elasticnet’, ‘l1’, ‘l2’, None]

lr_params = dict(penalty=penalty_range, C=C_range, max_iter=max_iter_range, class_weight=class_weight_range,
                 fit_intercept=[True], random_state=[1], solver=solver_range, multi_class=['ovr'])

#################################
# Parámetros de RandomForest
#################################
n_estimators_range = range(50, 100, 50)
criterion_range = ['entropy', 'gini', 'log_loss']
max_depth_range = range(7, 9)
min_samples_split_range = range(10, 20, 10)
bootstrap_range = [True, False]
class_weight_range = ['balanced', 'balanced_subsample']

n_estimators_range = range(200, 1000, 100)
criterion_range = ['entropy', 'gini', 'log_loss']
max_depth_range = range(2, 6)
min_samples_split_range = range(100, 400, 50)
bootstrap_range = [True, False]
class_weight_range = ['balanced', 'balanced_subsample']

rf_params = dict(n_estimators=n_estimators_range, criterion=criterion_range, max_depth=max_depth_range,
                 min_samples_split=min_samples_split_range,
                 bootstrap=bootstrap_range, n_jobs=[4], random_state=[1], class_weight=class_weight_range)

#################################
# Params SVC
#################################
C_range = np.linspace(0.1, 1, 10)
kernel_range = ['linear', 'poly', 'rbf', 'sigmoid']
# degree only with poly
degree_range = range(2, 10)
class_weight_range = ['balanced']

C_range = [0.1]
kernel_range = ['sigmoid']
degree_range = [3]
class_weight_range = ['balanced']
svc_grid = dict(kernel=kernel_range, class_weight=class_weight_range)

# Catboost
#################################
n_estimators_range = [800]
learning_rate: list[float] = [0.015]
max_depth_range = [3]
# range_scale_pos_weight = range(50, 130, 10)

cb_params = dict(n_estimators=n_estimators_range,
                 learning_rate=learning_rate,
                 max_depth=max_depth_range)

models = ['Logit', 'Random_Forest', 'SVC']
var = x_train.columns.to_list()
var.remove('order_uuid')
best_model = comparador_modelos(x_train, y_train, var, models, lr_grid=lr_params, rf_grid=rf_params,
                                svc_grid=svc_grid, cb_grid=cb_params)


# best = SVC()
# grid = GridSearchCV(best, svc_grid, cv=5, scoring='roc_auc')
# best.fit(x_train[var], y_train)
# balaceo de clases
# classes = np.unique(y_train)
# weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
# class_weights = dict(zip(classes, weights))
# best = ctb.CatBoostClassifier(eval_metric='AUC', class_weights=class_weights)
# cb_params = dict(n_estimators = n_estimators_range,
#                  learning_rate=learning_rate,
#                   max_depth=max_depth_range)
# grid = RandomizedSearchCV(best, cb_params, cv=5, scoring='roc_auc')
# grid.fit(x_train[var], y_train, sample_weight=peso)
# print('Roc_Auc score: ', grid.best_score_)
# print('Params: ', grid.best_params_)
# print('Estimator: ', grid.best_estimator_)
# compute accuracy of the model
# .best_score_(x_test[var], y_test)
# model = open_model(path, 'Random_Forest', nombre_modelo, num_ejecucion)
# y_pred_proba = model.predict_proba(x_test[var])[::,1]
# y_pred = model.predict(x_test[var])
# cm = confusion_matrix(y_test, y_pred)
# auc = metrics.roc_auc_score(y_test, y_pred_proba)
# metrics.f1_score(y_test, y_pred)

# Trabajar Con Catboost
model = ctb.CatBoostClassifier(eval_metric='Accuracy', scale_pos_weight=100)
model.fit(x_train[var], y_train, sample_weight=x_train.principal)

model.best_score_
y_pred_proba = model.predict_proba(x_test[var])[::, 1]
y_pred = model.predict(x_test[var])
cm = confusion_matrix(y_test, y_pred)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
metrics.f1_score(y_test, y_pred)
