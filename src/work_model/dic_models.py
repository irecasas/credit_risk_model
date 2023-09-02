import catboost as ctb
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.svm import SVC

from feature_engineering.utils import save_model


def comparador_modelos(x_train, y_train, var,
                       path, models, nombre_modelo, num_ejecucion,
                       lr_grid=dict(), rf_grid=dict(), svc_grid=dict(), cb_grid=dict(), cv_folds=5):
    best_scores = {}
    best_params = {}
    best_estimators = {}
    peso = x_train.principal

    print('\n*********** Busqueda de parametros con RandomizedSearchCV ***********')
    for model in models:
        # Regresión Logística
        if 'Logit' == model:
            print('\nLogisticRegression: ')
            best = LogisticRegression()
            grid = RandomizedSearchCV(best, lr_grid, cv=cv_folds, scoring='roc_auc', verbose=2)
            grid.fit(x_train[var], y_train, sample_weight=peso)

            print('Roc_Auc score: ', grid.best_score_)
            print('Params: ', grid.best_params_)
            print('Estimator: ', grid.best_estimator_)

            best_scores.update({model: grid.best_score_})
            best_params.update({model: grid.best_params_})
            best_estimators.update({model: grid.best_estimator_})

            print('\n*********** Coeficientes del modelo  ' + model + ' ***********')
            coeficientes = pd.DataFrame(list(zip(var, grid.best_estimator_.coef_.tolist()[0])),
                                        columns=['Variables', 'Coeficientes'])
            print(coeficientes)

            save_model(path, model, nombre_modelo, num_ejecucion, grid.best_estimator_)

        # Random_Forest
        if 'Random_Forest' == model:
            print('\nRandomForestClassifier: ')
            best = RandomForestClassifier()
            grid = RandomizedSearchCV(best, rf_grid, cv=cv_folds, scoring='roc_auc', verbose=2)
            grid.fit(x_train[var], y_train, sample_weight=peso)

            print('Roc_Auc score: ', grid.best_score_)
            print('Params: ', grid.best_params_)
            print('Estimator: ', grid.best_estimator_)

            best_scores.update({model: grid.best_score_})
            best_params.update({model: grid.best_params_})
            best_estimators.update({model: grid.best_estimator_})

            print('\n*********** Importancia de variables  ' + model + ' ***********')
            importancias = pd.DataFrame(list(zip(
                var, grid.best_estimator_.feature_importances_)), columns=['Variables', 'Importancia'])\
                .sort_values(by=['Importancia'], ascending=False, axis=0)
            print(importancias)

            save_model(path, model, nombre_modelo, num_ejecucion, grid.best_estimator_)

        # SVC
        if 'SVC' == model:
            print('\nSVC: ')
            best = SVC()
            grid = RandomizedSearchCV(best, svc_grid, cv=cv_folds, scoring='roc_auc')
            grid.fit(x_train[var], y_train, sample_weight=peso)

            print('Roc_Auc score: ', grid.best_score_)
            print('Params: ', grid.best_params_)
            print('Estimator: ', grid.best_estimator_)

            best_scores.update({model: grid.best_score_})
            best_params.update({model: grid.best_params_})
            best_estimators.update({model: grid.best_estimator_})

            print('\n*********** Importancia de variables  ' + model + ' ***********')
            importancias = pd.DataFrame(list(zip(
                var, grid.best_estimator_.feature_importances_)), columns=['Variables', 'Importancia']).\
                sort_values(by=['Importancia'], ascending=False, axis=0)
            print(importancias)

            save_model(path, model, nombre_modelo, num_ejecucion, grid.best_estimator_)

        if 'Catboost' == model:
            print('\nCatBoostClassifier: ')
            best = ctb.CatBoostClassifier(eval_metric='AUC')
            grid = RandomizedSearchCV(best, cb_grid, cv=cv_folds, scoring='f1')
            grid.fit(x_train[var], y_train, sample_weight=peso)

            print('Roc_Auc score: ', grid.best_score_)
            print('Params: ', grid.best_params_)
            print('Estimator: ', grid.best_estimator_)

            best_scores.update({model: grid.best_score_})
            best_params.update({model: grid.best_params_})
            best_estimators.update({model: grid.best_estimator_})

            print('\n*********** Importancia de variables ' + model + ' ***********')
            importancias = pd.DataFrame(list(zip(
                var, grid.best_estimator_.feature_importances_)), columns=['Variables', 'Importancia'])\
                .sort_values(by=['Importancia'], ascending=False, axis=0)
            print(importancias)

            save_model(path, model, nombre_modelo, num_ejecucion, grid.best_estimator_)

    print('\n*********** Cross Validation para mejores modelos ***********')
    lista = []
    print('entra en linea 90')
    for i in (best_estimators.values()):
        print('print i', i)
        scores = cross_val_score(i, x_train[var], y_train, cv=5, scoring='roc_auc')
        lista.append({'CV_Scores': scores, 'CV_Score_mean': scores.mean(), 'Cv_Score_Std': scores.std() * 2})

    tabla = pd.DataFrame(lista, index=models)
    print('print tabla shape', tabla.shape)
    tabla_CV = pd.DataFrame(tabla['CV_Scores'].tolist(), columns=['CV_1', 'CV_2', 'CV_3', 'CV_4', 'CV_5'],
                            index=best_estimators.keys())
    print(tabla_CV.round(4))

    tabla_plot = tabla_CV
    tabla_plot.columns = range(1, 6)
    plt.plot(tabla_plot.T)
    plt.ylim((0.5, 1))
    plt.title('Cross Validation')
    plt.xlabel('Número de folds')
    plt.ylabel('Roc_Auc_score')
    plt.legend(best_estimators.keys(), loc=0)
    plt.show()

    print('\n*********** Tabla resumen mejores modelos ***********')
    tabla['Best_Score'] = list(best_scores.values())
    tabla['Criterio_eleccion'] = ((tabla['Best_Score'] + tabla['CV_Score_mean']) / 2.0)
    print(tabla[['Best_Score', 'CV_Score_mean', 'Cv_Score_Std']].round(4))
    tabla['max'] = np.where(tabla['Criterio_eleccion'] == tabla['Criterio_eleccion'].max(), 1, 0)
    tabla['seleccion'] = np.where(tabla['max'] == 1, tabla['max'] - tabla['Cv_Score_Std'], tabla['max'])
    election = tabla['seleccion'].idxmax()

    print('\n*********** Eleccion ***********\n', election)
    return election
