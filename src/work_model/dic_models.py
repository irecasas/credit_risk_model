import catboost as ctb
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, cross_val_score, learning_curve
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
    tabla_CV = pd.DataFrame(tabla['CV_Scores'].tolist(),
                            columns=['CV_1', 'CV_2', 'CV_3', 'CV_4', 'CV_5'],
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


def learning_curves(modelo, x_train0, y_train0, cv):
    train_sizes, train_scores, test_scores = learning_curve(modelo, x_train0, y_train0,
                                                            train_sizes=[0.2, 0.4, 0.6, 0.8, 1],
                                                            cv=cv, scoring='roc_auc', n_jobs=-1)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.figure()
    plt.title("Curva de Aprendizaje")
    plt.legend(loc="best")
    plt.xlabel("Training examples")
    plt.ylabel("Score")

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.3, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.3, color="g")

    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    plt.legend(loc='best')

    plt.ylim(0.5, 1)
    plt.show()

def lift_number(df):
    """
    function that returns the order of the data entered according to the score,
    from highest to lowest probability, and calculates the lift
    the dataframe has to have the following columns: order_uuid, Decil, Probabilidad, target
    :param df:
    :return:
    """
    tabla1 = pd.DataFrame({'Orders': df.groupby('Decil')['order_uuid'].count()}).reset_index()
    tabla2 = pd.DataFrame({'Avg Proba': df.groupby('Decil')['Probabilidad'].mean()}).reset_index()
    df1 = pd.merge(tabla1, tabla2, how='left', on='Decil')
    tabla2 = pd.DataFrame({'Target': df.groupby('Decil')['target'].sum()}).reset_index()
    df1 = pd.merge(df1, tabla2, how='left', on='Decil')
    df1['% Target'] = (df1['Target'] / df1['Target'].sum()) * 100
    df1['Decil'] = df1['Decil'].astype('float')

    df1['Acum Target'] = 0.0
    for i in [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]:
        if i == 9:
            df1['Acum Target'][i] = df1['Target'][i]
        else:
            df1['Acum Target'][i] = df1['Acum Target'][i + 1] + df1['Target'][i]
    var = df1['Target'].sum() / 10
    df1['borrar1'] = [var, var, var, var, var, var, var, var, var, var]

    df1['borrar'] = 0.0
    for i in [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]:
        df1['borrar'][i] = df1['borrar1'][i] * df1['Decil'][i]

    df1['Lift'] = 0.0
    for i in [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]:
        df1['Lift'][i] = ((df1['Acum Target'][i]) / (df1['borrar'][i]))

    df1 = df1[['Decil', 'Orders', 'Avg Proba', 'Target', 'Acum Target', '% Target', 'Lift']]
    df1 = df1.sort_values(by=['Decil'], ascending=True)
    return df1

def lift_booking(df):
    """
    function that returns the order of the data entered according to the score,
    from highest to lowest probability, and calculates the lift
    the dataframe has to have the following columns: order_uuid, Decil, Probabilidad, target
    :param df:
    :return:
    """
    tabla1 = pd.DataFrame({'Orders': df.groupby('Decil')['order_uuid'].count()}).reset_index()
    tabla2 = pd.DataFrame({'Avg Proba': df.groupby('Decil')['Probabilidad'].mean()}).reset_index()
    df1 = pd.merge(tabla1, tabla2, how='left', on='Decil')
    tabla2 = pd.DataFrame({'Sum Egr': df.groupby('Decil')['egr_mob3'].sum()}).reset_index()
    df1 = pd.merge(df1, tabla2, how='left', on='Decil')
    tabla2 = pd.DataFrame({'O30 Mob3': df.groupby('Decil')['egr_over30mob3'].sum()}).reset_index()
    df1 = pd.merge(df1, tabla2, how='left', on='Decil')
    df1['% O30 Mob3'] = (df1['O30 Mob3'] / df1['O30 Mob3'].sum()) * 100
    df1['Decil'] = df1['Decil'].astype('float')

    df1['Acum Target'] = 0.0
    for i in [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]:
        if i == 9:
            df1['Acum Target'][i] = df1['O30 Mob3'][i]
        else:
            df1['Acum Target'][i] = df1['Acum Target'][i + 1] + df1['O30 Mob3'][i]
    var = df1['O30 Mob3'].sum() / 10
    df1['borrar1'] = [var, var, var, var, var, var, var, var, var, var]

    df1['borrar'] = 0.0
    for i in [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]:
        df1['borrar'][i] = df1['borrar1'][i] * df1['Decil'][i]

    df1['Lift'] = 0.0
    for i in [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]:
        df1['Lift'][i] = ((df1['Acum Target'][i]) / (df1['borrar'][i]))

    df1 = df1[['Decil', 'Orders', 'Avg Proba', 'Sum Egr', 'O30 Mob3', '% O30 Mob3', 'Lift']]
    df1 = df1.sort_values('Decil')
    return df1
