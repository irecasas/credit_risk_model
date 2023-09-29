import pandas as pd
from mlxtend.evaluate import accuracy_score
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import RandomizedSearchCV, RepeatedStratifiedKFold

from commons.constants import DATA_PATH_PREPROCESSED, DATA_PATH_RAW, MODELS_PATH
from feature_engineering.utils import open_model, save_model
from work_model.dic_models import lift_booking, lift_number


def train_RF():
    # Load data train & test
    x_train = pd.read_csv('data/preprocessed/train_preprocessed.csv', sep=';', decimal=',')
    print(x_train.shape)
    y_train = pd.read_csv('data/preprocessed/y_train_preprocessed.csv', sep=';', decimal=',')
    y_train = y_train.squeeze()

    # Variable definition
    var = x_train.columns.to_list()
    var.remove('order_uuid')
    peso = x_train.principal
    path = 'models/'
    nombre_modelo = 'tfm'
    num_ejecucion = 1

    # Model parameters
    n_estimators_range = range(200, 1000, 100)
    criterion_range = ['entropy', 'gini', 'log_loss']
    max_depth_range = range(2, 8)
    min_samples_split_range = range(50, 150, 50)
    bootstrap_range = [True, False]
    class_weight_range = ['balanced']

    rf_params = dict(n_estimators=n_estimators_range, criterion=criterion_range,
                     max_depth=max_depth_range, min_samples_split=min_samples_split_range,
                     bootstrap=bootstrap_range, n_jobs=[4], random_state=[1],
                     class_weight=class_weight_range)

    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=1)

    # Fit model
    best = RandomForestClassifier()
    grid = RandomizedSearchCV(best, rf_params, cv=cv, scoring='roc_auc', verbose=2)
    grid.fit(x_train[var], y_train, sample_weight=peso)

    print('Roc_Auc score: ', grid.best_score_)
    print('Params: ', grid.best_params_)
    print('Estimator: ', grid.best_estimator_)

    importancias = pd.DataFrame(list(zip(
        var, grid.best_estimator_.feature_importances_)), columns=['Variables', 'Importancia'])\
        .sort_values(by=['Importancia'], ascending=False, axis=0)
    print(importancias)

    # Save RF model
    save_model(path, 'RF', nombre_modelo, num_ejecucion, grid.best_estimator_)


def result_RF():
    # Load data train & test
    x_test = pd.read_csv(f'{DATA_PATH_PREPROCESSED}/test_preprocessed.csv', sep=';', decimal=',')
    y_test = pd.read_csv(f'{DATA_PATH_PREPROCESSED}/y_test_preprocessed.csv', sep=';', decimal=',')
    y_test = y_test.squeeze()

    # Variable definition
    var = x_test.columns.to_list()
    var.remove('order_uuid')
    nombre_modelo = 'tfm'
    num_ejecucion = 1

    # Open RF model
    model = open_model(MODELS_PATH, 'RF', nombre_modelo, num_ejecucion)

    # Learning curves
    # learning_curves(model, x_train[var], y_train, cv)

    # evaluate model
    y_pred_proba = model.predict_proba(x_test[var])[::, 1]
    y_pred = model.predict(x_test[var])

    print(classification_report(y_test, y_pred))
    print('accuracy score: ', accuracy_score(y_test, y_pred))
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)
    print('auc score: ', metrics.auc(fpr, tpr))

    # Preparing the basis for the LIFT calculation
    x_test = x_test[['order_uuid']]
    y_pred_proba = pd.DataFrame(y_pred_proba)
    x_test = pd.merge(x_test, y_pred_proba, how='left', left_index=True, right_index=True)

    cwd = f"{DATA_PATH_RAW}/dataset_total_2020_2023_raw.parquet"
    df = pd.read_csv(cwd, low_memory=False)
    df = df[['order_uuid', 'target', 'egr_mob3', 'egr_over30mob3']]
    x_test = pd.merge(x_test, df, how='left', on='order_uuid')
    x_test = x_test.rename(index=str, columns={0: 'Probabilidad'})

    l = list(range(1, 11))
    l.sort(reverse=True)
    x_test['Decil'] = pd.qcut(x_test.Probabilidad, 10, labels=l)

    # Sorting the database according to the number of applications
    df1 = lift_number(x_test)
    print("LIFT by target:")
    print(df1)

    # Sorting the database according to the amount borrowed
    df1 = lift_booking(x_test)
    print("LIFT by booking:")
    print(df1)


if __name__ == "__main__":
    result_RF()
