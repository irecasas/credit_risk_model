import pandas as pd
from mlxtend.evaluate import accuracy_score
from sklearn import metrics, svm
from sklearn.metrics import classification_report
from sklearn.model_selection import RandomizedSearchCV, RepeatedStratifiedKFold

from commons.constants import DATA_PATH_PREPROCESSED, MODELS_PATH
from feature_engineering.utils import open_model, save_model


def train_SVM():
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
    num_ejecucion = 3

    # Model parameters
    svm_grid = dict()
    svm_grid['kernel'] = ['poly']
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=1)

    # Fit model
    best = svm.SVC(class_weight='balanced', probability=True)
    grid = RandomizedSearchCV(best, svm_grid, cv=cv, scoring='roc_auc', verbose=2)
    grid.fit(x_train[var], y_train, sample_weight=peso)

    print('Roc_Auc score: ', grid.best_score_)
    print('Params: ', grid.best_params_)
    print('Estimator: ', grid.best_estimator_)

    # Save SVM model
    save_model(path, 'SVM', nombre_modelo, num_ejecucion, grid.best_estimator_)


def result_SVM():
    # Load data train & test
    x_test = pd.read_csv(f'{DATA_PATH_PREPROCESSED}/test_preprocessed.csv', sep=';', decimal=',')
    y_test = pd.read_csv(f'{DATA_PATH_PREPROCESSED}/y_test_preprocessed.csv', sep=';', decimal=',')
    y_test = y_test.squeeze()

    # Variable definition
    var = x_test.columns.to_list()
    var.remove('order_uuid')
    nombre_modelo = 'tfm'
    num_ejecucion = 2

    # Open LR model
    model = open_model(MODELS_PATH, 'SVM', nombre_modelo, num_ejecucion)

    # Learning curves
    # learning_curves(model, x_train[var], y_train, cv)

    # evaluate model
    # y_pred_proba = model.predict_proba(x_test[var])[::, 1]
    y_pred = model.predict(x_test[var])

    print(classification_report(y_test, y_pred))
    print('accuracy score: ', accuracy_score(y_test, y_pred))
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)
    print('auc score: ', metrics.auc(fpr, tpr))


if __name__ == "__main__":
    result_SVM()
