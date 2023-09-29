import pandas as pd
from mlxtend.evaluate import accuracy_score
from sklearn import metrics
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold

from commons.constants import DATA_PATH_PREPROCESSED, DATA_PATH_RAW, MODELS_PATH
from feature_engineering.utils import open_model, save_model
from work_model.dic_models import lift_booking, lift_number

pd.options.mode.chained_assignment = None  # default='warn'


def train_LDA():
    # Load data train & test
    x_train = pd.read_csv('data/preprocessed/train_preprocessed.csv', sep=';', decimal=',')
    y_train = pd.read_csv('data/preprocessed/y_train_preprocessed.csv', sep=';', decimal=',')
    y_train = y_train.squeeze()

    # Variable definition
    path = 'models/'
    nombre_modelo = 'tfm'
    num_ejecucion = 1
    var = x_train.columns.to_list()
    var.remove('order_uuid')

    # Model parameters
    lda_grid = dict()
    lda_grid['solver'] = ['svd', 'lsqr', 'eigen']
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=1)

    # Fit model
    model = LinearDiscriminantAnalysis()
    grid = GridSearchCV(model, lda_grid, scoring='accuracy', cv=cv, n_jobs=-1)
    grid.fit(x_train[var], y_train)

    print('Roc_Auc score: ', grid.best_score_)
    print('Params: ', grid.best_params_)
    print('Estimator: ', grid.best_estimator_)

    # Save LDA model
    save_model(path, 'LDA', nombre_modelo, num_ejecucion, grid.best_estimator_)

def result_LDA():
    # Load data train & test
    x_test = pd.read_csv(f'{DATA_PATH_PREPROCESSED}/test_preprocessed.csv', sep=';', decimal=',')
    y_test = pd.read_csv(f'{DATA_PATH_PREPROCESSED}/y_test_preprocessed.csv', sep=';', decimal=',')
    y_test = y_test.squeeze()

    # Variable definition
    nombre_modelo = 'tfm'
    num_ejecucion = 1
    var = x_test.columns.to_list()
    var.remove('order_uuid')

    # Open LDA model
    model = open_model(MODELS_PATH, 'LDA', nombre_modelo, num_ejecucion)

    # Learning curves
    # learning_curves(model, x_train[var], y_train, cv)

    # Metrics to evaluate model
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
    result_LDA()
