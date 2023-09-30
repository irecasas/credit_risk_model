import pandas as pd
from sklearn import metrics
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import RandomizedSearchCV, RepeatedStratifiedKFold
from sklearn.neural_network import MLPClassifier

from commons.constants import DATA_PATH_PREPROCESSED, DATA_PATH_RAW
from work_model.dic_models import lift_booking, lift_number


def result_mlp():
    # Load data train & test
    x_train = pd.read_csv(f'{DATA_PATH_PREPROCESSED}/train_preprocessed.csv', sep=';', decimal=',')
    y_train = pd.read_csv(f'{DATA_PATH_PREPROCESSED}/y_train_preprocessed.csv', sep=';', decimal=',')
    x_test = pd.read_csv(f'{DATA_PATH_PREPROCESSED}/test_preprocessed.csv', sep=';', decimal=',')
    y_train = y_train.squeeze()

    # Variable definition
    var = x_train.columns.to_list()
    var.remove('order_uuid')
    x_train = x_train[var]
    x_test = x_test[var]

    # parameters = {
    #    'hidden_layer_sizes': [(500, 400, 300), (20, 40, 50), (1000, 300, 100)],
    #    'solver': ['lbfgs', 'adam'],
    #    'activation': ['tanh', 'identity', 'logistic', 'relu'],
    #    'learning_rate_init': [0.0001, 0.01]
    # }

    parameters = {
        'hidden_layer_sizes': (500, 400, 300),
        'solver': ['lbfgs'],
        'activation': ['tanh'],
        'learning_rate_init': [0.01]
    }
    cv = RepeatedStratifiedKFold(n_splits=2, n_repeats=1, random_state=1)

    # Fit model
    model = MLPClassifier(verbose=True, early_stopping=True)
    randomSearch = RandomizedSearchCV(model, param_distributions=parameters, cv=cv)
    randomSearch.fit(x_train, y_train)

    y_pred = randomSearch.predict(x_test)
    y_pred_proba = randomSearch.predict_proba(x_test[var])[::, 1]

    y_test = pd.read_csv(f'{DATA_PATH_PREPROCESSED}/y_test_preprocessed.csv', sep=';', decimal=',')
    y_test = y_test.squeeze()

    print(classification_report(y_test, y_pred))
    print('accuracy score: ', accuracy_score(y_test, y_pred))
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)
    print('auc score: ', metrics.auc(fpr, tpr))
    # Preparing the basis for the LIFT calculation
    x_test = pd.read_csv(f'{DATA_PATH_PREPROCESSED}/test_preprocessed.csv', sep=';', decimal=',')
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
    print("LIFT by target:")
    print(df1)


if __name__ == "__main__":
    result_mlp()
