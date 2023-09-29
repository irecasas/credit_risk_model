import os

import numpy as np
import pandas as pd
from keras.layers import Flatten
from mlxtend.evaluate import accuracy_score
from sklearn import metrics
from sklearn.metrics import classification_report
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

from commons.constants import DATA_PATH_PREPROCESSED, DATA_PATH_RAW, MODELS_PATH
from work_model.dic_models import lift_booking, lift_number


def result_lrn():
    # Load data train & test
    x_train = pd.read_csv(f'{DATA_PATH_PREPROCESSED}/train_preprocessed.csv', sep=';', decimal=',')
    y_train = pd.read_csv(f'{DATA_PATH_PREPROCESSED}/y_train_preprocessed.csv', sep=';', decimal=',')
    x_test = pd.read_csv(f'{DATA_PATH_PREPROCESSED}/test_preprocessed.csv', sep=';', decimal=',')
    y_train = y_train.squeeze()
    y_test = pd.read_csv(f'{DATA_PATH_PREPROCESSED}/y_test_preprocessed.csv', sep=';', decimal=',')
    y_test = y_test.squeeze()

    # Variable definition
    checkpoint_path = f'{MODELS_PATH}/LR_tfm_network1.hdf5'
    os.path.dirname(checkpoint_path)

    var = x_train.columns.to_list()
    var.remove('order_uuid')
    x_train = x_train[var]
    x_test = x_test[var]
    y_train = np.asarray(y_train).astype('float32').reshape((-1, 1))
    y_test = np.asarray(y_test).astype('float32').reshape((-1, 1))

    # Model parameters
    number_of_classes = 2
    number_of_features = x_train.shape[1]

    # Fit model
    model = Sequential()
    model.add(Dense(number_of_classes, activation='sigmoid', input_dim=number_of_features))
    model.add(Flatten())
    model.compile(optimizer='adam', loss='binary_crossentropy')
    model.summary()

    early_stop = EarlyStopping(monitor='val_loss', patience=20, verbose=1)
    checkpoint = ModelCheckpoint(filepath=checkpoint_path, monitor='val_loss', verbose=0, save_best_only=True)
    model.fit(x_train, y_train, epochs=50, validation_data=(
        x_test, y_test), callbacks=[checkpoint, early_stop])

    y_pred_proba = model.predict(x_test)
    y_pred = pd.DataFrame(y_pred_proba).idxmax(axis=1)
    y_pred_proba = y_pred_proba[::, 1]

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
    result_lrn()
