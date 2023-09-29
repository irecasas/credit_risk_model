import pandas as pd

from feature_engineering import utils
from feature_engineering.constants import delete_correlation
from feature_engineering.utils import (
    dtypes_selector,
    feature_selection_rf,
    initial_cleaning,
    iv_woe,
    missing_values,
    splitting_data,
    standardization_continuous_features,
    woe_encoder,
)

# Read raw data
cwd = "/Users/irene.casas/PycharmProjects/credit_risk_model/data/raw/dataset_total_2020_2023_raw.parquet"
df = pd.read_csv(cwd, low_memory=False)
# Drop columns
df.drop(['uuid_y', 'Unnamed: 0'], axis=1, inplace=True)
# Delete months in which I do not have any targets
df = df.loc[(df.mes != '2023-05') & (df.mes != '2023-06')]

# INITIAL CLEANING
# First cleaning data
df = initial_cleaning(df)
# Missing treatment
dic_vars = dtypes_selector(df)
missing_values(df, dic_vars)
# Splitting data
x_train, y_train, x_test, y_test = splitting_data(df)

# STANDARDIZATION OF DATA
# Continuous variables:
x_train = standardization_continuous_features(x_train, dic_vars)
x_test = standardization_continuous_features(x_test, dic_vars)
# One Hot:
# x_train = onehot_encoding(x_train)
# x_test = onehot_encoding(x_test)
# Woe Encoder
x_train, x_test = woe_encoder(x_train, y_train, x_test)

# DIMENSION REDUCTION OF EXPERIMENTAL DATASET
# Random Forest
features = list(x_train.columns)
features.remove('order_uuid')
feature_importance = feature_selection_rf(x_train, features, y_train)
x_train = x_train[feature_importance]
x_test = x_test[feature_importance]
# Correlation Matrix
utils.corr_matrix(x_train)
# Drop columns by correlation
x_train.drop(delete_correlation, axis=1, inplace=True)
x_test.drop(delete_correlation, axis=1, inplace=True)

# STEPWISE
# Stepwise (below 0.05 / 0.1)
features = x_train.columns.to_list()
features.remove("order_uuid")
utils.get_stats(x_train, y_train, features)
features.remove("emailage_response__domain_creation_days")
features.remove("emailage_response__ip_anonymousdetected_trans")
features.append('order_uuid')
x_train = x_train[features]
x_test = x_test[features]

# INFORMATION VALUE
# Select IV values greater than 0.02
x_train = x_train.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)
df = pd.merge(x_train, y_train, how='left', left_index=True, right_index=True)
features = df.columns.to_list()
features.remove('order_uuid')
iv, woe = iv_woe(data=df[features], target='target')
delete_iv = ['iovation_ruleResults__score',
             'experian_ResumenCais__NumeroOperacionesImpagadas',
             'experian_ResumenCais__ImporteImpagado',
             'experian_ResumenCais__NumeroCuotasImpagadas',
             'experian_ResumenCais__FechaAltaOperacionMasAntigua_difference_with_created']
x_train.drop(delete_iv, axis=1, inplace=True)
x_test.drop(delete_iv, axis=1, inplace=True)

# SAVE DATASETS
x_train.to_csv('data/preprocessed/train_preprocessed.csv', sep=';', decimal=',', index=False)
print(x_train.shape)
y_train.to_csv('data/preprocessed/y_train_preprocessed.csv', sep=';', decimal=',', index=False)
x_test.to_csv('data/preprocessed/test_preprocessed.csv', sep=';', decimal=',', index=False)
y_test.to_csv('data/preprocessed/y_test_preprocessed.csv', sep=';', decimal=',', index=False)
