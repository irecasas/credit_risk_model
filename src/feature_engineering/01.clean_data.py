import os

import pandas as pd

from feature_engineering import utils
from feature_engineering.constants import delete_correlation, features_remove
from feature_engineering.utils import (
    dtypes_selector,
    feature_selection_rf,
    initial_cleaning,
    iv_woe,
    missing_values,
    onehot_encoding,
    splitting_data,
    standardization_continuous_features,
    woe_encoder,
)

# Read data
cwd = "/Users/irene.casas/PycharmProjects/credit_risk_model/"
path_data = "data/raw/dataset_total_2020_2023_raw.parquet"
df = pd.read_csv(os.path.join(cwd, path_data), low_memory=False)
df.drop(['uuid_y', 'Unnamed: 0'], axis=1, inplace=True)
# Quito los meses que aún no tengo target
df = df.loc[(df.mes != '2023-05') & (df.mes != '2023-06')]

# First cleaning data
df = initial_cleaning(df)

# missing treatment
dic_vars = dtypes_selector(df)
missing_values(df, dic_vars)

# Splitting data
x_train, y_train, x_test, y_test = splitting_data(df)

# standardization of data
# Continuous variables:
x_train = standardization_continuous_features(x_train, dic_vars)
x_test = standardization_continuous_features(x_test, dic_vars)

# One Hot:
x_train = onehot_encoding(x_train)
x_test = onehot_encoding(x_test)

# Woe Encoder
x_train, x_test = woe_encoder(x_train, y_train, x_test)

# Filtramos por las variables en común entre los dos conjuntos de datos
features_test = x_test.columns.to_list()
features_test = set(features_test).difference(set(features_remove))
x_train = x_train[features_test]
x_test = x_test[features_test]

# CONTEOS DE NULOS POR VARIABLES
# for i in x_train:
#    if pd.isnull(x_train[i]).sum() > 0:
#        print("La variable ", i, " tiene ", pd.isnull(x_train[i]).sum(), " valores nulos")
# for i in x_test:
#    if pd.isnull(x_test[i]).sum() > 0:
#        print("La variable ", i, " tiene ", pd.isnull(x_test[i]).sum(), " valores nulos")

# Dimension Reduction of Experimental Dataset Based on Pdc-RF Algorithm
# Random Forest
features = list(x_train.columns)
features.remove('order_uuid')
feature_importance = feature_selection_rf(x_train, features, y_train)

x_train = x_train[feature_importance]
x_test = x_test[feature_importance]

# Matriz de correlación
utils.corr_matrix(x_train)

# Drop columns by correlation
x_train.drop(delete_correlation, axis=1, inplace=True)
x_test.drop(delete_correlation, axis=1, inplace=True)

# Stepwise (below 0.05 / 0.1)
features = x_train.columns.to_list()
features.remove("order_uuid")

utils.get_stats(x_train, y_train, features)
features.remove("emailage_response__domaincategory_Legislation, Politics & Law")  # Primera eliminada
features.remove("emailage_response__domaincategory_Pharmaceuticals")
features.remove("emailage_response__domaincategory_Motorized Vehicles")
features.remove("emailage_response__domaincategory_Private Domain")
features.remove("emailage_response__domaincategory_IT Consulting")
features.remove("emailage_response__domaincategory_Software Development and Services")
features.remove("emailage_response__ip_corporateProxy_trans")
features.remove("industry_orders_timeline-ptg_orders_rejected_severity_high")
features.remove("experian_OrigenScore__Codigo_6.0")
features.remove("emailage_response__ip_netSpeedCell_ultrabb")
features.remove("emailage_response__domaincategory_Online Ads")
features.append('order_uuid')

x_train = x_train[features]
x_test = x_test[features]

# Select IV values greater than 0.02
x_train = x_train.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)
df = pd.merge(x_train, y_train, how='left', left_index=True, right_index=True)
features = df.columns.to_list()
features.remove('order_uuid')
iv, woe = iv_woe(data=df[features], target='target')

delete_iv = ['experian_ResumenCais__NumeroCuotasImpagadas',
             'minfraud_response__ip_address__traits__is_anonymous_trans',
             'experian_ResumenCais__NumeroOperacionesImpagadas',
             'emailage_response__domaincategory_Login Screens',
             'iovation_ruleResults__score',
             'minfraud_response__email__is_free_trans']
x_train.drop(delete_iv, axis=1, inplace=True)
x_test.drop(delete_iv, axis=1, inplace=True)

# Save datasets
x_train.to_csv('data/preprocessed/train_preprocessed.csv', sep=';', decimal=',', index=False)
print(x_train.shape)
y_train.to_csv('data/preprocessed/y_train_preprocessed.csv', sep=';', decimal=',', index=False)
x_test.to_csv('data/preprocessed/test_preprocessed.csv', sep=';', decimal=',', index=False)
y_test.to_csv('data/preprocessed/y_test_preprocessed.csv', sep=';', decimal=',', index=False)
