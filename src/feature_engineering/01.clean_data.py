import os

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from feature_engineering import utils
from feature_engineering.constants import delete_correlation, feature_importance
from feature_engineering.utils import (
    dtypes_selector,
    initial_cleaning,
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

# @TODO Para reporte, eliminar despues. Para ver proporciones de impagos
# tabla1 = pd.DataFrame(
#     {"Total Orders": df.groupby("mes")["order_uuid"].count()}
# ).reset_index()
# tabla2 = pd.DataFrame(
#     {
#         "Total Default": df.groupby("mes").apply(
#             lambda x: x[(x["target"] == 1)]["order_uuid"].count()
#         )
#     }
# ).reset_index()
# df1 = pd.merge(tabla1, tabla2, how="left", on="mes")
# df1["%Default"] = (df1["Total Default"] / df1["Total Orders"]) * 100

# First cleaning data
df = initial_cleaning(df)

# missing treatment
dic_vars = dtypes_selector(df)
missing_values(df, dic_vars)

# Splitting data
x_train, y_train, x_test, y_test, x_val, y_val = splitting_data(df)

# standardization of data
# Continuous variables:
x_train = standardization_continuous_features(x_train, dic_vars)
# One Hot:
x_train = onehot_encoding(x_train)
x_train = woe_encoder(x_train, y_train)

# CONTEOS DE NULOS POR VARIABLES
for i in x_train:
    if pd.isnull(x_train[i]).sum() > 0:
        print("La variable ", i, " tiene ", pd.isnull(x_train[i]).sum(), " valores nulos")

# Dimension Reduction of Experimental Dataset Based on Pdc-RF Algorithm
# Random Forest
features = list(x_train.columns)
features.remove('order_uuid')

rf = RandomForestClassifier(n_estimators=500, n_jobs=1, max_depth=10).fit(x_train[features], y_train)
rf_vip = pd.DataFrame(list(zip(x_train.columns, rf.feature_importances_)),
                      columns=['variables', 'importance']).\
    sort_values(by=['importance'], ascending=False, axis=0).set_index('variables')
# feature_importance = rf_vip.loc[rf_vip.importance>= 0.01]
# feature_importance = list(feature_importance.index)
# feature_importance.append('order_uuid')

x_train = x_train[feature_importance]


# Matriz de correlación
utils.corr_matrix(x_train)

# Drop columns by correlation
x_train.drop(delete_correlation, axis=1, inplace=True)
# x_test.drop(delete_correlation, axis=1, inplace=True)
# x_val.drop(delete_correlation, axis=1, inplace=True)

# Stepwise (below 0.05)
x_colums = x_train.columns.to_list()
x_colums.remove("order_uuid")

utils.get_stats(x_train, y_train, x_colums)
x_colums.remove("emailage_response__domainrisklevel")  # Primera eliminada
utils.get_stats(x_train, y_train, x_colums)
x_colums.remove("merchant_id")  # Segunda eliminada
utils.get_stats(x_train, y_train, x_colums)
x_colums.remove("emailage_response__ip_userType_dialup")
utils.get_stats(x_train, y_train, x_colums)
x_colums.remove("minfraud_response__ip_address__traits__user_type_business")
utils.get_stats(x_train, y_train, x_colums)
x_colums.remove("minfraud_response__ip_address__continent__code_nan")
utils.get_stats(x_train, y_train, x_colums)
x_colums.remove("minfraud_response__ip_address__traits__user_type_nan")
utils.get_stats(x_train, y_train, x_colums)
x_colums.remove("minfraud_response__ip_address__traits__user_type_hosting")
utils.get_stats(x_train, y_train, x_colums)

x_colums.append('order_uuid')
x_train = x_train[x_colums]

#############################
# Fisher's Exact Test
#############################

# for c in x_train[x_colums]:
#    col = x_train[c]
# data = pd.merge(col, y_train, how='left', left_index = True, right_index = True)
#    pd.crosstab(y_train, col)

# Perform Fisher's Exact Test
# odds_ratio, p_value = fisher_exact(data)
# Output the results
# print(f"Odds Ratio: {odds_ratio}")
# print(f"P-value: {p_value}")
# Interpret the results
# alpha = 0.05
# if p_value < alpha:
# print("Reject the null hypothesis -
# There is a significant association between the treatment types and success rates.")
# else:
#    print("Fail to reject the null hypothesis -
#    There is no significant association between the treatment types and success rates.")
