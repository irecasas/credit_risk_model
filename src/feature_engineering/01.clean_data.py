import os

import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.metrics import confusion_matrix

# from scipy.stats import chi2_contingency
from sklearn.preprocessing import KBinsDiscretizer

from feature_engineering import utils
from feature_engineering.constants import delete_correlation, feature_importance
from feature_engineering.utils import (
    dtypes_selector,
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

# CONTEOS DE NULOS POR VARIABLES
# for i in x_train:
#    if pd.isnull(x_train[i]).sum() > 0:
#        print("La variable ", i, " tiene ", pd.isnull(x_train[i]).sum(), " valores nulos")
# for i in x_val:
#    if pd.isnull(x_val[i]).sum() > 0:
#        print("La variable ", i, " tiene ", pd.isnull(x_val[i]).sum(), " valores nulos")

# Dimension Reduction of Experimental Dataset Based on Pdc-RF Algorithm
# Random Forest
# features = list(x_train.columns)
# features.remove('order_uuid')
# feature_importance = feature_selection_rf(x_train, features, y_train)

x_train = x_train[feature_importance]
x_test = x_test[feature_importance]

# Matriz de correlación
# utils.corr_matrix(x_train)

# Drop columns by correlation
x_train.drop(delete_correlation, axis=1, inplace=True)
x_test.drop(delete_correlation, axis=1, inplace=True)

# Stepwise (below 0.05)
features = x_train.columns.to_list()
features.remove("order_uuid")

utils.get_stats(x_train, y_train, features)
features.remove("emailage_response__domainrisklevel")  # Primera eliminada
features.remove("minfraud_response__ip_address__traits__user_type_business")
features.remove("merchant_id")
features.remove("minfraud_response__ip_address__continent__code_nan")
features.remove("minfraud_response__ip_address__traits__user_type_nan")
features.append('order_uuid')

x_train = x_train[features]
x_test = x_test[features]

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

#############################
# Paper 1
#############################

features = x_train.columns.to_list()
features.remove("order_uuid")

x_train = x_train[features]
x_test = x_test[features]

linear_discriminant = LinearDiscriminantAnalysis()
# User RepeatedStratifiedKFold because is an imbalance class
# cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# scores = cross_val_score(model_lda, x_train, y_train, scoring='accuracy', cv=cv, n_jobs=-1)
model_lda = linear_discriminant.fit(x_train, y_train)

print(model_lda.intercept_, model_lda.coef_)
y_pred = model_lda.predict(x_test)
cm = confusion_matrix(y_pred, y_test)

#############################
# Paper 2
#############################

# Teniendo toda la parte de one hot  y woe enconder,es necesaria la discretización?
# Añadir paper explicándola
continuous_vars = ['principal', 'downpayment_amount', 'age']
est = KBinsDiscretizer(encode='ordinal', strategy='uniform')
x_train[continuous_vars] = est.fit_transform(x_train[continuous_vars])
x_test[continuous_vars] = est.transform(x_test[continuous_vars])

# IV values greater than 0.02 are selected
x_train = x_train.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)
df = pd.merge(x_train, y_train, how='left', left_index=True, right_index=True)
features = df.columns.to_list()
features.remove('order_uuid')
iv, woe = iv_woe(data=df[features], target='target')

delete_iv = ['experian_ResumenCais__NumeroOperacionesImpagadas',
             'emailage_response__ip_risklevelid',
             'experian_documento__tipodocumento__descripcion_trans',
             'experian_ResumenCais__MaximoImporteImpagado',
             'experian_ResumenCais__NumeroCuotasImpagadas',
             'minfraud_response__ip_address__traits__user_type_hosting']
x_train.drop(delete_iv, axis=1, inplace=True)
x_test.drop(delete_iv, axis=1, inplace=True)

x_train.to_csv('data/preprocessed/train_preprocessed.csv', sep=';', decimal=',', index=False)
print(x_train.shape)
y_train.to_csv('data/preprocessed/y_train_preprocessed.csv', sep=';', decimal=',', index=False)
x_test.to_csv('data/preprocessed/test_preprocessed.csv', sep=';', decimal=',', index=False)
y_test.to_csv('data/preprocessed/y_test_preprocessed.csv', sep=';', decimal=',', index=False)
