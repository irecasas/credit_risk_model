import os

import numpy as np
import pandas as pd
from category_encoders.woe import WOEEncoder
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder

from feature_engineering import utils
from feature_engineering.constants import delete_correlation

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
df = utils.initial_cleaning(df)

# missing treatment
dic_vars = utils.dtypes_selector(df)
utils.missing_values(df, dic_vars)

# standardization of data
num_vars = []
for c in df:
    if c in dic_vars['numerical'] or c in dic_vars['numerical_trans']:
        num_vars.append(c)
binary_vars = []
for c in df:
    if c in dic_vars['boolean'] or c in dic_vars['binary']:
        binary_vars.append(c)
cat_vars = list(set(df.columns.to_list()).difference(num_vars))
cat_vars = list(set(cat_vars).difference(binary_vars))
base_vars = ['cancelled_at', 'mes', 'egr_over30mob3', 'order_uuid', 'created_at', 'egr_mob3']
cat_vars = list(set(cat_vars).difference(base_vars))

df_cat = df[cat_vars]
df_num = df[num_vars]

# continuous variables (min-max std)
min_max_scaler = preprocessing.MinMaxScaler()
df_num_minmax = min_max_scaler.fit_transform(df_num)
df_num_minmax = pd.DataFrame(df_num_minmax, columns=df_num.columns)

df.drop(num_vars, axis=1, inplace=True)
df = pd.merge(df, df_num_minmax, how='left', left_index=True, right_index=True)

# category variables - one hot encoder and WOE encoder
one_hot_vars = ['emailage_response__ip_userType', 'emailage_response__status',
                'minfraud_response__ip_address__traits__user_type',
                'experian_OrigenScore__Codigo', 'experian_Documento__TipoDocumento__Codigo',
                'emailage_response__ip_netSpeedCell', 'emailage_response__domaincategory',
                'emailage_response__domainrelevantinfoID',
                'minfraud_response__ip_address__continent__code']
woe_encoder = ['emailage_response__fraudRisk', 'experian_InformacionDelphi__Nota',
               'emailage_response__domainrisklevel',
               'emailage_response__EAAdvice', 'industry_id', 'merchant_id', 'emailage_response__domainname',
               'emailage_response__phonecarriername', 'minfraud_response__ip_address__traits__organization']

# ONEHOT
enc = OneHotEncoder()
enc_data = pd.DataFrame(enc.fit_transform(df_cat[one_hot_vars]).toarray(),
                        columns=enc.get_feature_names_out(one_hot_vars))
df.drop(one_hot_vars, axis=1, inplace=True)
df = pd.merge(df, enc_data, how='left', left_index=True, right_index=True)

# WOE ENCODER
WOE_encoder = WOEEncoder()
df_woe = WOE_encoder.fit_transform(df[woe_encoder], df.target)


# CONTEOS DE NULOS POR VARIABLES
# for i in df:
#    if pd.isnull(df[i]).sum() > 0:
#        print("La variable ", i, " tiene ", pd.isnull(df[i]).sum(), " valores nulos")

x_train, y_train, x_test, y_test, x_val, y_val = utils.splitting_data(df)

# preprocessing = Pipeline(steps=[('label_encoder', CustomLabelEncoder())])
# x_train = preprocessing.fit_transform(x_train, y_train)
# x_test = preprocessing.transform(x_test)
# x_val = preprocessing.transform(x_val)


# Drop columns by correlation
x_train.drop(delete_correlation, axis=1, inplace=True)
# x_test.drop(delete_correlation, axis=1, inplace=True)
# x_val.drop(delete_correlation, axis=1, inplace=True)
utils.corr_matrix(x_train)

# Stepwise (below 0.05)
x_colums = x_train.columns.to_list()
x_colums.remove("order_uuid")
x_colums.remove("created_at")

utils.get_stats(x_train, y_train, x_colums)
x_colums.remove("emailage_response__domainrisklevel")  # Primera eliminada
utils.get_stats(x_train, y_train, x_colums)
x_colums.remove("minfraud_response__billing_address__is_in_ip_country_trans")  # Segunda eliminada
utils.get_stats(x_train, y_train, x_colums)
x_colums.remove("emailage_response__ip_anonymousdetected_trans")
utils.get_stats(x_train, y_train, x_colums)
x_colums.remove("minfraud_response__shipping_address__is_in_ip_country_trans")
utils.get_stats(x_train, y_train, x_colums)
x_colums.remove("email_exists")
utils.get_stats(x_train, y_train, x_colums)
x_colums.remove("minfraud_response__email__is_disposable_trans")
utils.get_stats(x_train, y_train, x_colums)
x_colums.remove("emailage_response__shipforward_trans")
utils.get_stats(x_train, y_train, x_colums)
x_colums.remove("minfraud_response__ip_address__registered_country__is_in_european_union_trans")
utils.get_stats(x_train, y_train, x_colums)
x_colums.remove("experian_autonomo_trans")
utils.get_stats(x_train, y_train, x_colums)
x_colums.remove("emailage_response__domain_creation_days")
utils.get_stats(x_train, y_train, x_colums)
x_colums.remove("emailage_response__domainriskcountry_trans")
utils.get_stats(x_train, y_train, x_colums)
x_colums.remove("emailage_response__gender_female")
utils.get_stats(x_train, y_train, x_colums)
x_colums.remove("minfraud_response__email__is_high_risk_trans")
utils.get_stats(x_train, y_train, x_colums)

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


df.merchant_id = df.merchant_id.astype('object')

y = y_train
cat_cols = ['merchant_id']
encoder = WOEEncoder(cols=cat_cols, handle_unknown="value", handle_missing="value", return_df=True)
encoder.fit(x_train[cat_cols], y)

cat_cols_diff = list(set(cat_cols).difference(set(x_train.columns)))
cat_cols_remain = list(set(x_train.columns).intersection(set(cat_cols)))

if cat_cols_diff:
    to_add = pd.DataFrame(np.full((len(cat_cols_diff), x_train.shape[0]), "missing"), index=cat_cols_diff,
                          columns=x_train.index).T
    to_encode = x_train[cat_cols_remain].merge(to_add, left_index=True, right_index=True)
else:
    to_encode = x_train[cat_cols_remain]

cats = encoder.transform(to_encode)
df = pd.concat([x_train[set(x_train.columns).difference(set(cat_cols_remain))], cats], axis=1)

############################################################################################################
#############################################################################################################
############################################################################################################

df = df.head(1000)
cols_emailage = df.columns[df.columns.str.startswith("emailage")]
cols_emailage = cols_emailage.to_list()
cols_emailage.append('created_at')
cols_emailage.append('order_uuid')
df = df[cols_emailage]
