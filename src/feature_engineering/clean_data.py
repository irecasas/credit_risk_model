import os

import pandas as pd

from feature_engineering import utils
from feature_engineering.constants import list_delete, var_drop

# Load the dataset
cwd = "/Users/irene.casas/PycharmProjects/credit_risk_model/"
path_data = "data/raw/dataset_total_accepted_2020_2023.parquet"
df = pd.read_parquet(os.path.join(cwd, path_data))
df.drop(var_drop, axis=1, inplace=True)

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

df = utils.initial_cleaning(df)
dic_vars = utils.dtypes_selector(df)
utils.missing_values(df, dic_vars)
utils.age_customer(df, "birthday", "created_at")

# CONTEOS DE NULOS POR VARIABLES
for i in df:
    if pd.isnull(df[i]).sum() > 0:
        print("La variable ", i, " tiene ", pd.isnull(df[i]).sum(), " valores nulos")

x_train, y_train, x_test, y_test, x_val, y_val = utils.splitting_data(df)

utils.corr_matrix(x_train, "merchant_id")

x_train.drop(list_delete, axis=1, inplace=True)
x_test.drop(list_delete, axis=1, inplace=True)
x_val.drop(list_delete, axis=1, inplace=True)

utils.corr_matrix(x_train, "merchant_id")

x_colums = x_train.columns.to_list()
x_colums.remove("order_uuid")
x_colums.remove("merchant_id")
x_colums.remove("created_at")
x_colums.remove("n_total_ord")  # Primera eliminada
x_colums.remove("ido_cancelled_15d_ord_bool")
x_colums.remove("campaign_cancelled_ord_bool")
x_colums.remove("ido_completed_30d_ord_bool")
x_colums.remove("avg_principal_dropout_1d_ord")
x_colums.remove("ido_cancelled_1d_ord_bool")

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