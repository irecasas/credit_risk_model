import os

import pandas as pd

from feature_engineering.constants import var_drop, var_pending_analysis
from feature_engineering.new_features import (
    age_customer,
    email_features,
    experian_features,
    industries,
    minfraud_features,
)

# Load the dataset
cwd = "/Users/irene.casas/PycharmProjects/credit_risk_model/"
path_data = "data/raw/dataset_total_2020_2023.parquet"
df = pd.read_parquet(os.path.join(cwd, path_data))

# Drop features
df.drop(var_drop, axis=1, inplace=True)
df.drop(var_pending_analysis, axis=1, inplace=True)
for feature_name in ("minfraud_response__ip_address__country__names",
                     "minfraud_response__ip_address__city__names",
                     "minfraud_response__ip_address__continent__names",
                     "minfraud_response__ip_address__registered_country__names",
                     "minfraud_response__ip_address__subdivisions__0__names",
                     "minfraud_response__ip_address__subdivisions__1__names"):
    del_vars = df.columns[df.columns.str.startswith(feature_name)]
    df.drop(del_vars, axis=1, inplace=True)

del_vars = df.columns[df.columns.str.startswith('iovation')]
del_vars = del_vars.to_list()
del_vars.remove('iovation_ruleResults__score')
del_vars.remove('iovation_statedIp__botnet__score')
del_vars.remove('iovation_realIp__botnet__score')
df.drop(del_vars, axis=1, inplace=True)

# Create new features
minfraud_features(df)
age_customer(df, "birthday", "created_at")
email_features(df)
experian_features(df)
# iovation_features(df)
# time_reference(df, created_at)
industries(df)

# Save data
path_data = "data/raw/dataset_total_2020_2023_raw.parquet"
df.to_csv(os.path.join(cwd, path_data))
