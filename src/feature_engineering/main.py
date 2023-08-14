import pandas as pd

from feature_engineering import utils

# Total data, incluye denegados (data/raw/dataset_total_2020_2023.parquet)
# Usaremos solicitudes aceptadas para entrenar
path_data = "data/raw/dataset_total_accepted_2020_2023.parquet"
df = pd.read_parquet(path_data)

# Para ver proporciones de impagos
tabla1 = pd.DataFrame(
    {"Total Orders": df.groupby("mes")["order_uuid"].count()}
).reset_index()
tabla2 = pd.DataFrame(
    {
        "Total Default": df.groupby("mes").apply(
            lambda x: x[(x["target"] == 1)]["order_uuid"].count()
        )
    }
).reset_index()
df1 = pd.merge(tabla1, tabla2, how="left", on="mes")
df1["%Default"] = (df1["Total Default"] / df1["Total Orders"]) * 100

x_train, y_train, x_test, y_test, x_val, y_val = utils.splitting_data(df)

x_train = utils.initial_cleaning(x_train)
x_val = utils.initial_cleaning(x_val)
