import os

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

# Read data
cwd = "/Users/irene.casas/PycharmProjects/credit_risk_model/"
path_data = "data/raw/dataset_total_2020_2023_raw.parquet"
df = pd.read_csv(os.path.join(cwd, path_data), low_memory=False)
df.drop(['uuid_y', 'Unnamed: 0'], axis=1, inplace=True)
# Quito los meses que aún no tengo target
df = df.loc[(df.mes != '2023-05') & (df.mes != '2023-06')]


# GRAFICO PAPER
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


sns.set(style="white", rc={"lines.linewidth": 3})
fig, ax1 = plt.subplots(figsize=(10, 10))
ax2 = ax1.twinx()

sns.barplot(x=df1['mes'],
            y=df1['Total Orders'],
            color='#6ab0de',
            ax=ax1)

sns.lineplot(x=df1['mes'],
             y=df1['%Default'],
             color='#edbd2d',
             marker="o",
             ax=ax2)
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=90)
plt.show()
sns.set()
