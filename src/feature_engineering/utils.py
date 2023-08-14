from typing import Optional

import pandas as pd

from feature_engineering import constants


def splitting_data(df: pd.DataFrame):
    """
    Name: splitting_data

    Description: Function to separate the dataset between:
    training, validation and testing.

    Parameters: - df <pandas.DataFrame>

    Return: -
    """
    target = constants.target
    df = df.loc[df["mes"] <= "2023-04"]
    train = df.loc[(df["mes"] >= "2020-01") & (df["mes"] <= "2022-08")]
    # Quito en entrenamiento las operaciones aceptadas
    # y canceladas para reducir falsos negativos
    train = train.loc[pd.isnull(train["cancelled_at"])]
    train.drop(
        ["mes", "status", "customer_id", "num_recibos", "cancelled_at"],
        axis=1,
        inplace=True,
    )
    validation = df.loc[(df["mes"] >= "2022-09") & (df["mes"] <= "2023-01")]
    validation.drop(
        ["mes", "status", "customer_id", "num_recibos", "cancelled_at"],
        axis=1,
        inplace=True,
    )
    test = df.loc[(df["mes"] >= "2023-02")]
    test.drop(
        ["mes", "status", "customer_id", "num_recibos", "cancelled_at"],
        axis=1,
        inplace=True,
    )

    var_list = list(train.columns)
    var_list.remove(target)

    x_train = train[var_list]
    y_train = train[target]
    x_val = validation[var_list]
    y_val = validation[target]
    x_test = test[var_list]
    y_test = test[target]

    return x_train, y_train, x_test, y_test, x_val, y_val


def initial_cleaning(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Name: initial_cleaning

    Description: Eliminates variables with unique values

    Parameters: - df <pandas.DataFrame>

    Return: - df <pandas.DataFrame>
    """
    a = dataframe.shape[1]
    for i in dataframe:
        if len(dataframe[i].unique()) == 1:
            dataframe.drop(i, 1, inplace=True)
            print("The variable ", i, " is eliminated because it has a unique value")
    print(
        str(a - dataframe.shape[1])
        + " variables have been eliminated as they have unique values"
    )
    print(dataframe.shape)

    return dataframe


def correlation_matrix(
    df: pd.DataFrame, list_vars: Optional[list] = []
) -> pd.DataFrame:
    """
    Name: correlation matrix

    Description: Function that returns the correlation matrix between
    two variables according to Pearson's method.
    If the absolute value of the Pearson coefficient between two variables exceeds 0.8,
    collinearity exists.

    Parameters: - df <pandas.DataFrame>

    Return: - df <pandas.DataFrame>
    """

    cols_names = df.columns
    cols_names.remove(list_vars)
    df_aux = df[cols_names].corr(method="pearson")

    return df_aux
