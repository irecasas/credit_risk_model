from typing import Optional

import pandas as pd
import regex
import statsmodels.api as sm
from pandas.api.types import is_datetime64_any_dtype as is_datetime
from pandas.api.types import is_numeric_dtype

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
        [
            "mes",
            "customer_id",
            "cancelled_at",
            "first_payment_date",
            "egr_over30mob3",
            "egr_mob3",
        ],
        axis=1,
        inplace=True,
    )
    validation = df.loc[(df["mes"] >= "2022-09") & (df["mes"] <= "2023-01")]
    validation.drop(
        [
            "mes",
            "customer_id",
            "cancelled_at",
            "first_payment_date",
            "egr_over30mob3",
            "egr_mob3",
        ],
        axis=1,
        inplace=True,
    )
    test = df.loc[(df["mes"] >= "2023-02")]
    test.drop(
        [
            "mes",
            "customer_id",
            "cancelled_at",
            "first_payment_date",
            "egr_over30mob3",
            "egr_mob3",
        ],
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

    :param dataframe:
    :return:
    """

    a = dataframe.shape[0]
    b = dataframe.shape[1]
    lim_50 = a * 50 / 100

    for i in dataframe:
        if pd.isnull(dataframe[i]).sum() > lim_50:
            dataframe.drop(i, 1, inplace=True)
            print("The variable ", i, " is dropped because it has more than 50% nulls")

    for i in dataframe:
        if len(dataframe[i].unique()) == 1:
            dataframe.drop(i, 1, inplace=True)
            print("The variable ", i, " is eliminated because it has a unique value")
    print(
        str(b - dataframe.shape[1])
        + " variables have been eliminated"
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


def dtypes_selector(df: pd.DataFrame) -> dict:
    """
    Function to split the dataset columns according to their types.

    Returns
    -------
    feature_types (dict)
        A dictionary whose keys are the different types
        the columns of the data set could be divided into
        and whose values are the corresponding columns.
    """
    feature_types = {}
    dt_features, string_features, bool_features, numeric_features = [], [], [], []

    string_features = df.select_dtypes(
        exclude=["datetimetz", "datetime", "timedelta", "number", "bool"]
    ).columns.tolist()
    dt_cols0 = df.select_dtypes(
        include=["datetimetz", "datetime", "timedelta"]
    ).columns.tolist()
    num_cols = df.select_dtypes(
        include=["number", "int", "float", "float64"]
    ).columns.tolist()
    bool_features = df.select_dtypes(include=["bool"]).columns.tolist()
    dt_cols1 = [
        c
        for c in df.columns
        if regex.search(
            regex.compile(
                r"(date(time\w*)?|time(stamp)?|firstseen|created(_at)?)$", regex.I
            ),
            c,
        )
    ]
    relevant_cols = set(
        string_features + dt_cols0 + dt_cols1 + num_cols + bool_features
    )
    for c in relevant_cols:
        if is_numeric_dtype(df[c]):
            numeric_features.append(c)
        else:
            if is_datetime(df[c]):
                dt_features.append(c)
            else:
                most_freq_vals = list(map(str, df[c].value_counts().index[:7]))
                clean_freqs = regex.sub(
                    regex.compile(r"_*(missing|(np\.)?na?n|none)_*", regex.I),
                    "",
                    "__".join(most_freq_vals),
                ).strip(r"[\s_]+")
                if regex.search(
                    r"^\d{2,4}[-\/:]\d{2}[-\/:]\d{2,4}[\s\d\:\.TZ\+]*(__\d{2,4}[-\/\:]\d{2}[-\/\:]\d{2,"
                    r"4}[\s\d:\.TZ\+]*){5,}$",
                    clean_freqs,
                ):
                    dt_features.append(c)
                elif len(
                    regex.findall(
                        regex.compile(r"(-?\d+|true|false)+", regex.I), clean_freqs
                    )
                ) < len(regex.findall(r"[a-z]+", clean_freqs)):
                    string_features.append(c)
                elif regex.search(
                    regex.compile(
                        "^((true|false|[a-zA-Z]+)__)((true|false)_*)+$", regex.I
                    ),
                    clean_freqs,
                ):
                    bool_features.append(c)
                else:
                    cond0 = (
                        len(regex.findall(r"-?\d+([\,\.]\d+)?", clean_freqs))
                        > len(
                            regex.findall(regex.compile("[a-z]+", regex.I), clean_freqs)
                        )
                        + 2
                    )
                    cond1 = regex.search(
                        r"(?<=_|)(([7698|s]\d{8,}|\d{5}|(\d__)+\d_|\d{3,5}X\d{3,5})(\.\d{1,2})?_*){5,}$",
                        clean_freqs,
                    )
                    if cond0 and not cond1:
                        numeric_features.append(c)
                    else:
                        string_features.append(c)

    bool_list = [
        "rej_reason7_ord",
        "campaign_completed_ord_bool",
        "ido_dropout_15d_ord_bool",
        "ido_completed_15d_ord_bool",
        "ido_accepted_30d_ord",
        "ido_cancelled_7d_ord_bool",
        "ido_rejected_ord",
        "rej_reason2_ord",
        "rej_reason5_ord",
        "campaign_rejected_ord",
        "ido_rejected_30d_ord",
        "n_rechazados_ord_48h_bool",
        "ido_dropout_1d_ord_bool",
        "campaign_dropout_ord_bool",
        "rej_reason1_ord",
        "ido_accepted_7d_ord",
        "ido_cancelled_30d_ord_bool",
        "ido_cancelled_ord",
        "ido_cancelled_1d_ord_bool",
        "rej_reason6_ord",
        "ido_accepted_15d_ord",
        "ido_chargeoff_ord",
        "ido_dropout_7d_ord_bool",
        "ido_completed_ord",
        "ido_dropout_30d_ord_bool",
        "stripe_ord_max",
        "ido_total_ord",
        "campaign_cancelled_ord_bool",
        "ido_accepted_1d_ord",
        "ido_dropout_ord",
        "rej_reason3_ord",
        "ido_rejected_1d_ord",
        "ido_rejected_7d_ord",
        "ido_rejected_15d_ord",
        "rej_reason4_ord",
        "ido_accepted_ord",
        "ido_completed_30d_ord_bool",
        "n_aceptados_ord_48h_bool",
        "ido_cancelled_15d_ord_bool",
        "target",
    ]

    bool_features = bool_features + bool_list
    bool_features = list(set(bool_features))

    string_features = string_features + ["merchant_id", "customer_id"]
    string_features = list(set(string_features))

    for c in relevant_cols:
        if c in string_features and c in dt_features:
            string_features.remove(c)
        elif c in string_features and c in bool_features:
            string_features.remove(c)
        elif c in string_features and c in numeric_features:
            numeric_features.remove(c)
        elif c in bool_features and c in numeric_features:
            numeric_features.remove(c)
        elif c in bool_features and c in string_features:
            bool_features.remove(c)

    feature_types["datetime"] = dt_features
    feature_types["string"] = string_features
    feature_types["boolean"] = bool_features
    feature_types["numerical"] = numeric_features

    return feature_types


def missing_values(df: pd.DataFrame, dict: dict) -> pd.DataFrame:
    list_columns = df.columns.to_list()
    for c in list_columns:
        if c in dict["boolean"]:
            df[c] = df[c].astype("object").fillna("0").astype(int)
            df[c] = df[c].fillna(0).astype(int)
        elif c in dict["numerical"]:
            df[c] = df[c].astype(float).fillna(-9999)


def age_customer(df: pd.DataFrame, date_birthday: str, date_init: str):
    """

    :param df:
    :param date_birthday:
    :param date_init:
    :return:
    """
    df["birthday"] = pd.to_datetime(df["birthday"], format="%Y-%m-%d").dt.date
    df["created_at"] = pd.to_datetime(df["created_at"], format="%Y-%m-%d").dt.date
    df["age"] = ((df["created_at"] - df["birthday"]).dt.days) / 365
    df["age"] = df["age"].fillna(0)
    df.drop(["birthday"], axis=1, inplace=True)


def corr_matrix(df: pd.DataFrame, var_remove: Optional):
    """

    :param df:
    :param var_remove:
    :return:
    """
    list_corr = df.select_dtypes(
        exclude=["datetime", "timedelta", "object", "bool"]
    ).columns.tolist()
    list_corr.remove(var_remove)
    mat_corr = df[list_corr].corr()
    mat_corr = mat_corr.reset_index()
    # mat_corr.to_csv('matriz_cor.csv', sep=';', decimal=',')
    l_list = []
    for columns in [col for col in df[list_corr].columns if col != "index"]:
        for j in range(mat_corr.shape[0]):
            if mat_corr[columns].name != mat_corr["index"][j] and (
                mat_corr[columns][j] >= 0.8 or mat_corr[columns][j] <= -0.8
            ):
                short_list = [mat_corr[columns].name, mat_corr["index"][j]]
                short_list.sort()
                if short_list not in l_list:
                    l_list.append(short_list)
    return l_list


def get_stats(df: pd.DataFrame, df_target, list_colums: list):
    """

    :param df:
    :param df_target:
    :param list_colums:
    :return:
    """
    results = sm.OLS(df_target, df[list_colums]).fit()
    print(results.summary())
