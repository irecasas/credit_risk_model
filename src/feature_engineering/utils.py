from statistics import mode

import pandas as pd
import statsmodels.api as sm
from category_encoders.woe import WOEEncoder
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder

from feature_engineering import constants


def splitting_data(df: pd.DataFrame):
    """

    :param df:
    :return:
    """
    target = constants.name_target
    df = df.loc[df["mes"] <= "2023-04"]
    train = df.loc[(df["mes"] >= "2020-01") & (df["mes"] <= "2022-08")]
    # Quito en entrenamiento las operaciones aceptadas
    # y canceladas para reducir falsos negativos
    train = train.loc[pd.isnull(train["cancelled_at"])]
    train.drop(
        [
            "mes",
            "cancelled_at",
            "egr_over30mob3",
            "egr_mob3",
            "created_at"
        ],
        axis=1,
        inplace=True,
    )
    validation = df.loc[(df["mes"] >= "2022-09") & (df["mes"] <= "2023-01")]
    validation.drop(
        [
            "mes",
            "cancelled_at",
            "egr_over30mob3",
            "egr_mob3",
            "created_at"
        ],
        axis=1,
        inplace=True,
    )
    test = df.loc[(df["mes"] >= "2023-02")]
    test.drop(
        [
            "mes",
            "cancelled_at",
            "egr_over30mob3",
            "egr_mob3",
            "created_at"
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
    lim_95 = a * 50 / 100
    j = 'cancelled_at'

    for i in dataframe:
        if (pd.isnull(dataframe[i]).sum() > lim_50) and (str(i) != j):
            dataframe.drop(i, 1, inplace=True)
            print("The variable ", i, " is dropped because it has more than 50% nulls")

    for i in dataframe:
        if len(dataframe[i].unique()) == 1:
            dataframe.drop(i, 1, inplace=True)
            print("The variable ", i, " is eliminated because it has a unique value")

    for i in dataframe:
        if (pd.isnull(dataframe[i]).sum() > lim_95) and (str(i) != j):
            dataframe.drop(i, 1, inplace=True)
            print("The variable ", i, " is dropped because it has more than 95% nulls")

    print(
        str(b - dataframe.shape[1])
        + " variables have been eliminated"
    )
    print(dataframe.shape)

    return dataframe


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
    feature_types = dict(
        boolean=['is_offline'],
        binary=['emailage__ip_reputation', 'target',
                'emailage_response__domainriskcountry_trans',
                'emailage_response__domaincorporate_trans',
                'emailage_response__ip_corporateProxy_trans',
                'emailage_response__ip_anonymousdetected_trans',
                'emailage_response__domainExists_trans',
                'emailage_response__ipcountrymatch_trans',
                'emailage_response__shipforward_trans',
                'emailage_response__gender_female',
                'emailage_phone_status',
                'minfraud_response__ip_address__registered_country__is_in_european_union_trans',
                'minfraud_response__email__is_disposable_trans',
                'minfraud_response__email__is_high_risk_trans',
                'minfraud_response__email__is_free_trans',
                'minfraud_response__shipping_address__is_postal_in_city_trans',
                'minfraud_response__shipping_address__is_in_ip_country_trans',
                'minfraud_response__billing_address__is_postal_in_city_trans',
                'minfraud_response__billing_address__is_in_ip_country_trans',
                'minfraud_response__ip_address__traits__is_anonymous_trans',
                'email_exists',
                'experian_autonomo_trans',
                'experian_documento__tipodocumento__descripcion_trans'],
        numerical_trans=['experian_ResumenCais__FechaAltaOperacionMasAntigua_difference_with_created',
                         'experian_ResumenCais__FechaMaximoImporteImpagado_difference_with_created',
                         'experian_ResumenCais__FechaPeorSituacionPagoHistorica_difference_with_created',
                         'emailage_response__lastVerificationDate_difference_with_created',
                         'emailage_response__firstVerificationDate_difference_with_created'],
        category=['emailage_response__status', 'emailage_response__fraudRisk',
                  'minfraud_response__ip_address__traits__user_type',
                  'emailage_response__ip_netSpeedCell', 'emailage_response__ip_userType',
                  'emailage_response__domaincategory',
                  'minfraud_response__ip_address__traits__organization',
                  'experian_InformacionDelphi__Nota',
                  'emailage_response__domainname', 'emailage_response__EAAdvice', 'merchant_id',
                  'industry_id',
                  'minfraud_response__ip_address__continent__code',
                  'emailage_response__phonecarriername',
                  'emailage_response__domainrisklevel', 'experian_OrigenScore__Codigo',
                  'emailage_response__domainrelevantinfoID',
                  'experian_Documento__TipoDocumento__Codigo'],
        numerical=['minfraud_response__risk_score', 'emailage_response__ip_risklevelid',
                   'experian_InformacionDelphi__Percentil',
                   'experian_ResumenCais__NumeroCuotasImpagadas',
                   'emailage_response__domain_creation_days',
                   'experian_InformacionDelphi__Puntuacion', 'experian_InformacionDelphi__Decil',
                   'age', 'emailage_response__EAScore',
                   'experian_ResumenCais__MaximoImporteImpagado',
                   'experian_ResumenCais__NumeroOperacionesImpagadas',
                   'emailage_response__EARiskBandID', 'n_initial_instalments',
                   'emailage_response__domainrisklevelID',
                   'experian_ResumenCais__ImporteImpagado', 'principal', 'downpayment_amount',
                   'experian_InformacionDelphi__ProbabilidadIncumplimientoPorScore',
                   'emailage_response__first_seen_days',
                   'annual_percentage_rate'])

    return feature_types


def missing_values(df: pd.DataFrame, dict: dict) -> pd.DataFrame:
    list_columns = df.columns.to_list()
    for c in list_columns:
        if c in dict["boolean"]:
            df[c] = df[c].astype("object").fillna("0").astype(int)
            df[c] = df[c].fillna(0).astype(int)
        elif c in dict["numerical"]:
            df[c] = df[c].astype(float)
            inf_mode = mode(df[c])
            df[c] = df[c].fillna(inf_mode)
        elif c in dict["binary"]:
            df[c] = df[c].astype(int)
            df[c] = df[c].fillna(0).astype(int)
        elif c in dict["numerical_trans"]:
            df[c] = df[c].astype(int)
            df[c] = df[c].fillna(0).astype(int)


def corr_matrix(df: pd.DataFrame):
    """
    :type dict: object
    :param df:
    :param dict:
    :return:
    """
    list_columns = df.columns.to_list()
    var_remove = ['order_uuid', 'created_at']
    list_columns = [i for i in list_columns if i not in var_remove]
    mat_corr = df[list_columns].corr()
    mat_corr = mat_corr.reset_index()
    mat_corr.to_csv('matriz_correlacion.csv', sep=';', decimal=',')
    l_list = []
    for columns in [col for col in df[list_columns].columns if col != "index"]:
        for j in range(mat_corr.shape[0]):
            if (mat_corr[columns].name != mat_corr["index"][j]) and (
                    mat_corr[columns][j] >= 0.8 or mat_corr[columns][j] <= -0.8):
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
    y = list(df_target)
    results = sm.OLS(y, df[list_colums]).fit()
    print(results.summary())


def standardization_continuous_features(df: pd.DataFrame, dictionary: dict):
    num_vars = []
    for c in df:
        if c in dictionary['numerical'] or c in dictionary['numerical_trans']:
            num_vars.append(c)
    df_num = df[num_vars]

    # Min-Max Standardization
    min_max_scaler = preprocessing.MinMaxScaler()
    df_num_minmax = min_max_scaler.fit_transform(df_num)
    df_num_minmax = pd.DataFrame(df_num_minmax, columns=df_num.columns)

    df.drop(num_vars, axis=1, inplace=True)
    df = df.reset_index(drop=True)
    df = pd.merge(df, df_num_minmax, how='left', left_index=True, right_index=True)

    return df


def onehot_encoding(df: pd.DataFrame):

    one_hot_vars = ['emailage_response__ip_userType',
                    'emailage_response__status',
                    'minfraud_response__ip_address__traits__user_type',
                    'experian_OrigenScore__Codigo',
                    'experian_Documento__TipoDocumento__Codigo',
                    'emailage_response__ip_netSpeedCell',
                    'emailage_response__domaincategory',
                    'emailage_response__domainrelevantinfoID',
                    'minfraud_response__ip_address__continent__code']

    # One Hot Encoder
    enc = OneHotEncoder()
    enc_data = pd.DataFrame(enc.fit_transform(df[one_hot_vars]).toarray(),
                            columns=enc.get_feature_names_out(one_hot_vars))
    df.drop(one_hot_vars, axis=1, inplace=True)
    df = pd.merge(df, enc_data, how='left', left_index=True, right_index=True)

    return df

def woe_encoder(df: pd.DataFrame, y_target):

    woe_vars = ['emailage_response__fraudRisk', 'experian_InformacionDelphi__Nota',
                'emailage_response__domainrisklevel',
                'emailage_response__EAAdvice', 'industry_id', 'merchant_id', 'emailage_response__domainname',
                'emailage_response__phonecarriername', 'minfraud_response__ip_address__traits__organization']
    woe_encoder = WOEEncoder()
    df = df.reset_index(drop=True)
    y_target = y_target.reset_index(drop=True)
    df_woe = woe_encoder.fit_transform(df[woe_vars], y_target)
    df.drop(woe_vars, axis=1, inplace=True)
    df = pd.merge(df, df_woe, how='left', left_index=True, right_index=True)

    return df
