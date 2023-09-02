import pickle
from statistics import mode

import numpy as np
import pandas as pd
import statsmodels.api as sm
from category_encoders.woe import WOEEncoder
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder

from feature_engineering import constants


def splitting_data(df: pd.DataFrame):
    """

    :param df:
    :return:
    """
    target = constants.name_target
    train = df.loc[(df["mes"] >= "2020-01") & (df["mes"] <= "2022-10")]
    # Quito en entrenamiento las operaciones aceptadas
    # y canceladas para reducir falsos negativos
    train = train.loc[pd.isnull(train["cancelled_at"])]
    train.drop(
        [
            "mes",
            "uuid",
            "cancelled_at",
            "egr_over30mob3",
            "egr_mob3",
            "created_at"
        ],
        axis=1,
        inplace=True,
    )
    test = df.loc[(df["mes"] >= "2022-11")]
    test.drop(
        [
            "mes",
            "uuid",
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
    x_test = test[var_list]
    y_test = test[target]

    return x_train, y_train, x_test, y_test


def initial_cleaning(dataframe: pd.DataFrame) -> pd.DataFrame:
    """

    :param dataframe:
    :return:
    """

    a = dataframe.shape[0]
    b = dataframe.shape[1]
    a * 50 / 100
    lim_95 = a * 50 / 100
    j = 'cancelled_at'

    # for i in dataframe:
    #    if (pd.isnull(dataframe[i]).sum() > lim_50) and (str(i) != j):
    #        dataframe.drop(i, 1, inplace=True)
    #        print("The variable ", i, " is dropped because it has more than 50% nulls")

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
                'experian_documento__tipodocumento__descripcion_trans',
                'iovation_device__isNew_trans', 'iovation_device__browser__cookiesEnabled_trans',
                'high_risk_industry'],
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
                  'experian_Documento__TipoDocumento__Codigo',
                  'iovation_device__type', 'iovation_device__browser__type',
                  'iovation_device__os', 'iovation_realIp__source',
                  'iovation_device__blackboxMetadata__age',
                  'iovation_ruleResults__rulesMatched'],
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
                   'annual_percentage_rate', 'asnef-score', 'asnef-number_of_consumer_credit_operations',
                   'asnef-total_number_operations', 'asnef-total_unpaid_balance',
                   'asnef-number_of_telco_operations', 'industry_orders_timeline-num_orders',
                   'industry_orders_timeline-ptg_checkout_origin_offline',
                   'industry_orders_timeline-ptg_orders_dropout',
                   'industry_orders_timeline-ptg_orders_completed',
                   'industry_orders_timeline-ptg_orders_accepted_cancelled',
                   'industry_orders_timeline-ptg_orders_rejected_severity_high',
                   'industry_orders_timeline-ptg_orders_rejected_downpayment',
                   'industry_orders_timeline-ptg_orders_accepted_no_payments_among_accepted',
                   'industry_orders_timeline-ptg_orders_with_defaults_among_accepted',
                   'merchant_loanbook_timeline-seniority',
                   'merchant_loanbook_timeline-principal_lent',
                   'merchant_loanbook_timeline-ptg_principal_accrued_paid',
                   'merchant_loanbook_timeline-ptg_principal_not_accrued',
                   'merchant_loanbook_timeline-ptg_cancels_principal',
                   'merchant_loanbook_timeline-ptg_refunds_principal',
                   'merchant_loanbook_timeline-ptg_prepayments_principal',
                   'merchant_loanbook_timeline-ptg_campaigns_amount',
                   'merchant_loanbook_timeline-ptg_principal_accrued_unpaid_among_accrued',
                   'merchant_loanbook_timeline-diff_principal_lent',
                   'merchant_orders_timeline-num_orders',
                   'merchant_orders_timeline-ptg_checkout_origin_offline',
                   'merchant_orders_timeline-ptg_orders_dropout',
                   'merchant_orders_timeline-ptg_orders_completed',
                   'merchant_orders_timeline-ptg_orders_accepted_cancelled',
                   'merchant_orders_timeline-ptg_orders_rejected_severity_high',
                   'merchant_orders_timeline-ptg_orders_rejected_downpayment',
                   'merchant_orders_timeline-ptg_orders_accepted_no_payments_among_accepted',
                   'merchant_orders_timeline-ptg_orders_with_defaults_among_accepted',
                   'merchant_orders_timeline-diff_num_orders',
                   'merchant_orders_timeline-ptg_diff_orders_rejected_severity_high',
                   'iovation_ruleResults__score'])

    return feature_types


def missing_values(df: pd.DataFrame, dict: dict) -> pd.DataFrame:
    list_columns = df.columns.to_list()
    for c in list_columns:
        if c in dict["boolean"]:
            df[c] = df[c].astype("object").fillna("0").astype(int)
            df[c] = df[c].fillna(0).astype(int)
        elif (c in dict["numerical"]) or \
                (c in ['iovation_device__blackboxMetadata__age', 'iovation_ruleResults__rulesMatched']):
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
                    'experian_OrigenScore__Codigo',
                    'experian_Documento__TipoDocumento__Codigo',
                    'emailage_response__ip_netSpeedCell',
                    'emailage_response__domaincategory',
                    'emailage_response__domainrelevantinfoID',
                    'minfraud_response__ip_address__continent__code'
                    ]

    # One Hot Encoder
    enc = OneHotEncoder()
    enc_data = pd.DataFrame(enc.fit_transform(df[one_hot_vars]).toarray(),
                            columns=enc.get_feature_names_out(one_hot_vars))
    df.drop(one_hot_vars, axis=1, inplace=True)
    df = pd.merge(df, enc_data, how='left', left_index=True, right_index=True)

    return df


def woe_encoder(df: pd.DataFrame, y_target, df_validation: pd.DataFrame):
    woe_vars = ['emailage_response__fraudRisk', 'experian_InformacionDelphi__Nota',
                'emailage_response__domainrisklevel',
                'emailage_response__EAAdvice', 'industry_id', 'merchant_id', 'emailage_response__domainname',
                'emailage_response__phonecarriername', 'minfraud_response__ip_address__traits__organization',
                'minfraud_response__ip_address__traits__user_type']
    woe_encoder = WOEEncoder()

    df = df.reset_index(drop=True)
    y_target = y_target.reset_index(drop=True)
    df_validation = df_validation.reset_index(drop=True)

    df_woe = woe_encoder.fit_transform(df[woe_vars], y_target)
    df_woe_validation = woe_encoder.transform(df_validation[woe_vars])

    df.drop(woe_vars, axis=1, inplace=True)
    df_validation.drop(woe_vars, axis=1, inplace=True)

    df = pd.merge(df, df_woe, how='left', left_index=True, right_index=True)
    df_validation = pd.merge(df_validation, df_woe_validation, how='left', left_index=True, right_index=True)

    return df, df_validation


def iv_woe(data, target, bins=10):
    # Empty Dataframe
    newDF, woeDF = pd.DataFrame(), pd.DataFrame()

    # Extract Column Names
    cols = data.columns

    # Run WOE and IV on all the independent variables
    for ivars in cols[~cols.isin([target])]:
        if (data[ivars].dtype.kind in 'bifc') and (len(np.unique(data[ivars])) > 10):
            binned_x = pd.qcut(data[ivars], bins, duplicates='drop')
            d0 = pd.DataFrame({'x': binned_x, 'y': data[target]})
        else:
            d0 = pd.DataFrame({'x': data[ivars], 'y': data[target]})
        d = d0.groupby("x", as_index=False).agg({"y": ["count", "sum"]})
        d.columns = ['Cutoff', 'N', 'Events']
        d['% of Events'] = np.maximum(d['Events'], 0.5) / d['Events'].sum()
        d['Non-Events'] = d['N'] - d['Events']
        d['% of Non-Events'] = np.maximum(d['Non-Events'], 0.5) / d['Non-Events'].sum()
        d['WoE'] = np.log(d['% of Events'] / d['% of Non-Events'])
        d['IV'] = d['WoE'] * (d['% of Events'] - d['% of Non-Events'])
        d.insert(loc=0, column='Variable', value=ivars)
        print("Information value of " + ivars + " is " + str(round(d['IV'].sum(), 6)))
        temp = pd.DataFrame({"Variable": [ivars], "IV": [d['IV'].sum()]}, columns=["Variable", "IV"])
        newDF = pd.concat([newDF, temp], axis=0)
        woeDF = pd.concat([woeDF, d], axis=0)

    return newDF, woeDF


def feature_selection_rf(df: pd.DataFrame, features, y_target):
    rf = RandomForestClassifier(n_estimators=500, n_jobs=1, max_depth=10).fit(df[features], y_target)
    rf_vip = pd.DataFrame(list(zip(df.columns, rf.feature_importances_)),
                          columns=['variables', 'importance']). \
        sort_values(by=['importance'], ascending=False, axis=0).set_index('variables')
    feature_importance = rf_vip.loc[rf_vip.importance >= 0.01]
    feature_importance = list(feature_importance.index)
    feature_importance.append('order_uuid')

    return feature_importance


def save_model(path, model, nombre_modelo, num_ejecucion, estimator):
    output = open(path + model + '_' + nombre_modelo + '_' + str(num_ejecucion) + '.pkl', 'wb')
    pickle.dump(estimator, output, -1)
    output.close()


def open_model(path, model, nombre_modelo, num_ejecucion):
    pkl = open(path + model + '_' + nombre_modelo + '_' + str(num_ejecucion) + '.pkl', 'rb')
    model = pickle.load(pkl)
    pkl.close()

    return model
