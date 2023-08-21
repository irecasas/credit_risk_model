import pandas as pd


def age_customer(df: pd.DataFrame, date_birthday: str, date_init: str):
    """

    :param df:
    :param date_birthday:
    :param date_init:
    :return:
    """
    df[date_birthday] = pd.to_datetime(df[date_birthday], format="%Y-%m-%d").dt.date
    df[date_init] = pd.to_datetime(df[date_init], format="%Y-%m-%d").dt.date
    df["age"] = (df[date_init] - df[date_birthday]).dt.days / 365
    df["age"] = df["age"].fillna(0)
    df.drop([date_birthday], axis=1, inplace=True)


def email_features(df: pd.DataFrame):
    # Data features
    for date_feature_name in (
            'emailage_response__firstVerificationDate',
            'emailage_response__lastVerificationDate',
    ):
        if date_feature_name in df:
            engineered_feature_name = f'{date_feature_name}_difference_with_created'
            df[date_feature_name] = pd.to_datetime(
                df[date_feature_name]).apply(lambda d: d.replace(tzinfo=None))
            df['created_at'] = pd.to_datetime(df['created_at'])
            df[engineered_feature_name] = df[[date_feature_name, 'created_at']].apply(
                lambda row: (row['created_at'] - row[date_feature_name]).days if
                row['created_at'] >= row[date_feature_name] else 0, axis=1)
            df.drop([date_feature_name], axis=1, inplace=True)

    # Binary features
    for binary_feature_name in ('emailage_response__domainriskcountry',
                                'emailage_response__domaincorporate',
                                'emailage_response__ip_corporateProxy',
                                'emailage_response__ip_anonymousdetected',
                                'emailage_response__domainExists',
                                'emailage_response__ipcountrymatch',
                                'emailage_response__shipforward'):
        if binary_feature_name in df:
            engineered_feature = f'{binary_feature_name}_trans'
            df[engineered_feature] = 0
            df[engineered_feature] = df[[binary_feature_name]].apply(lambda row: 1 if (
                row[
                    binary_feature_name] == 'Yes') or (
                row[
                    binary_feature_name] == 'True') else 0,
                axis=1)
            df.drop([binary_feature_name], axis=1, inplace=True)

    # Email feature
    def emailexists_correspondence(x):
        if x == 'No':
            return 0.0
        elif x == 'Not Sure':
            return 0.2
        elif x == 'Not Anymore':
            return 0.4
        elif x == 'Yes':
            return 1
        else:
            return 0.0

    df['email_exists'] = df['emailage_response__emailExists'].apply(lambda x: emailexists_correspondence(x))
    df.drop(['emailage_response__emailExists'], axis=1, inplace=True)

    def estatus_id(x):
        if (x == 0) | (pd.isnull(x)):
            return -1
        elif x == 1:
            return 1
        elif x == 2:
            return 0.7
        elif x == 3:
            return 0.1
        elif x == 4:
            return 0.5
        elif x == 5:
            return 0

    df['emailage_eastatus_id'] = df['emailage_response__EAStatusID'].apply(lambda x: estatus_id(x))
    df.drop(['emailage_response__EAStatusID'], axis=1, inplace=True)

    # Email reputation feature
    df['emailage__ip_reputation'] = df[['emailage_response__ip_reputation']].apply(
        lambda row: 1 if row['emailage_response__ip_reputation'] == 'Good' else 0, axis=1)
    df.drop(['emailage_response__ip_reputation'], axis=1, inplace=True)
    # Gender feature
    df['emailage_response__gender_female'] = df[['emailage_response__gender']].apply(
        lambda row: 1 if row['emailage_response__gender'] == 'female' else 0, axis=1)
    df.drop(['emailage_response__gender'], axis=1, inplace=True)
    df['emailage_phone_status'] = df[['emailage_response__phone_status']].apply(
        lambda row: 1 if row['emailage_response__phone_status'] == 'Valid' else 0, axis=1)
    df.drop(['emailage_response__phone_status'], axis=1, inplace=True)


def minfraud_features(df: pd.DataFrame):
    # Binary features
    for binary_feature_name in ('minfraud_response__ip_address__registered_country__is_in_european_union',
                                'minfraud_response__email__is_disposable',
                                'minfraud_response__email__is_free',
                                'minfraud_response__email__is_high_risk',
                                'minfraud_response__shipping_address__is_postal_in_city',
                                'minfraud_response__shipping_address__is_in_ip_country',
                                'minfraud_response__billing_address__is_postal_in_city',
                                'minfraud_response__billing_address__is_in_ip_country',
                                'minfraud_response__ip_address__traits__is_anonymous'):
        if binary_feature_name in df:
            engineered_feature_name = f'{binary_feature_name}_trans'
            df[engineered_feature_name] = 0
            df[engineered_feature_name] = df[[binary_feature_name]].apply(lambda row: 1 if (
                row[
                    binary_feature_name] == 'Yes') or (
                row[
                    binary_feature_name] == 'True') else 0,
                axis=1)
            df.drop([binary_feature_name], axis=1, inplace=True)


def experian_features(df: pd.DataFrame):
    df['experian_autonomo_trans'] = df[['experian_AUTONOMO']].apply(
        lambda row: 1 if row['experian_AUTONOMO'] == '1' else 0, axis=1)
    df.drop(['experian_AUTONOMO'], axis=1, inplace=True)

    df['experian_documento__tipodocumento__descripcion_trans'] = \
        df[['experian_Documento__TipoDocumento__Descripcion']].apply(
            lambda row: 1 if row['experian_Documento__TipoDocumento__Descripcion'] == 'NIF' else 0, axis=1)
    df.drop(['experian_Documento__TipoDocumento__Descripcion'], axis=1, inplace=True)

    # Data features
    for date_feature_name in (
            'experian_ResumenCais__FechaMaximoImporteImpagado',
            'experian_ResumenCais__FechaPeorSituacionPagoHistorica',
            'experian_ResumenCais__FechaAltaOperacionMasAntigua'
    ):
        if date_feature_name in df:
            engineered_feature_name = f'{date_feature_name}_difference_with_created'
            df[date_feature_name] = df[[date_feature_name]].apply(
                lambda row: pd.to_datetime('1800-01-01') if (row[date_feature_name] == '0001-01-01')
                or (pd.isnull(row[date_feature_name]))
                or (row[date_feature_name] == 'nan')
                else pd.to_datetime(row[date_feature_name]), axis=1)
            df[engineered_feature_name] = df[[date_feature_name, 'created_at']].apply(
                lambda row: (row['created_at'] - row[date_feature_name]).days if
                (row['created_at'] >= row[date_feature_name]) and (
                    row[date_feature_name] != pd.to_datetime('1800-01-01')) else 0, axis=1)
            df.drop([date_feature_name], axis=1, inplace=True)
