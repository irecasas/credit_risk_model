from typing import Optional

import pandas as pd
'''
Name: initial_cleaning

Description: Eliminates variables with unique values

Parameters: - df <pandas.DataFrame>

Return: - df <pandas.DataFrame>
'''
def initial_cleaning(dataframe: pd.DataFrame) -> pd.DataFrame:
    a = dataframe.shape[1]
    for i in dataframe:
        if len(dataframe[i].unique() )==1:
            dataframe.drop(i, 1, inplace=True)
            print('The variable ', i, ' is eliminated because it has a unique value')
    print(str( a- dataframe.shape[1]) + ' variables have been eliminated as they have unique values')
    print(dataframe.shape)

    return dataframe


'''
Name: correlation matrix

Description: Function that returns the correlation matrix between two variables according to Pearson's method. 
If the absolute value of the Pearson coefficient between two variables exceeds 0.8, collinearity exists.

Parameters: - df <pandas.DataFrame>

Return: - df <pandas.DataFrame>
'''

def correlation_matrix(df: pd.DataFrame, list_vars: Optional[list] = []) -> pd.DataFrame:
    cols_names = df.columns
    cols_names.remove(list_vars)
    df_aux = df[cols_names].corr(method = 'pearson')

    return df_aux