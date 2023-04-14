from sklearn.model_selection import train_test_split
import math
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
import pycountry_convert as pc
from pycountry import pycountry


country_name_map = {
    'Saint Vincent and Grenadines': 'Saint Vincent and the Grenadines',
    'Republic of Korea': 'Korea, Republic of',
}


def load_suicide_data(min_data_years: int = 3, path="datasets/suicide_ds_2016.csv") -> pd.DataFrame:
    df = pd.read_csv(path)

    # drop data
    df = df.drop(['HDI for year', 'country-year'], axis='columns')

    # rename columns to be snake case conform and remove $ currency
    df = df.rename(columns={
        'suicides/100k pop': 'suicides_per_100k_pop',
        # the extra spaces in the name below are intended!
        ' gdp_for_year ($) ': 'gdp_for_year',
        'gdp_per_capita ($)': 'gdp_per_capita'
    })
    # rm 2016
    df = df[df['year'] != 2016]

    # get recorded years per country and remove countries below threshold
    temp = df.groupby(['country', 'year']).count().reset_index()
    temp = temp[['country', 'year']]
    country_years = temp.groupby('country').count()
    temp = country_years.reset_index()
    names = temp[temp.year <= min_data_years].country.values
    df = df[~df['country'].isin(names)]

    # add country code and continent
    df['country_code'] = [
        pc.country_name_to_country_alpha3(country_name_map.get(c, c)) for c in df.country
    ]
    df['continent'] = [
        pc.convert_continent_code_to_continent_name(
            pc.country_alpha2_to_continent_code(
                pycountry.countries.get(alpha_3=cc).alpha_2
            ))
        for cc in df.country_code
    ]

    return df


def load_gini(path="datasets/gini.csv") -> pd.DataFrame:
    df = pd.read_csv(path, delimiter=',')
    # melt df and remove years without data
    melted_df = df.melt(['Country Name', 'Country Code', 'Indicator Name',
                        'Indicator Code'], var_name="year", value_name='gini')
    melted_df = melted_df[melted_df['gini'].notna()]
    melted_df = melted_df[['Country Name', 'Country Code', 'year', 'gini']]
    melted_df = melted_df.rename(columns={
        'Country Name': 'country',
        'Country Code': 'country_code'
    })
    # change type and select year range
    melted_df.year = melted_df.year.astype(int)
    melted_df = melted_df[melted_df.year.between(2010, 2015)]

    return melted_df


def load_healthcare() -> pd.DataFrame:
    df = pd.read_csv('datasets/healtcare_coverage.csv')
    df = df[['COU', 'Country', 'Year', 'Value']]
    df = df.rename(columns={
        'COU': 'country_code',
        'Country': 'country',
        'Year': 'year',
        'Value': 'healthcare_coverage'
    })
    # limit to 2010-2015
    df = df[df.year.between(2010, 2015)]
    # rm spain and russia sicne they have missing data
    df = df[~df.country.isin(['Spain', 'Russia'])]
    return df


def load_suicide_healthcare_gini_df() -> pd.DataFrame:

    suicide_df = load_suicide_data()
    suicide_df = suicide_df[suicide_df.year.between(2010, 2015)]

    gini_df = load_gini()
    gini_df = gini_df[['country_code', 'year', 'gini']]
    health_df = load_healthcare()
    health_df = health_df[['country_code', 'year', 'healthcare_coverage']]
    gh = health_df.merge(gini_df, on=['year', 'country_code'])

    # not all countries have entries for all combinations of country, age, sex, year
    # but there are no missing entries in rows, so it should be fine
    df = suicide_df.merge(gh, on=['year', 'country_code'])
    return df


def get_train_test_split_legacy(X_cols=None):
    df = load_suicide_healthcare_gini_df()
    # transform text columns to categories
    df.country = pd.Categorical(df.country).codes
    df.continent = pd.Categorical(df.continent).codes
    df.sex = pd.Categorical(df.sex).codes
    df.age = pd.Categorical(df.age).codes

    # split data
    test_df = df[df.year == 2015]
    train_df = df[df.year < 2015]

    # extract values
    if not X_cols:
        # use default cols
        X_cols = ['country', 'continent', 'sex', 'age', 'year',
                  'gdp_per_capita', 'healthcare_coverage', 'gini', 'population']
    X_train = train_df[X_cols].values
    Y_train = train_df['suicides_no'].values

    X_test = test_df[X_cols].values
    Y_test = test_df['suicides_no'].values

    # scale data
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    scaler = StandardScaler()
    scaler.fit(X_train)

    # transform the training and test data using the scaler
    X_train_std = scaler.transform(X_train)
    X_test_std = scaler.transform(X_test)

    return X_train_std, X_test_std, Y_train, Y_test


def root_mean_squared_error(y_true, y_pred) -> float:
    return mean_squared_error(y_true=y_true, y_pred=y_pred, squared=False)


def get_train_val_test_split(X_cols=None):
    df = load_suicide_healthcare_gini_df()
    # transform text columns to categories
    df.country = pd.Categorical(df.country).codes
    df.continent = pd.Categorical(df.continent).codes
    df.sex = pd.Categorical(df.sex).codes
    df.age = pd.Categorical(df.age).codes

    # extract values
    if not X_cols:
        # use default cols
        X_cols = ['country', 'continent', 'sex', 'age', 'year',
                  'gdp_per_capita', 'healthcare_coverage', 'gini', 'population']
    X = df[X_cols].values
    y = df['suicides_no'].values

    # get train val test split

    train_ratio = 0.8
    validation_ratio = 0.10
    test_ratio = 0.10

    # train is now 80% of the entire data set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=1 - train_ratio)

    # test is now 10% of the initial data set
    # validation is now 10% of the initial data set
    X_val, X_test, y_val, y_test = train_test_split(
        X_test, y_test, test_size=test_ratio/(test_ratio + validation_ratio))

    # scale data
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    scaler = StandardScaler()
    scaler.fit(X_train)

    # transform the training and test data using the scaler
    X_train_std = scaler.transform(X_train)
    X_val_std = scaler.transform(X_val)
    X_test_std = scaler.transform(X_test)

    return {
        'train': {
            'X': X_train_std,
            'y': y_train
        },
        'val': {
            'X': X_val_std,
            'y': y_val
        },
        'test': {
            'X': X_test_std,
            'y': y_test
        },
    }
