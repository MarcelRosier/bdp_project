import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns


def clean_data(df: pd.DataFrame, min_data_years: int = 3) -> pd.DataFrame:
    # rm 2016
    df = df[df['year'] != 2016]

    # get recorded years per country and remove countries below threshold
    temp = df.groupby(['country', 'year']).count().reset_index()
    temp = temp[['country', 'year']]
    country_years = temp.groupby('country').count()
    temp = country_years.reset_index()
    names = temp[temp.year <= min_data_years].country.values
    df = df[~df['country'].isin(names)]
    return df
