# Big Data Processes Project

## Dataset

The dataset is a merge of 3 main sources:

- [Suicide Dataset](https://www.kaggle.com/datasets/russellyates88/suicide-rates-overview-1985-to-2016)
- [GINI Dataset](https://data.worldbank.org/indicator/SI.POV.GINI?end=2021&most_recent_value_desc=true&start=2021&view=map)
- [Healtcare Coverage Dataset](https://stats.oecd.org/Index.aspx?ThemeTreeId=9)

The base Suicide Dataset was extended to grant more features for modeling.
For this purpose we searched for possible datasets that could match factors listed here: https://www.cdc.gov/suicide/factors/index.html.
For most features it's hard to find proper datasets that cover a variety of countries and years.
As a compromise we added the Healthcare dataset (access to healthcare is listed as a factor) and additionaly added the GINI dataset to account for factors that might be related with social inequality.

## EDA

Refer to EDA notebook:

- [Suicide Dataset](mental_health/eda_suicide.ipynb)
- [GINI Dataset](mental_health/eda_gini.ipynb)
- [Healtcare Coverage Dataset](mental_health/eda_hc_coverage.ipynb)

## Data cleaning

All data cleaning and loading code is in [utils.py](mental_health/utils.py)

### Suicide Dataset | `load_suicide_data()`

- Drop columns:
  - 'HDI for year': too much missing data
  - 'country-year': redundant
- Rename columns:
  - 'suicides/100k pop' => 'suicides_per_100k_pop'
  - ' gdp_for_year ($) ' => 'gdp_for_year',
  - 'gdp_per_capita ($)' => 'gdp_per_capita'
- Drop data from 2016 due to missing datapoints
- Drop all datapoints of countries that have less than 3 years of data
- Add columns (done using [pycountry](https://pypi.org/project/pycountry/) and [pycountry_convert](https://pypi.org/project/pycountry-convert/)):
  - 'country_code'(useful for joing with oder datasets)
  - 'continent' (useful for visualizing)

### GINI Dataset | `load_gini()`

- Melt the dataframe to get tidy columns (from a column for each year to a single year column)
- Rename columns:
  - 'Country Name' => 'country',
  - 'Country Code' => 'country_code'
- Set type of year column to int (was string before)
- Filter to only contain years $\in [2010; 2015]$

### Healthcare Dataset | `load_healthcare()`

- Remove data for Spain and Russia due to missing entries
- Rename columns:
  - 'COU' => 'country_code',
  - 'Country' => 'country',
  - 'Year' => 'year',
  - 'Value' => 'healthcare_coverage'
- Filter to only contain years $\in [2010; 2015]$

## Data Loading

### Merge datasets | `load_suicide_healthcare_gini_df()`

- Load suicide dataset and filter to range $\in [2010; 2015]$
- Load GINI data and only select columns: ['country_code', 'year', 'gini']
- Load Healthcare and only select columns: ['country_code', 'year', 'healthcare_coverage']
- Merge Healthcare and GINI data on 'year' and 'country_code'
- Merge filtered Suicide df with previous merge results on 'year' and 'country_code'

### Preprocessing

To make the data suitable as input for our models some columns have to be converted to numerical values. Additionaly the data has to be separated into feature and label parts and split into a train, valdiation and test set.
Code in [utils.py](mental_health/utils.py).

- Conversions:  
  country, continent, sex and age are converted to numerical values
  ```python
  # example for country column
  df.country = pd.Categorical(df.country).codes
  ```
- Feature/ Label split:  
   We want to predict the number of suicides based on a set of input features. We experimented with choosing only a subset of available features but using all yielded the best results. Hence we chose:

  - Features:  
     `['country', 'continent', 'sex', 'age', 'year',
'gdp_per_capita', 'healthcare_coverage', 'gini', 'population']`
  - Label: `['suicides_no']`

- Train/ Test split:
  Currently 2 implementations:
  - `get_train_test_split_legacy()`:
    Split data into train (up to 2014) and test (2015):
    - Training: data $\in [2010; 2014]$; 1788 samples ~ 86,6%
    - Test: data $\in [2015]$; 276 samples ~ 13,4%
  - `get_train_val_test_split`:
    Perform a random split of the data into train, val & test
    - Training: 1651 samples ~ 80%
    - Validation: 206 samples ~ 10%
    - Test: 207 samples ~ 10%

## Modeling

### Model types

- Linear model
  - LinearRegression (sklearn)
- Tree model
  - DecisionTreeRegressor (sklearn)
- Neighbour model
  - KNeighborsRegressor (sklearn)
- Ensemble methods
  - RandomForestRegressor (sklearn)
  - XGBRegressor (xgboost.sklearn)
- Neural networks
  - MLPRegressor (sklearn)
  - Custom MLP (PyTorch)

### Metrics

- mean_squared_error
- root_mean_squared_error
- r2_score
- mean_absolute_error
- max_error

### Model evaluation and comparison

`TODO`
