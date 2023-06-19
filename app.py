# Fetching all Air Quality datasets into their dataframes
# Perform immediate concatenation per year
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import joblib

def gatherData(stateName, dataDir):
    # Initialize accumulator datasets using CO California datasets
    # NOTE: Dataset fragments are all in './StateAirData/'
    colidx = [0,2,4,17]
    innerkeys = ['Date', 'Site ID', 'COUNTY']       # Merge on Date, Site ID, and County
    dataA2020 = pd.read_csv(dataDir + stateName + '-2020-co.csv', parse_dates=True, usecols=colidx)
    dataA2021 = pd.read_csv(dataDir + stateName + '-2021-co.csv', parse_dates=True, usecols=colidx)
    dataA2022 = pd.read_csv(dataDir + stateName + '-2022-co.csv', parse_dates=True, usecols=colidx)

    # Iterate through the directory using os.scandir, and then get all the datasets there
    # Merge the remaining datasets other than ca-year-co to dataAyear
    with os.scandir(dataDir) as datasets:
        for dataset in datasets:
            if dataset.is_file() and 'co' not in dataset.name:
                temp = pd.read_csv(dataset, parse_dates=True, usecols=colidx)
                if stateName + '-2020' in dataset.name:
                    dataA2020 = pd.merge(dataA2020, temp, how='outer', on=innerkeys)
                elif stateName + '-2021' in dataset.name:
                    dataA2021 = pd.merge(dataA2021, temp, how='outer', on=innerkeys)
                elif stateName + '-2022' in dataset.name:
                    dataA2022 = pd.merge(dataA2022, temp, how='outer', on=innerkeys)

    # At this point, dataA2020, dataA2021, and dataA2022 have accumulated the dataset year fragments.

    # Parse Date to date
    dataA2020['Date'] = pd.to_datetime(dataA2020['Date'])
    dataA2021['Date'] = pd.to_datetime(dataA2021['Date'])
    dataA2022['Date'] = pd.to_datetime(dataA2022['Date'])

    # Group data by Date and Site ID, then by Date again to remove the Site ID feature
    # Result would be mean measurements per day
    dataA2020 = dataA2020.drop(columns=['Site ID'], axis=1).groupby(by=['Date'], as_index=False).mean()
    dataA2021 = dataA2021.drop(columns=['Site ID'], axis=1).groupby(by=['Date'], as_index=False).mean()
    dataA2022 = dataA2022.drop(columns=['Site ID'], axis=1).groupby(by=['Date'], as_index=False).mean()

    dataA = pd.concat([dataA2020, dataA2021, dataA2022])    # Combine the three datasets

    # Append State column to the data
    # dataA['State'] = stateName
    dataA.insert(1, 'State', stateName.upper(), True)

    return dataA

dataA = gatherData('ca', 'StateAirData/')
dataA = pd.concat([dataA, gatherData('ny', 'StateAirData/')])
dataA = pd.concat([dataA, gatherData('tx', 'StateAirData/')])

new_names = ['Date', 'State', 'CO conc (ppm)', 'NO2 conc (ppb)', 'O3 conc (ppm)',
             'Pb conc (ug/m3 SC)', 'PM10 conc (ug/m3 SC)',
             'PM2.5 conc (ug/m3 LC)', 'SO2 conc (ppb)']

# Rename columns
for i in range(len(new_names)):
    dataA.rename(columns={dataA.columns[i]: new_names[i]}, inplace=True)

# Deciding whether to drop or impute null values, so we check how many null values there are.
# Dataset A impute
dataA.dropna(inplace=True)

print("A: Number of entries with null values:", dataA.isna().any(axis=1).sum())
print("A: Number of entries:", dataA.shape[0])

# These imports are important, imputer relies on them.

from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer   # Important!
from sklearn.impute import IterativeImputer     # default imputer is BayesianRidge

# Other estimators (estimator = func()) to try
from sklearn.linear_model import BayesianRidge

# Initialize imputer
# NOTE: DIFFERENT ESTIMATORS WERE TRIED HERE, from DecisionTreeRegressor() to KNeighborsRegressor()
imp = IterativeImputer(max_iter=100, random_state=1, verbose=True, estimator=BayesianRidge())

# Perform imputation (note that convergence was quickly reached, indicating a good imputation)
# Imputation done separately from dataB to ensure impartiality of resulting values
backup = pd.DataFrame().assign(Date=dataA['Date'], State=dataA['State'])
dataA.drop(['Date', 'State'], axis=1, inplace=True)

dataA[:] = imp.fit_transform(dataA)

print("After imputation:")
dataA.head()
print("A: Number of entries with null values after impute:", dataA.isna().any(axis=1).sum())
print("A: Number of entries:", dataA.shape[0])

dataA.insert(0, 'Date', backup.Date)
dataA.insert(1, 'State', backup.State)

dataA

colidx = [0,1,2,3,5,6]     # column indexes to use (based on preemptively looking at dataB)
dataB = pd.read_csv('datasets/us_covid_cases_and_deaths_by_state.csv', parse_dates=True, usecols=colidx)

# ASSUMPTION: Let us treat all probable cases as an actual new case.
# We can do this because there are many dates where there are zero infections in dataB, but science would say that there's always background transmission going on, especially in the case of COVID-19 which is airborne and isn't always symptomatic.
sum_new_cases = dataB['new_case'] + dataB['pnew_case']
dataB.drop(['new_case', 'pnew_case'], axis=1, inplace=True)
dataB['sum_new_cases'] = sum_new_cases
dataB.head()

# Rename dataB columns to names comparable to dataA
# Note that we designated new_case + pnew_case as the column 'Sum New Cases'
dataB.columns = ['Date', 'State', 'Total Cases', 'Confirmed Cases', 'Sum New Cases']

dataB['Date'] = pd.to_datetime(dataB['Date'])

# Filter dataset B to just the states of CA and NY
dataB = dataB[(dataB['State'] == 'CA') | (dataB['State'] == 'NY') | (dataB['State'] == 'TX')]
dataB.head()

dataB.dropna(inplace=True)

print("B: Number of entries with null values:", dataB.isna().any(axis=1).sum())
print("B: Number of entries:", dataB.shape[0])

# First, impute NAN values
imp = IterativeImputer(max_iter=100, random_state=1, verbose=True, estimator=BayesianRidge())
backup = pd.DataFrame().assign(Date=dataB['Date'], State=dataB['State'])
dataB.drop(['Date', 'State'], axis=1, inplace=True)
dataB[:] = imp.fit_transform(dataB)

# Then, impute 0 values
imp = IterativeImputer(max_iter=100, random_state=1, verbose=True, estimator=BayesianRidge(), missing_values=0, tol=1e-3)
dataB[:] = imp.fit_transform(dataB)

dataB.insert(0, 'Date', backup.Date)
dataB.insert(1, 'State', backup.State)

# Filter dataA with temporal restriction given by dataB
dataA = dataA[(dataA.Date >= dataB.Date.min()) &
              (dataA.Date <= dataB.Date.max())]

print("Filtered Dataset A")
dataA.head()

# Merging the two datasets (dataA & dataB)
data = dataA.merge(dataB, on=['Date', 'State'])

# Shift Date by 1 so that model is using the number of infected from the day before, this makes the model more valuable for predicting new COVID-19 cases.
data['Total Cases'].shift(1)
data['Confirmed Cases'].shift(1)
data.dropna(inplace=True)

# Take a peak at both contents and shape
data.head()
data.info()

# The purpose of Date and State is done, drop them
data.drop(['Date', 'State'], axis=1, inplace=True)

# GENERATE SCATTERPLOTS FOR EDUCATED GUESSES OF WHAT COLUMNS TO USE
print("Scatterplots before pruning:")
for label in data.columns:
    if label in ['Sum New Cases', 'Total Cases', 'Confirmed Cases']: continue
    sns.set_style('dark')

    sns.relplot(x=label, y='Sum New Cases', data=data, height=3.8, aspect=1.8, kind='scatter')

print("Number of entries remaining BEFORE pruning:", data.shape[0])

# DYNAMIC PRUNING
# IDEA: We focus on ambient level of pollutants and ignore sudden spikes in COVID-19 case data (i.e. data dumps).
# We also remove Dates where there are no new infections as their volume skews the data alot.
# Currently visible values are most likely plateaus after painstaking tuning.
data = data[(data['Sum New Cases'] <= 73000) & (data['Sum New Cases'] > 1)]
data = data[(data['Total Cases'] > 0)]
data = data[(data['Confirmed Cases'] > 0)]
data = data[data['CO conc (ppm)'] <= 1.5]
# data = data[data['NO2 conc (ppb)'] > 1]   # pruning based on some columns hurts the metrics
data = data[data['O3 conc (ppm)'] > 0.035]
data = data[data['Pb conc (ug/m3 SC)'] < 0.04]
data = data[data['PM10 conc (ug/m3 SC)'] < 100]
data = data[data['PM2.5 conc (ug/m3 LC)'] < 15]
data = data[data['SO2 conc (ppb)'] < 2.0]

# DROP COLUMNS HERE
data.drop(columns=['PM10 conc (ug/m3 SC)'], axis=1, inplace=True)
# data = data[['PM10 conc (ug/m3 SC)', 'Sum New Cases']]

# SHOW scatterplots of pruned columns
print("Scatterplots after pruning")
for label in data.columns:
    if label in ['Sum New Cases', 'Total Cases', 'Confirmed Cases']: continue
    sns.set_style('dark')

    sns.relplot(x=label, y='Sum New Cases', data=data, height=3.8, aspect=1.8, kind='scatter')

print("Number of entries remaining AFTER pruning:", data.shape[0])

fig, axs = plt.subplots(nrows=len(data.columns)-3, ncols=1, figsize=(16,64))

i = 0
for label in data.columns:
    if label in ["Sum New Cases", "Total Cases", "Confirmed Cases"]: continue
    sns.regplot(x=label, y='Sum New Cases', data=data, ci=95, scatter_kws={'s':100, 'facecolor':'red'}, ax=axs[i])
    i += 1

# Linear Regression
# PREPARE FEATURES AND TARGET DATA (standardize first)
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import normalize
scaler = MinMaxScaler()

# SPLIT DATA TO FEATURE SET AND TARGET
X = data.iloc[:,0:-1] # feature matrix
X = scaler.fit_transform(X)
y = data.iloc[:,-1] # target vector

# PREPARE TRAINING AND TESTING DATA
# NOTE: test_size is reduced to 20% because we have few records to work with especially after pruning
# (around 499 left from roughly 1000)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import Ridge
import sklearn.linear_model as sklm
regressor = LinearRegression()
alt1 = Ridge(alpha=0.5)
alt2 = SGDRegressor()
regressor.fit(X_train, y_train)
alt1.fit(X_train, y_train)
alt2.fit(X_train, y_train)

print("Coefficients:", regressor.coef_)
print("Intercept:", regressor.intercept_)

y_pred = regressor.predict(X_test)
y_pred

y_pred1 = alt1.predict(X_test)
y_pred2 = alt2.predict(X_test)

joblib_file = "cs180.joblib"
joblib.dump(regressor, joblib_file)