""" 
Author: Matheus Nogueira 

Description: Python code to implement time series models in order to forecast COS emitions from Rwanda.
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error

def RMSE(y, y_hat):
    return np.sqrt(mean_squared_error(y, y_hat))

" Importing data "
path = os.getcwd() + "\\CO2PredictionsRwanda\\"
df_train = pd.read_csv(path+"train.csv")
df_test = pd.read_csv(path+"test.csv")

print("Size df_train = ", df_train.shape)
print("Size df_test = ", df_test.shape)

" Just a bit of EDA "

for col in df_train.columns:
    na_count = sum(df_train[col].isna())
    T        = len(df_train[col])
    if na_count > 0:
        print(col," -> %d (%.2f%%)"%(na_count, 100*na_count/T))

" Lets get rid of the columns with a lot of missing values and than the rows with missing values "

df_train_clean = df_train.copy(deep=True)

for col in df_train_clean.columns:
    na_count   = sum(df_train[col].isna())
    T          = len(df_train[col])
    na_percent = 100*na_count/T
    if na_percent > 90:
        print("Deleting column %s"%col)
        df_train_clean.drop(columns = col, inplace=True)

print("Size df_train = ", df_train.shape)
print("Size df_train_clean = ", df_train_clean.shape)

df_train_clean.fillna(0, inplace = True)
df_train_clean.reset_index(drop=True, inplace = True)
print("Size df_train_clean = ", df_train_clean.shape)

" Now lets divide our dataset into a train and validation set -> 2019 and 2020 for training and 2021 for validation "

df_fit = df_train_clean.where(df_train_clean.year <= 2020).dropna()
df_val = df_train_clean.where(df_train_clean.year == 2021).dropna()

print("Size df_fit = ", df_fit.shape)
print("Size df_val = ", df_val.shape)
print("Sum -> %d = %d"%(df_fit.shape[0]+df_val.shape[0], df_train_clean.shape[0]))

" Lets treat each location as a separate time series "

unique_locations = df_fit[['latitude', 'longitude']].drop_duplicates()
print("Number of unique locations = %d"%len(unique_locations))

df_fit_dict = {}
df_val_dict = {}

for idx, row in unique_locations.iterrows():
    lat, lon = row['latitude'], row['longitude']
    
    df_fit_location = df_fit[(df_fit['latitude'] == lat) & (df_fit['longitude'] == lon)].copy()
    df_val_location = df_val[(df_val['latitude'] == lat) & (df_val['longitude'] == lon)].copy()
    
    key = f"{lat}:{lon}"
    df_fit_dict[key] = df_fit_location
    df_val_dict[key] = df_val_location

print("Number of keys fit = %d ? %r"%(len(unique_locations),len(unique_locations)==len(df_fit_dict.keys())))
print("Number of keys val = %d ? %r"%(len(unique_locations),len(unique_locations)==len(df_val_dict.keys())))

for key in df_fit_dict.keys():
    df_corr = df_val_dict[key]
    X = df_corr[df_corr.columns[5:-1]]
    print(X.shape)

" It seems that this wont be a good ideia, since each time series is super short and has more exogenous variables than its'own lenght"

" Lets treat each location as a feature for our model, so that we have a lot of data to train it "

" Lets create a new feature -> the t-1 value of emission considering, of course, each individual location"

for key in df_fit_dict.keys():
    df_fit_corr = df_fit_dict[key]
    df_fit_corr.insert(df_fit_corr.shape[1]-1,"emission_lag", df_fit_corr.emission.shift(1))
    df_fit_dict[key] = df_fit_corr
    df_val_corr = df_val_dict[key]
    df_val_corr.insert(df_val_corr.shape[1]-1,"emission_lag", df_val_corr.emission.shift(1))
    df_val_corr.emission_lag.iloc[0] = df_fit_corr.emission.iloc[-1]
    df_val_dict[key] = df_val_corr


" Now we shall create our X and y datasets"

cols = df_fit_dict['-3.299:30.301'].columns

X_fit = pd.DataFrame(columns=cols)
X_val = pd.DataFrame(columns=cols)
y_fit = pd.Series("emission")
y_val = pd.Series("emission")

for (key, df) in df_fit_dict.items():
    X_fit = pd.concat([X_fit, df], axis=0, ignore_index=True)
    X_val = pd.concat([X_val, df_val_dict[key]], axis=0, ignore_index=True)

X_fit.dropna(inplace=True)

cols_drop = ['ID_LAT_LON_YEAR_WEEK', 'year', "emission"]

y_fit = X_fit.emission
y_val = X_val.emission
X_fit.drop(columns=cols_drop, inplace=True)
X_val.drop(columns=cols_drop, inplace=True)

" Now, it is time to fit some models "

# lasso = Lasso()
# params = {"alpha":np.linspace(1,10,20), "max_iter":[2000], "random_state":[0, 42]}
# grid  = GridSearchCV(lasso, params, cv=10)
# grid.fit(X_fit, y_fit)

lasso = Lasso(alpha = 1, max_iter = 2000, random_state = 0)
lasso.fit(X_fit, y_fit)

print("Columns selected by Lasso")
print(X_fit.columns[lasso.coef_ != 0])

print("Columns not selected by Lasso")
print(X_fit.columns[lasso.coef_ == 0])

r2_fit_lasso   = lasso.score(X_fit, y_fit)
y_hat_fit      = lasso.predict(X_fit)
rmse_fit_lasso = RMSE(y_fit, y_hat_fit)

y_hat_val      = lasso.predict(X_val)
rmse_val_lasso = RMSE(y_val, y_hat_val)
r2_val_lasso   = lasso.score(X_val, y_val)

" Finally lets predict the test values "

# We need to create the lagged emission feature

cols_test = df_fit.columns.drop("emission")
df_test   = df_test.loc[:,cols_test]
df_test.fillna(0, inplace = True)

y_hat_test = np.array([])

for idx, row in df_test.iterrows():
    lat, long  = row["latitude"], row["longitude"]
    year, week = row["year"], row["week_no"]
    location   = f"{lat}:{long}"
    if week == 0:
        m1 = df_val.year == year - 1
        m2 = df_val.week_no == 52
        m3 = df_val.latitude == lat
        m4 = df_val.longitude == long
        m = m1 & m2 & m3 & m4
        last_emission = df_val[m].emission.values[0]
    else:
        last_emission = y_hat_test[idx-1]
    row_test = pd.concat([row, pd.Series([last_emission])])
    row_test.drop(["ID_LAT_LON_YEAR_WEEK","year"], inplace = True)
    X_test = row_test.values.reshape(1,-1)
    y_hat = lasso.predict(X_test)
    y_hat_test = np.append(y_hat_test, y_hat)

