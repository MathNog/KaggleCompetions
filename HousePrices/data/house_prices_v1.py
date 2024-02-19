"""
Python code for the Housing Price Regression Competition: https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques

Author: Matheus Nogueira
"""

import pandas as pd
import numpy as np
import os

from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder
import xgboost as xgb


def RMSE(y, y_hat):
    return np.sqrt(mean_squared_error(y, y_hat))

def MAPE(y, y_hat):
    return mean_absolute_percentage_error(y_pred=y_hat, y_true=y)

" Importing data "
path = os.getcwd() + "\\HousePrices\\data\\"
df_train = pd.read_csv(path+"train.csv")
df_test = pd.read_csv(path+"test.csv")
df_test_id = df_test.Id
"""
First things first, lets check for missing values in the dataset
Then, I will drop every collumn with more than 10% of missing values and fill the remaining ones with the column median
"""
cols_to_keep = []
for col in df_train.columns:
    N       = len(df_train[col])
    num_na  = df_train[col].isna().sum()
    perc_na = 100*num_na/N
    print("%s -> %.2f"%(col, perc_na))
    if perc_na <= 10:
        cols_to_keep.append(col)

df_train = df_train.loc[:,cols_to_keep]
df_test  = df_test.loc[:,cols_to_keep[0:-1]]

print("Shape dt_train = (%d, %d)"%(df_train.shape[0],df_train.shape[1]))
print("Shape dt_test = (%d, %d)"%(df_test.shape[0],df_test.shape[1]))



" Now I will fill every missing value from a numeric column as the column's median wile also filling every string missing value as NA"

for col in df_train.columns:
    if df_train[col].isna().sum() > 0 :
        print("NA prior = ",df_train[col].isna().sum())
        if isinstance(df_train[col].iloc[0], (int, float)):
            print("%s fill with median"%col)
            train_median = np.nanmedian(df_train[col])
            df_train[col] = df_train[col].astype('float64')
            df_train[col].fillna(train_median, inplace = True)
        else:
            print("%s fill with NA as string"%col)
            df_train[col].fillna("NA", inplace = True)
        print("NA after = ",df_train[col].isna().sum())

for col in df_test.columns:
    if df_test[col].isna().sum() > 0 :
        if isinstance(df_test[col].iloc[0], (int, float)):
            print("%s fill with median"%col)
            test_median  = np.nanmedian(df_test[col])
            df_test[col] = df_test[col].astype('float64')
            df_test[col].fillna(test_median, inplace = True)
        else:
            print("%s fill with NA as string"%col)
            df_test[col].fillna("NA", inplace = True)



" We have to take care of the string columns - lets do a One Hot Encoding to each of them "
for col in df_train.columns:
    if not pd.api.types.is_numeric_dtype(df_train[col][0]):
        print(col)
        encoder = OneHotEncoder()
        df_aux = pd.DataFrame(encoder.fit_transform(df_train[[col]]).toarray())
        unique_values = df_train[col].unique()
        df_aux.columns = [col+"_"+unique_values[i] for i in range(0,len(unique_values))]
        print("Len new columns = ",len(df_aux.columns))
        print("Total columns prior drop = ",df_train.shape[1])
        df_train.drop(col, axis = 1, inplace = True)
        print("Total columns after drop = ",df_train.shape[1])
        df_train = pd.concat([df_train, df_aux], axis = 1)
        print("Total columns after add = ",df_train.shape[1])

for col in df_test.columns:
    if not pd.api.types.is_numeric_dtype(df_test[col][0]):
        print(col)
        encoder = OneHotEncoder()
        df_aux = pd.DataFrame(encoder.fit_transform(df_test[[col]]).toarray())
        unique_values = df_test[col].unique()
        print(unique_values)
        df_aux.columns = [col+"_"+unique_values[i] for i in range(0,len(unique_values))]
        print("Len new columns = ",len(df_aux.columns))
        print("Total columns prior drop = ",df_test.shape[1])
        df_test.drop(col, axis = 1, inplace = True)
        print("Total columns after drop = ",df_test.shape[1])
        df_test = pd.concat([df_test, df_aux], axis = 1)
        print("Total columns after add = ",df_test.shape[1])

common_cols = list(set(df_test.columns) & set(df_train.columns))
sale_price = df_train.SalePrice

df_test = df_test.loc[:, common_cols]
df_train = df_train.loc[:, common_cols]
df_train['SalePrice'] = sale_price

print("Shape dt_train = (%d, %d)"%(df_train.shape[0],df_train.shape[1]))
print("Shape dt_test = (%d, %d)"%(df_test.shape[0],df_test.shape[1]))

" Now that our dataset is clean, lets divide the train dataset into a fit and validation datasets"

X = df_train.drop(columns="SalePrice")
y = df_train.SalePrice

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.33, random_state=42)

" Lets apply PCA into the dataset "
pca = PCA()
pca.fit(X_train)
X_train_pca = pd.DataFrame(pca.transform(X_train))
X_val_pca   = pd.DataFrame(pca.transform(X_val))

" And fit a LASSO model "
params = {"alpha":np.linspace(0,10,11), "random_state":[0, 123], "max_iter":[1000]}
lasso  = Lasso()
grid   = GridSearchCV(lasso, params, cv=5)
grid.fit(X_train_pca, y_train)

best_alpha = grid.best_params_["alpha"]
best_random_state = grid.best_params_["random_state"]

lasso = Lasso(alpha = best_alpha, random_state = best_random_state)
lasso.fit(X_train_pca, y_train)

y_hat_train = lasso.predict(X_train_pca)
y_hat_val   = lasso.predict(X_val_pca)

print(MAPE(y_train, y_hat_train))
print(MAPE(y_val, y_hat_val))

print(RMSE(y_train, y_hat_train))
print(RMSE(y_val, y_hat_val))

" Finally, lets fit the lasso model into the full train dataset and predict for the test dataset"

pca = PCA()
pca.fit(X)
X_pca = pd.DataFrame(pca.transform(X))

params = {"alpha":np.linspace(0,10,11), "random_state":[0, 123], "max_iter":[1000]}
lasso  = Lasso()
grid   = GridSearchCV(lasso, params, cv=5)
grid.fit(X_pca, y)

best_alpha = grid.best_params_["alpha"]
best_random_state = grid.best_params_["random_state"]

lasso = Lasso(alpha = best_alpha, random_state = best_random_state)
lasso.fit(X_pca, y)

y_hat = lasso.predict(X_pca)

print(MAPE(y, y_hat))
print(RMSE(y, y_hat))

X_test     = df_test
X_test_pca = pca.fit_transform(X_test)

y_hat_test = lasso.predict(X_test_pca)

prediction = pd.DataFrame({"Id":df_test_id, "SalePrice":y_hat_test})
prediction.to_csv("HousePrices\\test_prediction_v0.csv", index=False)

