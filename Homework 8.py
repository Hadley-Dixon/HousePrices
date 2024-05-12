#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Homework 8, BSDS 200

Created on Sat May 11 18:11:19 2024

@author: hadleydixon
"""

# Link to GitHub repo: https://github.com/Hadley-Dixon/HousePrices

#%%

import pandas as pd
import numpy as np
import sklearn as sk
import sklearn.model_selection
import sklearn.tree
from xgboost import XGBClassifier
import sklearn.ensemble

#%%

df_labeled = pd.read_csv('/Users/hadleydixon/Desktop/Homework 8/HousePrices/house-prices-advanced-regression-techniques/train.csv')
df_test = pd.read_csv('/Users/hadleydixon/Desktop/Homework 8/HousePrices/house-prices-advanced-regression-techniques/test.csv')

# Function combines train/test dataset to handle missing/categorical data
def whole(df1, df2):
    return pd.concat((df1, df2), ignore_index = True)

# Function seperates train/test dataset after special cases handles
def seperate(df, df1_len, df2_len):
    df1 = df.iloc[:df1_len]
    df2 = df.iloc[df1_len:df1_len + df2_len]
    return df1, df2

#%%

# Impute values for missing data

whole_df = whole(df_labeled, df_test)

# Judgement call to drop columns from dataframe due to the lack of predictive power
whole_df.drop(columns=["MiscFeature"], inplace = True)
whole_df.drop(columns=["MiscVal"], inplace = True)

numerical = whole_df.select_dtypes(include=['int64', 'float64']).columns
for col in numerical:
    median = whole_df[col].median()
    whole_df[col].fillna(median, inplace=True)
    
categorical = whole_df.select_dtypes(include=['object']).columns
for col in categorical:
    mode = whole_df[col].mode()[0]
    whole_df[col].fillna(mode, inplace=True)
    
#%%

# Handle categorical data correctly

# Used the following source to learn about the .get_dummies() method in the pandas library
# SOURCE: https://www.geeksforgeeks.org/python-pandas-get_dummies-method/

whole_df = pd.get_dummies(whole_df, columns = categorical)
boolean = whole_df.select_dtypes(include=['bool']).columns
whole_df[boolean] = whole_df[boolean].astype(int)

#%%

# Return dataframe to its proper train/test split

labeled_len = len(df_labeled)
test_len = len(df_test)

df_labeled, df_test = seperate(whole_df, labeled_len, test_len)
df_test.drop(columns=["SalePrice"], inplace = True)
df_labeled.drop(columns=["Id"], inplace = True)

#%%

# Split the labeled data into training and validation subsets.
df_train, df_val = sk.model_selection.train_test_split(df_labeled, train_size = 0.8)

#df_train['SalePrice'] = df_train['SalePrice'].astype('category')
#df_val['SalePrice'] = df_val['SalePrice'].astype('category')

X_train = df_train.loc[:, df_train.columns != "SalePrice"]
X_val = df_val.loc[:, df_val.columns != "SalePrice"]

y_train = df_train["SalePrice"]
y_val = df_val["SalePrice"]

#%%
  
# Model 1: Decision Tree
tree = sk.tree.DecisionTreeClassifier().fit(X_train, y_train)
y_pred1 = tree.predict(X_val)

rmse1 = np.sqrt(np.mean((y_pred1 - y_val) ** 2))
print("-----------------------------------------")
print("RMSE (Decision Tree):", rmse1)
print("-----------------------------------------")

#%%

# Model 2: Random Forest
clf = sk.ensemble.RandomForestClassifier(n_estimators = 100)
clf.fit(X_train, y_train)
y_pred2 = clf.predict(X_val)

rmse2 = np.sqrt(np.mean((y_pred2 - y_val) ** 2))
print("-----------------------------------------")
print("RMSE:", rmse2)

#%%

# Model 3: Gradient Boosting
gb = XGBClassifier(n_estimators = 200, max_depth = 3, learning_rate = 0.1, subsample = 0.8, colsample_bytree = 0.8)
gb.fit(X_train, y_train)
y_pred3 = gb.predict(X_val)

rmse3 = np.sqrt(np.mean((y_pred3 - y_val) ** 2))
print("-----------------------------------------")
print("RMSE (Gradient Boosting):", rmse3)
print("-----------------------------------------")