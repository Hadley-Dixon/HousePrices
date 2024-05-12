#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Homework 8, BSDS 200

Created on Sat May 11 18:11:19 2024

@author: hadleydixon
"""

# Link to GitHub repo: TBD

#%%

import pandas as pd
import numpy as np
import sklearn as sk
import sklearn.model_selection
import sklearn.tree
from xgboost import XGBClassifier

#%%

df_labeled = pd.read_csv('/Users/hadleydixon/Desktop/BSDS200/Homework /Homework 7/spaceship-titanic/train.csv')
df_test = pd.read_csv('/Users/hadleydixon/Desktop/BSDS200/Homework /Homework 7/spaceship-titanic/test.csv')