# import libraries necessary for this project
import os, sys, pickle

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import seaborn as sns

from datetime import date

from sklearn.model_selection import KFold, train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, roc_auc_score, auc, roc_curve
from sklearn.preprocessing import MinMaxScaler

import xgboost as xgb
import lightgbm as lgb

dfoff = pd.read_csv('datalab/ccf_offline_stage1_train.csv')
dftest = pd.read_csv('datalab/ccf_offline_stage1_test_revised.csv')

dfon = pd.read_csv('datalab/ccf_online_stage1_train.csv')

dfoff.head(5)
print('有优惠券，购买商品条数', dfoff[(dfoff['Date_received'].notnull()) & (dfoff['Date'].notnull())].shape[0])
print('无优惠券，购买商品条数', dfoff[(dfoff['Date_received'].isnull()) & (dfoff['Date'].notnull())].shape[0])
print('有优惠券，不购买商品条数', dfoff[(dfoff['Date_received'].notnull()) & (dfoff['Date'].isnull())].shape[0])
print('无优惠券，不购买商品条数', dfoff[(dfoff['Date_received'].isnull()) & (dfoff['Date'].isnull())].shape[0])
# 在测试集中出现的用户但训练集没有出现
print('1. User_id in training set but not in test set', set(dftest['User_id']) - set(dfoff['User_id']))
# 在测试集中出现的商户但训练集没有出现
print('2. Merchant_id in training set but not in test set', set(dftest['Merchant_id']) - set(dfoff['Merchant_id']))
print('Discount_rate 类型:',dfoff['Discount_rate'].unique())
print('Distance 类型:', dfoff['Distance'].unique())


# convert Discount_rate and Distance

def getDiscountType(row):
    if row != row:
        return 'null'
    elif ':' in row:
        return 1
    else:
        return 0


def convertRate(row):
    """Convert discount to rate"""

    #nan != nan
    if row != row:
        return 1.0
    elif ':' in row:
        rows = row.split(':')
        return 1.0 - float(rows[1]) / float(rows[0])
    else:
        return float(row)


def getDiscountMan(row):
    if row == row and ':' in row:
        rows = row.split(':')
        return int(rows[0])
    else:
        return 0


def getDiscountJian(row):
    if row == row and ':' in row:
        rows = row.split(':')
        return int(rows[1])
    else:
        return 0


def processData(df):
    # convert discunt_rate
    df['discount_rate'] = df['Discount_rate'].apply(convertRate)
    df['discount_man'] = df['Discount_rate'].apply(getDiscountMan)
    df['discount_jian'] = df['Discount_rate'].apply(getDiscountJian)
    df['discount_type'] = df['Discount_rate'].apply(getDiscountType)
    print(df['discount_rate'].unique())

    # convert distance
    df['distance'] = df['Distance'].replace(np.nan, -1).astype(int)
    print(df['distance'].unique())
    return df


dfoff = processData(dfoff)
dftest = processData(dftest)
a =1