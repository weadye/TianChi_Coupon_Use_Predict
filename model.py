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
print("Data:")
print(dfoff.head(2))
print(dftest.head(2))

date_received = dfoff['Date_received'].unique()
date_received = sorted(date_received[date_received == date_received])

date_buy = dfoff['Date'].unique()
date_buy = sorted(date_buy[date_buy == date_buy])

print('优惠券收到日期从',date_received[0],'到', date_received[-1])
print('消费日期从', date_buy[0], '到', date_buy[-1])
tmp1 = dfoff[dfoff['Date_received'] == dfoff['Date_received']][['Date_received', 'Date']]
tmp1.Date = 1
couponbydate = tmp1.groupby(['Date_received'], as_index=False).count()
couponbydate.columns = ['Date_received','count']
tmp = dfoff[(dfoff['Date'] == dfoff['Date']) & (dfoff['Date_received'] == dfoff['Date_received'])]
buybydate = dfoff[(dfoff['Date'] == dfoff['Date']) & (dfoff['Date_received'] == dfoff['Date_received'])][['Date_received', 'Date']].groupby(['Date_received'], as_index=False).count()
buybydate.columns = ['Date_received','count']
sns.set_style('ticks')
sns.set_context("notebook", font_scale= 1.4)
plt.figure(figsize = (12,8))
date_received_dt = pd.to_datetime(date_received, format='%Y%m%d')

plt.subplot(211)
plt.bar(date_received_dt, couponbydate['count'], label = 'number of coupon received' )
plt.bar(date_received_dt, buybydate['count'], label = 'number of coupon used')
plt.yscale('log')
plt.ylabel('Count')
plt.legend()

plt.subplot(212)
plt.bar(date_received_dt, buybydate['count']/couponbydate['count'])
plt.ylabel('Ratio(coupon used/coupon received)')
plt.tight_layout()
plt.show()

def getWeekday(row):
    if row == 'null':
        return row
    else:
        return date(int(row[0:4]), int(row[4:6]), int(row[6:8])).weekday() + 1

dfoff['weekday'] = dfoff['Date_received'].astype(str).apply(getWeekday)
dftest['weekday'] = dftest['Date_received'].astype(str).apply(getWeekday)
# weekday_type :  周六和周日为1，其他为0
dfoff['weekday_type'] = dfoff['weekday'].apply(lambda x : 1 if x in [6,7] else 0 )
dftest['weekday_type'] = dftest['weekday'].apply(lambda x : 1 if x in [6,7] else 0 )