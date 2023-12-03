import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import seaborn as sns
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error as MSE


random.seed(42)
np.random.seed(42)

df_train = pd.read_csv('https://raw.githubusercontent.com/hse-mlds/ml/main/hometasks/HT1/cars_train.csv')
df_test = pd.read_csv('https://raw.githubusercontent.com/hse-mlds/ml/main/hometasks/HT1/cars_test.csv')

l1 = [x for x in np.array(df_train.columns) if x != 'selling_price']
df_train = df_train.drop_duplicates(l1, keep='first')
df_train = df_train.reset_index(drop=True)

df_tmp = df_train[df_train['torque'].notna()]['torque']
df_tmp = df_tmp.str.replace(',', '.')
match_3 = df_tmp.str.extract('(\d+\.\d+|\d+)[^.\d]+(\d+\.\d+|\d+)[^.\d]+(\d+\.\d+|\d+)')
match_2 = df_tmp.str.extract('(\d+\.\d+|\d+)[^.\d]+(\d+\.\d+|\d+)')
match_1 = df_tmp.str.extract('(\d+\.\d+|\d+)')
match_3[0] = match_3[0].fillna(match_2[0])
match_3[1] = match_3[1].fillna(match_2[1])
match_3[0] = match_3[0].fillna(match_1[0])
match_3[1] = match_3[1].str.replace('.', '').astype(float)
match_3[2] = match_3[2].str.replace('.', '').astype(float)
match_3.loc[df_tmp.str.lower().str.contains('kgm'), 0] = match_3[0].astype(float) * 9.81
match_3['torque'] = match_3[0]
match_3['max_torque_rpm'] = match_3[[1,2]].max(axis=1)
df_train.drop(['torque'], axis=1, inplace=True)
df_train['max_torque_rpm'] = match_3['max_torque_rpm']
df_train['torque'] = match_3['torque'].astype(float)

df_tmp = df_test[df_test['torque'].notna()]['torque']
df_tmp = df_tmp.str.replace(',', '.')
match_3 = df_tmp.str.extract('(\d+\.\d+|\d+)[^.\d]+(\d+\.\d+|\d+)[^.\d]+(\d+\.\d+|\d+)')
match_2 = df_tmp.str.extract('(\d+\.\d+|\d+)[^.\d]+(\d+\.\d+|\d+)')
match_1 = df_tmp.str.extract('(\d+\.\d+|\d+)')
match_3[0] = match_3[0].fillna(match_2[0])
match_3[1] = match_3[1].fillna(match_2[1])
match_3[0] = match_3[0].fillna(match_1[0])
match_3[1] = match_3[1].str.replace('.', '').astype(float)
match_3[2] = match_3[2].str.replace('.', '').astype(float)
match_3.loc[df_tmp.str.lower().str.contains('kgm'), 0] = match_3[0].astype(float) * 9.81
match_3['torque'] = match_3[0]
match_3['max_torque_rpm'] = match_3[[1,2]].max(axis=1)
df_test.drop(['torque'], axis=1, inplace=True)
df_test['max_torque_rpm'] = match_3['max_torque_rpm']
df_test['torque'] = match_3['torque'].astype(float)


df_train['mileage'] = df_train['mileage'].str.split(' ', expand=True)[0].astype(float)
df_train['engine'] = df_train['engine'].str.split(' ', expand=True)[0].astype(float)
df_train['max_power'] = df_train['max_power'].str.split(' ', expand=True)[0].replace('', None).astype(float)

df_test['mileage'] = df_test['mileage'].str.split(' ', expand=True)[0].astype(float)
df_test['engine'] = df_test['engine'].str.split(' ', expand=True)[0].astype(float)
df_test['max_power'] = df_test['max_power'].str.split(' ', expand=True)[0].replace('', None).astype(float)

gaps = df_train.isna().sum()
gaps = list(gaps[gaps != 0].index)
dict_gaps = dict(df_train[gaps].median())
with open('gaps_to_median.pkl', 'wb') as handle:
    pickle.dump(dict_gaps, handle)

df_test[gaps] = df_test[gaps].fillna(df_train[gaps].median())
df_train[gaps] = df_train[gaps].fillna(df_train[gaps].median())

df_train = df_train.astype({'engine': 'int', 'seats': 'int'})
df_test = df_test.astype({'engine': 'int', 'seats': 'int'})

y_train = df_train['selling_price']
X_train = df_train.select_dtypes(include='number').drop('selling_price', axis=1)

y_test = df_test['selling_price']
X_test = df_test.select_dtypes(include='number').drop('selling_price', axis=1)

cols = X_train.columns
scaler = StandardScaler()
scaler.fit(X_train)
features = scaler.transform(X_train)
X_train = pd.DataFrame(features, columns = cols)
X_test = pd.DataFrame(scaler.transform(X_test), columns = cols)

pickle.dump(scaler, open('scaler.pkl','wb'))

X_train_cat = df_train.drop(['name', 'selling_price'], axis=1)
X_test_cat = df_test.drop(['name', 'selling_price'], axis=1)

one = OneHotEncoder()

one.fit(X_train_cat.select_dtypes(include='object'))
res = one.transform(X_train_cat.select_dtypes(include='object')).toarray()
X_train_cat = pd.concat([X_train, pd.DataFrame(columns=np.arange(res.shape[1]).astype(str), data=res)], axis=1)
res = one.transform(X_test_cat.select_dtypes(include='object')).toarray()
X_test_cat = pd.concat([X_test, pd.DataFrame(columns=np.arange(res.shape[1]).astype(str), data=res)], axis=1)
cat_columns = np.arange(res.shape[1]).astype(str)
pickle.dump(cat_columns, open('cat_columns.pkl','wb'))
pickle.dump(one, open('one.pkl','wb'))

X_train_cat['power/engine'] = X_train_cat['max_power'] / X_train_cat['engine']
X_test_cat['power/engine'] = X_test_cat['max_power'] / X_test_cat['engine']

X_train_cat['year_squared'] = X_train_cat['year'] ** 2
X_test_cat['year_squared'] = X_test_cat['year'] ** 2

y_train = np.log(y_train + 1)
y_test = np.log(y_test + 1)

rdg = Ridge(alpha = 298.364724028334)
rdg.fit(X_train_cat, y_train)
tr_pred = rdg.predict(X_train_cat)

with open('rgd.pkl', 'wb') as f:
    pickle.dump(rdg, f)

print(MSE(y_test, te_pred), r2_score(y_test, te_pred))