from sklearn import cross_validation, grid_search, metrics, ensemble
import xgboost as xgb
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import warnings
from sklearn.externals import joblib

pd.set_option('display.max_columns', None)
#显示所有行
pd.set_option('display.max_rows', None)
#设置value的显示长度为100，默认为50
pd.set_option('max_colwidth',100)
warnings.filterwarnings('ignore')
plt.style.use('ggplot')

df = pd.read_csv('train.csv')
df = df.sample(frac=0.001)

songs = pd.read_csv('songs.csv')
df = pd.merge(df, songs, on= 'song_id', how = 'left')
del songs

members = pd.read_csv('members.csv')
df = pd.merge(df, members, on = 'msno', how = 'left')
del members
# df.head()

df.isnull().sum()/df.isnull().count()*100

for i in df.select_dtypes(include = ['object']).columns:
    df[i][df[i].isnull()] = 'unknown'
df = df.fillna(value=0)
# print(df.info())

# 创建时间栏
df.registration_init_time = pd.to_datetime(df.registration_init_time, format='%Y%m%d', errors='raise')
df['registration_init_time_year'] = df['registration_init_time'].dt.year
df['registration_init_time_month'] = df['registration_init_time'].dt.month
df['registration_init_time_day'] = df['registration_init_time'].dt.day

df.expiration_date = pd.to_datetime(df.expiration_date,  format='%Y%m%d', errors='ignore')
df['expiration_date_year'] = df['expiration_date'].dt.year
df['expiration_date_month'] = df['expiration_date'].dt.month
df['expiration_date_day'] = df['expiration_date'].dt.day

df['registration_init_time'] = df['registration_init_time'].astype('category')
df['expiration_date'] = df['expiration_date'].astype('category')
# print(df.head(5))

for col in df.select_dtypes(include=['object']).columns:
    df[col] = df[col].astype('category')

# Encoding categorical features
for col in df.select_dtypes(include=['category']).columns:
    df[col] = df[col].cat.codes

fig1 = plt.figure(figsize=[7,5])
sns.heatmap(df.corr())

df.drop(['expiration_date', 'lyricist'], 1)

# 随机森林

model = ensemble.RandomForestClassifier(n_estimators=250, max_depth=25)
model.fit(df[df.columns[df.columns != 'target']], df.target)
joblib.dump(model,'RandomForest.m')
# print(cross_validation.cross_val_score(model,df[df.columns[df.columns != 'target']],df.target,n_jobs=-1,cv=3))

df_plot = pd.DataFrame({'features':df.columns[df.columns != 'target'], 'importances':model.feature_importances_})
df_plot = df_plot.sort_values('importances', ascending=False)
# print(df_plot)

fig2 = plt.figure(figsize=[11,5])
sns.barplot(x = df_plot.importances, y = df_plot.features)
plt.title('Importance of Features Plot')
plt.show()


df.drop(df_plot.features[df_plot.importances < 0.04].tolist(), 1)
# print(df.head())

# XGboost

target = df.pop('target')
train_data, test_data, train_labels, test_labels = cross_validation.train_test_split(df, target, test_size=0.3)
del df

model = xgb.XGBClassifier(learning_rate=0.1, max_depth=15, min_child_weight=5, n_estimators=250)
model.fit(train_data, train_labels)
joblib.dump(model,'XGboost.m')
predict_labels = model.predict(test_data)
print(metrics.classification_report(test_labels, predict_labels))

