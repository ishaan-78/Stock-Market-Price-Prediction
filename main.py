import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import ConfusionMatrixDisplay
from xgboost import XGBClassifier
from sklearn import metrics

import warnings
warnings.filterwarnings('ignore')

col_names = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']
df = pd.read_csv("prices.csv", header=None, names=col_names)
# df['Date'] = pd.to_datetime(df['Date'])
# df = df.sort_values('Date')
# print(df.head())
# print(df.shape)
# print(df.describe())
# print(df.info())

plt.figure(figsize=(12,5))
plt.plot(df['Close'])
plt.title('Stock Close Price', fontsize=15)
plt.ylabel('Price in dollars.')
# plt.show()

if (df[df['Close']==df['Adj Close']].shape == df.shape):
    df = df.drop(['Adj Close'], axis=1)

# print(df.isnull().sum())

features = ['Open','High','Low','Close','Volume']
plt.subplots(figsize=(13,6))
for i, col in enumerate(features):
    plt.subplot(2,3,i+1)
    sb.distplot(df[col])
# plt.show()

plt.subplots(figsize=(13,6))
for i, col in enumerate(features):
    plt.subplot(2,3,i+1)
    sb.boxplot(x=df[col])
# plt.show()

splitted = df['Date'].str.split('/', expand=True)
df['day'] = splitted[1].astype('int')
df['month'] = splitted[0].astype('int')
df['year'] = splitted[2].astype('int')
# print(df.head())

df['is_quarter_end'] = np.where(df['month']%3==0,1,0)
# print(df.head())

data_grouped = df.drop('Date',axis=1).groupby('year').mean()
plt.subplots(figsize=(12,5))
for i, col in enumerate(['Open','High','Low','Close']):
    plt.subplot(2,2,i+1)
    data_grouped[col].plot.bar()
# plt.show()

# print(df.drop('Date', axis=1).groupby('is_quarter_end').mean())

df['open-close']  = df['Open'] - df['Close']
df['low-high']  = df['Low'] - df['High']
df['target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)

plt.figure(figsize=(5,5))
plt.pie(df['target'].value_counts().values, labels=[0, 1], autopct='%1.1f%%')
# plt.show()

plt.figure(figsize=(10,10))
sb.heatmap(df.drop('Date',axis=1).corr() > 0.9, annot=True, cbar=False)
# plt.show()

features = df[['open-close','low-high','is_quarter_end']]
target = df['target']
scalar = StandardScaler()
features = scalar.fit_transform(features)
X_train, X_valid, Y_train, Y_valid = train_test_split(features, target, test_size=0.1, random_state=2022)

models = [LogisticRegression(), SVC(kernel='poly',probability=True), XGBClassifier()]
for i in range(len(models)):
    models[i].fit(X_train, Y_train)
    print(f'{models[i]} : ')
    print('Training Accuracy : ', metrics.roc_auc_score(Y_train,models[i].predict_proba(X_train)[:,1]))
    print('Validation Accuracy : ', metrics.roc_auc_score(Y_valid, models[i].predict_proba(X_valid)[:,1]))
    print()
for i in range(3):
    ConfusionMatrixDisplay.from_estimator(models[i], X_valid, Y_valid)
plt.show()

# N = 7
# latest_data = df.iloc[-N:].copy()
#
# latest_data['open-close'] = latest_data['Open'] - latest_data['Close']
# latest_data['low-high'] = latest_data['Low'] - latest_data['High']
# latest_data['is_quarter_end'] = np.where(latest_data['month'] % 3 == 0, 1, 0)
#
# future_features = latest_data[['open-close', 'low-high', 'is_quarter_end']]
# future_scaled = scalar.transform(future_features)
# predictions = models[0].predict(future_scaled)
#
# # Print predictions
# print("\nFuture Predictions (1 = Up, 0 = Down):")
# for i, pred in enumerate(predictions):
#     date = pd.to_datetime(latest_data['Date'].iloc[i])
#     print(f"{date.date()} -> {'Up' if pred == 1 else 'Down'}")