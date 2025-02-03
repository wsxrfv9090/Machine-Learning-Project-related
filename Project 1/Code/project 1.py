import pandas as pd
import numpy as np

#churn_df = pd.read_csv('bank.data.csv')
file_path = r'D:\ImportanFiles\Coding Related\Repositories\Machine Learning project related\Project 1\Code\bank.data.csv'
churn_df = pd.read_csv(file_path)

#print(churn_df.head())
#print(churn_df.info())
#print(churn_df.nunique())

y = churn_df['Exited']

#print(churn_df.isnull().sum())
#print(churn_df[['CreditScore', 'Age', 'Tenure', 'NumOfProducts', 'Balance', 'EstimatedSalary']].describe())

# import matplotlib.pyplot as plt
# import seaborn as sns
# _,axss = plt.subplots(2, 3, figsize = [20, 10]) 
# sns.boxplot(x = 'Exited', y = 'CreditScore', data = churn_df, ax = axss[0][0])
# sns.boxplot(x = 'Exited', y = 'Age', data = churn_df, ax = axss[0][1])
# sns.boxplot(x = 'Exited', y = 'Tenure', data = churn_df, ax = axss[0][2])
# sns.boxplot(x = 'Exited', y = 'NumOfProducts', data = churn_df, ax = axss[1][0])
# sns.boxplot(x = 'Exited', y = 'Balance', data = churn_df, ax = axss[1][1])
# sns.boxplot(x = 'Exited', y = 'EstimatedSalary', data = churn_df, ax = axss[1][2])
# plt.show()

# _,axss = plt.subplots(2, 2, figsize = [20, 10]) 
# sns.countplot(x = 'Exited', hue = 'Geography', data = churn_df, ax = axss[0][0])
# sns.countplot(x = 'Exited', hue = 'Gender', data = churn_df, ax = axss[0][1])
# sns.countplot(x = 'Exited', hue = 'HasCrCard', data = churn_df, ax = axss[1][0])
# sns.countplot(x = 'Exited', hue = 'IsActiveMember', data = churn_df, ax = axss[1][1])
# plt.show()

to_drop = ['RowNumber', 'CustomerId', 'Surname', 'Exited']
X = churn_df.drop(to_drop, axis = 1)
#print(X.head())
#print(X.dtypes)
cat_cols = X.columns[X.dtypes == 'object']
num_cols = X.columns[(X.dtypes == 'float64') | (X.dtypes == 'int64')]
#print(num_cols)
#print(cat_cols)

from sklearn import model_selection

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = 0.25, stratify = y, random_state = 1)

#print('Training data has ' + str(X_train.shape[0]) + ' observation with ' + str(X_train.shape[1]) + ' features')
#print('Test data has ' + str(X_test.shape[0]) + ' observation with ' + str(X_test.shape[1]) + ' features')

#print(X_train.head())


#one hot encoding
#another way: get_dummies
from sklearn.preprocessing import OneHotEncoder

def OneHotEncoding(df, enc, categories):
    transformed = pd.DataFrame(enc.transform(df[categories]).toarray(), columns = enc.get_feature_names_out(categories))
    return pd.concat([df.reset_index(drop = True), transformed], axis = 1).drop(categories, axis = 1)

categories = ['Geography']
enc_ohe = OneHotEncoder()
enc_ohe.fit(X_train[categories])

X_train = OneHotEncoding(X_train, enc_ohe, categories)
X_test = OneHotEncoding(X_test, enc_ohe, categories)

#print(X_train.head())

from sklearn.preprocessing import OrdinalEncoder

categories = ['Gender']
enc_oe = OrdinalEncoder()
enc_oe.fit(X_train[categories])

X_train[categories] = enc_oe.transform(X_train[categories])
X_test[categories] = enc_oe.transform(X_test[categories])

#print(X_train.head())

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train[num_cols])

X_train[num_cols] = scaler.transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

#print(X_train.head())

