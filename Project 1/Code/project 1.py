import pandas as pd
import numpy as np

#Stores the file as a pandas object
churn_df = pd.read_csv('bank.data.csv')

#print(churn_df.head())
#print(churn_df.info())
#print(churn_df.nunique())

#Get target variable, Meaning that the user have churned or not, this is the answer for the machine
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

#Get feature space by dropping useless features
to_drop = ['RowNumber', 'CustomerId', 'Surname', 'Exited']
X = churn_df.drop(to_drop, axis = 1)
#print(X.head())
#print(X.dtypes)
cat_cols = X.columns[X.dtypes == 'object']
num_cols = X.columns[(X.dtypes == 'float64') | (X.dtypes == 'int64')]
#print(num_cols)
#print(cat_cols)