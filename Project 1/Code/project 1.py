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


#ordinal encoder
from sklearn.preprocessing import OrdinalEncoder

categories = ['Gender']
enc_oe = OrdinalEncoder()
enc_oe.fit(X_train[categories])

X_train[categories] = enc_oe.transform(X_train[categories])
X_test[categories] = enc_oe.transform(X_test[categories])

#print(X_train.head())

#Scaling
# Scale the data, using standardization
# standardization (x-mean)/std
# normalization (x-x_min)/(x_max-x_min) ->[0,1]

# 1. speed up gradient descent
# 2. same scale
# 3. algorithm requirments

# for example, use training data to train the standardscaler to get mean and std
# apply mean and std to both training and testing data.
# fit_transform does the training and applying, transform only does applying.
# Because we can't use any info from test, and we need to do the same modification
# to testing data as well as training data

# https://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html#sphx-glr-auto-examples-preprocessing-plot-all-scaling-py
# https://scikit-learn.org/stable/modules/preprocessing.html

# min-max example: (x-x_min)/(x_max-x_min)
# [1,2,3,4,5,6,100] -> fit(min:1, max:6) (scalar.min = 1, scalar.max = 6) -> transform [(1-1)/(6-1),(2-1)/(6-1)..]
# scalar.fit(train) -> min:1, max:100
# scalar.transform(apply to x) -> apply min:1, max:100 to X_train
# scalar.transform -> apply min:1, max:100 to X_test

# scalar.fit -> mean:1, std:100
# scalar.transform -> apply mean:1, std:100 to X_train
# scalar.transform -> apply mean:1, std:100 to X_test
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train[num_cols])

X_train[num_cols] = scaler.transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

#print(X_train.head())

#@title build models
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

# Logistic Regression
classifier_logistic = LogisticRegression()

# K Nearest Neighbors
classifier_KNN = KNeighborsClassifier()

# Random Forest
classifier_RF = RandomForestClassifier()

# Train the model
classifier_logistic.fit(X_train, y_train)

# Prediction of test data
classifier_logistic.predict(X_test)

# Accuracy of test data
print(classifier_logistic.score(X_test, y_test))

#Finding the optimal parameters:
#Loss/cost function --> (wx + b - y) ^2 + ƛ * |w| --> ƛ is a hyperparameter
from sklearn.model_selection import GridSearchCV

# helper function for printing out grid search results
def print_grid_search_metrics(gs):
    print ("Best score: " + str(gs.best_score_))
    print ("Best parameters set:")
    best_parameters = gs.best_params_
    for param_name in sorted(best_parameters.keys()):
        print(param_name + ':' + str(best_parameters[param_name]))
        

# Possible hyperparamter options for Logistic Regression Regularization
# Penalty is choosed from L1 or L2
# C is the 1/lambda value(weight) for L1 and L2
# solver: algorithm to find the weights that minimize the cost function

# ('l1', 0.01)('l1', 0.05) ('l1', 0.1) ('l1', 0.2)('l1', 1)
# ('12', 0.01)('l2', 0.05) ('l2', 0.1) ('l2', 0.2)('l2', 1)
parameters = {
    'penalty':('l2','l1'),
    'C':(0.01, 0.05, 0.1, 0.2, 1)
}

Grid_LR = GridSearchCV(LogisticRegression(solver='liblinear'),parameters, cv = 5)
Grid_LR.fit(X_train, y_train)

# the best hyperparameter combination
# C = 1/lambda
print_grid_search_metrics(Grid_LR)