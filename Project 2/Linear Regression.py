# This file is not fully finished, but it contained some of the thoughts I was thinking while doing this, so I kept it in the repo.
import pandas as pd
import math, datetime
import numpy as np
from sklearn import preprocessing, model_selection
from sklearn.linear_model import LinearRegression
# Use to plot the data
import matplotlib.pyplot as plt
# Use to make data look better
from matplotlib import style

# Use to specify the style of the plot
style.use('ggplot')


#Selecting files
file_path = r'Project 2\Data\600873_metadata_utf8.csv'
stock_df = pd.read_csv(file_path)

#Calculating high-low percentage and percentage change
stock_df['HL_PCT'] = (stock_df['Highest Price'] - stock_df['Closing Price']) / stock_df['Closing Price'] * 100.0
stock_df['PCT_change'] = (stock_df['Closing Price'] - stock_df['Opening Price']) / stock_df['Opening Price'] * 100.0

#Drop useless data
stock_df = stock_df[['Closing Price', 'HL_PCT', 'PCT_change', 'Trading Volume']]

# Remove commas and convert to numeric, since in metadata the trading volume contain commas
stock_df['Trading Volume'] = stock_df['Trading Volume'].replace({',': ''}, regex=True).astype(float)

#Choose which data column is the one that will be predicted
forecast_col = 'Closing Price'

#Fill NaN values in case pandas can't handle them
#stock_df.fillna(-999999, inplace = True)

#Give a row number to shift, in this case: len(stock_df) = 6884, math.ceil make it 7, then type convert to integer, meaning that later .shift function will shift closing price upwards 7 rows and fill them into Label rows. Basically we are using the datas to predict the closing price 7 days later
forecast_out = int(math.ceil(0.001*len(stock_df)))

#Anwers
stock_df['Label'] = stock_df[forecast_col].shift(-forecast_out)
print(forecast_out)

# Split the where X is the features and y is the answer
X = np.array(stock_df.drop(['Label'], axis = 1))
X = preprocessing.scale(X)
X = X[:-forecast_out]
X_lately = X[-forecast_out:]

stock_df.dropna(inplace = True)
y = np.array(stock_df['Label'])

# Scale the data

# Split the data, 80% for training and 20% for testing (randomly)
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = 0.2)

# Getting the classifier
clf = LinearRegression(n_jobs = -1)
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)

forecast_set = clf.predict(X_lately)

print("Forecast result: " + str(forecast_set) + " with accuracy of " + str(accuracy) + " and forecast out of " + str(forecast_out) + " days.")

stock_df['Forecast'] = np.nan

