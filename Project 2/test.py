import pandas as pd
import math
import numpy as np
from sklearn import preprocessing, model_selection
from sklearn.linear_model import LinearRegression
import joblib


#Selecting files
file_path = r'D:\ImportanFiles\Coding Related\Repositories\Machine Learning project related\Project 2\Data\600873_metadata_utf8.csv'
stock_df = pd.read_csv(file_path)

#Calculating high-low percentage and percentage change
stock_df['HL_PCT'] = (stock_df['Highest Price'] - stock_df['Closing Price']) / stock_df['Closing Price'] * 100.0
stock_df['PCT_change'] = (stock_df['Closing Price'] - stock_df['Opening Price']) / stock_df['Opening Price'] * 100.0

#Drop useless data
stock_df = stock_df[['Closing Price', 'HL_PCT', 'PCT_change', 'Trading Volume']]

#Choose which data column is the one that will be predicted
forecast_col = 'Closing Price'

#Fill NaN values in case pandas can't handle them
#stock_df.fillna(-999999, inplace = True)

#Give a row number to shift, in this case: len(stock_df) = 6884, math.ceil make it 7, then type convert to integer, meaning that later .shift function will shift closing price upwards 7 rows and fill them into Label rows. Basically we are using the datas to predict the closing price 7 days later
forecast_out = int(math.ceil(0.0002*len(stock_df)))
stock_df['Label'] = stock_df[forecast_col].shift(-forecast_out)
print(forecast_out)

stock_df.dropna(inplace = True)

# Remove commas and convert to numeric, since in metadata the trading volume contain commas
stock_df['Trading Volume'] = stock_df['Trading Volume'].replace({',': ''}, regex=True).astype(float)

# Split the data by time, 75% for training and 25% for testing
train_size = int(0.70 * len(stock_df))
test_size = int(0.70 * len(stock_df) + 0.10 * len(stock_df))
train_data = stock_df[:train_size]
test_data = stock_df[train_size:test_size]
to_be_predicted = stock_df[test_size:]

# Prepare the features and labels
X_train = np.array(train_data.drop(['Label'], axis=1))
y_train = np.array(train_data['Label'])
X_test = np.array(test_data.drop(['Label'], axis=1))
y_test = np.array(test_data['Label'])
z_predict = np.array(to_be_predicted.drop(['Label'], axis=1))
z_actual_answer = np.array(to_be_predicted['Label'])

# Train the model
clf = LinearRegression(n_jobs=-1)
clf.fit(X_train, y_train)

# Evaluate the model
accuracy = clf.score(X_test, y_test)

# Save the trained model
#joblib.dump(clf, 'stock_prediction_model.pkl')

print(accuracy)

z_predict = clf.predict(z_predict)

mean_difference = (z_actual_answer - z_predict)/z_actual_answer
print(mean_difference)

flag = 0

for i in range(len(mean_difference)):
    if mean_difference[i] > 0.03 or mean_difference[i] <-0.03:
        flag += 1

print(len(mean_difference))
print(flag)