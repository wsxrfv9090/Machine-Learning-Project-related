# Assume you have new data for prediction (same format as training data)
import joblib
import pandas as pd
from sklearn import preprocessing


# Load the saved model from the file
clf = joblib.load('stock_prediction_model.pkl')

file_path = r'D:\ImportanFiles\Coding Related\Repositories\Machine Learning project related\Project 2\Data\test.csv'
stock_df = pd.read_csv(file_path)

stock_df['HL_PCT'] = (stock_df['Highest Price'] - stock_df['Closing Price']) / stock_df['Closing Price'] * 100.0
stock_df['PCT_change'] = (stock_df['Closing Price'] - stock_df['Opening Price']) / stock_df['Opening Price'] * 100.0

stock_df = stock_df[['Closing Price', 'HL_PCT', 'PCT_change', 'Trading Volume']]

stock_df['Trading Volume'] = stock_df['Trading Volume'].replace({',': ''}, regex=True).astype(float)

print(stock_df)

# Preprocess the new data (same as the training data)
new_data_scaled = preprocessing.scale(stock_df)

# Use the loaded model to predict the label (closing price 7 days later)
predicted_price = clf.predict(stock_df)

print(f"Predicted closing price 7 days later: {predicted_price[0]}")
