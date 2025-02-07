#data #data_processing #data_preprocessing #data_conversion #file_manipulation #py_lib_pandas #py_lib_math #financial_prediction #financial_index #featuer_processing #numerical_features #NaN #py_lib_numpy #py_lib_sklearn
## 1. Data Preprocessing
```python
import pandas as pd
import math
  
file_path = r'D:\ImportanFiles\Coding Related\Repositories\Machine Learning project related\Project 2\Data\600873_metadata_utf8.csv'
stock_df = pd.read_csv(file_path)
  
stock_df['HL_PCT'] = (stock_df['Highest Price'] - stock_df['Closing Price']) / stock_df['Closing Price'] * 100.0
stock_df['PCT_change'] = (stock_df['Closing Price'] - stock_df['Opening Price']) / stock_df['Opening Price'] * 100.0
  
stock_df = stock_df[['Trading Time', 'Closing Price', 'HL_PCT', 'PCT_change', 'Trading Volume']]
```
In this code:
**HL_PCT (High-Low Percentage):**  
This feature is calculated as the percentage difference between the highest price and the closing price. It provides insight into the daily volatility of a stock. In some cases, stock prices can swing between the opening and closing prices, and the high-low range often gives a better idea of price movement. The **HL_PCT** can act as a measure of the "spread" or "range" for the day, which might provide more meaningful input to the model compared to using just the opening or highest price directly.
$$
\text{HL\_PCT} = \frac{\text{Highest Price} - \text{Closing Price}}{\text{Closing Price}} \times 100
$$
**PCT_change (Percentage Change):**  
This feature shows the percentage difference between the closing and opening prices. It captures how much the price has changed during the trading day in relative terms, which is important to understand the stock's daily price movement. The teacher likely included this because it highlights how volatile the stock's price is and reflects daily trends that are predictive of future price movements.
$$
\text{PCT\_change} = \frac{\text{Closing Price} - \text{Opening Price}}{\text{Opening Price}} \times 100
$$
Using these indicators allows the model to focus on the relationship between price changes rather than just absolute prices, making the model more sensitive to price fluctuations and trends. While the opening or highest prices might tell you some things, percentage-based features like **HL_PCT** and **PCT_change** can offer better relative measures of stock price dynamics.

## 2. Data Features and Labels
```python
forecast_col = 'Closing Price'
stock_df.fillna(-999999, inplace = True)
  
forecast_out = int(math.ceil(0.001*len(stock_df)))
  
stock_df['Label'] = stock_df[forecast_col].shift(-forecast_out)
stock_df.dropna(inplace = True)

print(stock_df.head())
print(stock_df.tail())

# Remove commas and convert to numeric
stock_df['Trading Volume'] = stock_df['Trading Volume'].replace({',': ''}, regex=True).astype(float)
```
### Managing `NaN` values
For pandas, you can't really work with `NaN` values, so the best method is to always make sure you don't have `NaN` values, and in this code, this was done in this code by `.dropna()` and `.fillna`
Important notes, make sure to drop tail useless information which doesn't belong to the data before handling this, you can use `dropna` directly to that, but you have to be careful about the potential data corruption, since if there's some `NaN` in the data column that you do not wish to drop, so the best way to do this, is to drop specific lines, for example to drop the last few lines with `NaN` values:
```python
df.iloc[-10:] = df.iloc[-10:].dropna()
```
Using `iloc` like this selects the last ten lines, and `dropna` will drop the ones in this range that contains `NaN` values.

```python
inplace = True
```
In this code, this means to modify and drop the data directly from the `stock_df` instead of making a copy.


### Numerical Features Handling.

###### Output:
Head:

| Trading Time | Closing Price | HL_PCT   | PCT_change | Trading Volume | Label |
| ------------ | ------------- | -------- | ---------- | -------------- | ----- |
| 2/17/1995    | 5.95          | 5.042017 | 8.181818   | 11,954,900     | 8.10  |
| 2/20/1995    | 5.76          | 3.298611 | -0.860585  | 3,859,200      | 8.23  |
| 2/21/1995    | 5.88          | 0.000000 | 2.260870   | 2,091,500      | 8.18  |
| 2/22/1995    | 8.32          | 4.447115 | 41.979522  | 16,299,000     | 8.90  |
| 2/23/1995    | 9.99          | 0.000000 | 25.187970  | 18,309,200     | 10.53 |
Tail:

| Trading Time | Closing Price | HL_PCT    | PCT_change | Trading Volume | Label |
|--------------|---------------|-----------|------------|----------------|-------|
| 1/10/2025    | 9.44          | 1.800847  | -1.666667  | 14,697,048     | 9.72  |
| 1/13/2025    | 9.61          | 0.416233  | 2.125399   | 25,370,178     | 9.78  |
| 1/14/2025    | 9.67          | 1.034126  | 0.729167   | 20,879,042     | 9.76  |
| 1/15/2025    | 9.79          | 0.817160  | 1.240951   | 26,232,207     | 9.83  |
| 1/16/2025    | 9.78          | 1.635992  | -0.710660  | 18,791,900     | 10.03 |

## 3. Training and Testing
```python
X = np.array(stock_df.drop(['Label'], axis = 1))
y = np.array(stock_df['Label'])
  
X = preprocessing.scale(X)
  
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = 0.2)
  
clf = LinearRegression()
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)
  
print(accuracy)
```
`np.array` sets the the chosen column into an array in linear algebra.
