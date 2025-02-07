### 1. Adjust the Model
1. The score is really high, but the outcome is not, to solve this, this Linear Regression's normal distribution should be adjusted to get validate score, more explanation here: [[No.1 Regression Intro - Some explanations]]
2. After 1st step the accuracy would be significantly low
3. Use a time based function to split instead of using `train_test_split` which is random to get more validate result.
4. After this, if the model is still bad, can add some new days to train, for now the model is using 1 day's data to predict closing price 7 days later, you can either make it train to use 2 consecutive days to predict 7 days later or make it train to use 1 day to predict 2 days later, to improve the accuracy

### 2. `fillna` Usage
I set the `NaN` values to -999999 which is impractical, to fix this, more explanation here: [[No.1 Regression Intro - Some explanations]]

### 3. Model selection
If the model is wrong or overfitting or under-fitting, try to use other models. 

### 4. Feature processing
Consider technical indicators like the Relative Strength Index (RSI), Moving Average Convergence Divergence (MACD), or Exponential Moving Average (EMA) for enhanced prediction.

### 5. Scaling Method
More in [[No.1 Regression Intro - Some explanations]].
Note that after scale you have to record the mean and std used to scale
Because the model is trained in scaled data, and when actually use it to predict, it will predict the scaled closing price, you have to un scale it to get the actual prediction.

