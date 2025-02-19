import numpy as np
import pandas as pd

df = pd.read_csv(r'Practice projects\Data\coffee.csv')

# 1. Create a numpy array of the column 'Age'
# 2. Change the coffee type into binary values
df['Coffee Type'] = df['Coffee Type'].map({'Espresso': 1, 'Latte': 0})

# One-hot encode the 'Day' column
day_dummies = pd.get_dummies(df['Day'], prefix='Day')
df = pd.concat([df, day_dummies], axis=1)

df.drop('Day', axis=1, inplace=True)

# Change the data from false and true to 1 and 0
df = df.replace({False: 0, True: 1})

dfMatrix = df.to_numpy()
print(dfMatrix)

# 3. Add a column of index values to the numpy array
index = np.arange(1, len(df) + 1)
index = index.reshape(len(df), 1)
dfMatrix = np.hstack((index, dfMatrix))
print(dfMatrix)


