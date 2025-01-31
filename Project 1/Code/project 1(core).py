import pandas as pd
import numpy as np

churn_df = pd.read_csv('bank.data.csv')
y = churn_df['Exited']

to_drop = ['RowNumber', 'CustomerId', 'Surname', 'Exited']
X = churn_df.drop(to_drop, axis = 1)

cat_cols = X.columns[X.dtypes == 'object']
num_cols = X.columns[(X.dtypes == 'float64') | (X.dtypes == 'int64')]