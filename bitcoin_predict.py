import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

#reading the data and converting to a dataframe
df = pd.read_csv('bitcoin_data.csv')

df = df[['Price', 'Open', 'High', 'Low', 'Change %']]
df['HL_PCT'] = ((df['High'] - df['Low']) / df['Low']) * 100
df = df[['Price', 'HL_PCT', 'Change %']]

# the coloumn which we want to predict
forecast_coloumn = 'Price'
df.fillna(value = -99999, inplace=True)

# the number of days to predict
forecast_out = 7
df['Label'] = df[forecast_coloumn].shift(forecast_out)

# Drop specified labels from rows or columns
X = np.array(df.drop(['Label'], 1))

X = preprocessing.scale(X)

X_current = X[:forecast_out]

X = X[forecast_out:]

df.dropna(inplace=True)

y = np.array(df['Label'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
'''
@param n_jobs
The number of jobs to use for the computation. 
This will only provide speedup for n_targets > 1 and sufficient large 
problems. None means 1 unless in a joblib.parallel_backend context. 
-1 means using all processors.
'''
clf = LinearRegression(n_jobs = -1)
clf.fit(X_train, y_train)
confidence = clf.score(X_test, y_test)
forecast_set = clf.predict(X_current)
print(', '.join([str(x) for x in forecast_set]))
