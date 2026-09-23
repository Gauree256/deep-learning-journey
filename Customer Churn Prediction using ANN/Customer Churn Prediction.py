import pandas as pd
df = pd.read_csv(r'C:\Users\DikGauree\OneDrive\Document\deep-learning-journey\Customer Churn Prediction using ANN\Churn_Modelling.csv')

df.drop(columns=['RowNumber','CustomerId','Surname'], inplace=True, errors='ignore')
df = pd.get_dummies(df, columns=['Geography', 'Gender'], drop_first=True)

X = df.drop(columns=['Exited'])
y = df['Exited']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Now, The whole data became Numerical.
# Training and testing data seperated
# Also Scalling is done.
# Now, we are ready to build our model.

import tensorflow
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense