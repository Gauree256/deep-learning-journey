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

import numpy as np
import tensorflow
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()

model.add(Dense(3,activation='sigmoid',input_dim = 11))
model.add(Dense(1,activation='sigmoid'))

model.summary()
print("_______________________________________________________________")

model.compile(loss='binary_crossentropy',optimizer='Adam', metrics=['accuracy'])
history = model.fit(X_train_scaled,y_train,epochs=10, validation_split=0.2)
model.layers[0].get_weights()
y_log = model.predict(X_test_scaled)
y_pred = np.where(y_log > 0.5, 1,0)

print("________________________________________________________________")
print(y_pred)
print("________________________________________________________________")

# Check accuracy of model

from sklearn.metrics import accuracy_score
print("Model Accuracy : ",accuracy_score(y_test, y_pred))
print("________________________________________________________________")

import matplotlib.pyplot as plt

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss vs Validation Loss')
plt.legend()
plt.show()