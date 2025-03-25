# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. <b>Load and preprocess data:</b> Fetch dataset, create DataFrame, split into features and targets.
2. <b>Split and scale data:</b> Train-test split, apply standard scaling to features and targets.
3. <b>Initialize and train model:</b> Set up SGDRegressor with MultiOutputRegressor, fit on training data.
4. <b>Predict and evaluate:</b> Predict on test data, inverse transform, calculate mean squared error.

## Program and Outputs:
```
/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: NAVEEN KUMAR S
RegisterNumber: 212223040129
*/
```
```
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
```

```
# Load the California Housing dataset
data = fetch_california_housing()
print(data)
```
![image](https://github.com/user-attachments/assets/8779f21d-c716-42df-be2e-eb3d3d0f1d96)
```
import pandas as pd
df=pd.DataFrame(data.data,columns=data.feature_names)
df['target']=data.target
print(df.head())
```
![image](https://github.com/user-attachments/assets/01f10470-41d5-41ea-9923-c341accc9704)

```
df.info()
```
![image](https://github.com/user-attachments/assets/de3b958a-d5a2-4c95-8b50-e61c56255bb0)

```
X = df.drop(columns=['AveOccup','target'])
X.info()
```
![image](https://github.com/user-attachments/assets/66364678-a223-496c-bd3c-abca73da84e0)

```
Y = df[['AveOccup' , 'target']]
Y.info()
```
![image](https://github.com/user-attachments/assets/201b6918-3998-4d00-8810-d799676d2028)

```
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=11)
X.head()
```
![image](https://github.com/user-attachments/assets/ac00262c-4ec5-4f2c-963e-4b0d82833e4c)

```
scaler_X = StandardScaler()
scaler_Y = StandardScaler()

X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)
Y_train = scaler_Y.fit_transform(Y_train)
Y_test = scaler_Y.transform(Y_test)
```
```
sgd = SGDRegressor(max_iter=1000, tol=1e-3)
multi_output_sgd = MultiOutputRegressor(sgd)
multi_output_sgd.fit(X_train, Y_train)
```
![image](https://github.com/user-attachments/assets/adb917c6-4e48-4291-b87e-3413dceb22a5)

```
Y_pred = multi_output_sgd.predict(X_test)
Y_pred = scaler_Y.inverse_transform(Y_pred)
Y_test = scaler_Y.inverse_transform(Y_test)
mse = mean_squared_error(Y_test, Y_pred)
print("Mean Squared Error:", mse)
print("\nPredictions:\n", Y_pred[:5])
```
![image](https://github.com/user-attachments/assets/bc3c8dca-8eda-489d-8bd1-3e81fb7ecd2a)

## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
