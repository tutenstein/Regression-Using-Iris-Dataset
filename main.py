import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np

iris = load_iris()  # load to  Iris flower data set

feature_df = pd.DataFrame(data = iris.data, columns = iris.feature_names)  # crate fatures
label_df = pd.DataFrame(data= iris.target, columns= ['species']) # keep the target numbers in the 'species' column

iris_df = pd.concat([feature_df, label_df], axis= 1) #concat dataframe to gather full dataset

X = iris_df.select_dtypes("float64").drop("petal width (cm)", axis = 1)  #independent variables
y = iris_df["petal width (cm)"] #depended variables

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state = 7) #split the data into 2 at a certain rate for training and testing

linear_model = LinearRegression()
ridge_model = Ridge() #L2
lasso_model = Lasso() #L1

linear_model.fit(X_train, y_train) #train according to the "linear Regression" model
ridge_model.fit(X_train, y_train) #train according to the "Lasso Regression" model
lasso_model.fit(X_train, y_train) #train according to the "Lasso Regression" model


value_to_predict = [[3.7, 5, 0.9]]
lin_pred = linear_model.predict(value_to_predict)
ridge_pred = ridge_model.predict(value_to_predict)
lasso_pred = lasso_model.predict(value_to_predict)
pred_dict = {"Linear": lin_pred, "Ridge":ridge_pred, "Lasso": lasso_pred}


print(pred_dict)