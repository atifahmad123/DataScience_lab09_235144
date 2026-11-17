import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

cars = pd.read_csv("auto_mpg.csv", na_values="?")
print("Rows in the UCI file:", len(cars))
print("Missing horsepower values:", cars["horsepower"].isna().sum())
cars = cars[["horsepower", "mpg"]].dropna()

X = cars[["horsepower"]] # Double brackets keep X two-dimensional.
y = cars["mpg"] # The target is one-dimensional.
X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.20, random_state=42
)
print("Training rows:", len(X_train))
print("Test rows:", len(X_test))


