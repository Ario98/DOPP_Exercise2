from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
from xgboost.sklearn import XGBRegressor
from numpy import sqrt

# Read the data
df = pd.read_csv("data/preprocessed_full.csv")

# Split the data into training and test sets
X = df.drop('Positive', axis=1)
y = df['Positive']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the pipeline
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=0.95)),
    ('regressor', LinearRegression())
])

# Fit the pipeline to the training data
pipe.fit(X_train, y_train)

# Make predictions on the test data
y_pred = pipe.predict(X_test)

# Measure the performance of the model
r2 = cross_val_score(pipe, X, y, cv=5, scoring='r2')
mse = cross_val_score(pipe, X, y, cv=5, scoring='neg_mean_squared_error')
mae = cross_val_score(pipe, X, y, cv=5, scoring='neg_mean_absolute_error')

# Define the steps names for later use
scaler_name = type(pipe.named_steps['scaler']).__name__
model_name = type(pipe.named_steps['regressor']).__name__

print("Metrics for Model: {} using Scaler: {}".format(model_name, scaler_name))
print("Average CV Score (r2): {:.2f}".format(r2.mean()))
print("Average CV Score (MSE): {:.2f}".format(mse.mean()))
print("Average CV Score (MAE): {:.2f}".format(mae.mean()))