import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import metrics

# Load the dataset using NumPy
data = np.genfromtxt("world_health_statistics_indicators_zaf.xls", delimiter=',', skip_header=1)
#Classification
# Define the columns for y classification
column_index_1 = 2  # Risk factor target column
column_index_2 = 3  # Treatment success rate
column_index_3 = 4  # Health deterioration rate
# Prepare the data for the classification task
X_classification = data[:, :-4]  # Excluding the last 3 columns as they are target columns
y_classification_1 = data[:, column_index_1]
y_classification_2 = data[:, column_index_2]
y_classification_3 = data[:, column_index_3]

# Split data into training and testing sets for all target columns
X_train, X_test, y_train_1, y_test_1, y_train_2, y_test_2, y_train_3, y_test_3 = train_test_split(
X_classification, y_classification_1, y_classification_2, y_classification_3, test_size=0.2, random_state=42
)

# Create and train the Gaussian Naive Bayes model for all target columns
model_1 = GaussianNB()
model_1.fit(X_train, y_train_1)

model_2 = GaussianNB()
model_2.fit(X_train, y_train_2)

model_3 = GaussianNB()
model_3.fit(X_train, y_train_3)

# Make necessary predictions for all target columns
y_pred_1 = model_1.predict(X_test)
y_pred_2 = model_2.predict(X_test)
y_pred_3 = model_3.predict(X_test)

# Evaluate the models' performance for all target columns
accuracy_1 = metrics.accuracy_score(y_test_1, y_pred_1)
precision_1 = metrics.precision_score(y_test_1, y_pred_1, average='macro')
recall_1 = metrics.recall_score(y_test_1, y_pred_1, average='macro')
f1_1 = metrics.f1_score(y_test_1, y_pred_1, average='macro')

accuracy_2 = metrics.accuracy_score(y_test_2, y_pred_2)
precision_2 = metrics.precision_score(y_test_2, y_pred_2, average='macro')
recall_2 = metrics.recall_score(y_test_2, y_pred_2, average='macro')
f1_2 = metrics.f1_score(y_test_2, y_pred_2, average='macro')

accuracy_3 = metrics.accuracy_score(y_test_3, y_pred_3)
precision_3 = metrics.precision_score(y_test_3, y_pred_3, average='macro')
recall_3 = metrics.recall_score(y_test_3, y_pred_3, average='macro')
f1_3 = metrics.f1_score(y_test_3, y_pred_3, average='macro')

print("Metrics for Risk Factor:")
print(f"Accuracy: {accuracy_1}")
print(f"Precision: {precision_1}")
print(f"Recall: {recall_1}")
print(f"F1 Score: {f1_1}")

print("\nMetrics for :Treatment success rate")
print(f"Accuracy: {accuracy_2}")
print(f"Precision: {precision_2}")
print(f"Recall: {recall_2}")
print(f"F1 Score: {f1_2}")

print("\nMetrics for :Health deterioration rate")
print(f"Accuracy: {accuracy_3}")
print(f"Precision: {precision_3}")
print(f"Recall: {recall_3}")
print(f"F1 Score: {f1_3}")



# Regression
# Prepare data for the regression task
X_regression = data[:, :-3]  # Exclude the last three columns as they are targets
y_regression_1 = data[:, -3]  # First additional target
y_regression_2 = data[:, -2]  # Second additional target
y_regression_3 = data[:, -1]  # Third additional target

# Split data into training and testing sets for each target
X_train, X_test, y_train_1, y_test_1, y_train_2, y_test_2, y_train_3, y_test_3 = train_test_split(
    X_regression, y_regression_1, y_regression_2, y_regression_3, test_size=0.2, random_state=42
)

# Define a list of regression models
models = [LinearRegression(), GradientBoostingRegressor()]

for model in models:
#Risk Factor Column
    # Create and Train the regression model
    model.fit(X_train, y_train_1)

    # Make predictions
    y_pred_1 = model.predict(X_test)

    # Evaluate the model 
    mae_1 = metrics.mean_absolute_error(y_test_1, y_pred_1)
    mse_1 = metrics.mean_squared_error(y_test_1, y_pred_1)
    rmse_1 = np.sqrt(mse_1)
    r2_1 = metrics.r2_score(y_test_1, y_pred_1)
    
    # Treatment success rate column
    # Create and Train the regression model
    model.fit(X_train, y_train_2)

    # Make predictions
    y_pred_2 = model.predict(X_test)

    # Evaluate the model
    mae_2 = metrics.mean_absolute_error(y_test_2, y_pred_2)
    mse_2 = metrics.mean_squared_error(y_test_2, y_pred_2)
    rmse_2 = np.sqrt(mse_2)
    r2_2 = metrics.r2_score(y_test_2, y_pred_2)
    
    # Health deterioration rate
    # Create and Train the regression model
    model.fit(X_train, y_train_3)

    # Make predictions
    y_pred_3 = model.predict(X_test)

    # Evaluate the model
    mae_3 = metrics.mean_absolute_error(y_test_3, y_pred_3)
    mse_3 = metrics.mean_squared_error(y_test_3, y_pred_3)
    rmse_3 = np.sqrt(mse_3)
    r2_3 = metrics.r2_score(y_test_3, y_pred_3)

    model_name = model._class.__name_
    print(f"{model_name} Metrics for the Risk Factor:")
    print(f"Mean Absolute Error (MAE): {mae_1}")
    print(f"Mean Squared Error (MSE): {mse_1}")
    print(f"Root Mean Squared Error (RMSE): {rmse_1}")
    print(f"R-squared (R2) Score: {r2_1}\n")

    print(f"{model_name} Metrics for the Treatment Success rate:")
    print(f"Mean Absolute Error (MAE): {mae_2}")
    print(f"Mean Squared Error (MSE): {mse_2}")
    print(f"Root Mean Squared Error (RMSE): {rmse_2}")
    print(f"R-squared (R2) Score: {r2_2}\n")

    print(f"{model_name} Metrics for the Health deterioration:")
    print(f"Mean Absolute Error (MAE): {mae_3}")
    print(f"Mean Squared Error (MSE): {mse_3}")
    print(f"Root Mean Squared Error (RMSE): {rmse_3}")
    print(f"R-squared (R2) Score: {r2_3}\n")
