import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import mlflow
import mlflow.sklearn
import seaborn as sns
from mlflow.models.signature import infer_signature

mlflow.set_experiment("My_ML_Experiment")


# Download dataset using seaborn
data = sns.load_dataset('mpg').dropna()

# Convert integer columns to floats to handle potential missing values
for col in data.select_dtypes(include=['int']).columns:
        data[col] = data[col].astype(float)

# Prepare the dataset (dropping irrelevant columns like 'name')
X = data.drop(columns=['mpg', 'name', 'origin'])  # Features
y = data['mpg']  # Target

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Input example for model signature (using one row of the training data)
input_example = X_test.head(1)

# Variables to store the best model and its MSE
best_model = None
best_mse = float('inf')
best_model_name = ""

# Start MLflow tracking
with mlflow.start_run(run_name='Linear_Regression'):
        # Train a linear regression model
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        y_pred_lr = lr.predict(X_test)

        # Calculate MSE
        mse_lr = mean_squared_error(y_test, y_pred_lr)

        # Log parameters and metrics to MLflow
        mlflow.log_param('model_type', 'Linear_Regression')
        mlflow.log_metric('mse', mse_lr)

        # Infer the model signature
        signature = infer_signature(X_test, y_pred_lr)
        # Log the model with input example and signature
        mlflow.sklearn.log_model(lr, "model", input_example=input_example, signature=signature)

        # Check if this is the best model so far
        if mse_lr < best_mse:
            best_mse = mse_lr
            best_model = lr
            best_model_name = 'Linear_Regression'                                 

with mlflow.start_run(run_name='Random_Forest'):
        # Train a random forest model
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        y_pred_rf = rf.predict(X_test)

        # Calculate MSE
        mse_rf = mean_squared_error(y_test, y_pred_rf)

        # Log parameters and metrics to MLflow
        mlflow.log_param('model_type', 'Random_Forest')
        mlflow.log_metric('mse', mse_rf)

                                    
        # Infer the model signature 
        signature = infer_signature(X_test, y_pred_rf)
        # Log the model with input example and signature
        mlflow.sklearn.log_model(rf, "model", input_example=input_example, signature=signature)
        
        # Check if this is the best model so far
        if mse_rf < best_mse:
            best_mse = mse_rf
            best_model = rf
            best_model_name = 'Random_Forest'

# Save the best model in MLflow's Model Registry
if best_model is not None:
        with mlflow.start_run(run_name='Best_Model'):
             # Log the best model with input example and signature
             signature = infer_signature(X_test, best_model.predict(X_test))
             mlflow.sklearn.log_model(best_model, "best_model", input_example=input_example, signature=signature)
             # Optionally, register the model in the Model Registry
             model_uri = f"runs:/{mlflow.active_run().info.run_id}/best_model"
             mlflow.register_model(model_uri, f"Best_{best_model_name}")

print(f"Best model: {best_model_name} with MSE: {best_mse} logged in MLflow.")

print("Training completed and results logged in MLflow.")
                                                   
