import os

import mlflow
from mlflow.tracking import MlflowClient

# Print the tracking URI
print(f"MLFLOW_TRACKING_URI: {os.environ.get('MLFLOW_TRACKING_URI')}")

try:
    # Initialize MLflow client
    client = MlflowClient()

    # Try to create an experiment
    experiment_name = "test_experiment"
    try:
        experiment_id = mlflow.create_experiment(experiment_name)
        print(f"Created new experiment with ID: {experiment_id}")
    except mlflow.exceptions.MlflowException as e:
        if "already exists" in str(e):
            experiment = mlflow.get_experiment_by_name(experiment_name)
            experiment_id = experiment.experiment_id if experiment else None
            print(f"Using existing experiment with ID: {experiment_id}")
        else:
            raise

    # Start a run
    with mlflow.start_run(experiment_id=experiment_id) as run:
        run_id = run.info.run_id
        print(f"Started run with ID: {run_id}")

        # Log a parameter
        mlflow.log_param("test_param", "test_value")
        print("Logged parameter successfully")

        # Log a metric
        mlflow.log_metric("test_metric", 1.0)
        print("Logged metric successfully")

    print("MLflow connection test successful!")
except Exception as e:
    print(f"MLflow connection test failed: {str(e)}")
