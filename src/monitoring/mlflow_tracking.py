import mlflow
import os

def init_mlflow(experiment_name, run_name=None):
    """
    Initialize MLflow run.
    """
    mlflow.set_experiment(experiment_name)
    return mlflow.start_run(run_name=run_name)

def log_params(params):
    """
    Log parameters to MLflow.
    """
    mlflow.log_params(params)

def log_metrics(metrics, step=None):
    """
    Log metrics to MLflow.
    """
    mlflow.log_metrics(metrics, step=step)

def log_model(model, artifact_path):
    """
    Log model to MLflow.
    """
    mlflow.pytorch.log_model(model, artifact_path)
