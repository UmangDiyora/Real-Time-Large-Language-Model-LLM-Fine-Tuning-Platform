import wandb
import os

def init_wandb(project_name, run_name, config):
    """
    Initialize Weights & Biases run.
    """
    wandb.init(
        project=project_name,
        name=run_name,
        config=config
    )

def log_metrics(metrics):
    """
    Log metrics to W&B.
    """
    wandb.log(metrics)

def save_model_artifact(model_path, artifact_name):
    """
    Save model as W&B artifact.
    """
    artifact = wandb.Artifact(artifact_name, type='model')
    artifact.add_dir(model_path)
    wandb.log_artifact(artifact)
