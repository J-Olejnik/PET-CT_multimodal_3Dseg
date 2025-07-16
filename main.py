import optuna
import wandb
from base.train import optuna_objective
import glob2
import os
import yaml
import argparse

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def main(config):
    def objective(trial):
        return optuna_objective(trial, config)
    
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5)
    study = optuna.create_study(
        directions=["minimize", "maximize", "maximize", "minimize"],  # For valid loss, dice, surface dice, hausdorff
        study_name=f'{config["type"]}_full_optimization',
        pruner=pruner,
        storage=f'sqlite:///optuna_study_{config["type"]}_full.db',
        load_if_exists=True
    )
    study.optimize(objective, n_trials=config['n_trials'])

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Training Script")
    parser.add_argument("--type", type=str, required=True, help="CT_only, PET_CTcat")
    args = parser.parse_args()

    with open(r"config.yaml", 'r') as file:
        config = yaml.safe_load(file)
    
    if isinstance(config.get("Dataset"), str):
        config["Dataset"] = glob2.glob(config["Dataset"])

    if isinstance(config.get("type"), str):
        config["type"] = args.type

    if isinstance(config.get("hu"), list):
        config["hu"] = [tuple(x) for x in config["hu"]]

    wandb.login(key="your_API_key_here")

    print(f'Successfully found {len(config["Dataset"])} patients')
    
    main(config)