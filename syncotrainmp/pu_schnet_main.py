
import os
from pathlib import Path
import sys
import argparse
import json
import time
import random

import numpy as np
import pandas as pd
import schnetpack as spk
import torch
import torchmetrics
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping

from syncotrainmp.experiment_setup import current_setup, str_to_bool
from syncotrainmp.pu_schnet.pu_learn import int2metric
from syncotrainmp.pu_schnet.pu_learn.Datamodule4PU import DataModuleWithPred
from syncotrainmp.pu_schnet.pu_learn.schnet_funcs import directory_setup, predProb


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Semi-Supervised ML for Synthesizability Prediction"
    )
    parser.add_argument(
        "--experiment",
        default="schnet0",
        help="name of the experiment and corresponding config files.",
    )
    parser.add_argument(
        "--ehull015",
        type=str_to_bool,
        default=False,
        help="Predicting stability to evaluate PU Learning's efficacy with 0.015eV cutoff.",
    )
    parser.add_argument(
        "--small_data",
        type=str_to_bool,
        default=False,
        help="This option selects a small subset of data for checking the workflow faster.",
    )
    parser.add_argument(
        "--startIt",
        type=int,
        default=0,
        help="Starting iteration No."
    )
    parser.add_argument(
        "--gpu_id",
        type=int,
        default=3,
        help="GPU ID to use for training."
    )
    return parser.parse_args(sys.argv[1:])


def load_config(config_path):
    """Load the configuration from the specified JSON file."""
    with open(config_path, "r") as read_file:
        print("Read Experiment configuration")
        return json.load(read_file)


def setup_environment(gpu_id):
    """Set up the environment for GPU usage."""
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)


def initialize_experiment(args):
    """Initialize the experiment setup based on command line arguments."""
    cs = current_setup(small_data=args.small_data, experiment=args.experiment, ehull015=args.ehull015)
    return cs


def prepare_dataframes(propDFpath, TARGET):
    """Load the property DataFrame and prepare targets."""
    crysdf = pd.read_pickle(propDFpath)
    crysdf["targets"] = crysdf[TARGET].map(lambda target: np.array(target).flatten())
    crysdf["targets"] = crysdf.targets.map(lambda target: {prop: np.array(target)})
    return crysdf


def initialize_training_config(config, small_data, startIt):
    """Initialize training configurations based on the config settings."""
    epoch_num = config["epoch_num"]
    if small_data:
        epoch_num = int(epoch_num * 0.5)
    config["start_iter"] = startIt
    return epoch_num


def create_directories(config, data_prefix, experiment, ehull015):
    """Create output directories based on experiment settings."""
    save_dir = os.path.join(config["schnetDirectory"], f'PUOutput_{data_prefix}{experiment}')
    if ehull015:
        save_dir = os.path.join(config["schnetDirectory"], f'PUehull015_{experiment}')
    return save_dir


def initialize_model(config, crysdf):
    """Initialize the model and optimizer based on configuration."""
    model = spk.representation.SchNet(
        n_interactions=config["n_interactions"],
        n_atom_basis=config["n_atom_basis"],
        n_output=config["n_output"],
        cutoff=config["cutoff"],
        num_radial=config["num_radial"],
        n_gaussians=config["n_gaussians"],
    )

    # Using Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    
    return model, optimizer


def prepare_dataloaders(crysdf, batch_size):
    """Prepare training and testing dataloaders."""
    train_data = DataModuleWithPred(crysdf, batch_size=batch_size, split='train')
    test_data = DataModuleWithPred(crysdf, batch_size=batch_size, split='test')
    
    return train_data, test_data


def train_model(task, crysData, trainer):
    """Train the model using the specified trainer and task."""
    trainer.fit(task, datamodule=crysData)


def evaluate_model(trainer, task, crysTest, prop):
    """Evaluate the model on the test dataset and return predictions."""
    predictions = trainer.predict(model=task, dataloaders=crysTest.predict_dataloader(), return_predictions=True)
    results = []
    for batch in predictions:
        for datum in batch[prop]:
            results.append(predProb(datum.float()))
    return results


def log_remaining_time(it, start_iter, num_iter, start_time, data_prefix, experiment, prop):
    """Log the remaining time for the experiment."""
    elapsed_time = time.time() - start_time
    remaining_iterations = num_iter - it - 1
    time_per_iteration = elapsed_time / (it - start_iter + 1)
    estimated_remaining_time = remaining_iterations * time_per_iteration
    remaining_days = int(estimated_remaining_time // (24 * 3600))
    remaining_hours = int((estimated_remaining_time % (24 * 3600)) // 3600)

    time_log_path = os.path.join('time_logs', f'schnet_remaining_time_{data_prefix}{experiment}_{prop}.txt')
    with open(time_log_path, 'w') as file:
        file.write(f"Iterations completed: {it - start_iter}\n")
        file.write(f"Iterations remaining: {remaining_iterations}\n")
        file.write(f"Estimated remaining time: {remaining_days} days, {remaining_hours} hours\n")

    print(f"Iteration {it} completed. Remaining time: {remaining_days} days, {remaining_hours} hours")


def main():
    args = parse_arguments()
    setup_environment(args.gpu_id)

    # Load configuration and initialize experiment
    config_path = 'pu_schnet/schnet_configs/pu_config_schnetpack.json'
    config = load_config(config_path)
    cs = initialize_experiment(args)
    propDFpath = cs["propDFpath"]
    data_prefix = cs["dataPrefix"]
    TARGET = cs["TARGET"]

    print(f"Running the {data_prefix}{args.experiment} experiment for the {cs['prop']} property.")
    start_time = time.time()

    # Prepare DataFrames
    crysdf = prepare_dataframes(propDFpath, TARGET)

    # Initialize training configuration
    epoch_num = initialize_training_config(config, args.small_data, args.startIt)

    # Create directories for saving results
    save_dir = create_directories(config, data_prefix, args.experiment, args.ehull015)
    res_dir = os.path.join(save_dir, 'res_df')

    # Prepare dataloaders
    batch_size = config.get("batch_size", 32)  # Default batch size if not specified
    crysData, crysTest = prepare_dataloaders(crysdf, batch_size)

    # Initialize model and optimizer
    model, optimizer = initialize_model(config, crysdf)

    # Set up trainer
    trainer = pl.Trainer(
        max_epochs=epoch_num,
        callbacks=[EarlyStopping(monitor='val_loss', patience=5)],
        gpus=[args.gpu_id] if torch.cuda.is_available() else None,
    )

    # Training loop
    for it in range(config["start_iter"], config["num_iter"]):
        print(f'Starting iteration {it}')

        # Train the model
        train_model(model, crysData, trainer)

        # Evaluate the model
        results = evaluate_model(trainer, model, crysTest, cs["prop"])

        # Log remaining time
        log_remaining_time(it, config["start_iter"], config["num_iter"], start_time, data_prefix, args.experiment, cs["prop"])

    # Final summary
    elapsed_time = time.time() - start_time
    elapsed_days = int(elapsed_time // (24 * 3600))
    elapsed_hours = int((elapsed_time % (24 * 3600)) // 3600)
    print(f"PU Learning completed. Total time taken: {elapsed_days} days, {elapsed_hours} hours")


if __name__ == "__main__":
    main()
