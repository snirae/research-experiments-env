# A parent interface for all experiments.
# It contains the common methods and attributes that are shared among all experiments.
# It is inherited by all the experiment classes.

import random
import torch
import numpy as np
import wandb

from model.benchmarks.training import load_optimizer

from abc import ABC, abstractmethod


class Experiment:
    def __init__(self, args):
        args.dataset_name = args.data_path.split("/")[-1].split(".")[0]
        self.args = args

        # random seed
        fix_seed = args.seed

        print(f"setting random seed to {fix_seed}")
        random.seed(fix_seed)
        torch.manual_seed(fix_seed)
        np.random.seed(fix_seed)

        # optimizer
        print(f"Loading optimizer '{args.optimizer}'")
        optimizer = load_optimizer(args.optimizer)
        optimizer_kwargs = {"lr": args.lr, "weight_decay": args.weight_decay}

        self.optimizer = optimizer
        self.optimizer_kwargs = optimizer_kwargs

    # train and test abstract methods
    @abstractmethod
    def train(self):
        raise NotImplementedError

    @abstractmethod
    def test(self):
        raise NotImplementedError

    def run(self):
        print("\n\nStarting the experiment...")

        if self.args.train:
            print(f"Training the models {self.args.models}")
            print(f"Saving checkpoints to '{self.args.save_dir}'")
            print(f"Training for {self.args.max_steps} steps / {self.args.max_epochs} epochs\n")

            self.train()

            print("Training finished.")

        if self.args.test:
            print("\n\nTesting the model...")
            
            self.test()

            print("\nTesting finished.")

        print("\nExperiment finished.")

        if self.args.logger == "wandb":
            wandb.finish()
