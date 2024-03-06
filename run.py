# A script to run the pretraining, given a model and other parameters

import argparse
from utils.experiment import Experiment


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pretraining script")

    ##### Required arguments #####
    parser.add_argument("--model-name", type=str, help="Name of the model to use, as defined in the implementation")
    parser.add_argument("--model-params", nargs='+', metavar='key=value', default=[], help="Model parameters as key-value pairs")
    parser.add_argument("--models-path", type=str, default="./models", help="Path to the models directory")
    parser.add_argument("--data-path", type=str, default="./data", help="Path to the data directory")
    parser.add_argument("--task", type=str, default="forecasting", help="Task to train the model on")

    ##### Optional arguments #####
    # seed
    parser.add_argument("--seed", type=int, default=2024, help="Random seed")

    # data
    parser.add_argument("--val-split", type=float, default=0.1, help="Proportion of the data to use for validation")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of workers for data loading")
    parser.add_argument("--lookback", type=int, default=96, help="Number of time steps to look back")
    parser.add_argument("--horizon", type=int, default=96, help="Number of time steps to forecast (for forecasting task)")
    parser.add_argument("--mask-prob", type=float, default=0.1, help="Probability of masking the data (for imputation task)")

    # training
    parser.add_argument("--max-epochs", type=int, default=10, help="Number of epochs to train for")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--optimizer", type=str, default="adam", help="Optimizer to use")
    parser.add_argument("--losses", nargs='+', default=["mse"], help="Loss functions to use")

    # callbacks
    parser.add_argument("--early-stopping", type=bool, default=True, help="Whether to use early stopping")
    parser.add_argument("--patience", type=int, default=5, help="Patience for early stopping")
    parser.add_argument("--min-improvement", type=float, default=0.01, help="Minimum improvement for early stopping")
    
    # logging
    parser.add_argument("--logger", type=str, default="tensorboard", help="Logger to use (wandb, tensorboard)")
    parser.add_argument("--api-key-file", type=str, default="./api_key.txt", help="Path to the file containing the API key")
    parser.add_argument("--project", type=str, default="pretraining", help="Project name for logging")
    parser.add_argument("--entity", type=str, default="pretraining", help="Entity name for logging")
    parser.add_argument("--log-interval", type=int, default=10, help="Interval for logging by steps")
    parser.add_argument("--log-dir", type=str, default="./logs", help="Directory to save logs")
    parser.add_argument("--save-dir", type=str, default="./checkpoints", help="Directory to save models")

    # hardware
    parser.add_argument("--accelerator", type=str, default="auto", help="Device to use for training")
    parser.add_argument("--devices", type=int, default=1, help="Number of devices to use")

    ####################################################################################################################
    
    args = parser.parse_args()

    # running the experiment
    exp = Experiment(args)
    exp.run()
    