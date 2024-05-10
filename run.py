# A script to run the pretraining, given a model and other parameters

import argparse
from utils.experiment import Experiment


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pretraining script")

    ##### Required arguments #####
    parser.add_argument("--model-name", type=str, help="Name of the model to use, as defined in the filename and implementation")
    parser.add_argument("--model-config", type=str, default='./models/config.json', help="Path to the model configuration file (JSON)")
    parser.add_argument("--ckpt-path", type=str, default="", help="Path to a checkpoint to load the model from")
    parser.add_argument("--data-path", type=str, default="./data", help="Path to the dataset file")
    parser.add_argument("--task", type=str, default="forecasting", help="Task to train the model on")

    ##### Optional arguments #####
    # seed
    parser.add_argument("--seed", type=int, default=2024, help="Random seed")

    # data
    parser.add_argument("--val-split", type=float, default=0.1, help="Proportion of the data to use for validation")
    parser.add_argument("--test-split", type=float, default=0.1, help="Proportion of the data to use for testing")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of workers for data loading")
    parser.add_argument("--lookback", type=int, default=96, help="Number of time steps to look back")
    parser.add_argument("--horizon", type=int, default=96, help="Number of time steps to forecast (for forecasting task)")
    parser.add_argument("--gap", type=int, default=0, help="Gap to forecast after (for forecasting task)")
    parser.add_argument("--mask-perc", type=float, default=0.1, help="Percentage of the data to mask (for imputation task)")

    # training
    parser.add_argument("--train", type=int, default=1, help="Whether to train the model (0-False, 1-True)")
    parser.add_argument("--max-epochs", type=int, default=10, help="Number of epochs to train for")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--optimizer", type=str, default="adam", help="Optimizer to use")
    parser.add_argument("--losses", nargs='+', default=["mse"], help="Loss functions to use")

    # callbacks
    parser.add_argument("--early-stopping", type=int, default=1, help="Whether to use early stopping (0-False, 1-True)")
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

    # testing
    parser.add_argument("--test", type=int, default=1, help="Whether to test the model (0-False, 1-True)")
    parser.add_argument("--test-data", type=str, default='./testing/data', help="Path to the test data directory")
    parser.add_argument("--save-plots", type=int, default=1, help="Whether to save forecasting/imputation plots (0-False, 1-True)")

    # hardware
    parser.add_argument("--accelerator", type=str, default="auto", help="Device to use for training")
    parser.add_argument("--devices", type=int, default=1, help="Number of devices to use")

    ####################################################################################################################
    
    args = parser.parse_args()

    # cast boolean arguments
    args.train = bool(args.train)
    args.early_stopping = bool(args.early_stopping)
    args.test = bool(args.test)
    args.save_plots = bool(args.save_plots)

    # running the experiment
    exp = Experiment(args)
    exp.run()
    