import argparse
import os


def write_config(config, data_path, is_local, lr=1e-3, path='.'):
    dataset = data_path.split('/')[-1].split('.')[0]

    size, lora, batch_size, train = config
    with open(f'{path}/config_{dataset}.yaml', 'w') as f:
        f.write(f"""
models:
  - "MOIRAI": "./moirai_{dataset}.yaml"

data_path: "path/{data_path}"
time_col: "date"
is_local: {is_local}
batch_size: {batch_size}

max_steps: 5000
max_epochs: 100

logger: "wandb"

horizon: 96
lookback: 96

patience: 10

train: {train}
lr: {lr}
                """)


def write_moirai_config(config, dataset, path='.'):
    size, lora, batch_size, train = config
    with open(f'{path}/moirai_{dataset}.yaml', 'w') as f:
        f.write(f"""
size: '{size}'
patch_size: 'auto'
num_samples: 100
target_dim: 2

train_dataloader:
  shuffle: true
  num_batches_per_epoch: 100

val_dataloader:
  shuffle: false

lora: {lora}
                """)
        

def run_exp(data_path, is_local, run_path='.'):
    dataset = data_path.split('/')[-1].split('.')[0]

    sizes = ['small', 'base', 'large'] * 3
    loras = ['false'] * 6 + ['true'] * 3
    batch_sizes = [128, 32, 8] * 3
    trains = ['false'] * 3 + ['true'] * 6

    cfgs = list(zip(sizes, loras, batch_sizes, trains))

    for i, config in enumerate(cfgs):
        if config[-1] == 'true':
            lrs = [10**-3, 10**-4, 10**-5, 10**-6]

            for lr in lrs:
                write_config(config, data_path, is_local, lr, run_path)
                write_moirai_config(config, dataset, run_path)

                print(f"\n\n\n\nRunning experiment {i + 1}/{len(cfgs)}")
                size, lora, batch_size, train = config
                print(f"Size: {size}, Lora: {lora}, Batch size: {batch_size}, Train: {train}\n\n\n")
                
                # os.system("conda activate snir-env")
                os.system(f"python {run_path}/run.py --config {run_path}/config_{dataset}.yaml")
        else:
            write_config(config, data_path, is_local, path=run_path)
            write_moirai_config(config, dataset, path=run_path)

            print(f"\n\n\n\nRunning experiment {i + 1}/{len(cfgs)}")
            size, lora, batch_size, train = config
            print(f"Size: {size}, Lora: {lora}, Batch size: {batch_size}, Train: {train}\n\n\n")
            
            # os.system("conda activate snir-env")
            os.system(f"python {run_path}/run.py --config {run_path}/config_{dataset}.yaml")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pretraining script")

    parser.add_argument("--data-path", type=str, default="./data/ETTh1.csv", help="Path to data")
    parser.add_argument("--is-local", type=bool, default=1, help="Run locally or on cloud")
    parser.add_argument("--run-path", type=str, default="./", help="Path of run.py")

    args = parser.parse_args()
    args.is_local = bool(args.is_local)

    run_exp(args.data_path, args.is_local, args.run_path)
