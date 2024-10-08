import argparse
import yaml
from train import train_simclr, train_classifiers
from evaluate import evaluate_model
from ablation_study import run_ablation_study
import torch
from torch.multiprocessing import spawn


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    config['training']['lr'] = float(config['training']['lr'])
    config['training']['weight_decay'] = float(config['training']['weight_decay'])
    config['training']['simclr_temperature'] = float(config['training']['simclr_temperature'])

    return config

def main(args):
    config = load_config(args.config)

    if args.mode == "ablation":
        print("Running ablation study...")
        run_ablation_study(config)
    if args.mode == "train_simclr":
        print("Starting SimCLR pre-training...")
        world_size = torch.cuda.device_count()
        spawn(train_simclr, args=(world_size, config), nprocs=world_size, join=True)
    elif args.mode == "train_classifiers":
        print("Starting classifier training...")
        world_size = torch.cuda.device_count()
        spawn(train_classifiers, args=(world_size, config, True), nprocs=world_size, join=True)
    elif args.mode == "train_classifiers_no_correlation":
        print("Starting classifier training without correlation...")
        world_size = torch.cuda.device_count()
        spawn(train_classifiers, args=(world_size, config, False), nprocs=world_size, join=True)
    elif args.mode == "evaluate":
        print("Evaluating model...")
        evaluate_model(config, "best_model_with_correlation.pth")
    elif args.mode == "ablation":
        print("Running ablation study...")
        run_ablation_study(config)
    else:
        print("Invalid mode: {}".format(args.mode))



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ChestX-ray14 Multi-Label Classification")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to configuration file")
    parser.add_argument("--mode", type=str, choices=["train_simclr", "train_classifiers", "train_classifiers_no_correlation", "evaluate", "ablation"], required=True,
                        help="Mode of operation")
    args = parser.parse_args()

    main(args)

