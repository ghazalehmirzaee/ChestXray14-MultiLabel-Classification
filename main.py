import argparse
import yaml
from train import train_simclr, train_classifiers
from evaluate import evaluate_model
from ablation_study import run_ablation_study


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Convert relevant string values to float
    config['training']['lr'] = float(config['training']['lr'])
    config['training']['weight_decay'] = float(config['training']['weight_decay'])
    config['training']['simclr_temperature'] = float(config['training']['simclr_temperature'])

    return config


def main(args):
    config = load_config(args.config)

    if args.mode == "train":
        print("Starting SimCLR pre-training...")
        train_simclr(config)
        print("Starting classifier training...")
        train_classifiers(config)
    elif args.mode == "evaluate":
        print("Evaluating model...")
        evaluate_model(config)
    elif args.mode == "ablation":
        print("Running ablation study...")
        run_ablation_study(config)
    else:
        print("Invalid mode: {}".format(args.mode))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ChestX-ray14 Multi-Label Classification")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to configuration file")
    parser.add_argument("--mode", type=str, choices=["train", "evaluate", "ablation"], required=True,
                        help="Mode of operation")
    args = parser.parse_args()

    main(args)

