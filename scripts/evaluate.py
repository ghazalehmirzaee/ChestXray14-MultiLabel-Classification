import argparse
import torch
import wandb
from tqdm import tqdm

from data.dataset import get_data_loaders
from models.efficient_net import EfficientNetModel
from utils.metrics import calculate_metrics

def evaluate(args):
    # Initialize wandb
    wandb.init(project="chest-xray-classification", config=args)

    # Set up data loader
    _, _, test_loader = get_data_loaders(
        args.data_dir, args.batch_size, args.num_workers
    )

    # Load model
    model = EfficientNetModel(num_classes=14).to(args.device)
    model.load_state_dict(torch.load(args.model_path))
    model.eval()

    test_preds, test_targets = [], []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images, labels = images.to(args.device), labels.to(args.device)
            outputs = model(images)

            test_preds.extend(outputs.cpu().numpy())
            test_targets.extend(labels.cpu().numpy())

    test_metrics = calculate_metrics(test_preds, test_targets)

    # Log metrics to wandb
    wandb.log({f"test_{k}": v for k, v in test_metrics.items()})

    print("Test Results:")
    for k, v in test_metrics.items():
        print(f"{k}: {v:.4f}")

    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Chest X-ray Classification Model")
    parser.add_argument("--data_dir", type=str, default="/NewRaidData/ghazal/data/ChestX-ray14", help="Path to dataset")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use for evaluation")

    args = parser.parse_args()
    evaluate(args)

    