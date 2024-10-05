import wandb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import savgol_filter


def list_runs(entity, project):
    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}")
    print("Available runs:")
    for run in runs:
        print(f"Run ID: {run.id}, Name: {run.name}, Status: {run.state}")
    return runs


def analyze_training_progress(run):
    # Get the history of the run
    history = run.scan_history()
    df = pd.DataFrame(history)

    # Basic statistics
    print("\nTraining Progress Summary:")
    print(f"Total Epochs: {df['epoch'].max()}")
    print(f"Current SimCLR Loss: {df['simclr_loss'].iloc[-1]:.4f}")
    print(f"Best SimCLR Loss: {df['simclr_loss'].min():.4f}")

    # Plot SimCLR Loss
    plt.figure(figsize=(12, 6))
    sns.lineplot(x='epoch', y='simclr_loss', data=df)
    plt.title('SimCLR Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    # Add smoothed trend line
    smooth_loss = savgol_filter(df['simclr_loss'], window_length=min(21, len(df)), polyorder=3)
    plt.plot(df['epoch'], smooth_loss, color='red', label='Trend')
    plt.legend()
    plt.savefig('simclr_loss_trend.png')
    plt.close()

    # Analyze learning rate if available
    if 'learning_rate' in df.columns:
        plt.figure(figsize=(12, 6))
        sns.lineplot(x='epoch', y='learning_rate', data=df)
        plt.title('Learning Rate over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.savefig('learning_rate.png')
        plt.close()

    # Check for convergence
    recent_loss = df['simclr_loss'].tail(20)  # Last 20 epochs
    loss_std = recent_loss.std()
    print(f"\nRecent Loss Standard Deviation: {loss_std:.6f}")
    if loss_std < 0.01:  # You can adjust this threshold
        print("The loss appears to be converging.")
    else:
        print("The loss is still fluctuating significantly.")

    # Estimate remaining time
    if len(df) > 1:
        time_per_epoch = (df['_timestamp'].iloc[-1] - df['_timestamp'].iloc[0]) / len(df)
        remaining_epochs = max(0, 100 - df['epoch'].max())  # Assuming 100 total epochs
        estimated_time = remaining_epochs * time_per_epoch
        print(f"\nEstimated time remaining: {estimated_time:.2f} seconds")

    return df


# Usage
entity = "mirzaeeghazal"
project = "ChestXray14_MultiLabel_Classification"

runs = list_runs(entity, project)

if runs:
    # Analyze the most recent run
    most_recent_run = max(runs, key=lambda run: run.created_at)
    print(f"\nAnalyzing most recent run: {most_recent_run.name} (ID: {most_recent_run.id})")
    df = analyze_training_progress(most_recent_run)
else:
    print("No runs found in the project.")
