import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import torch


def analyze_training_progress(loss_history, lr_history=None, window_length=11, polyorder=3):
    """
    Analyzes the training progress and provides insights.

    Args:
    loss_history (list): List of loss values for each epoch
    lr_history (list): List of learning rate values for each epoch (optional)
    window_length (int): The length of the filter window for smoothing (must be odd)
    polyorder (int): The order of the polynomial used to fit the samples

    Returns:
    dict: A dictionary containing analysis results
    """
    epochs = list(range(1, len(loss_history) + 1))

    # Smooth the loss curve
    smooth_loss = savgol_filter(loss_history, window_length, polyorder)

    # Calculate statistics
    current_loss = loss_history[-1]
    best_loss = min(loss_history)
    best_epoch = loss_history.index(best_loss) + 1

    # Check for convergence
    recent_losses = loss_history[-20:]
    loss_std = np.std(recent_losses)
    is_converging = loss_std < 0.01  # You can adjust this threshold

    # Prepare the plot
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot loss
    ax1.plot(epochs, loss_history, label='Loss')
    ax1.plot(epochs, smooth_loss, color='red', label='Smoothed Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.tick_params(axis='y')
    ax1.legend(loc='upper left')

    # Plot learning rate if provided
    if lr_history:
        ax2 = ax1.twinx()
        ax2.plot(epochs, lr_history, color='green', label='Learning Rate')
        ax2.set_ylabel('Learning Rate')
        ax2.tick_params(axis='y')
        ax2.legend(loc='upper right')

    plt.title('Training Progress')
    plt.tight_layout()

    # Save the plot
    plt.savefig('training_progress.png')
    plt.close()

    # Prepare analysis results
    analysis = {
        'total_epochs': len(loss_history),
        'current_loss': current_loss,
        'best_loss': best_loss,
        'best_epoch': best_epoch,
        'is_converging': is_converging,
        'loss_std': loss_std,
    }

    return analysis

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.delta = delta
        self.path = path
        self.loss_history = []
        self.lr_history = []

    def __call__(self, val_loss, model, optimizer):
        score = -val_loss
        self.loss_history.append(val_loss)
        self.lr_history.append(optimizer.param_groups[0]['lr'])

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

    def analyze_progress(self):
        return analyze_training_progress(self.loss_history, self.lr_history)

