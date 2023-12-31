import os
import torch
import logging


class EarlyStopping:
    def __init__(
        self,
        patience=7,
        verbose=False,
        delta=0,
        trace_func=logging.info,
    ):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, logs a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pth'
            trace_func (function): trace terminal output function.
                            Default: logging.info
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float("inf")
        self.delta = delta
        self.path = (
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            + "/checkpoint.pth"
        )
        self.trace_func = trace_func

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(
                f"EarlyStopping counter: {self.counter} out of {self.patience}"
            )
            # Triggering of Early Stopping process, returning the checkpoint path.
            if self.counter >= self.patience:
                self.early_stop = True
                return self.get_checkpoint_path()

        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

        return None

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decreases."""
        if self.verbose:
            self.trace_func(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
            )
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

    def get_checkpoint_path(self):
        return self.path
