import torch


class EarlyStopping:
    def __init__(self, output_path, patience=10, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_acc_min = 0
        self.delta = delta
        self.output_path = output_path

    def __call__(self, val_acc, model):
        score = val_acc
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_acc, model)
        elif score >= self.best_score + self.delta:
            self.counter += 1
            print(f"Early stopping counter: {self.counter} out of {self.patience}.")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_acc, model)
            self.counter = 0
        return self.early_stop
    
    def save_checkpoint(self, val_acc, model):
        torch.save(model.state_dict(), self.output_path)
        self.val_acc_min = val_acc