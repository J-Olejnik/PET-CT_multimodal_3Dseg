import torch.nn as nn
from monai.losses.dice import DiceLoss
    
class Loss_function(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss(sigmoid=True)

    def forward(self, pred, target):

        target = target.unsqueeze(1).float()
        
        L1 = self.bce_loss(pred, target)
        L2 = self.dice_loss(pred, target)

        loss = L1 + L2
        return loss
    
class EarlyStopping:
    def __init__(self, patience=1, delta=0.0001):
        self.patience = patience
        self.delta = delta
        self.best_loss = None
        self.no_improvement_count = 0
        self.stop_training = False
        self.best_model_weights = None
    
    def check(self, val_loss, model):
        if self.best_loss is None or val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.no_improvement_count = 0
            self.best_model_weights = model.state_dict().copy()
        else:
            self.no_improvement_count += 1
            if self.no_improvement_count >= self.patience:
                self.stop_training = True
                model.load_state_dict(self.best_model_weights)