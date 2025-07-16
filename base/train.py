from .dataset import NiftiDataset
import os
import torch
import wandb
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from .models import SwinUNetR, UNet
from .utils import Loss_function, EarlyStopping
from monai.metrics import DiceMetric, SurfaceDiceMetric, HausdorffDistanceMetric, ConfusionMatrixMetric  
from torch.amp import autocast, GradScaler
import gc
import numpy as np
import random

torch.backends.cudnn.benchmark = True
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def optuna_objective(trial, config):
    """Optuna optimization objective function."""
    # model_name = trial.suggest_categorical("model", config["models"])
    lr = trial.suggest_float("lr", config["lr"][0], config["lr"][1], log=True)
    eff_batch_size = trial.suggest_categorical("batch_size", config["batch_size"])
    dropout_rate = trial.suggest_float("dropout", config["dropout"][0], config["dropout"][1])
    hu = trial.suggest_categorical("hu", config["hu"])

    # Randomly select 3 unique folds
    selected_folds = sorted(random.sample(range(config["splits"]), 3))
    print(f'Selected folds: {selected_folds}')

    # Initialize wandb
    wandb.init(
        project=f"PAINT-ITV-Full",
        config={"type": config["type"], "learning_rate": lr, "batch_size": eff_batch_size, "dropout_rate": dropout_rate, "HU (level, window)": hu},
        reinit=True,
        name=f'trial_{trial.number}_{config["type"]}'
    )
    wandb.define_metric("epoch")
    wandb.define_metric("trial")

    for i,_ in enumerate(selected_folds):
        wandb.define_metric(f'Fold_{i} valid_loss', step_metric="epoch")
        wandb.define_metric(f'Fold_{i} dice_score', step_metric="epoch")
        wandb.define_metric(f'Fold_{i} srf_dice', step_metric="epoch")
        wandb.define_metric(f'Fold_{i} h_95', step_metric="epoch")

    wandb.define_metric(f"Avg validation loss", step_metric="trial")
    wandb.define_metric(f"Avg dice score", step_metric="trial")
    wandb.define_metric(f"Avg surface dice", step_metric="trial")
    wandb.define_metric(f"Avg Hausdorff_95", step_metric="trial")  

    # Setup
    batch_size = 2 # Fixed size on GPU
    accumulation_steps = eff_batch_size // batch_size
    scaler = GradScaler()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    avg_val_loss = 0.0
    avg_dice = 0.0
    avg_srf = 0.0
    avg_h95 = 0.0

    # Set determinism
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    kf = KFold(n_splits=config["splits"], shuffle=False)
    splits = list(kf.split(range(len(config["Dataset"]))))

    # Clear cache
    torch.cuda.empty_cache()
    gc.collect()

    for i,fold in enumerate(selected_folds):

        # Initialize data
        train_idx, val_idx = splits[fold]

        if config["type"] == "CT_only":
            train_dataset = NiftiDataset([config["Dataset"][i] for i in train_idx], hu)
            val_dataset = NiftiDataset([config["Dataset"][i] for i in val_idx], hu) 
            in_channels = 1
        elif config["type"] == "PET_CTcat":
            train_dataset = NiftiDataset([config["Dataset"][i] for i in train_idx], hu, include_pet=True, concat=True)
            val_dataset = NiftiDataset([config["Dataset"][i] for i in val_idx], hu, include_pet=True, concat=True) 
            in_channels = 2
        else:
            raise Exception(f'Model type: {config["type"]} is undefined')       
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

        # Initialize Model
        # if model_name == "3DUNet":
        #     model = UNet(in_channels=1,num_classes=2)
        # elif model_name == "LSTMUNet":
        #     model = UNet(in_channels=1,num_classes=2, use_addons=True)
        # elif model_name == "SwinUNETR":
        #     model = SwinUNetR(in_channels=1,num_classes=2)
        # else:
        #     raise Exception(f'Model name: {model_name} is undefined')

        model = SwinUNetR(in_channels=in_channels, num_classes=1, dropout_rate=dropout_rate).to(device)

        # Modify Dropout
        # for module in model.modules():
        #     if isinstance(module, torch.nn.Dropout):
        #         module.p = dropout_rate

        # Loss & Optimizer
        criterion = Loss_function()
        optimizer = torch.optim.Adam(list(model.parameters()), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)

        # Early stopping
        early_stopping = EarlyStopping(patience=20, delta=0.001)

        # MONAI metrics per batch (SurfDSC 2 voxel (4mm) tolerance, HD 95th percentile)
        dice_metric = DiceMetric(include_background=False, reduction="mean_batch")
        surface_metric = SurfaceDiceMetric(class_thresholds=[2.0], include_background=False, reduction="mean_batch")
        h95_metric = HausdorffDistanceMetric(include_background=False,  percentile=95, reduction="mean_batch")
        prec_metric = ConfusionMatrixMetric(include_background=False, reduction="mean_batch", metric_name="precision")
        rec_metric = ConfusionMatrixMetric(include_background=False, reduction="mean_batch", metric_name="recall")

        best_fold_val_loss = float('inf')

        for epoch in range(100):

            # Training
            model.train()
            optimizer.zero_grad()

            for step, (images, labels) in enumerate(train_loader):
                images, labels = images.contiguous().to('cuda'), labels.contiguous().to('cuda')

                # Mixed precision
                with autocast(device_type="cuda", dtype=torch.float16):
                    outputs = model(images)
                    loss = criterion(outputs, labels) / accumulation_steps
                scaler.scale(loss).backward()

                # Gradient accumulation
                if (step + 1) % accumulation_steps == 0 or (step + 1) == len(train_loader):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

            # Validation
            model.eval()
            val_loss = 0.0
            dice_metric.reset()
            surface_metric.reset()
            h95_metric.reset()
            prec_metric.reset()
            rec_metric.reset()
        
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.contiguous().to('cuda'), labels.contiguous().to('cuda')

                    with autocast(device_type="cuda", dtype=torch.float16):
                        outputs = model(images)
                        val_loss += criterion(outputs, labels).item()

                    outputs = torch.sigmoid(outputs)
                    outputs = (outputs > 0.5).float()
                    labels = labels.unsqueeze(1).float()

                    dice_metric(outputs, labels)
                    surface_metric(outputs, labels, spacing=(2.0, 2.0, 2.0))
                    h95_metric(outputs, labels, spacing=(2.0, 2.0, 2.0))
                    prec_metric(outputs, labels)
                    rec_metric(outputs, labels)

            val_loss /= len(val_loader)

            # Warm-up period of 20 epochs
            if epoch >= 20:
                scheduler.step(val_loss)

            dice = dice_metric.aggregate().item()
            srf = surface_metric.aggregate().item()
            h95 = h95_metric.aggregate().item()
            prec = prec_metric.aggregate()[0].item()
            rec = rec_metric.aggregate()[0].item()

            if epoch == 0 or val_loss < best_fold_val_loss:
                best_fold_val_loss = val_loss
                best_fold_dice = dice
                best_fold_srf = srf
                best_fold_h95 = h95
                best_fold_prec = prec
                best_fold_rec = rec
            
            wandb.log({f'Fold_{i} valid_loss': val_loss, f'Fold_{i} dice_score': dice, f'Fold_{i} srf_dice': srf, f'Fold_{i} h_95': h95, f'epoch': epoch})

            early_stopping.check(val_loss, model)

            if early_stopping.stop_training or epoch == 99:
                if epoch != 99:
                    print(f'Early stopping at epoch {epoch}. Restoring and saving weights')
                os.makedirs('./Models', exist_ok=True)
                torch.save(model.state_dict(), f'./Models/{config["type"]}_trial{trial.number}_fold{i}.pt')
                break
        
        avg_val_loss += best_fold_val_loss
        avg_dice += best_fold_dice 
        avg_srf += best_fold_srf
        avg_h95 += best_fold_h95

        print("")
        print(f'Fold_{i} Dice: {best_fold_dice}, Srf: {best_fold_srf}, H95: {best_fold_h95}, Precision: {best_fold_prec}, Recall: {best_fold_rec}')
        print("")
        print(f'Fold {i} Validation indices: {val_idx.min()} - {val_idx.max()}')
        print("")

        # Clear cache
        torch.cuda.synchronize()
        del model, train_dataset, val_dataset, train_loader, val_loader
        torch.cuda.empty_cache()
        gc.collect()

    avg_val_loss /= len(selected_folds)
    avg_dice /= len(selected_folds)
    avg_srf /= len(selected_folds)
    avg_h95 /= len(selected_folds)

    # Clear cache
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    gc.collect()
    
    wandb.log({"Avg validation loss": avg_val_loss, "Avg dice score": avg_dice, "Avg surface dice": avg_srf, "Avg Hausdorff_95": avg_h95, "trial": trial.number})
    wandb.finish()

    return avg_val_loss, avg_dice, avg_srf, avg_h95