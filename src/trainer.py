import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import os
from src.metrics import calculate_metrics, aggregate_metrics

class Trainer:
    """
    Handles the training and validation loops with Mixed Precision (AMP).
    """
    def __init__(self, model, optimizer, loss_fn, train_loader, val_loader, 
                 device='cuda', scheduler=None, num_classes=6, checkpoint_dir='checkpoints'):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.scheduler = scheduler
        self.num_classes = num_classes
        self.use_amp = (self.device.type == 'cuda') if isinstance(self.device, torch.device) else (self.device == 'cuda')
        self.scaler = GradScaler(enabled=self.use_amp)
        self.checkpoint_dir = checkpoint_dir
        
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} Training")
        
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            
            if self.use_amp:
                with autocast():
                    output = self.model(data)
                    loss = self.loss_fn(output, target)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                output = self.model(data)
                loss = self.loss_fn(output, target)
                loss.backward()
                self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'batch_loss': f"{loss.item():.4f}"})
            
        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def validate(self, epoch):
        self.model.eval()
        total_loss = 0
        all_metrics = {'precision': [], 'recall': [], 'f1': [], 'iou': []}
        
        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch} Validation")
        for data, target in pbar:
            data, target = data.to(self.device), target.to(self.device)
            
            if self.use_amp:
                with autocast():
                    output = self.model(data)
                    loss = self.loss_fn(output, target)
            else:
                output = self.model(data)
                loss = self.loss_fn(output, target)
            
            total_loss += loss.item()
            
            # Compute segmentation metrics
            preds = torch.argmax(output, dim=1)
            batch_metrics = calculate_metrics(preds, target, num_classes=self.num_classes)
            for k, v in batch_metrics.items():
                all_metrics[k].append(v)
        
        # Aggregate results
        agg = aggregate_metrics(all_metrics)
        avg_loss = total_loss / len(self.val_loader)
        
        print(f"\nValidation Result (Epoch {epoch}):")
        print(f"  - Avg Loss: {avg_loss:.4f}")
        print(f"  - mIoU: {agg['mIoU']:.4f}")
        print(f"  - Mean F1 (Dice): {agg['mean_f1']:.4f}")
        print(f"  - Global Precision: {agg['mean_precision']:.4f}")
        print(f"  - Global Recall: {agg['mean_recall']:.4f}")
        
        return avg_loss, agg

    def save_checkpoint(self, epoch, mIoU, is_best=False):
        state = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'mIoU': mIoU,
        }
        filename = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
        torch.save(state, filename)
        if is_best:
            best_filename = os.path.join(self.checkpoint_dir, 'best_model.pth')
            torch.save(state, best_filename)
            print(f"   --> Best model saved with mIoU: {mIoU:.4f}")

    def fit(self, epochs):
        best_mIoU = 0.0
        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(epoch)
            val_loss, val_metrics = self.validate(epoch)
            
            if self.scheduler:
                self.scheduler.step(val_loss)
            
            is_best = val_metrics['mIoU'] > best_mIoU
            if is_best:
                best_mIoU = val_metrics['mIoU']
            
            self.save_checkpoint(epoch, val_metrics['mIoU'], is_best=is_best)
            print(f"Epoch {epoch} complete. Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}\n")
