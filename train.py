import argparse
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os

from src.model import UNet3D
from src.dataset import get_dataloader
from src.loss import HybridLoss
from src.trainer import Trainer

def main(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load Data Loaders
    print(f"Initializing Data Loaders (Dummy Mode: {args.dummy})...")
    train_loader = get_dataloader(args.train_dir, batch_size=args.batch_size, 
                                  patch_size=(128, 128, 128), is_training=True, dummy=args.dummy)
    
    val_loader = get_dataloader(args.val_dir, batch_size=args.batch_size, 
                                patch_size=(128, 128, 128), is_training=False, dummy=args.dummy)

    # Initialize Model
    print(f"Initializing 3D U-Net with {args.num_classes} classes...")
    model = UNet3D(in_channels=1, out_channels=args.num_classes)
    
    # Optimizer & Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # Loss Function (with optional class weights)
    weights = None
    if args.weights:
        weights = torch.tensor(args.weights).float().to(device)
    
    loss_fn = HybridLoss(weight=weights, num_classes=args.num_classes)

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        scheduler=scheduler,
        num_classes=args.num_classes,
        checkpoint_dir=args.checkpoint_dir
    )

    # Start Training
    print("Starting training loop...")
    trainer.fit(epochs=args.epochs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="3D Seismic Facies Segmentation Trainer")
    parser.add_argument("--train_dir", type=str, default="train", help="Path to training .npz files (keys: seis, fault)")
    parser.add_argument("--val_dir", type=str, default="validation", help="Path to validation .npz files")
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--dummy", action="store_true", help="Run with synthetic dummy data")
    parser.add_argument("--weights", type=float, nargs='+', help="Class weights for Loss (CE)")

    args = parser.parse_args()
    main(args)
