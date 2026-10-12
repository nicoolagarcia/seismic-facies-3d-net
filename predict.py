import torch
import numpy as np
import argparse
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.model import UNet3D
from src.dataset import SeismicDataset
from src.visualizer import plot_seismic_slices

def inference_sliding_window(model, volume, patch_size=(128, 128, 128), stride=(64, 64, 64),
                              num_classes=6, device='cuda', return_probs=False,
                              progress_callback=None, cancel_event=None):
    """
    Performs sliding window inference on a large volume using overlapping patches.

    Args:
        model: trained UNet3D model.
        volume: (D, H, W) numpy array with the full seismic cube.
        patch_size, stride: sliding-window geometry.
        num_classes: number of output channels/facies classes.
        device: torch device or string ('cuda', 'cpu').
        return_probs: if True, also returns the averaged softmax probability
            volume (num_classes, D, H, W) alongside the hard argmax mask.
        progress_callback: optional callback func(step, total_steps, message)
        cancel_event: optional threading.Event to abort inference gracefully.

    Returns:
        pred (np.ndarray): argmax segmentation mask, shape (D, H, W).
        probs (np.ndarray, optional): softmax probability volume, shape
            (num_classes, D, H, W). Only returned if return_probs=True.
    """
    if isinstance(device, str):
        if device == 'cuda' and not torch.cuda.is_available():
            device = torch.device('cpu')
        else:
            device = torch.device(device)

    model.eval()
    model.to(device)

    orig_d, orig_h, orig_w = volume.shape

    # Handle volume dimensions smaller than patch_size by padding
    pad_d = max(0, patch_size[0] - orig_d)
    pad_h = max(0, patch_size[1] - orig_h)
    pad_w = max(0, patch_size[2] - orig_w)
    if pad_d > 0 or pad_h > 0 or pad_w > 0:
        vol_working = np.pad(volume, ((0, pad_d), (0, pad_h), (0, pad_w)), mode='reflect')
    else:
        vol_working = volume

    d, h, w = vol_working.shape

    # Initialize output score maps and count map for averaging
    output = torch.zeros((num_classes, d, h, w), device=device)
    counts = torch.zeros((d, h, w), device=device)

    # Slide across dimensions
    z_steps = list(range(0, d - patch_size[0] + 1, stride[0]))
    y_steps = list(range(0, h - patch_size[1] + 1, stride[1]))
    x_steps = list(range(0, w - patch_size[2] + 1, stride[2]))

    # Add final steps if volume isn't divisible by stride
    if not z_steps or z_steps[-1] != d - patch_size[0]:
        z_steps.append(d - patch_size[0])
    if not y_steps or y_steps[-1] != h - patch_size[1]:
        y_steps.append(h - patch_size[1])
    if not x_steps or x_steps[-1] != w - patch_size[2]:
        x_steps.append(w - patch_size[2])

    total_patches = len(z_steps) * len(y_steps) * len(x_steps)
    step_count = 0

    with torch.no_grad():
        for z in z_steps:
            for y in y_steps:
                for x in x_steps:
                    if cancel_event is not None and cancel_event.is_set():
                        return (None, None) if return_probs else None

                    # Extract patch
                    patch = vol_working[z:z+patch_size[0], y:y+patch_size[1], x:x+patch_size[2]]

                    # Normalize patch
                    patch = (patch - patch.mean()) / (patch.std() + 1e-8)

                    # To tensor: (1, 1, D, H, W)
                    patch_tensor = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).float().to(device)

                    # Predict
                    logits = model(patch_tensor)
                    probs = torch.softmax(logits, dim=1).squeeze(0)  # (C, D, H, W)

                    # Accumulate
                    output[:, z:z+patch_size[0], y:y+patch_size[1], x:x+patch_size[2]] += probs
                    counts[z:z+patch_size[0], y:y+patch_size[1], x:x+patch_size[2]] += 1

                    step_count += 1
                    if progress_callback is not None:
                        progress_callback(step_count, total_patches, f"Inferencia: parche {step_count}/{total_patches}")

    # Average overlapping areas
    counts = torch.clamp(counts, min=1.0)
    output /= counts.unsqueeze(0)
    probs_np = output.cpu().numpy()

    # Crop unpadded region if needed
    if pad_d > 0 or pad_h > 0 or pad_w > 0:
        probs_np = probs_np[:, :orig_d, :orig_h, :orig_w]

    pred = np.argmax(probs_np, axis=0)

    if return_probs:
        return pred, probs_np
    return pred

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading model from {args.checkpoint} on {device}...")
    
    # Initialize model
    model = UNet3D(in_channels=1, out_channels=args.num_classes).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load sample data (Dummy or real)
    if args.dummy:
        print("Using synthetic dummy volume for inference test...")
        vol_shape = (256, 256, 256)
        seis = np.random.randn(*vol_shape).astype(np.float32)
        fault = np.random.randint(0, 6, vol_shape).astype(np.int64)
    else:
        # Load files from prediction_dir
        files = [f for f in os.listdir(args.prediction_dir) if f.endswith(('.npz', '.dat'))]
        if not files: raise FileNotFoundError(f"No .npz or .dat files in {args.prediction_dir}")

        filename = files[0]
        filepath = os.path.join(args.prediction_dir, filename)
        print(f"Loading volume: {filename}...")

        if filename.endswith('.npz'):
            data = np.load(filepath)
            seis = data['seis']
            fault = data['fault'] if 'fault' in data else None
        else: # .dat file
            seis = np.fromfile(filepath, dtype=np.float32).reshape(args.vol_shape)
            fault = None

    # Run Inference
    print("Starting sliding window inference...")
    pred = inference_sliding_window(model, seis, patch_size=(args.patch_size,)*3, 
                                     stride=(args.stride,)*3, num_classes=args.num_classes, device=device)
    
    # Visualize results
    print("Generating visualization...")
    fig = plot_seismic_slices(seis, fault, pred, num_classes=args.num_classes)
    plt.savefig(args.output_img)
    print(f"Visualized results saved to {args.output_img}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="3D Seismic Facies Inference")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best_model.pth")
    parser.add_argument("--prediction_dir", type=str, default="prediction/f3d", help="Directory with .dat or .npz volumes")
    parser.add_argument("--vol_shape", type=int, nargs=3, default=[128, 128, 128], help="Shape for .dat files (D H W)")
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--patch_size", type=int, default=128)
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--dummy", action="store_true")
    parser.add_argument("--output_img", type=str, default="prediction_faults.png")
    
    args = parser.parse_args()
    main(args)
    """
    python predict.py --checkpoint checkpoints/best_model.pth --prediction_dir validation/ --output_img validation_test.png
    """