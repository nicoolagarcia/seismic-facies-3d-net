import matplotlib.pyplot as plt
import numpy as np

def plot_seismic_slices(seis, facies=None, pred=None, slice_idx=None, num_classes=6):
    """
    Plots Inline, Crossline, and Depth/Time slices.
    - seis: (D, H, W) volumetric data
    - facies: (D, H, W) ground truth labels
    - pred: (D, H, W) predictions
    - slice_idx: (z_idx, y_idx, x_idx) coordinates for the slices
    """
    if slice_idx is None:
        # Default to middle slices
        sz, sy, sx = seis.shape
        slice_idx = (sz // 2, sy // 2, sx // 2)

    z_idx, y_idx, x_idx = slice_idx
    
    # Define which slices to show
    titles = ['Inline (X)', 'Crossline (Y)', 'Depth (Z)']
    slices_seis = [seis[:, :, x_idx], seis[:, y_idx, :], seis[z_idx, :, :]]
    
    rows = 1
    if facies is not None: rows += 1
    if pred is not None: rows += 1
    
    fig, axes = plt.subplots(rows, 3, figsize=(15, 5 * rows))
    if rows == 1: axes = np.expand_dims(axes, 0)
    
    # Color map for facies (0 to num_classes - 1)
    cmap = plt.cm.get_cmap('viridis', num_classes)

    for i in range(3):
        # Row 0: Seismic
        axes[0, i].imshow(slices_seis[i], cmap='gray')
        axes[0, i].set_title(f"{titles[i]} Seismic")
        axes[0, i].axis('off')
        
        # Row 1: Ground Truth
        if facies is not None:
            slices_facies = [facies[:, :, x_idx], facies[:, y_idx, :], facies[z_idx, :, :]]
            im = axes[1, i].imshow(slices_facies[i], cmap=cmap, vmin=0, vmax=num_classes-1)
            axes[1, i].set_title(f"{titles[i]} Faults (GT)")
            axes[1, i].axis('off')
            if i == 2:
                fig.colorbar(im, ax=axes[1, i], fraction=0.046, pad=0.04)
        
        # Row 2: Prediction
        if pred is not None:
            slices_pred = [pred[:, :, x_idx], pred[:, y_idx, :], pred[z_idx, :, :]]
            im_pred = axes[rows-1, i].imshow(slices_pred[i], cmap=cmap, vmin=0, vmax=num_classes-1)
            axes[rows-1, i].set_title(f"{titles[i]} Prediction")
            axes[rows-1, i].axis('off')
            if i == 2:
                fig.colorbar(im_pred, ax=axes[rows-1, i], fraction=0.046, pad=0.04)

    plt.tight_layout()
    return fig

if __name__ == "__main__":
    # Test visualization with dummy data
    dummy_seis = np.random.randn(128, 128, 128)
    dummy_facies = np.random.randint(0, 6, (128, 128, 128))
    dummy_pred = np.random.randint(0, 6, (128, 128, 128))
    
    fig = plot_seismic_slices(dummy_seis, dummy_facies, dummy_pred)
    plt.savefig('test_slices.png')
    print("Saved test slice visualization to test_slices.png")
