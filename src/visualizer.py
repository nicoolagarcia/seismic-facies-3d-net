import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

def get_colormap(cmap_name='viridis', num_classes=6):
    """Safely retrieves a colormap across different matplotlib versions."""
    try:
        return plt.colormaps[cmap_name].resampled(num_classes)
    except (AttributeError, KeyError):
        try:
            return plt.get_cmap(cmap_name, num_classes)
        except Exception:
            return plt.cm.get_cmap('viridis', num_classes)

def plot_seismic_slices(seis, facies=None, pred=None, slice_idx=None, num_classes=6, fig=None, cmap_name='viridis'):
    """
    Plots Inline, Crossline, and Depth/Time slices in rows:
    - Row 0: Seismic amplitude (grayscale)
    - Row 1: Ground truth facies/faults (if present)
    - Row 2: Model prediction (if present)

    Args:
        seis: (D, H, W) volumetric data
        facies: (D, H, W) ground truth labels
        pred: (D, H, W) predictions
        slice_idx: (z_idx, y_idx, x_idx) coordinates for the slices
        num_classes: number of facies/fault classes
        fig: optional existing matplotlib Figure to render into
        cmap_name: colormap name for facies/faults
    """
    if slice_idx is None:
        sz, sy, sx = seis.shape
        slice_idx = (sz // 2, sy // 2, sx // 2)

    z_idx, y_idx, x_idx = slice_idx
    sz, sy, sx = seis.shape
    z_idx = max(0, min(z_idx, sz - 1))
    y_idx = max(0, min(y_idx, sy - 1))
    x_idx = max(0, min(x_idx, sx - 1))

    titles = [f'Inline (X={x_idx})', f'Crossline (Y={y_idx})', f'Depth (Z={z_idx})']
    slices_seis = [seis[:, :, x_idx], seis[:, y_idx, :], seis[z_idx, :, :]]

    rows = 1
    if facies is not None:
        rows += 1
    if pred is not None:
        rows += 1

    if fig is None:
        fig, axes = plt.subplots(rows, 3, figsize=(15, 4.2 * rows))
    else:
        fig.clf()
        axes = fig.subplots(rows, 3)

    if rows == 1:
        axes = np.expand_dims(axes, 0)

    cmap = get_colormap(cmap_name, num_classes)

    # Compute sensible vmin/vmax for seismic
    seis_vmin, seis_vmax = np.percentile(seis, 1), np.percentile(seis, 99)

    for i in range(3):
        # Row 0: Seismic
        axes[0, i].imshow(slices_seis[i], cmap='gray', vmin=seis_vmin, vmax=seis_vmax, aspect='auto')
        axes[0, i].set_title(f"{titles[i]} - Sísmica", fontsize=10, fontweight='bold')
        axes[0, i].axis('off')

        curr_row = 1
        # Row 1: Ground Truth (if provided)
        if facies is not None:
            slices_facies = [facies[:, :, x_idx], facies[:, y_idx, :], facies[z_idx, :, :]]
            im = axes[curr_row, i].imshow(slices_facies[i], cmap=cmap, vmin=0, vmax=num_classes - 1, aspect='auto')
            axes[curr_row, i].set_title(f"{titles[i]} - Ground Truth", fontsize=10, fontweight='bold')
            axes[curr_row, i].axis('off')
            if i == 2:
                fig.colorbar(im, ax=axes[curr_row, i], fraction=0.046, pad=0.04, label='Clase')
            curr_row += 1

        # Row 2: Prediction (if provided)
        if pred is not None:
            slices_pred = [pred[:, :, x_idx], pred[:, y_idx, :], pred[z_idx, :, :]]
            im_pred = axes[curr_row, i].imshow(slices_pred[i], cmap=cmap, vmin=0, vmax=num_classes - 1, aspect='auto')
            axes[curr_row, i].set_title(f"{titles[i]} - Predicción", fontsize=10, fontweight='bold')
            axes[curr_row, i].axis('off')
            if i == 2:
                fig.colorbar(im_pred, ax=axes[curr_row, i], fraction=0.046, pad=0.04, label='Clase')

    fig.tight_layout()
    return fig

def plot_overlay_slices(seis, pred=None, facies=None, slice_idx=None, alpha=0.45, num_classes=6,
                        fig=None, cmap_name='tab10', show_gt=False):
    """
    Overlays predicted facies / faults (or ground truth) directly on the seismic amplitude
    with adjustable alpha transparency.
    """
    if slice_idx is None:
        sz, sy, sx = seis.shape
        slice_idx = (sz // 2, sy // 2, sx // 2)

    z_idx, y_idx, x_idx = slice_idx
    sz, sy, sx = seis.shape
    z_idx = max(0, min(z_idx, sz - 1))
    y_idx = max(0, min(y_idx, sy - 1))
    x_idx = max(0, min(x_idx, sx - 1))

    titles = [f'Inline (X={x_idx})', f'Crossline (Y={y_idx})', f'Depth (Z={z_idx})']
    slices_seis = [seis[:, :, x_idx], seis[:, y_idx, :], seis[z_idx, :, :]]

    target_vol = facies if (show_gt and facies is not None) else pred
    target_label = "Ground Truth" if (show_gt and facies is not None) else "Predicción"

    if fig is None:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    else:
        fig.clf()
        axes = fig.subplots(1, 3)

    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    seis_vmin, seis_vmax = np.percentile(seis, 1), np.percentile(seis, 99)
    cmap = get_colormap(cmap_name, num_classes)

    for i in range(3):
        # Base seismic
        axes[i].imshow(slices_seis[i], cmap='gray', vmin=seis_vmin, vmax=seis_vmax, aspect='auto')

        if target_vol is not None:
            slices_target = [target_vol[:, :, x_idx], target_vol[:, y_idx, :], target_vol[z_idx, :, :]]
            t_slice = slices_target[i]

            # Mask out background if binary (class 0)
            masked_target = np.ma.masked_where(t_slice == 0, t_slice)
            im = axes[i].imshow(masked_target, cmap=cmap, vmin=0, vmax=max(1, num_classes - 1),
                                alpha=alpha, aspect='auto')
            if i == 2:
                fig.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04, label=f'Clase {target_label}')

        axes[i].set_title(f"{titles[i]} - Overlay ({target_label})", fontsize=10, fontweight='bold')
        axes[i].axis('off')

    fig.tight_layout()
    return fig

def plot_probability_slices(seis, prob_map, mask=None, slice_idx=None, fig=None,
                            cmap_prob='inferno', class_name="Falla"):
    """
    Visualizes the raw seismic amplitude, continuous softmax probability heatmap,
    and thresholded segmentation mask.
    """
    if slice_idx is None:
        sz, sy, sx = seis.shape
        slice_idx = (sz // 2, sy // 2, sx // 2)

    z_idx, y_idx, x_idx = slice_idx
    sz, sy, sx = seis.shape
    z_idx = max(0, min(z_idx, sz - 1))
    y_idx = max(0, min(y_idx, sy - 1))
    x_idx = max(0, min(x_idx, sx - 1))

    titles = [f'Inline (X={x_idx})', f'Crossline (Y={y_idx})', f'Depth (Z={z_idx})']
    slices_seis = [seis[:, :, x_idx], seis[:, y_idx, :], seis[z_idx, :, :]]
    slices_prob = [prob_map[:, :, x_idx], prob_map[:, y_idx, :], prob_map[z_idx, :, :]]

    rows = 2 if mask is None else 3
    if fig is None:
        fig, axes = plt.subplots(rows, 3, figsize=(15, 4.2 * rows))
    else:
        fig.clf()
        axes = fig.subplots(rows, 3)

    seis_vmin, seis_vmax = np.percentile(seis, 1), np.percentile(seis, 99)

    for i in range(3):
        # Row 0: Seismic
        axes[0, i].imshow(slices_seis[i], cmap='gray', vmin=seis_vmin, vmax=seis_vmax, aspect='auto')
        axes[0, i].set_title(f"{titles[i]} - Sísmica", fontsize=10, fontweight='bold')
        axes[0, i].axis('off')

        # Row 1: Probability Heatmap
        im_p = axes[1, i].imshow(slices_prob[i], cmap=cmap_prob, vmin=0.0, vmax=1.0, aspect='auto')
        axes[1, i].set_title(f"{titles[i]} - Probabilidad ({class_name})", fontsize=10, fontweight='bold')
        axes[1, i].axis('off')
        if i == 2:
            fig.colorbar(im_p, ax=axes[1, i], fraction=0.046, pad=0.04, label='Probabilidad [0, 1]')

        # Row 2: Mask (if provided)
        if mask is not None:
            slices_mask = [mask[:, :, x_idx], mask[:, y_idx, :], mask[z_idx, :, :]]
            axes[2, i].imshow(slices_mask[i], cmap='gray', vmin=0, vmax=1, aspect='auto')
            axes[2, i].set_title(f"{titles[i]} - Máscara Binaria", fontsize=10, fontweight='bold')
            axes[2, i].axis('off')

    fig.tight_layout()
    return fig

if __name__ == "__main__":
    dummy_seis = np.random.randn(128, 128, 128)
    dummy_facies = np.random.randint(0, 6, (128, 128, 128))
    dummy_pred = np.random.randint(0, 6, (128, 128, 128))
    fig = plot_seismic_slices(dummy_seis, dummy_facies, dummy_pred)
    plt.savefig('test_slices.png')
    print("Saved test slice visualization to test_slices.png")
