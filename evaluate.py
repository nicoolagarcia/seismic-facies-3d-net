"""
evaluate.py
-----------
Evaluación extendida del modelo 3D U-Net para interpretación de facies/fallas
sísmicas, en respuesta directa a los puntos señalados en `mejoras.txt`:

1. Métricas por clase y matriz de confusión (evita depender de un promedio
   global que puede ocultar el desbalance entre clases).
2. Barrido de umbrales de probabilidad (no solo el corte fijo de 0.5) para
   trazar la curva precisión/recall y justificar el umbral final elegido.
3. Visualización de mapas de probabidad continuos, además de la máscara
   binaria, para distinguir zonas de alta confianza de zonas ambiguas.
4. Post-procesamiento morfológico opcional (apertura/cierre binario) para
   evaluar si mejora la coherencia estructural de las fallas detectadas.

La evaluación se realiza sobre volúmenes 3D completos (no sobre parches
sueltos) leídos de un directorio de validación independiente del de
entrenamiento (`--val_dir`), evitando así el leakage espacial entre
entrenamiento y validación mencionado en `mejoras.txt`.

Uso típico:
    python evaluate.py --checkpoint checkpoints/best_model.pth \
        --val_dir validation/ --num_classes 2 --output_dir eval_results
"""
import argparse
import os
import csv

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy import ndimage

from src.model import UNet3D
from src.metrics import (
    calculate_metrics, aggregate_metrics, confusion_matrix,
    metrics_from_confusion_matrix, binary_metrics_at_threshold,
)
from predict import inference_sliding_window


def load_volume(filepath, vol_shape=None, dtype=np.float32):
    """Loads a single seismic volume (+ ground truth if present) from .npz/.dat."""
    if filepath.endswith('.npz'):
        data = np.load(filepath)
        seis = data['seis']
        fault = data['fault'] if 'fault' in data else None
    else:
        if vol_shape is None:
            raise ValueError(f"vol_shape must be provided for binary files (.dat): {filepath}")
        seis = np.fromfile(filepath, dtype=dtype).reshape(vol_shape)
        fault = None
    return seis, fault


def plot_confusion_matrix(cm, class_names, out_path):
    """Saves a heatmap of the confusion matrix (row-normalized for readability)."""
    cm = np.asarray(cm, dtype=np.float64)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = cm / np.clip(row_sums, 1e-9, None)

    fig, ax = plt.subplots(figsize=(1.5 * len(class_names) + 2, 1.5 * len(class_names) + 2))
    im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel('Clase Predicha')
    ax.set_ylabel('Clase Real')
    ax.set_title('Matriz de Confusión (normalizada por fila)')

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, f"{cm[i, j]:.0f}\n({cm_norm[i, j]:.2f})",
                    ha='center', va='center',
                    color='white' if cm_norm[i, j] > 0.5 else 'black', fontsize=9)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_threshold_sweep(thresholds, sweep_metrics, out_path):
    """Plots precision, recall, F1 and IoU as a function of the decision threshold."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for key, label in [('precision', 'Precisión'), ('recall', 'Recall'),
                        ('f1', 'F1 (Dice)'), ('iou', 'IoU')]:
        ax.plot(thresholds, sweep_metrics[key], marker='o', label=label)
    ax.set_xlabel('Umbral de probabilidad (clase Falla)')
    ax.set_ylabel('Valor de la métrica')
    ax.set_title('Barrido de umbrales de decisión')
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_threshold_sweep_comparison(thresholds, sweep_falla, sweep_macro, out_path):
    """
    Plots F1(Dice) and IoU side by side under two conventions, so the curve
    can be read either the way trainer.py reports it during training (macro
    average across the 2 classes: 'f1_dice', 'mIoU' -- the exact same
    quantity printed as "Dice (F1)"/"mIoU" in curvas_evolucion.txt) or
    isolated to the positive "Falla" class alone ('f1', 'iou'), which is what
    actually matters when judging fault-detection quality on its own.

    Args:
        thresholds: array of decision thresholds swept.
        sweep_falla: dict with 'f1'/'iou' (and 'precision'/'recall') arrays,
            one value per threshold, for the isolated Falla class.
        sweep_macro: dict with 'f1_dice'/'mIoU' (and 'mean_precision'/
            'mean_recall') arrays, one value per threshold, matching the
            aggregate_metrics() convention used in src/metrics.py /
            trainer.py during training-time validation (same 'Dice (F1)'
            reported per epoch in curvas_evolucion.txt).
        out_path: output image path.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].plot(thresholds, sweep_falla['f1'], marker='o', label='F1 (solo clase Falla)')
    axes[0].plot(thresholds, sweep_macro['f1_dice'], marker='s',
                 label='Dice (F1) -- promedio macro 2 clases, igual que en entrenamiento')
    axes[0].set_xlabel('Umbral de probabilidad (clase Falla)')
    axes[0].set_ylabel('F1 / Dice')
    axes[0].set_title('F1 (clase Falla) vs. Dice (F1) de estilo curvas_evolucion.txt')
    axes[0].set_ylim(0, 1.02)
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(thresholds, sweep_falla['iou'], marker='o', label='IoU (solo clase Falla)')
    axes[1].plot(thresholds, sweep_macro['mIoU'], marker='s',
                 label='mIoU -- promedio macro 2 clases, igual que en entrenamiento')
    axes[1].set_xlabel('Umbral de probabilidad (clase Falla)')
    axes[1].set_ylabel('IoU')
    axes[1].set_title('IoU (clase Falla) vs. mIoU de estilo curvas_evolucion.txt')
    axes[1].set_ylim(0, 1.02)
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_probability_maps(seis, prob_fault, mask, gt_mask=None, slice_idx=None,
                           out_path='probability_maps.png'):
    """
    Visualizes, for a single slice per orientation, the raw seismic amplitude,
    the continuous fault-probability map, the predicted binary mask and
    (when available) the ground-truth binary mask, one below the other.

    Stacking the predicted mask directly above the real one lets the two be
    compared voxel-for-voxel on the same slice, instead of having to infer
    the difference visually from the probability map alone: this is what
    makes false positives/negatives visible as a mismatch between the last
    two rows of the figure.

    Args:
        seis: (D, H, W) seismic amplitude volume.
        prob_fault: (D, H, W) softmax probability of the "Falla" class.
        mask: (D, H, W) predicted binary mask (e.g. prob_fault >= threshold).
        gt_mask: (D, H, W) ground-truth binary mask, or None if not available
            (e.g. for a true prediction volume without labels). When
            provided, a fourth row with the real fault mask is added below
            the predicted one for direct comparison.
        slice_idx: (z, y, x) slice indices; defaults to the volume's center.
        out_path: output image path.
    """
    if slice_idx is None:
        sz, sy, sx = seis.shape
        slice_idx = (sz // 2, sy // 2, sx // 2)
    z_idx, y_idx, x_idx = slice_idx

    titles = ['Inline (X)', 'Crossline (Y)', 'Depth (Z)']
    seis_slices = [seis[:, :, x_idx], seis[:, y_idx, :], seis[z_idx, :, :]]
    prob_slices = [prob_fault[:, :, x_idx], prob_fault[:, y_idx, :], prob_fault[z_idx, :, :]]
    mask_slices = [mask[:, :, x_idx], mask[:, y_idx, :], mask[z_idx, :, :]]

    n_rows = 4 if gt_mask is not None else 3
    fig, axes = plt.subplots(n_rows, 3, figsize=(15, 5 * n_rows))

    for i in range(3):
        axes[0, i].imshow(seis_slices[i], cmap='gray')
        axes[0, i].set_title(f"{titles[i]} - Sísmica")
        axes[0, i].axis('off')

        im1 = axes[1, i].imshow(prob_slices[i], cmap='inferno', vmin=0, vmax=1)
        axes[1, i].set_title(f"{titles[i]} - Mapa de Probabilidad (Falla)")
        axes[1, i].axis('off')
        if i == 2:
            fig.colorbar(im1, ax=axes[1, i], fraction=0.046, pad=0.04)

        axes[2, i].imshow(mask_slices[i], cmap='gray', vmin=0, vmax=1)
        axes[2, i].set_title(f"{titles[i]} - Máscara Predicha")
        axes[2, i].axis('off')

        if gt_mask is not None:
            gt_slices = [gt_mask[:, :, x_idx], gt_mask[:, y_idx, :], gt_mask[z_idx, :, :]]
            axes[3, i].imshow(gt_slices[i], cmap='gray', vmin=0, vmax=1)
            axes[3, i].set_title(f"{titles[i]} - Máscara Real (Ground Truth)")
            axes[3, i].axis('off')

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def morphological_postprocess(binary_mask, opening_size=1, closing_size=1):
    """
    Applies binary opening followed by binary closing to the predicted fault
    mask, as a lightweight structural clean-up step (mejoras.txt item 5):
    opening removes isolated speckle/noise voxels, closing bridges small gaps
    along otherwise continuous fault planes.

    A true skeletonization (1-voxel-thin fault surfaces) requires
    scikit-image (`skimage.morphology.skeletonize`), which is not part of
    `requirements.txt`. If a thinner, structurally-consistent representation
    is needed for the thesis figures, install scikit-image and call
    `skimage.morphology.skeletonize_3d` on the returned mask.

    Args:
        binary_mask: (D, H, W) boolean/int array.
        opening_size, closing_size: structuring element radius (in voxels)
            for `scipy.ndimage.binary_opening/closing` with a cube structure.

    Returns:
        np.ndarray (bool) of the same shape, post-processed mask.
    """
    mask = binary_mask.astype(bool)
    if opening_size > 0:
        struct_o = ndimage.generate_binary_structure(3, 1)
        mask = ndimage.binary_opening(mask, structure=struct_o, iterations=opening_size)
    if closing_size > 0:
        struct_c = ndimage.generate_binary_structure(3, 1)
        mask = ndimage.binary_closing(mask, structure=struct_c, iterations=closing_size)
    return mask


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Cargando modelo desde {args.checkpoint} en {device}...")
    model = UNet3D(in_channels=1, out_channels=args.num_classes).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    if args.dummy:
        print("Usando volumen sintético (dummy) para prueba de humo del pipeline...")
        vol_shape = (160, 160, 160)
        volumes = [("dummy",
                    np.random.randn(*vol_shape).astype(np.float32),
                    np.random.randint(0, args.num_classes, vol_shape).astype(np.int64))]
    else:
        files = [f for f in os.listdir(args.val_dir) if f.endswith(('.npz', '.dat'))]
        if not files:
            raise FileNotFoundError(f"No se encontraron volúmenes .npz/.dat en {args.val_dir}")
        volumes = []
        for filename in files:
            seis, fault = load_volume(os.path.join(args.val_dir, filename), vol_shape=args.vol_shape)
            if fault is None:
                print(f"  [aviso] {filename} no tiene ground truth ('fault'); se omite de la evaluación cuantitativa.")
                continue
            volumes.append((filename, seis, fault))

    if not volumes:
        raise RuntimeError("Ningún volumen con ground truth disponible para evaluar.")

    class_names = [f"C{c}" for c in range(args.num_classes)]
    global_cm = np.zeros((args.num_classes, args.num_classes), dtype=np.int64)
    thresholds = np.round(np.arange(args.thr_min, args.thr_max + 1e-9, args.thr_step), 3)
    # Métricas aisladas de la clase Falla (positiva) en cada umbral...
    sweep_accum = {k: [] for k in ('precision', 'recall', 'f1', 'iou')}
    # ...y, en paralelo, el promedio macro entre las 2 clases (No Falla/Falla)
    # calculado con la MISMA metodología que trainer.py/aggregate_metrics, de
    # modo que cada umbral tenga también su "mean_precision/mean_recall/
    # f1_dice/mIoU" directamente comparable con curvas_evolucion.txt.
    # 'f1_dice' es exactamente la misma cantidad que la columna "Dice (F1)"
    # de curvas_evolucion.txt (internamente, aggregate_metrics la llama
    # 'mean_f1'; aquí se renombra a 'f1_dice' para que no se confunda con un
    # simple promedio y quede claro que es el coeficiente de Dice).
    sweep_macro_accum = {k: [] for k in ('mean_precision', 'mean_recall', 'f1_dice', 'mIoU')}

    for filename, seis, fault in volumes:
        print(f"\n=== Evaluando volumen completo: {filename} (shape={seis.shape}) ===")
        pred, probs = inference_sliding_window(
            model, seis, patch_size=(args.patch_size,) * 3, stride=(args.stride,) * 3,
            num_classes=args.num_classes, device=device, return_probs=True,
        )

        # --- 1. Matriz de confusión / métricas por clase (umbral por defecto = argmax) ---
        cm = confusion_matrix(pred, fault, num_classes=args.num_classes)
        global_cm += cm
        per_class = metrics_from_confusion_matrix(cm)
        print("Métricas por clase (argmax, equivalente a umbral 0.5):")
        for c in range(args.num_classes):
            print(f"  Clase {c}: precision={per_class['precision'][c]:.4f} "
                  f"recall={per_class['recall'][c]:.4f} f1={per_class['f1'][c]:.4f} "
                  f"iou={per_class['iou'][c]:.4f}")

        cm_path = os.path.join(args.output_dir, f"confusion_matrix_{filename}.png")
        plot_confusion_matrix(cm, class_names, cm_path)

        # --- 2. Barrido de umbrales (solo tiene sentido binario: Falla vs No Falla) ---
        if args.num_classes == 2:
            prob_fault = probs[1]
            gt_binary = (fault > 0).astype(np.uint8)
            for t in thresholds:
                # (a) Métrica binaria aislada de la clase Falla (positiva)
                m = binary_metrics_at_threshold(prob_fault, gt_binary, threshold=t)
                for k in sweep_accum:
                    sweep_accum[k].append(m[k])

                # (b) Promedio macro 2 clases con la misma convención que
                #     trainer.py: se binariza la predicción en ese umbral,
                #     se arma la matriz de confusión 2x2 (No Falla/Falla) y
                #     se promedian las métricas por clase sin ponderar.
                thr_pred = (prob_fault >= t).astype(np.int64)
                cm_t = confusion_matrix(thr_pred, gt_binary, num_classes=2)
                per_class_t = metrics_from_confusion_matrix(cm_t)
                sweep_macro_accum['mean_precision'].append(float(np.mean(per_class_t['precision'])))
                sweep_macro_accum['mean_recall'].append(float(np.mean(per_class_t['recall'])))
                sweep_macro_accum['f1_dice'].append(float(np.mean(per_class_t['f1'])))
                sweep_macro_accum['mIoU'].append(float(np.mean(per_class_t['iou'])))

            # Mapa de probabilidad + máscara binaria en el umbral por defecto,
            # junto con la máscara real (ground truth) para comparación directa
            default_mask = prob_fault >= 0.5
            prob_map_path = os.path.join(args.output_dir, f"probability_map_{filename}.png")
            plot_probability_maps(seis, prob_fault, default_mask, gt_mask=gt_binary,
                                   out_path=prob_map_path)

            # --- 4. Post-procesamiento morfológico opcional ---
            if args.morph_postprocess:
                post_mask = morphological_postprocess(
                    default_mask, opening_size=args.morph_opening, closing_size=args.morph_closing)
                before = binary_metrics_at_threshold(default_mask.astype(np.float64), gt_binary, threshold=0.5)
                after = binary_metrics_at_threshold(post_mask.astype(np.float64), gt_binary, threshold=0.5)
                print(f"  Post-procesamiento morfológico (opening={args.morph_opening}, closing={args.morph_closing}):")
                print(f"    Antes  -> precision={before['precision']:.4f} recall={before['recall']:.4f} f1={before['f1']:.4f} iou={before['iou']:.4f}")
                print(f"    Después-> precision={after['precision']:.4f} recall={after['recall']:.4f} f1={after['f1']:.4f} iou={after['iou']:.4f}")
        else:
            print("  [aviso] El barrido de umbrales y el post-procesamiento binario se omiten "
                  "porque --num_classes > 2 (no es un problema estrictamente binario Falla/No-Falla).")

    # --- Resultados agregados sobre todos los volúmenes evaluados ---
    print("\n=== Matriz de confusión global (todos los volúmenes) ===")
    print(global_cm)
    global_metrics = metrics_from_confusion_matrix(global_cm)

    # Nota metodológica: `trainer.py` (Trainer.validate -> aggregate_metrics)
    # reporta mIoU/mean_f1/mean_precision/mean_recall como el PROMEDIO MACRO
    # simple entre todas las clases (num_classes=2 en este proyecto), es
    # decir, promedia por igual la clase "No Falla" (fácil, mayoritaria,
    # F1~0.98) y la clase "Falla" (difícil, minoritaria). Esa es la razón por
    # la que las cifras de curvas_evolucion.txt (p. ej. Dice=0.7729 en la
    # época 10) lucen más altas que las de esta tabla o que el barrido de
    # umbrales: aquí se reporta cada clase por separado, sin diluir el
    # desempeño de "Falla" con el de la clase de fondo. La fila
    # "macro_promedio_2clases" reconstruye exactamente el número que hubiera
    # reportado curvas_evolucion.txt para este mismo checkpoint, como
    # verificación cruzada de que ambos cálculos son consistentes.
    macro_precision = float(np.mean(global_metrics['precision']))
    macro_recall = float(np.mean(global_metrics['recall']))
    macro_f1 = float(np.mean(global_metrics['f1']))
    macro_iou = float(np.mean(global_metrics['iou']))
    print("\n=== Reconciliación con la metodología de trainer.py (promedio macro 2 clases) ===")
    print(f"  precision={macro_precision:.4f}  recall={macro_recall:.4f}  "
          f"F1(Dice)={macro_f1:.4f}  mIoU={macro_iou:.4f}")
    print("  (compárese con la fila correspondiente al último checkpoint evaluado en curvas_evolucion.txt)")

    with open(os.path.join(args.output_dir, "per_class_metrics.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["clase", "precision", "recall", "f1", "iou"])
        for c in range(args.num_classes):
            writer.writerow([c, global_metrics['precision'][c], global_metrics['recall'][c],
                              global_metrics['f1'][c], global_metrics['iou'][c]])
        writer.writerow(["macro_promedio_2clases", macro_precision, macro_recall, macro_f1, macro_iou])
    plot_confusion_matrix(global_cm, class_names, os.path.join(args.output_dir, "confusion_matrix_global.png"))

    if args.num_classes == 2 and sweep_accum['precision']:
        n_vols = len(volumes)
        n_thr = len(thresholds)
        avg_sweep = {k: np.array(v).reshape(n_vols, n_thr).mean(axis=0) for k, v in sweep_accum.items()}
        avg_sweep_macro = {k: np.array(v).reshape(n_vols, n_thr).mean(axis=0) for k, v in sweep_macro_accum.items()}

        sweep_path = os.path.join(args.output_dir, "threshold_sweep.csv")
        with open(sweep_path, "w", newline="") as f:
            writer = csv.writer(f)
            # Columnas "_falla": métrica binaria aislada de la clase Falla.
            # Columnas "mean_precision"/"mean_recall"/"f1_dice"/"mIoU":
            # promedio macro 2 clases, misma convención que
            # trainer.py/aggregate_metrics; "f1_dice" es literalmente la
            # columna "Dice (F1)" de curvas_evolucion.txt (comparable
            # directamente para tau=0.5).
            writer.writerow(["threshold", "precision_falla", "recall_falla", "f1_falla", "iou_falla",
                              "mean_precision", "mean_recall", "f1_dice", "mIoU"])
            for i, t in enumerate(thresholds):
                writer.writerow([t, avg_sweep['precision'][i], avg_sweep['recall'][i],
                                  avg_sweep['f1'][i], avg_sweep['iou'][i],
                                  avg_sweep_macro['mean_precision'][i], avg_sweep_macro['mean_recall'][i],
                                  avg_sweep_macro['f1_dice'][i], avg_sweep_macro['mIoU'][i]])
        plot_threshold_sweep(thresholds, avg_sweep, os.path.join(args.output_dir, "threshold_sweep.png"))
        plot_threshold_sweep_comparison(thresholds, avg_sweep, avg_sweep_macro,
                                         os.path.join(args.output_dir, "threshold_sweep_comparativo.png"))

        best_idx = int(np.argmax(avg_sweep['f1']))
        print(f"\nMejor umbral según F1 (clase Falla): {thresholds[best_idx]:.3f} "
              f"(F1_falla={avg_sweep['f1'][best_idx]:.4f}, precision_falla={avg_sweep['precision'][best_idx]:.4f}, "
              f"recall_falla={avg_sweep['recall'][best_idx]:.4f}, iou_falla={avg_sweep['iou'][best_idx]:.4f} | "
              f"Dice(F1)={avg_sweep_macro['f1_dice'][best_idx]:.4f}, mIoU={avg_sweep_macro['mIoU'][best_idx]:.4f})")

        best_idx_macro = int(np.argmax(avg_sweep_macro['f1_dice']))
        print(f"Mejor umbral según Dice (F1) -- promedio macro, estilo curvas_evolucion.txt: "
              f"{thresholds[best_idx_macro]:.3f} (Dice(F1)={avg_sweep_macro['f1_dice'][best_idx_macro]:.4f}, "
              f"mIoU={avg_sweep_macro['mIoU'][best_idx_macro]:.4f}, "
              f"precision_falla={avg_sweep['precision'][best_idx_macro]:.4f}, "
              f"recall_falla={avg_sweep['recall'][best_idx_macro]:.4f})")
        print("  [nota] Las columnas '_falla' miden solo la clase positiva; 'mean_precision'/'mean_recall'/"
              "'f1_dice'/'mIoU' son el promedio macro de 2 clases, exactamente comparable con "
              "curvas_evolucion.txt en tau=0.5 (f1_dice == columna 'Dice (F1)' de ese archivo). "
              "Ambos criterios de 'mejor umbral' pueden no coincidir, ya que optimizan cantidades distintas.")

    print(f"\nResultados guardados en: {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluación extendida: métricas por clase, matriz de "
                                                  "confusión, barrido de umbrales y mapas de probabilidad.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Ruta a best_model.pth")
    parser.add_argument("--val_dir", type=str, default="validation", help="Directorio con volúmenes .npz/.dat de validación (independiente del de entrenamiento)")
    parser.add_argument("--vol_shape", type=int, nargs=3, default=[128, 128, 128], help="Shape para archivos .dat")
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--patch_size", type=int, default=128)
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--thr_min", type=float, default=0.1)
    parser.add_argument("--thr_max", type=float, default=0.9)
    parser.add_argument("--thr_step", type=float, default=0.05)
    parser.add_argument("--morph_postprocess", action="store_true", help="Aplica apertura+cierre binario a la máscara del umbral 0.5")
    parser.add_argument("--morph_opening", type=int, default=1)
    parser.add_argument("--morph_closing", type=int, default=1)
    parser.add_argument("--output_dir", type=str, default="eval_results")
    parser.add_argument("--dummy", action="store_true", help="Ejecuta con un volumen sintético para verificar el pipeline")

    args = parser.parse_args()
    main(args)
    """
    python evaluate.py --checkpoint checkpoints/best_model.pth --val_dir validation/ \
        --num_classes 2 --morph_postprocess --output_dir eval_results
    """
