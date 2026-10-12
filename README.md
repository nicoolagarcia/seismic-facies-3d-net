# Seismic Facies 3D Net

Pipeline de segmentación 3D basado en un U-Net para la interpretación de facies geológicas (y fallas) en volúmenes sísmicos, implementado en PyTorch.

## ¿En qué consiste?

El proyecto entrena y evalúa una red **3D U-Net** (encoder-decoder simétrico de 4 niveles, skip connections, `BatchNorm3D`) que clasifica cada voxel de un cubo sísmico en una clase de facies/falla. Incluye:

- **Loss híbrida**: Cross Entropy + Dice Loss (0.5/0.5) para mitigar el desbalance de clases.
- **Mixed Precision Training (AMP)** para entrenar parches de 128×128×128 en GPUs estándar.
- **Inferencia por ventana deslizante** (sliding window con solapamiento y promediado) para eliminar artefactos de borde al reconstruir el cubo completo.
- **Generador de datos sintéticos** (`--dummy`) para probar el pipeline sin el dataset completo.
- **Evaluación** con métricas por clase (Precision, Recall, F1, mIoU), matrices de confusión, mapas de probabilidad y barrido de umbrales.

## Estructura del proyecto

```
src/
  model.py       # Arquitectura 3D U-Net
  dataset.py     # Carga de datos, normalización Z-score y augmentations 3D
  loss.py        # Loss híbrida (CE + Dice)
  metrics.py     # Precision, Recall, F1, mIoU
  trainer.py     # Loop de entrenamiento, AMP y checkpointing
  visualizer.py  # Visualización de slices (Inline/Crossline/Depth) con overlay de GT
train.py         # Entrenamiento
predict.py       # Inferencia y visualización
evaluate.py      # Evaluación cuantitativa sobre el set de validación
inspect_data.py  # Inspección rápida de un volumen .npz
gui_app.py       # Interfaz gráfica (GUI) interactiva de predicción y visualización
```

Los directorios `train/`, `validation/`, `data/` y `prediction/` (datos crudos) y `checkpoints/` (pesos entrenados) no se versionan en este repositorio por su tamaño. Ver [Modelos y datos](#modelos-y-datos-pesados).

## Instalación

```bash
pip install -r requirements.txt
```

Dependencias principales: `torch`, `torchvision`, `numpy`, `matplotlib`, `tqdm`, `scipy`, `customtkinter`.

## Uso

### Interfaz Gráfica (GUI Interactiva)

Inicia la interfaz de escritorio para seleccionar modelo, cargar volumen y visualizar la predicción:
```bash
python gui_app.py
```
*(En Windows también puedes hacer doble clic en `run_gui.bat`)*

### Entrenamiento

Prueba rápida con datos sintéticos:
```bash
python train.py --dummy --epochs 2 --batch_size 1
```

Entrenamiento real:
```bash
python train.py --train_dir train/ --val_dir validation/ --epochs 20 --batch_size 2
```

Si hay clases sub-representadas, se pueden ponderar con `--weights` (una por clase, ej. `--weights 1.0 2.0 2.0 1.5 1.5 3.0`).

### Inferencia por Consola

```bash
python predict.py --checkpoint checkpoints/best_model.pth --prediction_dir prediction/f3d --vol_shape 128 128 128
```

### Evaluación

```bash
python evaluate.py --checkpoint checkpoints/best_model.pth --val_dir validation/ --output_dir eval_results
```

Genera matrices de confusión, mapas de probabilidad y métricas por clase (`eval_results/per_class_metrics.csv`).

### Inspección de un volumen

```bash
python inspect_data.py
```

Imprime la forma del volumen, el rango de valores sísmicos y la distribución de facies del ground truth.

## Notas de memoria

Si aparecen errores de "Out of Memory" (OOM), reduce `--batch_size` a 1 o disminuye `init_features` en `src/model.py` (por defecto 32).

## Modelos y datos pesados

Los checkpoints entrenados (`checkpoints/*.pth`, ~259 MB cada uno) y los volúmenes de datos (`train/`, `validation/`, `prediction/`) no están incluidos en este repositorio por superar el límite de tamaño de GitHub. Están disponibles en Google Drive:

https://drive.google.com/drive/folders/1Kqs-8dyDTqoUh61Cauxtiv_R0TG0R8hU?usp=sharing
