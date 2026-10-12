"""
gui_app.py
----------
Interfaz Gráfica de Usuario (GUI) moderna para el pipeline de segmentación 3D
de Facies y Fallas Sísmicas con 3D U-Net.

Permite:
1. Seleccionar o cargar un modelo PyTorch (.pth / .pt) o inicializar pesos de prueba.
2. Seleccionar un volumen de entrada (.npz o .dat) o generar uno sintético al vuelo.
3. Ejecutar la inferencia por ventana deslizante en un hilo de fondo sin bloquear la GUI.
4. Visualizar los resultados interactivamente en la interfaz mediante el módulo src.visualizer:
   - Cortes ortogonales 3D: Inline (X), Crossline (Y) y Depth (Z).
   - Sliders de navegación de cortes en tiempo real.
   - Modos de visualización: Comparativo (Filas), Overlay con transparencia, y Mapa de Probabilidad continua.
   - Umbral de decisión dinámico y cálculo de métricas (F1/Dice, IoU, Precision, Recall).
   - Exportación de visualizaciones e inferencias.
"""

import os
import sys
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import numpy as np
import torch

try:
    import customtkinter as ctk
    ctk.set_appearance_mode("Dark")
    ctk.set_default_color_theme("blue")
    CTK_AVAILABLE = True
except ImportError:
    CTK_AVAILABLE = False

from src.model import UNet3D
from src.visualizer import plot_seismic_slices, plot_overlay_slices, plot_probability_slices, get_colormap
from predict import inference_sliding_window


class SeismicGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Seismic Facies 3D Net - Predicción y Visualización de Facies Sísmicas")
        self.root.geometry("1380x880")
        self.root.minsize(1100, 750)

        # Estado de datos y modelo
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.model_path = None
        self.num_classes = 2

        self.volume_path = None
        self.seis_volume = None       # Array 3D (D, H, W)
        self.gt_volume = None         # Array 3D (D, H, W) o None
        self.pred_volume = None       # Array 3D (D, H, W)
        self.probs_volume = None      # Array 4D (C, D, H, W)

        # Estado de visualización
        self.slice_z = 0
        self.slice_y = 0
        self.slice_x = 0
        self.view_mode = "comparative"  # 'comparative', 'overlay', 'probability'
        self.overlay_alpha = 0.50
        self.threshold = 0.50
        self.colormap_name = "viridis"

        # Control de hilo de inferencia
        self.is_inferring = False
        self.cancel_event = threading.Event()

        # Construir Interfaz
        self._build_ui()

    def _build_ui(self):
        # Configurar grid principal: panel izquierdo de controles (380px), panel derecho de visualización
        self.root.grid_columnconfigure(0, weight=0, minsize=380)
        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_rowconfigure(0, weight=1)

        # -------------------------------------------------------------
        # PANEL IZQUIERDO: CONTROLES Y PARÁMETROS
        # -------------------------------------------------------------
        if CTK_AVAILABLE:
            self.left_frame = ctk.CTkScrollableFrame(self.root, width=370, corner_radius=0)
        else:
            self.left_frame = tk.Frame(self.root, width=370, bg="#2b2b2b")
        self.left_frame.grid(row=0, column=0, sticky="nsew", padx=(10, 5), pady=10)

        self._build_header_card()
        self._build_model_card()
        self._build_volume_card()
        self._build_inference_card()

        # -------------------------------------------------------------
        # PANEL DERECHO: VISUALIZACIÓN INTERACTIVA
        # -------------------------------------------------------------
        if CTK_AVAILABLE:
            self.right_frame = ctk.CTkFrame(self.root, corner_radius=10)
        else:
            self.right_frame = tk.Frame(self.root, bg="#1e1e1e")
        self.right_frame.grid(row=0, column=1, sticky="nsew", padx=(5, 10), pady=10)

        self._build_visualization_panel()

    # -----------------------------------------------------------------
    # CONSTRUCCIÓN DE COMPONENTES DEL PANEL IZQUIERDO
    # -----------------------------------------------------------------
    def _build_header_card(self):
        title_box = ctk.CTkFrame(self.left_frame, fg_color="#1f2937", corner_radius=8) if CTK_AVAILABLE else tk.Frame(self.left_frame, bg="#1f2937")
        title_box.pack(fill="x", padx=5, pady=(0, 10))

        if CTK_AVAILABLE:
            title_lbl = ctk.CTkLabel(title_box, text="3D Seismic Facies Net", font=ctk.CTkFont(size=18, weight="bold"), text_color="#38bdf8")
            title_lbl.pack(anchor="w", padx=12, pady=(10, 2))
            sub_lbl = ctk.CTkLabel(title_box, text="Inferencia 3D U-Net y Visualización Sísmica", font=ctk.CTkFont(size=11), text_color="#9ca3af")
            sub_lbl.pack(anchor="w", padx=12, pady=(0, 8))
            dev_text = f"Dispositivo de cálculo: {self.device.type.upper()}"
            dev_color = "#4ade80" if self.device.type == "cuda" else "#f59e0b"
            dev_lbl = ctk.CTkLabel(title_box, text=dev_text, font=ctk.CTkFont(size=11, weight="bold"), text_color=dev_color)
            dev_lbl.pack(anchor="w", padx=12, pady=(0, 10))
        else:
            title_lbl = tk.Label(title_box, text="3D Seismic Facies Net", font=("Arial", 14, "bold"), fg="#38bdf8", bg="#1f2937")
            title_lbl.pack(anchor="w", padx=10, pady=(8, 2))

    def _build_model_card(self):
        card = ctk.CTkFrame(self.left_frame, corner_radius=8) if CTK_AVAILABLE else tk.LabelFrame(self.left_frame, text="1. Modelo", bg="#2b2b2b", fg="white")
        card.pack(fill="x", padx=5, pady=6)

        if CTK_AVAILABLE:
            header = ctk.CTkLabel(card, text="1. Selección de Modelo", font=ctk.CTkFont(size=14, weight="bold"))
            header.pack(anchor="w", padx=10, pady=(8, 4))

            # Botón seleccionar archivo
            btn_load_model = ctk.CTkButton(card, text="📂 Seleccionar Modelo (.pth)", command=self._select_model_file, fg_color="#2563eb", hover_color="#1d4ed8")
            btn_load_model.pack(fill="x", padx=10, pady=4)

            # Botón modelo de prueba / pesos iniciales
            btn_init_model = ctk.CTkButton(card, text="⚡ Usar Modelo Demo / Pesos Iniciales", command=self._init_demo_model, fg_color="#374151", hover_color="#4b5563")
            btn_init_model.pack(fill="x", padx=10, pady=2)

            # Etiqueta estado del modelo
            self.lbl_model_status = ctk.CTkLabel(card, text="Modelo: No seleccionado", font=ctk.CTkFont(size=11), text_color="#f87171", wraplength=340, justify="left")
            self.lbl_model_status.pack(anchor="w", padx=10, pady=(4, 6))

            # Selector de clases
            row_classes = ctk.CTkFrame(card, fg_color="transparent")
            row_classes.pack(fill="x", padx=10, pady=4)
            ctk.CTkLabel(row_classes, text="Número de Clases:", font=ctk.CTkFont(size=12)).pack(side="left")
            self.combo_classes = ctk.CTkOptionMenu(row_classes, values=["2 (Fallas / Background)", "6 (Facies Geológicas)"], command=self._on_classes_change, width=170)
            self.combo_classes.set("2 (Fallas / Background)")
            self.combo_classes.pack(side="right")
        else:
            btn_load_model = tk.Button(card, text="Seleccionar Modelo (.pth)", command=self._select_model_file)
            btn_load_model.pack(fill="x", padx=5, pady=5)
            self.lbl_model_status = tk.Label(card, text="Modelo: No seleccionado", fg="red", bg="#2b2b2b")
            self.lbl_model_status.pack(padx=5, pady=2)

    def _build_volume_card(self):
        card = ctk.CTkFrame(self.left_frame, corner_radius=8) if CTK_AVAILABLE else tk.LabelFrame(self.left_frame, text="2. Volumen", bg="#2b2b2b", fg="white")
        card.pack(fill="x", padx=5, pady=6)

        if CTK_AVAILABLE:
            header = ctk.CTkLabel(card, text="2. Selección de Volumen Sísmico", font=ctk.CTkFont(size=14, weight="bold"))
            header.pack(anchor="w", padx=10, pady=(8, 4))

            # Botón seleccionar archivo de volumen
            btn_load_vol = ctk.CTkButton(card, text="📁 Seleccionar Archivo (.npz / .dat)", command=self._select_volume_file, fg_color="#0d9488", hover_color="#0f766e")
            btn_load_vol.pack(fill="x", padx=10, pady=4)

            # Botón generar sintético de demostración
            btn_synth = ctk.CTkButton(card, text="✨ Generar Volumen Sintético 3D", command=self._generate_synthetic_volume, fg_color="#374151", hover_color="#4b5563")
            btn_synth.pack(fill="x", padx=10, pady=2)

            # Tarjeta de metadatos del volumen
            self.vol_meta_box = ctk.CTkFrame(card, fg_color="#111827", corner_radius=6)
            self.vol_meta_box.pack(fill="x", padx=10, pady=6)

            self.lbl_vol_status = ctk.CTkLabel(self.vol_meta_box, text="Volumen: Ninguno cargado", font=ctk.CTkFont(size=11, weight="bold"), text_color="#f87171", justify="left")
            self.lbl_vol_status.pack(anchor="w", padx=8, pady=(4, 2))

            self.lbl_vol_shape = ctk.CTkLabel(self.vol_meta_box, text="Dimensiones: --", font=ctk.CTkFont(size=11), text_color="#9ca3af")
            self.lbl_vol_shape.pack(anchor="w", padx=8, pady=1)

            self.lbl_vol_range = ctk.CTkLabel(self.vol_meta_box, text="Amplitud Sísmica: --", font=ctk.CTkFont(size=11), text_color="#9ca3af")
            self.lbl_vol_range.pack(anchor="w", padx=8, pady=1)

            self.lbl_vol_gt = ctk.CTkLabel(self.vol_meta_box, text="Ground Truth: No detectado", font=ctk.CTkFont(size=11), text_color="#9ca3af")
            self.lbl_vol_gt.pack(anchor="w", padx=8, pady=(1, 4))
        else:
            btn_load_vol = tk.Button(card, text="Seleccionar Volumen (.npz)", command=self._select_volume_file)
            btn_load_vol.pack(fill="x", padx=5, pady=5)
            self.lbl_vol_status = tk.Label(card, text="Volumen: Ninguno", bg="#2b2b2b", fg="red")
            self.lbl_vol_status.pack(padx=5, pady=2)

    def _build_inference_card(self):
        card = ctk.CTkFrame(self.left_frame, corner_radius=8) if CTK_AVAILABLE else tk.LabelFrame(self.left_frame, text="3. Inferencia", bg="#2b2b2b", fg="white")
        card.pack(fill="x", padx=5, pady=6)

        if CTK_AVAILABLE:
            header = ctk.CTkLabel(card, text="3. Inferencia de Predicción", font=ctk.CTkFont(size=14, weight="bold"))
            header.pack(anchor="w", padx=10, pady=(8, 4))

            # Selector de Patch Size
            row_patch = ctk.CTkFrame(card, fg_color="transparent")
            row_patch.pack(fill="x", padx=10, pady=2)
            ctk.CTkLabel(row_patch, text="Patch Size:", font=ctk.CTkFont(size=12)).pack(side="left")
            self.combo_patch = ctk.CTkOptionMenu(row_patch, values=["64", "128"], width=110)
            self.combo_patch.set("64")
            self.combo_patch.pack(side="right")

            # Selector de Stride
            row_stride = ctk.CTkFrame(card, fg_color="transparent")
            row_stride.pack(fill="x", padx=10, pady=2)
            ctk.CTkLabel(row_stride, text="Stride (Paso):", font=ctk.CTkFont(size=12)).pack(side="left")
            self.combo_stride = ctk.CTkOptionMenu(row_stride, values=["64", "32", "128"], width=110)
            self.combo_stride.set("64")
            self.combo_stride.pack(side="right")

            # Botón de Ejecutar Predicción
            self.btn_run_pred = ctk.CTkButton(
                card, text="▶ Realizar Predicción", font=ctk.CTkFont(size=13, weight="bold"),
                command=self._start_inference, fg_color="#10b981", hover_color="#059669", height=38
            )
            self.btn_run_pred.pack(fill="x", padx=10, pady=(10, 4))

            # Botón de Cancelar
            self.btn_cancel_pred = ctk.CTkButton(
                card, text="⏹ Cancelar", command=self._cancel_inference,
                fg_color="#ef4444", hover_color="#dc2626", state="disabled"
            )
            self.btn_cancel_pred.pack(fill="x", padx=10, pady=(2, 6))

            # Barra de progreso y estado
            self.progress_bar = ctk.CTkProgressBar(card)
            self.progress_bar.set(0)
            self.progress_bar.pack(fill="x", padx=10, pady=4)

            self.lbl_infer_status = ctk.CTkLabel(card, text="Listo para inferir", font=ctk.CTkFont(size=11), text_color="#9ca3af")
            self.lbl_infer_status.pack(anchor="w", padx=10, pady=(0, 8))
        else:
            self.btn_run_pred = tk.Button(card, text="Realizar Predicción", command=self._start_inference, bg="#10b981", fg="white")
            self.btn_run_pred.pack(fill="x", padx=5, pady=5)
            self.lbl_infer_status = tk.Label(card, text="Listo", bg="#2b2b2b", fg="white")
            self.lbl_infer_status.pack(padx=5, pady=2)

    # -----------------------------------------------------------------
    # CONSTRUCCIÓN DEL PANEL DERECHO DE VISUALIZACIÓN
    # -----------------------------------------------------------------
    def _build_visualization_panel(self):
        # Barra superior de visualización: Selector de Modo, Colormap y Botones de Exportar
        top_bar = ctk.CTkFrame(self.right_frame, fg_color="#1f2937", corner_radius=8) if CTK_AVAILABLE else tk.Frame(self.right_frame, bg="#1f2937")
        top_bar.pack(fill="x", padx=8, pady=(8, 4))

        if CTK_AVAILABLE:
            ctk.CTkLabel(top_bar, text="Modo de Visualización:", font=ctk.CTkFont(size=12, weight="bold")).pack(side="left", padx=(10, 4), pady=6)
            self.combo_mode = ctk.CTkOptionMenu(
                top_bar,
                values=["Comparativa (Filas)", "Superposición (Overlay)", "Mapa de Probabilidad"],
                command=self._on_mode_change,
                width=190
            )
            self.combo_mode.set("Comparativa (Filas)")
            self.combo_mode.pack(side="left", padx=4, pady=6)

            ctk.CTkLabel(top_bar, text="Colormap:", font=ctk.CTkFont(size=12)).pack(side="left", padx=(12, 4), pady=6)
            self.combo_cmap = ctk.CTkOptionMenu(
                top_bar,
                values=["viridis", "inferno", "plasma", "tab10", "magma", "cividis"],
                command=self._on_cmap_change,
                width=110
            )
            self.combo_cmap.set("viridis")
            self.combo_cmap.pack(side="left", padx=4, pady=6)

            # Botones de exportación a la derecha
            btn_save_img = ctk.CTkButton(top_bar, text="💾 Guardar Imagen", command=self._save_visualization, width=120, fg_color="#374151", hover_color="#4b5563")
            btn_save_img.pack(side="right", padx=10, pady=6)

            btn_export_pred = ctk.CTkButton(top_bar, text="📦 Exportar .npz", command=self._export_prediction_npz, width=110, fg_color="#374151", hover_color="#4b5563")
            btn_export_pred.pack(side="right", padx=4, pady=6)

        # -------------------------------------------------------------
        # Canvas Embebido de Matplotlib
        # -------------------------------------------------------------
        canvas_container = ctk.CTkFrame(self.right_frame, corner_radius=6) if CTK_AVAILABLE else tk.Frame(self.right_frame, bg="black")
        canvas_container.pack(fill="both", expand=True, padx=8, pady=4)

        # Matplotlib figure con fondo oscuro adaptado
        plt.style.use("dark_background")
        self.fig = plt.Figure(figsize=(10, 6), dpi=100)
        self.fig.patch.set_facecolor("#111827")
        self.ax_initial = self.fig.add_subplot(111)
        self.ax_initial.text(0.5, 0.5, "Seleccione un modelo y un volumen sísmico\nluego presione 'Realizar Predicción'",
                             ha='center', va='center', color='#9ca3af', fontsize=13)
        self.ax_initial.axis("off")

        self.canvas = FigureCanvasTkAgg(self.fig, master=canvas_container)
        self.canvas.draw()
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill="both", expand=True)

        # Barra de herramientas de Matplotlib (zoom, pan, save)
        self.toolbar_frame = ctk.CTkFrame(self.right_frame, fg_color="transparent") if CTK_AVAILABLE else tk.Frame(self.right_frame)
        self.toolbar_frame.pack(fill="x", padx=8, pady=1)
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.toolbar_frame)
        self.toolbar.config(background="#1f2937")
        self.toolbar.update()

        # -------------------------------------------------------------
        # Barra Inferior: Sliders de Navegación 3D (Inline, Crossline, Depth)
        # -------------------------------------------------------------
        sliders_box = ctk.CTkFrame(self.right_frame, fg_color="#182234", corner_radius=8) if CTK_AVAILABLE else tk.Frame(self.right_frame, bg="#182234")
        sliders_box.pack(fill="x", padx=8, pady=(4, 8))

        # Fila 1 de sliders: Inline X, Crossline Y, Depth Z
        row_slices = ctk.CTkFrame(sliders_box, fg_color="transparent") if CTK_AVAILABLE else tk.Frame(sliders_box, bg="#182234")
        row_slices.pack(fill="x", padx=10, pady=4)

        if CTK_AVAILABLE:
            # Inline (X)
            ctk.CTkLabel(row_slices, text="Inline (X):", font=ctk.CTkFont(size=11, weight="bold"), text_color="#38bdf8").grid(row=0, column=0, padx=4)
            self.slider_x = ctk.CTkSlider(row_slices, from_=0, to=100, number_of_steps=100, command=self._on_slice_change)
            self.slider_x.set(50)
            self.slider_x.grid(row=0, column=1, sticky="ew", padx=6)
            self.lbl_x_val = ctk.CTkLabel(row_slices, text="50", width=35, font=ctk.CTkFont(size=11))
            self.lbl_x_val.grid(row=0, column=2, padx=2)

            # Crossline (Y)
            ctk.CTkLabel(row_slices, text="Crossline (Y):", font=ctk.CTkFont(size=11, weight="bold"), text_color="#4ade80").grid(row=0, column=3, padx=4)
            self.slider_y = ctk.CTkSlider(row_slices, from_=0, to=100, number_of_steps=100, command=self._on_slice_change)
            self.slider_y.set(50)
            self.slider_y.grid(row=0, column=4, sticky="ew", padx=6)
            self.lbl_y_val = ctk.CTkLabel(row_slices, text="50", width=35, font=ctk.CTkFont(size=11))
            self.lbl_y_val.grid(row=0, column=5, padx=2)

            # Depth (Z)
            ctk.CTkLabel(row_slices, text="Depth (Z):", font=ctk.CTkFont(size=11, weight="bold"), text_color="#f472b6").grid(row=0, column=6, padx=4)
            self.slider_z = ctk.CTkSlider(row_slices, from_=0, to=100, number_of_steps=100, command=self._on_slice_change)
            self.slider_z.set(50)
            self.slider_z.grid(row=0, column=7, sticky="ew", padx=6)
            self.lbl_z_val = ctk.CTkLabel(row_slices, text="50", width=35, font=ctk.CTkFont(size=11))
            self.lbl_z_val.grid(row=0, column=8, padx=2)

            btn_center = ctk.CTkButton(row_slices, text="Centrar", command=self._center_slices, width=65, fg_color="#374151")
            btn_center.grid(row=0, column=9, padx=6)

            row_slices.grid_columnconfigure(1, weight=1)
            row_slices.grid_columnconfigure(4, weight=1)
            row_slices.grid_columnconfigure(7, weight=1)

            # Fila 2 de controles: Alpha y Umbral
            row_tuning = ctk.CTkFrame(sliders_box, fg_color="transparent")
            row_tuning.pack(fill="x", padx=10, pady=(2, 6))

            # Opacidad Alpha
            ctk.CTkLabel(row_tuning, text="Opacidad Overlay (Alpha):", font=ctk.CTkFont(size=11)).pack(side="left", padx=4)
            self.slider_alpha = ctk.CTkSlider(row_tuning, from_=0.0, to=1.0, number_of_steps=20, command=self._on_alpha_change, width=140)
            self.slider_alpha.set(0.5)
            self.slider_alpha.pack(side="left", padx=4)
            self.lbl_alpha_val = ctk.CTkLabel(row_tuning, text="0.50", width=35, font=ctk.CTkFont(size=11))
            self.lbl_alpha_val.pack(side="left", padx=2)

            # Umbral de Probabilidad
            ctk.CTkLabel(row_tuning, text="Umbral Decisión (Threshold):", font=ctk.CTkFont(size=11)).pack(side="left", padx=(16, 4))
            self.slider_thresh = ctk.CTkSlider(row_tuning, from_=0.05, to=0.95, number_of_steps=18, command=self._on_thresh_change, width=140)
            self.slider_thresh.set(0.50)
            self.slider_thresh.pack(side="left", padx=4)
            self.lbl_thresh_val = ctk.CTkLabel(row_tuning, text="0.50", width=35, font=ctk.CTkFont(size=11))
            self.lbl_thresh_val.pack(side="left", padx=2)

            # Métricas en vivo
            self.lbl_metrics_summary = ctk.CTkLabel(
                row_tuning, text="Métricas: --", font=ctk.CTkFont(size=11, weight="bold"), text_color="#38bdf8"
            )
            self.lbl_metrics_summary.pack(side="right", padx=10)

    # -----------------------------------------------------------------
    # GESTIÓN Y CARGA DE MODELO
    # -----------------------------------------------------------------
    def _select_model_file(self):
        filetypes = [("Model Checkpoints (*.pth;*.pt)", "*.pth;*.pt"), ("Todos los archivos", "*.*")]
        initial_dir = os.path.join(os.getcwd(), "checkpoints") if os.path.exists("checkpoints") else os.getcwd()
        path = filedialog.askopenfilename(title="Seleccionar archivo del modelo", initialdir=initial_dir, filetypes=filetypes)
        if not path:
            return

        self._load_checkpoint(path)

    def _load_checkpoint(self, path):
        try:
            self.model_path = path
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)

            # Detectar número de clases a partir del checkpoint si es posible
            state_dict = checkpoint.get("model_state_dict", checkpoint)
            if "final_conv.weight" in state_dict:
                detected_classes = state_dict["final_conv.weight"].shape[0]
                self.num_classes = detected_classes
                if detected_classes == 2:
                    self.combo_classes.set("2 (Fallas / Background)")
                elif detected_classes == 6:
                    self.combo_classes.set("6 (Facies Geológicas)")
                else:
                    self.combo_classes.set(f"{detected_classes} clases")

            self.model = UNet3D(in_channels=1, out_channels=self.num_classes).to(self.device)
            self.model.load_state_dict(state_dict)
            self.model.eval()

            fname = os.path.basename(path)
            num_params = sum(p.numel() for p in self.model.parameters()) / 1e6
            status_text = f"Cargado: {fname}\nUNet3D ({num_params:.1f}M params) | {self.num_classes} Clases"
            self.lbl_model_status.configure(text=status_text, text_color="#4ade80")
            messagebox.showinfo("Modelo Cargado", f"Modelo cargado correctamente desde:\n{fname}\nDispositivo: {self.device}")

        except Exception as e:
            messagebox.showerror("Error al cargar modelo", f"No se pudo cargar el checkpoint:\n{str(e)}")
            self.lbl_model_status.configure(text=f"Error: {str(e)[:30]}...", text_color="#f87171")

    def _init_demo_model(self):
        """Inicializa un 3D U-Net con pesos demo para pruebas sin necesidad de checkpoint externo."""
        try:
            self.model = UNet3D(in_channels=1, out_channels=self.num_classes).to(self.device)
            self.model.eval()
            self.model_path = "DEMO (Pesos Iniciales)"
            num_params = sum(p.numel() for p in self.model.parameters()) / 1e6
            status_text = f"Modelo Demo Activo\nUNet3D ({num_params:.1f}M params) | {self.num_classes} Clases"
            self.lbl_model_status.configure(text=status_text, text_color="#38bdf8")
            messagebox.showinfo("Modelo Demo Inicializado", "3D U-Net inicializado correctamente con pesos de demostración para pruebas rápidas.")
        except Exception as e:
            messagebox.showerror("Error", f"Error al inicializar modelo demo: {str(e)}")

    def _on_classes_change(self, choice):
        if "2" in choice:
            self.num_classes = 2
        elif "6" in choice:
            self.num_classes = 6

    # -----------------------------------------------------------------
    # GESTIÓN Y CARGA DE VOLUMEN SÍSMICO
    # -----------------------------------------------------------------
    def _select_volume_file(self):
        filetypes = [
            ("Archivos sísmicos 3D (*.npz;*.dat)", "*.npz;*.dat"),
            ("Archivos NumPy (*.npz)", "*.npz"),
            ("Archivos Binarios Raw (*.dat)", "*.dat"),
            ("Todos los archivos", "*.*")
        ]
        path = filedialog.askopenfilename(title="Seleccionar volumen sísmico para predicción", filetypes=filetypes)
        if not path:
            return

        self._load_volume(path)

    def _load_volume(self, path):
        try:
            filename = os.path.basename(path)
            if path.endswith(".npz"):
                data = np.load(path)
                if "seis" in data:
                    self.seis_volume = data["seis"].astype(np.float32)
                elif "data" in data:
                    self.seis_volume = data["data"].astype(np.float32)
                else:
                    first_key = list(data.keys())[0]
                    self.seis_volume = data[first_key].astype(np.float32)

                # Buscar ground truth si existe
                self.gt_volume = None
                for key in ["fault", "facies", "labels", "gt", "mask"]:
                    if key in data:
                        self.gt_volume = data[key].astype(np.int64)
                        break

            elif path.endswith(".dat"):
                # Carga binaria: por defecto volumen cúbico 128x128x128
                raw = np.fromfile(path, dtype=np.float32)
                expected_shape = (128, 128, 128)
                if raw.size == 128 * 128 * 128:
                    self.seis_volume = raw.reshape(expected_shape)
                else:
                    # Intento de inferir cubo perfecto
                    side = int(round(raw.size ** (1 / 3.0)))
                    if side ** 3 == raw.size:
                        self.seis_volume = raw.reshape((side, side, side))
                    else:
                        raise ValueError(f"No se pudo determinar la forma del archivo .dat ({raw.size} valores).")
                self.gt_volume = None

            self.volume_path = path
            d, h, w = self.seis_volume.shape

            # Actualizar sliders
            self._update_slider_ranges(d, h, w)
            self._center_slices()

            # Limpiar predicción anterior
            self.pred_volume = None
            self.probs_volume = None

            # Actualizar tarjeta de información
            self.lbl_vol_status.configure(text=f"Volumen: {filename}", text_color="#4ade80")
            self.lbl_vol_shape.configure(text=f"Dimensiones: D={d}, H={h}, W={w} voxels")
            s_min, s_max = np.min(self.seis_volume), np.max(self.seis_volume)
            self.lbl_vol_range.configure(text=f"Amplitud Sísmica: [{s_min:.2f}, {s_max:.2f}]")

            gt_info = f"Ground Truth: DETECTADO ({'Fallas' if np.max(self.gt_volume) <= 1 else 'Facies'})" if self.gt_volume is not None else "Ground Truth: No presente"
            gt_color = "#38bdf8" if self.gt_volume is not None else "#9ca3af"
            self.lbl_vol_gt.configure(text=gt_info, text_color=gt_color)

            # Renderizar vista previa solo de la sísmica
            self._render_visualization()
            messagebox.showinfo("Volumen Cargado", f"Volumen cargado exitosamente:\n{filename}\nDimensiones: {d} x {h} x {w}")

        except Exception as e:
            messagebox.showerror("Error al cargar volumen", f"No se pudo procesar el archivo:\n{str(e)}")

    def _generate_synthetic_volume(self):
        """Genera un cubo sísmico 3D sintético fotorrealista con reflectores plegados y fallas."""
        try:
            shape = (128, 128, 128)
            d, h, w = shape

            z = np.linspace(0, 8 * np.pi, d)[:, None, None]
            y = np.linspace(0, 3 * np.pi, h)[None, :, None]
            x = np.linspace(0, 3 * np.pi, w)[None, None, :]

            # Estructura estratigráfica con leve buzamiento
            dip = 0.05 * (x + y)
            fault_plane = (x[:, :, :] + 0.4 * y[:, :, :] - 0.6 * z[:, :, :]) > (w * 0.35)

            # Desplazamiento de las capas a través de la falla
            seis = np.sin(z + dip + fault_plane.astype(float) * 0.9)
            noise = np.random.normal(0, 0.12, shape)
            seis = (seis + noise).astype(np.float32)

            # Ground truth
            from scipy.ndimage import binary_dilation, binary_erosion
            fault_mask = binary_dilation(fault_plane, iterations=2) ^ binary_erosion(fault_plane, iterations=1)
            gt = fault_mask.astype(np.int64)

            self.seis_volume = seis
            self.gt_volume = gt
            self.volume_path = "SINTÉTICO (3D Con Fallas)"
            self.pred_volume = None
            self.probs_volume = None

            self._update_slider_ranges(d, h, w)
            self._center_slices()

            self.lbl_vol_status.configure(text="Volumen: Sintético 3D (Demo)", text_color="#38bdf8")
            self.lbl_vol_shape.configure(text=f"Dimensiones: D={d}, H={h}, W={w} voxels")
            self.lbl_vol_range.configure(text=f"Amplitud Sísmica: [{np.min(seis):.2f}, {np.max(seis):.2f}]")
            self.lbl_vol_gt.configure(text="Ground Truth: DETECTADO (Fallas Sintéticas)", text_color="#38bdf8")

            self._render_visualization()
            messagebox.showinfo("Volumen Sintético Creado", "Cubo sísmico 3D sintético de 128x128x128 generado exitosamente.")
        except Exception as e:
            messagebox.showerror("Error", f"Error generando volumen sintético: {str(e)}")

    def _update_slider_ranges(self, d, h, w):
        self.slider_z.configure(from_=0, to=d - 1, number_of_steps=d - 1)
        self.slider_y.configure(from_=0, to=h - 1, number_of_steps=h - 1)
        self.slider_x.configure(from_=0, to=w - 1, number_of_steps=w - 1)

    def _center_slices(self):
        if self.seis_volume is not None:
            d, h, w = self.seis_volume.shape
            self.slice_z = d // 2
            self.slice_y = h // 2
            self.slice_x = w // 2
        else:
            self.slice_z = 50
            self.slice_y = 50
            self.slice_x = 50

        self.slider_z.set(self.slice_z)
        self.slider_y.set(self.slice_y)
        self.slider_x.set(self.slice_x)
        self.lbl_z_val.configure(text=str(self.slice_z))
        self.lbl_y_val.configure(text=str(self.slice_y))
        self.lbl_x_val.configure(text=str(self.slice_x))
        self._render_visualization()

    # -----------------------------------------------------------------
    # INFERENCIA DE PREDICCIÓN (SLIDING WINDOW EN SEGUNDO PLANO)
    # -----------------------------------------------------------------
    def _start_inference(self):
        if self.model is None:
            messagebox.showwarning("Atención", "Por favor seleccione un modelo (.pth) o active el modelo Demo primero.")
            return

        if self.seis_volume is None:
            messagebox.showwarning("Atención", "Por favor seleccione un volumen de entrada (.npz/.dat) o genere uno sintético.")
            return

        if self.is_inferring:
            return

        self.is_inferring = True
        self.cancel_event.clear()
        self.btn_run_pred.configure(state="disabled")
        self.btn_cancel_pred.configure(state="normal")
        self.progress_bar.set(0)
        self.lbl_infer_status.configure(text="Iniciando inferencia...", text_color="#38bdf8")

        patch_size = int(self.combo_patch.get())
        stride = int(self.combo_stride.get())

        # Iniciar hilo de inferencia
        thread = threading.Thread(
            target=self._run_inference_worker,
            args=(patch_size, stride),
            daemon=True
        )
        thread.start()

    def _cancel_inference(self):
        if self.is_inferring:
            self.cancel_event.set()
            self.lbl_infer_status.configure(text="Cancelando inferencia...", text_color="#f87171")

    def _run_inference_worker(self, patch_size, stride):
        try:
            t0 = time.time()

            def progress_cb(step, total, msg):
                frac = step / max(1, total)
                pct = int(frac * 100)
                # Actualizar GUI en el hilo principal
                self.root.after(0, lambda: self._update_progress_ui(frac, f"{msg} ({pct}%)"))

            pred, probs = inference_sliding_window(
                model=self.model,
                volume=self.seis_volume,
                patch_size=(patch_size, patch_size, patch_size),
                stride=(stride, stride, stride),
                num_classes=self.num_classes,
                device=self.device,
                return_probs=True,
                progress_callback=progress_cb,
                cancel_event=self.cancel_event
            )

            if self.cancel_event.is_set() or pred is None:
                self.root.after(0, lambda: self._on_inference_cancelled())
                return

            elapsed = time.time() - t0
            self.pred_volume = pred
            self.probs_volume = probs

            self.root.after(0, lambda: self._on_inference_completed(elapsed))

        except Exception as e:
            self.root.after(0, lambda: self._on_inference_error(str(e)))

    def _update_progress_ui(self, fraction, msg):
        self.progress_bar.set(fraction)
        self.lbl_infer_status.configure(text=msg)

    def _on_inference_cancelled(self):
        self.is_inferring = False
        self.btn_run_pred.configure(state="normal")
        self.btn_cancel_pred.configure(state="disabled")
        self.lbl_infer_status.configure(text="Inferencia cancelada por el usuario", text_color="#f87171")

    def _on_inference_completed(self, elapsed_sec):
        self.is_inferring = False
        self.btn_run_pred.configure(state="normal")
        self.btn_cancel_pred.configure(state="disabled")
        self.progress_bar.set(1.0)
        self.lbl_infer_status.configure(
            text=f"Predicción finalizada en {elapsed_sec:.1f}s", text_color="#4ade80"
        )

        # Calcular métricas si hay GT disponible
        self._calculate_metrics()

        # Renderizar la visualización completa de la predicción
        self._render_visualization()
        messagebox.showinfo("Predicción Completa", f"¡La predicción ha finalizado con éxito en {elapsed_sec:.1f} segundos!\nRevise la visualización 3D interactiva.")

    def _on_inference_error(self, err_msg):
        self.is_inferring = False
        self.btn_run_pred.configure(state="normal")
        self.btn_cancel_pred.configure(state="disabled")
        self.lbl_infer_status.configure(text=f"Error en inferencia: {err_msg[:30]}...", text_color="#f87171")
        messagebox.showerror("Error en inferencia", f"Ocurrió un error durante la inferencia:\n{err_msg}")

    # -----------------------------------------------------------------
    # MÉTRICAS Y EVALUACIÓN
    # -----------------------------------------------------------------
    def _calculate_metrics(self):
        if self.gt_volume is None or self.pred_volume is None:
            self.lbl_metrics_summary.configure(text="Métricas: No hay GT")
            return

        try:
            y_true = (self.gt_volume > 0).astype(np.uint8)
            y_pred = (self.pred_volume > 0).astype(np.uint8)

            intersection = np.sum(y_true * y_pred)
            total_pred = np.sum(y_pred)
            total_true = np.sum(y_true)

            precision = intersection / (total_pred + 1e-8)
            recall = intersection / (total_true + 1e-8)
            dice_f1 = (2.0 * intersection) / (total_pred + total_true + 1e-8)
            iou = intersection / (total_pred + total_true - intersection + 1e-8)

            metrics_txt = f"Dice (F1): {dice_f1:.3f} | IoU: {iou:.3f} | Prec: {precision:.3f} | Rec: {recall:.3f}"
            self.lbl_metrics_summary.configure(text=metrics_txt, text_color="#38bdf8")
        except Exception:
            self.lbl_metrics_summary.configure(text="Métricas: Error")

    # -----------------------------------------------------------------
    # CONTROL DE EVENTOS DE VISUALIZACIÓN
    # -----------------------------------------------------------------
    def _on_slice_change(self, value=None):
        self.slice_x = int(round(self.slider_x.get()))
        self.slice_y = int(round(self.slider_y.get()))
        self.slice_z = int(round(self.slider_z.get()))

        self.lbl_x_val.configure(text=str(self.slice_x))
        self.lbl_y_val.configure(text=str(self.slice_y))
        self.lbl_z_val.configure(text=str(self.slice_z))

        self._render_visualization()

    def _on_mode_change(self, choice):
        if "Comparativa" in choice:
            self.view_mode = "comparative"
        elif "Superposición" in choice:
            self.view_mode = "overlay"
        elif "Probabilidad" in choice:
            self.view_mode = "probability"
        self._render_visualization()

    def _on_cmap_change(self, choice):
        self.colormap_name = choice
        self._render_visualization()

    def _on_alpha_change(self, val):
        self.overlay_alpha = float(val)
        self.lbl_alpha_val.configure(text=f"{self.overlay_alpha:.2f}")
        if self.view_mode == "overlay":
            self._render_visualization()

    def _on_thresh_change(self, val):
        self.threshold = float(val)
        self.lbl_thresh_val.configure(text=f"{self.threshold:.2f}")

        # Si tenemos mapa de probabilidades continuo, recalcular máscara binaria instantáneamente
        if self.probs_volume is not None:
            # Clase 1 (Falla o positiva)
            target_class = 1 if self.probs_volume.shape[0] > 1 else 0
            self.pred_volume = (self.probs_volume[target_class] >= self.threshold).astype(np.int64)
            self._calculate_metrics()
            self._render_visualization()

    # -----------------------------------------------------------------
    # MOTOR DE RENDERIZADO VISUAL
    # -----------------------------------------------------------------
    def _render_visualization(self):
        if self.seis_volume is None:
            return

        slice_coords = (self.slice_z, self.slice_y, self.slice_x)

        if self.view_mode == "comparative":
            plot_seismic_slices(
                seis=self.seis_volume,
                facies=self.gt_volume,
                pred=self.pred_volume,
                slice_idx=slice_coords,
                num_classes=self.num_classes,
                fig=self.fig,
                cmap_name=self.colormap_name
            )

        elif self.view_mode == "overlay":
            plot_overlay_slices(
                seis=self.seis_volume,
                pred=self.pred_volume,
                facies=self.gt_volume,
                slice_idx=slice_coords,
                alpha=self.overlay_alpha,
                num_classes=self.num_classes,
                fig=self.fig,
                cmap_name=self.colormap_name
            )

        elif self.view_mode == "probability":
            if self.probs_volume is not None:
                target_class = 1 if self.probs_volume.shape[0] > 1 else 0
                prob_map = self.probs_volume[target_class]
            else:
                # Si aún no hay probabilidades calculadas, mostrar ceros
                prob_map = np.zeros_like(self.seis_volume)

            plot_probability_slices(
                seis=self.seis_volume,
                prob_map=prob_map,
                mask=self.pred_volume,
                slice_idx=slice_coords,
                fig=self.fig,
                cmap_prob=self.colormap_name,
                class_name="Falla" if self.num_classes == 2 else "Clase 1"
            )

        # Ajuste de fondo para coincidir con el tema oscuro
        self.fig.patch.set_facecolor("#111827")
        self.canvas.draw_idle()

    # -----------------------------------------------------------------
    # EXPORTACIÓN DE RESULTADOS
    # -----------------------------------------------------------------
    def _save_visualization(self):
        if self.seis_volume is None:
            messagebox.showwarning("Atención", "No hay visualización para guardar.")
            return

        filetypes = [
            ("Imagen PNG (*.png)", "*.png"),
            ("Imagen JPEG (*.jpg)", "*.jpg"),
            ("Documento PDF (*.pdf)", "*.pdf"),
            ("Vectorial SVG (*.svg)", "*.svg")
        ]
        out_path = filedialog.asksaveasfilename(
            title="Guardar visualización actual",
            defaultextension=".png",
            filetypes=filetypes
        )
        if out_path:
            try:
                self.fig.savefig(out_path, dpi=200, bbox_inches="tight")
                messagebox.showinfo("Imagen Guardada", f"Visualización guardada con éxito en:\n{out_path}")
            except Exception as e:
                messagebox.showerror("Error al guardar", f"No se pudo guardar la imagen:\n{str(e)}")

    def _export_prediction_npz(self):
        if self.pred_volume is None:
            messagebox.showwarning("Atención", "No se ha generado ninguna predicción aún para exportar.")
            return

        filetypes = [("NumPy Archive (*.npz)", "*.npz")]
        out_path = filedialog.asksaveasfilename(
            title="Exportar volumen de predicción",
            defaultextension=".npz",
            filetypes=filetypes
        )
        if out_path:
            try:
                save_dict = {
                    "pred": self.pred_volume.astype(np.int32),
                    "seis": self.seis_volume
                }
                if self.probs_volume is not None:
                    save_dict["probs"] = self.probs_volume.astype(np.float32)
                if self.gt_volume is not None:
                    save_dict["fault"] = self.gt_volume

                np.savez_compressed(out_path, **save_dict)
                messagebox.showinfo("Exportación Exitosa", f"Volumen de predicción 3D exportado a:\n{out_path}")
            except Exception as e:
                messagebox.showerror("Error al exportar", f"No se pudo exportar el archivo .npz:\n{str(e)}")


def main():
    if CTK_AVAILABLE:
        root = ctk.CTk()
    else:
        root = tk.Tk()

    app = SeismicGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
