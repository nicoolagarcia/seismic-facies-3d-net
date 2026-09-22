import numpy as np
import os

data_path = "c:/Users/Nicola/Desktop/tesis/validation/0.npz"
if os.path.exists(data_path):
    data = np.load(data_path)
    seis = data['seis']
    facies = data['facies'] if 'facies' in data else None
    
    print(f"Volume Shape: {seis.shape}")
    print(f"Seismic Range: {seis.min():.4f} to {seis.max():.4f}")
    
    if facies is not None:
        unique, counts = np.unique(facies, return_counts=True)
        counts_dict = {int(u): int(c) for u, c in zip(unique, counts)}
        print(f"Facies Distribution (Ground Truth): {counts_dict}")
    else:
        print("No facies found in .npz file")
else:
    print(f"File {data_path} not found")
