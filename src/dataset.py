import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import random

class SeismicDataset(Dataset):
    """
    3D Seismic Facies Dataset loader.
    Handles storage of volumetric data and extraction of sub-volumes (patches).
    Supports .npz (metadata included) and .dat (raw binary) formats.
    """
    def __init__(self, data_path, patch_size=(128, 128, 128), stride=(64, 64, 64), 
                 is_training=True, transform=True, dummy=False, 
                 vol_shape=None, dtype=np.float32):
        self.data_path = data_path
        self.patch_size = patch_size
        self.stride = stride
        self.is_training = is_training
        self.transform = transform
        self.dummy = dummy
        self.vol_shape = vol_shape # Needed for .dat files
        self.dtype = dtype
        
        # Load filenames if not in dummy mode
        if not dummy:
            # Look for both .npz and .dat files
            self.filenames = [f for f in os.listdir(data_path) if f.endswith(('.npz', '.dat'))]
            if not self.filenames:
                raise FileNotFoundError(f"No .npz or .dat files found in {data_path}")
            self.patches = self._generate_patch_list()
        else:
            self.filenames = ["dummy"]
            self.patches = [(0, (0, 0, 0))] # One dummy patch index

    def _generate_patch_list(self):
        """
        Pre-calculates all valid patch starting coordinates within each seismic volume.
        """
        patch_list = []
        for idx, filename in enumerate(self.filenames):
            if filename.endswith('.npz'):
                data = np.load(os.path.join(self.data_path, filename))
                shape = data['seis'].shape
            else: # .dat file
                if self.vol_shape is None:
                    raise ValueError(f"vol_shape must be provided for binary files (.dat): {filename}")
                shape = self.vol_shape
            
            # Sub-divide the volume based on stride
            for i in range(0, shape[0] - self.patch_size[0] + 1, self.stride[0]):
                for j in range(0, shape[1] - self.patch_size[1] + 1, self.stride[1]):
                    for k in range(0, shape[2] - self.patch_size[2] + 1, self.stride[2]):
                        patch_list.append((idx, (i, j, k)))
        return patch_list

    def __len__(self):
        if self.dummy:
            return 100 # Artificial length for testing logic
        return len(self.patches)

    def _normalize(self, x):
        """Z-score normalization"""
        mean = np.mean(x)
        std = np.std(x)
        return (x - mean) / (std + 1e-8)

    def _augment(self, seis, facies):
        """3D augmentation: 180 and 90 degree rotations and flips"""
        # Flip along different axes
        flip_axis = random.randint(0, 3)
        if flip_axis > 0:
            seis = np.flip(seis, axis=flip_axis-1)
            if facies is not None:
                facies = np.flip(facies, axis=flip_axis-1)
            
        # Optional: 90 degree rotation in random plane
        rot_axes = random.choice([(0, 1), (1, 2), (0, 2)])
        rot_times = random.randint(0, 3)
        seis = np.rot90(seis, k=rot_times, axes=rot_axes)
        if facies is not None:
            facies = np.rot90(facies, k=rot_times, axes=rot_axes)
        
        return seis.copy(), (facies.copy() if facies is not None else None)

    def __getitem__(self, idx):
        if self.dummy:
            seis = np.random.randn(*self.patch_size).astype(np.float32)
            fault = np.random.randint(0, 6, size=self.patch_size).astype(np.int64)
        else:
            vol_idx, (sx, sy, sz) = self.patches[idx]
            filename = self.filenames[vol_idx]
            filepath = os.path.join(self.data_path, filename)

            if filename.endswith('.npz'):
                data = np.load(filepath)
                full_seis = data['seis']
                full_fault = data['fault'] if 'fault' in data else None
            else: # .dat file
                # For raw binary, we load the whole volume then slice
                # Performance optimization: if files are very large, consider memmap or random access
                full_seis = np.fromfile(filepath, dtype=self.dtype).reshape(self.vol_shape)
                full_fault = None # Prediction .dat files typically don't have GT

            seis = full_seis[sx:sx+self.patch_size[0],
                             sy:sy+self.patch_size[1],
                             sz:sz+self.patch_size[2]]

            if full_fault is not None:
                fault = full_fault[sx:sx+self.patch_size[0],
                                     sy:sy+self.patch_size[1],
                                     sz:sz+self.patch_size[2]]
            else:
                fault = None
        
        # Preprocessing
        seis = self._normalize(seis)
        
        if self.is_training and self.transform:
            seis, fault = self._augment(seis, fault)
            
        # To Tensor
        seis_tensor = torch.from_numpy(seis).float().unsqueeze(0)
        
        if fault is not None:
            fault_tensor = torch.from_numpy(fault).long() 
        else:
            # For prediction where no GT exists, return empty tensor or dummy
            fault_tensor = torch.zeros(self.patch_size).long()
        
        return seis_tensor, fault_tensor

def get_dataloader(data_path, batch_size=2, patch_size=(128, 128, 128), 
                   stride=(64, 64, 64), is_training=True, dummy=False, vol_shape=None):
    dataset = SeismicDataset(data_path, patch_size=patch_size, stride=stride, 
                              is_training=is_training, dummy=dummy, vol_shape=vol_shape)
    
    # Optimize worker count based on system capabilities
    if dummy:
        num_workers = 0
    else:
        # Avoid the "max number of worker" warning by capping at cpu_count
        cpu_count = os.cpu_count() or 1
        num_workers = min(2, cpu_count) # Conservative default to avoid slow-downs
        
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=is_training,
                        num_workers=num_workers, pin_memory=torch.cuda.is_available())
    return loader

if __name__ == "__main__":
    # Test dummy loader
    example_loader = get_dataloader(None, batch_size=1, dummy=True)
    seis, facies = next(iter(example_loader))
    print(f"Loaded seismic batch shape: {seis.shape}")
    print(f"Loaded facies batch shape: {facies.shape}")
