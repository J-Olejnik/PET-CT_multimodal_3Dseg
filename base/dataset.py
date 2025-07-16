import os
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset

class NiftiDataset(Dataset):
    def __init__(self, paths, hu, include_pet = False, concat = False):

        self.paths = paths
        self.hu = hu
        self.include_pet = include_pet
        self.concat = concat
    
    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):

        ct_path = os.path.join(self.paths[idx], "CT.nii.gz")
        label_path = os.path.join(self.paths[idx], "ITV.nii.gz")

        ct = np.copy(nib.load(ct_path).get_fdata())
        label = np.copy(nib.load(label_path).get_fdata())

        # Reorder to (D, H, W) then add Channel dimension (C, D, H, W)
        ct = torch.tensor(ct.transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0)  
        label = torch.tensor(label.transpose(2, 0, 1), dtype=torch.long) 
        
        ct = self.hu_window(ct)
        ct = self.z_normalize(ct)
            
        if self.include_pet:
            pt_path = os.path.join(self.paths[idx], "PT_NoHeart.nii.gz")
            pt = np.copy(nib.load(pt_path).get_fdata())
            pt = torch.tensor(pt.transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0)
            pt = self.z_normalize(pt)

            if self.concat:
                sample = (torch.cat([ct,pt], dim=0), label)
            else:
                sample = (ct, pt, label)
        else: 
            sample = (ct, label)
        
        return sample
    
    def hu_window(self, ct):
        level, window = self.hu
        lower = level - window/2
        upper = level + window/2
        return torch.clamp(ct, min=lower, max=upper)

    @staticmethod
    def z_normalize(image):
        mean = image.mean()
        std = image.std()
        return (image - mean) / (std + 1e-8)