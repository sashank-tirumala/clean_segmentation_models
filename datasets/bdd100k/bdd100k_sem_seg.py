import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision
from pathlib import Path
from PIL import Image
import torchvision.transforms as ttf


class BDD100k_sem_seg(Dataset):
    def __init__(self, cfg):
        self.root_dir = Path(cfg.root_dir)
        self.split = Path(cfg.split)
        self.version = cfg.version
        self.input_dir = self.root_dir / self.split
        self.files = [x.name.strip(".jpg") for x in self.input_dir.glob('*.jpg')] 
        self.gt_dir = Path(cfg.gt_dir) / self.split

    def __len__(self):
        return len(self.input_files)
    
    def __getitem__(self, idx):
        if self.version == 0:
            cur_file = self.files[idx]
            x = Image.open(self.input_dir / (cur_file+".jpg"))
            y = Image.open(self.gt_dir / (cur_file +".png"))
            transforms = ttf.Compose([ttf.ToTensor()])
            x = transforms(x)
            y = torch.Tensor(np.array(y))
            return x,y
        else:
            raise NotImplementedError 


