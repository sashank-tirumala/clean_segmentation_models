import torch 
from torch.utils.data import Dataset
import pickle
class ADE20K(Dataset):
    def __init__(self, dataset_dir, pkl_file_path):
        self.ADE20K_dir = dataset_dir
        self.pickle_file = pkl_file_path
        with open(self.pickle_file, 'rb') as f:
            index_ade20k = pickle.load(f)
        breakpoint()
        pass
