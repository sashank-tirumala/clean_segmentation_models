import torch 
from datasets.ade20k_dataset import ADE20K
from pathlib import Path
def test_ADE20K():
    folder_dir = Path("/home/sashank/fiftyone/ADE20K_2021_17_01")
    pkl_file_path = Path("index_ade20k.pkl") 
    ds = ADE20K(folder_dir, folder_dir / pkl_file_path) 
    pass

if __name__ == "__main__":
    test_ADE20K() 
    pass
