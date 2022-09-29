import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import torch
import numpy as np
from datasets.bdd100k_sem_seg import BDD100k_sem_seg

def test_bdd100k_dataset():
    dataset = BDD100k_sem_seg()
    breakpoint()
    pass

if __name__ == "__main__":
    test_bdd100k_dataset()
    pass
