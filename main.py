import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from datasets.bdd100k.bdd100k_sem_seg import BDD100k_sem_seg

@hydra.main(version_base = None, config_path="config", config_name="config")
def main(cfg : DictConfig):
    breakpoint()
    bdd = BDD100k_sem_seg(cfg.dataset)
    pass

if __name__ == "__main__":
    main()
