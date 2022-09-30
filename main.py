import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from datasets.bdd100k.bdd100k_sem_seg import BDD100k_sem_seg
from models.UNet.model import UNet

@hydra.main(version_base = None, config_path="config", config_name="config")
def main(cfg : DictConfig):
    breakpoint()
    bdd = BDD100k_sem_seg(cfg.dataset)
    net = UNet(cfg)
    inp = torch.rand(6,3,512,512)
    outp = net(inp)
    pass

if __name__ == "__main__":
    main()
