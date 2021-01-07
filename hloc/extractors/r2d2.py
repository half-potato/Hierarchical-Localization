import os
import math
import torch
from torch import nn
import torch.nn.functional as F

from ..utils.base_model import BaseModel
import numpy as np

import sys
from pathlib import Path
third_party_folder = Path(__file__).parent / '../../third_party'
sys.path.append(str(third_party_folder / 'r2d2'))
import extract as r2d2
from tools.dataloader import RGB_mean, RGB_std
import torchvision.transforms as tvf

norm_RGB = tvf.Normalize(mean=RGB_mean, std=RGB_std)
# (N, C, H, W)

class R2D2(BaseModel):
    default_config = {
        "maxpool_radius": 3,
        "remove_borders": 4,
        "max_keypoints": 4096, # most closely reflects the original
        "scale_f": 2**0.25,
        "min_scale": 0,
        "max_scale": 1,
        "min_size": 256,
        "max_size": 1024,
        "reliability_thr": 0.7,
        "repeatability_thr": 0.7,
    }
    def _init(self, config):
        self.config = {**self.default_config, **config}
        self.net = r2d2.load_network(str(third_party_folder / 'r2d2/models/r2d2_WAF_N16.pt'))
        self.detector = r2d2.NonMaxSuppression(
            rel_thr = self.config["reliability_thr"],
            rep_thr = self.config["repeatability_thr"])
        self.RGB_mean_t = torch.Tensor(RGB_mean).reshape(1, -1, 1, 1)
        self.RGB_std_t = torch.Tensor(RGB_std).reshape(1, -1, 1, 1)

    def _forward(self, data):
        x = data["image"]
        img = (x - self.RGB_mean_t)/self.RGB_std_t
        # XYS: (N, 3)
        # desc: (N, 128)
        # scores: (N)
        XYS, desc, scores = r2d2.extract_multiscale(self.net, img, self.detector,
            scale_f   = self.config["scale_f"],
            min_scale = self.config["min_scale"],
            max_scale = self.config["max_scale"],
            min_size  = self.config["min_size"],
            max_size  = self.config["max_size"])
        if "threshold" in self.config:
            # Get points above a threshold, unless there are too many points
            N = len(scores > self.config["threshold"])
            if N > self.config["max_keypoints"]:
                idxs = scores.argsort()[-self.config["max_keypoints"] or None:]
            else:
                idxs = scores.argsort()[-N or None:]
        else:
            # Always get max number of points, like in the original code
            idxs = scores.argsort()[-self.config["max_keypoints"] or None:]

        return {
            'keypoints': XYS[idxs, :2],
            'scores': scores[idxs],
            'descriptors': desc[idxs, :],
        }

    def to(self, device):
        self.detector = self.detector.to(device)
        self.net = self.net.to(device)
        self.RGB_mean_t = self.RGB_mean_t.to(device)
        self.RGB_std_t = self.RGB_std_t.to(device)
        return self
