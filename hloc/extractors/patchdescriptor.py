import cv2
import os
import math
import torch
import kornia
from torch import nn
import torch.nn.functional as F
from ..utils.base_model import BaseModel
import numpy as np
import sys
from pathlib import Path
import matplotlib.pyplot as plt

# Descriptors:
# Input: (B, 1, patch_size, patch_size)
# Output: (B, dim)
DESCRIPTORS = {
    "sift": {
        "patch_size": 16,
        "dim": 128,
        "net": kornia.feature.SIFTDescriptor(patch_size=16),
    },
    # Input: (B, 1, 32, 32)
    # Output: (B, num_ang_bins * num_spatial_bins ** 2)
    "hardnet": {
        "patch_size": 32,
        "dim": 128,
        "net": kornia.feature.HardNet(pretrained=True),
    }
}

class PatchDescriptor(BaseModel):
    default_conf = {
    }
    def _init(self, config):
        self.config = {**config, **DESCRIPTORS[config["patch_descriptor_name"]]}
        self.descriptor = self.config["net"]
        self.patch_size = self.config["patch_size"]

    def _describe(self, data, output):
        # torch where over batch
        B = len(output['keypoints'])
        imgs = data["image"]
        B, C, H, W = imgs.shape

        descs = []
        # Accumulate patches associated with each keypoint
        # Right now, we assume that all keypoints have the same scale
        # all of this is on the gpu right now
        rad = self.patch_size // 2
        s = self.patch_size
        # (s, s)
        x, y = torch.meshgrid(torch.linspace(-rad, s-rad, s),
                              torch.linspace(-rad, s-rad, s))
        x = x.to(imgs.device)
        y = y.to(imgs.device)
        for i in range(B):
            # Points are assumed to be in (col, row) format
            # It's the same x, y form as opencv KeyPoint
            pts = output['keypoints'][i] # (N, 2)
            img = imgs[i].unsqueeze(0)
            # We want to broadcast add the point position along the first dimension
            # New size: (N, 1, s*s, 1)
            nx = (y.view(1, 1, -1, 1) + pts[:, 0].view(1, -1, 1, 1))/W*2 - 1
            ny = (x.view(1, 1, -1, 1) + pts[:, 1].view(1, -1, 1, 1))/H*2 - 1
            points = torch.cat((nx, ny), dim=-1)
            patches = F.grid_sample(img, points, mode="bilinear", align_corners=True)
            patches = patches.view(-1, C, s, s)

            img = (img.view(H, W, 1).cpu().numpy()*255).astype(np.uint8)
            kpts = []
            for pt in pts:
                kpts.append(cv2.KeyPoint(float(pt[0].cpu()-0.5), float(pt[1].cpu()-0.5), _size=int(float(pt[1])/50), _angle=0))

            drawn = img.copy()
            print(img.dtype, img.shape)
            drawn = cv2.drawKeypoints(img, kpts, drawn, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            plt.imshow(drawn)
            plt.show()

            # Looks good
            #  plt.figure()
            #  plt.imshow(imgs[i].squeeze().cpu())
            #  plt.figure()
            #  plt.imshow(patches[0].squeeze().cpu())
            #  print(pts[0])
            #  plt.show()

            # (N, dim)
            desc = self.descriptor(patches)
            descs.append(desc.T)

        output['descriptors'] = descs
        return output
