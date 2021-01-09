import os
import math
import torch
from torch import nn
import torch.nn.functional as F

from ..utils.base_model import BaseModel, dynamic_load
from .. import extractors as base

import numpy as np

import sys
from pathlib import Path
third_party_folder = Path(__file__).parent / '../../third_party'
sys.path.append(str(third_party_folder / 'r2d2'))
import extract as r2d2
from tools.dataloader import RGB_mean, RGB_std
import torchvision.transforms as tvf

# (N, C, H, W)

class R2D2(BaseModel):
    default_conf = {
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
        self.config = config
        self.net = r2d2.load_network(str(third_party_folder / 'r2d2/models/r2d2_WAF_N16.pt'))
        self.selector = r2d2.NonMaxSuppression(
            rel_thr = self.config["reliability_thr"],
            rep_thr = self.config["repeatability_thr"])
        self.RGB_mean_t = torch.Tensor(RGB_mean).reshape(1, -1, 1, 1)
        self.RGB_std_t = torch.Tensor(RGB_std).reshape(1, -1, 1, 1)
        self.detector = None
        if "detector_name" in self.config:
            Detector = dynamic_load(base, self.config["detector_name"])
            self.detector = Detector(self.config)


    def _forward(self, data):
        x = data["image"]
        img = (x - self.RGB_mean_t)/self.RGB_std_t
        # XYS: (N, 3)
        # desc: (N, 128)
        # scores: (N)
        XYS, desc, scores = extract_multiscale(self.net, self.detector, img, self.selector,
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
            'keypoints': [XYS[idxs, :2]],
            'scores': [scores[idxs]],
            'descriptors': [desc[idxs, :].T],
        }

    def to(self, device):
        self.selector = self.selector.to(device)
        self.net = self.net.to(device)
        self.RGB_mean_t = self.RGB_mean_t.to(device)
        self.RGB_std_t = self.RGB_std_t.to(device)
        if self.detector is not None:
            self.detector = self.detector.to(device)
        return self

def extract_multiscale( net, detector, img, selector, scale_f=2**0.25,
                        min_scale=0.0, max_scale=1,
                        min_size=256, max_size=1024,
                        verbose=False):
    old_bm = torch.backends.cudnn.benchmark
    torch.backends.cudnn.benchmark = False # speedup

    # extract keypoints at multiple scales
    B, three, H, W = img.shape
    assert B == 1 and three == 3, "should be a batch with a single RGB image"

    assert max_scale <= 1
    s = 1.0 # current scale factor

    X,Y,S,C,Q,D = [],[],[],[],[],[]
    while  s+0.001 >= max(min_scale, min_size / max(H,W)):
        if s-0.001 <= min(max_scale, max_size / max(H,W)):
            nh, nw = img.shape[2:]
            if verbose: print(f"extracting at scale x{s:.02f} = {nw:4d}x{nh:3d}")
            # extract descriptors
            with torch.no_grad():
                res = net(imgs=[img])
                # get output and reliability map
                descriptors = res['descriptors'][0]
                reliability = res['reliability'][0]
                if detector is not None:
                    output = detector({"image": img})
                    repeatability = output["heatmap"]
                else:
                    repeatability = res['repeatability'][0]

            # normalize the reliability for nms
            # extract maxima and descs
            y,x = selector(**res) # nms
            c = reliability[0,0,y,x]
            q = repeatability[0,0,y,x]
            d = descriptors[0,:,y,x].t()
            n = d.shape[0]

            # accumulate multiple scales
            X.append(x.float() * W/nw)
            Y.append(y.float() * H/nh)
            S.append((32/s) * torch.ones(n, dtype=torch.float32, device=d.device))
            C.append(c)
            Q.append(q)
            D.append(d)
        s /= scale_f

        # down-scale the image for next iteration
        nh, nw = round(H*s), round(W*s)
        img = F.interpolate(img, (nh,nw), mode='bilinear', align_corners=False)

    # restore value
    torch.backends.cudnn.benchmark = old_bm

    Y = torch.cat(Y)
    X = torch.cat(X)
    S = torch.cat(S) # scale
    scores = torch.cat(C) * torch.cat(Q) # scores = reliability * repeatability
    XYS = torch.stack([X,Y,S], dim=-1)
    D = torch.cat(D)
    return XYS, D, scores
