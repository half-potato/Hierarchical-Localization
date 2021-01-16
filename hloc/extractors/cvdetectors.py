import cv2
import os
import math
import torch
from torch import nn
import torch.nn.functional as F
from ..utils.base_model import BaseModel
import numpy as np
import sys
from pathlib import Path
import matplotlib.pyplot as plt

FEATURES = {
    "fast": cv2.FastFeatureDetector_create(),
    "orb": cv2.ORB_create(nfeatures=2000),
    #  "sift": cv2.SIFT(),
}

if hasattr(cv2, "xfeatures2d"):
    FEATURES["sift"] = cv2.xfeatures2d.SIFT_create()
    FEATURES["surf"] = cv2.xfeatures2d.SURF_create()
    #  FEATURES["brief"] = cv2.xfeatures2d.DescriptorExtractor_create("BRIEF")
    FEATURES["brief"] = cv2.xfeatures2d.BriefDescriptorExtractor_create()

class CVDetectors(BaseModel):
    default_conf = {
    }
    def _init(self, config):
        self.config = config
        self.detector = FEATURES[self.config["cvdetector_name"]]
        self.descriptor = FEATURES[self.config["cvdescriptor_name"]]

    def _forward(self, data):
        x = data["image"]

        if x.shape[1] > 1:
            # convert to bw
            x = (x[:, 0:1, :, :] + x[:, 1:2, :, :] + x[:, 2:3, :, :])/3

        B, C, H, W = x.shape
        pts = []
        scores = []
        sampled = []
        for img in x:
            img = (img.view(H, W, 1).cpu().numpy()*255).astype(np.uint8)
            kpts = self.detector.detect(img, None)
            kpts, desc = self.descriptor.compute(img, kpts)
            l_pts = []
            l_scores = []
            for kp in kpts:
                l_pts.append([int(kp.pt[1]+0.5), int(kp.pt[0]+0.5)])
                l_scores.append(kp.response)
            print(len(l_pts))
            pts.append(torch.tensor(l_pts)) # (N, 2)
            scores.append(torch.tensor(l_scores)) # (N)
            sampled.append(torch.tensor(desc.T)) # (256, N)

        return {
            'keypoints': pts,
            'scores': scores,
            'descriptors': sampled,
        }

    def _describe(self, data, output):
        # torch where over batch
        B = len(output['keypoints'])
        x = data["image"]
        B, C, H, W = x.shape

        points = []
        scores = []
        sampled = []
        for i in range(B):
            img = x[i]
            img = (img.view(H, W, 1).cpu().numpy()*255).astype(np.uint8)
            pts = output['keypoints'][i].float() # (N, 2)
            ss = output['scores'][i].float() # (N)
            # convert to keypoints
            kpts = []
            for (pt, s) in zip(pts, ss):
                kpts.append(cv2.KeyPoint(float(pt[1]-0.5), float(pt[0]-0.5), _size=31))

            kpts, desc = self.descriptor.compute(img, kpts)
            l_pts = []
            l_scores = []
            for kp in kpts:
                l_pts.append([int(kp.pt[1]+0.5), int(kp.pt[0]+0.5)])
                l_scores.append(kp.response)
            points.append(torch.tensor(l_pts)) # (N, 2)
            scores.append(torch.tensor(l_scores)) # (N)
            sampled.append(torch.tensor(desc.T)) # (32, N)

        return {
            'keypoints': points,
            'scores': scores,
            'descriptors': sampled,
        }
