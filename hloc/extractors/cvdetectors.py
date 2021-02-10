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
    "fast": cv2.FastFeatureDetector_create(threshold=30),
    "orb": cv2.ORB_create(nfeatures=6000),
    #  "sift": cv2.SIFT(),
    #  "sift-noscale": cv2.SIFT(nOctaveLayers=1),
}
DIM = {
    "fast": 128,
    "orb": 128,
    "brief": 128,
    "sift": 128,
}

DEFAULT_SIZE = {
    "fast": 7,
    "sift": 3,
    "orb": 31,
}

if hasattr(cv2, "xfeatures2d"):
    FEATURES["sift"] = cv2.xfeatures2d.SIFT_create()
    FEATURES["surf"] = cv2.xfeatures2d.SURF_create()
    #  FEATURES["brief"] = cv2.xfeatures2d.DescriptorExtractor_create("BRIEF")
    FEATURES["brief"] = cv2.xfeatures2d.BriefDescriptorExtractor_create()

def compute_orientations(image, radius):
    batch = image.view(-1, 1, image.shape[-2], image.shape[-1])
    B, _, H, W = batch.shape
    # generate kernel
    x, y, = torch.meshgrid(
        torch.linspace(-radius-0.5, radius+0.5, radius*2+1),
        torch.linspace(-radius-0.5, radius+0.5, radius*2+1)
    )
    x = x.view(1, 1, radius*2+1, radius*2+1).to(batch.device)
    y = y.view(1, 1, radius*2+1, radius*2+1).to(batch.device)
    m_10 = F.conv2d(batch, x, padding=radius)
    m_01 = F.conv2d(batch, y, padding=radius)
    #  m_00 = F.conv2d(image, torch.ones_like(x), pad=radius)
    angle = torch.atan2(m_01, m_10)
    angle[torch.isnan(angle)] = 0
    return angle.squeeze().cpu().numpy() * 180 / math.pi

class CVDetectors(BaseModel):
    default_conf = {
        "allow_scale_in_desc": True,
        "default_size": 15,
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
            l_pts = []
            for kp in kpts:
                l_pts.append([kp.pt[0], kp.pt[1], kp.response])

            l_pts = torch.tensor(l_pts)
            if "max_keypoints" in self.config:
                sorted_vals = -torch.sort(-l_pts[:, 2]).values
                desired = min(self.config["max_keypoints"], len(l_pts)-1)
                threshold = sorted_vals[desired]
                inds, = torch.where(l_pts[:, 2]>threshold)
                l_pts = l_pts[inds, :]
                kpts = [kpt for kpt in kpts if kpt.response > threshold]

            if not self.config["allow_scale_in_desc"]:
                for kp in kpts:
                    kp.size = DEFAULT_SIZE[self.config["cvdetector_name"]]
                    # if sift, set octave to 0
                    if self.config["cvdetector_name"] == "sift":
                        #  octave = kp.octave & 255
                        layer = (kp.octave >> 8) & 0xFF
                        #  octave = octave if octave < 128 else (-128 | octave)
                        kp.octave = (layer << 8)

            #  print(sizes)
            kpts, desc = self.descriptor.compute(img, kpts)
            #  drawn = img.copy()
            #  drawn = cv2.drawKeypoints(img, kpts, drawn, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            #  plt.imshow(drawn)
            #  plt.show()
            if desc is None:
                print(f"No points found for image: {data['name']}")
                pts.append(torch.empty((0, 2)))
                scores.append(torch.empty((0)))
                sampled.append(torch.empty((DIM[self.config["cvdescriptor_name"]], 0)))
            else:
                scores.append(l_pts[:, 2].squeeze())
                pts.append(l_pts[:, :2]) # (N, 2)
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
            pts = output['keypoints'][i].float().cpu().numpy() # (N, 2)
            ss = output['scores'][i].float().cpu().numpy() # (N)
            # compute orientation
            size = 16
            orientations = compute_orientations(x[i], size//2)
            # convert to keypoints
            kpts = []
            for (pt, s) in zip(pts, ss):
                kpts.append(cv2.KeyPoint(pt[0], pt[1], _size=size, _angle=orientations[int(pt[1]), int(pt[0])]))

            #  drawn = img.copy()
            #  drawn = cv2.drawKeypoints(img, kpts, drawn, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            #
            #  plt.figure()
            #  plt.imshow(drawn)
            #  plt.show()

            kpts, desc = self.descriptor.compute(img, kpts)
            l_pts = []
            l_scores = []
            for kp in kpts:
                l_pts.append([int(kp.pt[0]+0.5), int(kp.pt[1]+0.5)])
                l_scores.append(kp.response)
            if desc is None:
                print(f"No points found for image: {data['name']}. {pts.shape}")
                points.append(torch.empty((0, 2)))
                scores.append(torch.empty((0)))
                sampled.append(torch.empty((DIM[self.config["cvdescriptor_name"]], 0)))
            else:
                points.append(torch.tensor(l_pts)) # (N, 2)
                scores.append(torch.tensor(l_scores)) # (N)
                sampled.append(torch.tensor(desc.T)) # (32, N)

        return {
            'keypoints': points,
            'scores': scores,
            'descriptors': sampled,
        }
