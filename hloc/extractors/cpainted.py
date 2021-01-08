import os
import math
import torch
from torch import nn
import torch.nn.functional as F
from ..utils.base_model import BaseModel
import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / '../../third_party'))
import SuperGluePretrainedNetwork.models.superpoint as SP


dii_filter = torch.tensor([
    [1., -2., 1.],
    [2., -4., 2.],
    [1., -2., 1.]]).view(1, 1, 3, 3).float().cuda()

djj_filter = torch.tensor([
    [1., 2., 1.],
    [-2., -4., -2.],
    [1., 2., 1.]]).view(1, 1, 3, 3).float().cuda()


def block(in_ch, out_ch, pool=True, pad=True):
    layers = [
        nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1 if pad else 0),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    ]
    if pool:
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

    return layers


class SuperPointNet(nn.Module):
    def __init__(self, config):
        super(SuperPointNet, self).__init__()
        self.scaling_steps = 3
        self.cell = 2**self.scaling_steps
        self.config = config

        self.common = nn.Sequential(
            nn.ReflectionPad2d([1]*4),
            # block 1
            *block(1, 64, pad=False),
            # block 2
            *block(64, 64),
            # block 3
            *block(64, 128),
        )

        self.det_head = nn.Sequential(
            # block 4
            *block(128, 128, pool=False),
            # Rest of normal det head
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, self.cell**2+1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.cell**2+1)
        )

        self.desc_head = nn.Sequential(
            # block 4
            *block(128, 128, pool=False),
            # Rest of normal desc head
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(256)
        )

        #  N, D, H, W = 64, 256, 30, 30
        #  desc = nn.Parameter(torch.rand(N, D, H, W))
        #  self.register_parameter("desc", desc)

    def _forward(self, x):
        # semi: Output point pytorch tensor shaped N x 65 x H/8 x W/8.
        # desc: Output descriptor pytorch tensor shaped N x 256 x H/8 x W/8.

        x = self.common(x)
        semi = self.det_head(x)  # (N, 65, H/8, W/8)
        desc = self.desc_head(x)  # (N, 256, H/8, W/8)
        #  desc = self.desc[:N]
        desc = F.normalize(desc, p=2, dim=1)
        return semi, desc


class SuperPointTrainable(SuperPointNet):

    def forward(self, x):
        # The input size must be a multiple of 8
        N, _, H, W = x.shape

        semi, desc = self._forward(x)
        heatmap = self.semi_to_heatmap(semi, (N, 1, H, W))
        return {
            "heatmap": heatmap,
            "raw_desc": desc,
        }

    def semi_to_heatmap(self, semi, net_input_shape):
        N, _, H, W = net_input_shape
        # Reshape detector head
        #  dense = torch.exp(semi) # Softmax.
        #  print(dense.shape)
        #  print((torch.sum(dense, dim=1)+.00001).shape)
        #  dense = dense / (torch.sum(dense, dim=1)+.00001) # Should sum to 1.
        dense = F.softmax(semi, dim=1)
        # Remove dustbin.
        nodust = dense[:, :-1, :, :]  # N x 64 x Hc x Wc
        # Reshape to get full resolution heatmap.
        Hc = int(H / self.cell)
        Wc = int(W / self.cell)
        nodust = nodust.permute(0, 2, 3, 1)  # N x Hc x Wc
        heatmap = nodust.reshape([N, Hc, Wc, self.cell, self.cell])
        heatmap = heatmap.permute(0, 1, 3, 2, 4)
        heatmap = heatmap.reshape([N, 1, H, W])
        return heatmap

    def load_default_state_dict(self):
        path = self.config["checkpoint"]
        ckpt = torch.load(path)
        if "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            state_dict = ckpt

        state_dict = {".".join(k.split(".")[1:]): v for k, v in state_dict.items()}
        self.load_state_dict(state_dict)

class CPainted(BaseModel):
    default_config = {
        "threshold": 0.03,
        "maxpool_radius": 3,
        "remove_borders": 4,
        "max_keypoints": 2048,
        #  "checkpoint": "/app/outputs/checkpoints/no_forest_low_thres/models/checkpoint001.pth",
        #  "checkpoint": "/app/outputs/checkpoints/run_9_24/models/checkpoint003.pth",
        #  "checkpoint": "/app/outputs/checkpoints/run-8-20-unreal-blended_05/models/checkpoint005.pth",
        #  "checkpoint": "/app/outputs/checkpoints/run-10-23-unreal-blended_10/models/checkpoint002.pth",
        "checkpoint": "outputs/checkpoints/run-10-23-unreal-blended_08/models/checkpoint005.pth",
        #  "checkpoint": "/app/outputs/checkpoints/run-11-10-unreal-blended_08/models/checkpoint005.pth",
    }
    def _init(self, config):
        self.config = {**self.default_config, **config}
        self.net = SuperPointTrainable(self.config)
        self.net.load_default_state_dict()
        self.desc_net = SP.SuperPoint(self.config)

    def _forward(self, data):
        x = data["image"]
        # Resize image such that it is a multiple of the cell size
        old_size = x.shape
        B, C, H, W = old_size
        input_size_multiple = self.net.cell

        x = pad_to_multiple(x, input_size_multiple)
        # padding: lp, rp, tp, bp
        padding = get_padding_for_multiple(old_size, input_size_multiple)

        # Run model
        result = self.net.forward(x)
        heatmap = result["heatmap"]
        desc = result["raw_desc"]
        D = desc.shape[1]
        heatmap = unpad_multiple(heatmap, old_size, input_size_multiple)

        # remove border, apply nms + threshold
        # Shape: (3, N)
        mask1 = mask_border(heatmap, border=self.config["remove_borders"])
#         mask2 = mask_max(heatmap, radius=self.config["maxpool_radius"])

#         heatmap = score_gaussian_peaks(heatmap)
        mask2 = mask_max(heatmap, radius=self.config["maxpool_radius"])

        pooled = mask1 * mask2 * heatmap

        # torch where over batch
        pts = []
        scores = []
        sampled = []
        for i in range(B):
            y, x = torch.where(pooled[i].squeeze() > self.config["threshold"])
            if len(y) > self.config["max_keypoints"]:
                threshold, _ = torch.sort(pooled[i].flatten(), descending=True)
                threshold = threshold[self.config["max_keypoints"]]
                y, x = torch.where(pooled[i].squeeze() > threshold)
            l_pts = torch.stack((y, x), dim=1)
            l_scores = heatmap[i].squeeze()[l_pts[:, 0], l_pts[:, 1]]
            flipped = torch.flip(l_pts, [1]).float()

            l_sampled = sample_descriptors(
                    desc[i].unsqueeze(0), H, W, l_pts.view(1, -1, 2), padding).squeeze(0).T
            #  l_sampled = SP.sample_descriptors(flipped.view(1, -1, 2), descriptors[i][None], 8)[0]

            pts.append(flipped) # (N, 2)
            scores.append(l_scores) # (N)
            sampled.append(l_sampled) # (256, N)

        return {
            'keypoints': pts,
            'scores': scores,
            'descriptors': sampled,
        }

# Util for extracting features

# pytorch
def mask_border(score_map, border=4):
    batch = score_map.view(-1, 1, score_map.shape[-2], score_map.shape[-1])

    N, _, H, W = batch.shape
    mask = torch.ones((N, 1, H-border*2, W-border*2)).float().to(score_map.device)
    mask = F.pad(mask, [border]*4, mode="constant", value=0).view(N, H, W)
    return mask

# pytorch
def mask_max(score_map, radius=8):
    batch = score_map.view(-1, 1, score_map.shape[-2], score_map.shape[-1])
    N, _, H, W = batch.shape

    l_max = F.max_pool2d(
            F.pad(batch, [radius] * 4, mode='constant', value=0.),
            radius*2+1, stride=1
        )
    mask = l_max == batch
    return mask.view(N, H, W)

def score_gaussian_peaks(score_map):
    batch = score_map.view(1, 1, score_map.shape[-2], score_map.shape[-1]).float()
    batch = gf(batch)
    dii = F.conv2d(batch, dii_filter, padding=1, dilation=1, stride=1)
    djj = F.conv2d(batch, djj_filter, padding=1, dilation=1, stride=1)
    score = torch.min(-dii, -djj)/2
    return score


def gaussian_filter(kernel_size, sigma, channels=1):
    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_cord = torch.arange(kernel_size)
    x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1)

    mean = (kernel_size)/2.
    variance = sigma**2.

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = (1./(2.*math.pi*variance)) *\
                      torch.exp(
                          -torch.sum((xy_grid - mean)**2., dim=-1) /\
                          (2*variance)
                      )
    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1).cuda()

    gaussian_filter = nn.Conv2d(in_channels=channels, out_channels=channels, padding=kernel_size//2,
                                kernel_size=kernel_size, groups=channels, bias=False)

    gaussian_filter.weight.data = gaussian_kernel
    gaussian_filter.weight.requires_grad = False
    return gaussian_filter

gf = gaussian_filter(3, 1).cuda()

def get_padding_for_multiple(imshape, multiple):
    N, C, H, W = imshape
    # New size
    nH = math.ceil(H/multiple)*multiple
    nW = math.ceil(W/multiple)*multiple
    # Paddings
    tp = math.ceil((nH-H)/2)
    bp = math.floor((nH-H)/2)

    lp = math.ceil((nW-W)/2)
    rp = math.floor((nW-W)/2)
    return lp, rp, tp, bp

# pytorch
def unpad_multiple(images, old_shape, multiple):
    _, _, nH, nW = images.shape
    lp, rp, tp, bp = get_padding_for_multiple(old_shape, multiple)
    return images[:, :, tp:nH-bp, lp:nW-rp]


# pytorch
def pad_to_multiple(images, multiple, mode="reflect"):
    padding = get_padding_for_multiple(images.shape, multiple)
    padded = F.pad(images, padding, mode=mode)
    return padded

def sample_descriptors(descriptor, H, W, pts, padding=[0,0,0,0], normalize=True):
    # descriptor (N, C, H, W)
    # pts: (N, M, 2) in image coordinates (row col)
    # padding: lp, rp, tp, bp
    N, M, _ = pts.shape
    tens_pts = pts.float()
    norm_pts = torch.zeros_like(tens_pts) # (N, M, 2)
    norm_pts[:, :, 0] = (tens_pts[:, :, 1]+padding[0])/W*2 - 1
    norm_pts[:, :, 1] = (tens_pts[:, :, 0]+padding[2])/H*2 - 1

    grid = norm_pts.view(N, 1, M, 2)

    descs = F.grid_sample(descriptor, grid, mode="bilinear", align_corners=False)
    # descs has shape (N, C, 1, M)
    if normalize:
        descs = F.normalize(descs, p=2, dim=1)
    return descs.view(N, -1, M).transpose(1, 2)

