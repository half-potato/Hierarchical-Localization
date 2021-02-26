import sys
from pathlib import Path

import torch
from ..utils.base_model import BaseModel
import torch.nn.functional as F

#  sys.path.append(str(Path(__file__).parent / '../../third_party'))
#  import hardnet.examples.extract_DenseHardNet as libhardnet
from kornia.feature import hardnet


class HardNet(BaseModel):
    default_conf = {
        #  "upscale": False
        "window_size": 32,
    }
    required_inputs = ['image']

    def _init(self, conf):
        self.config = conf
        #  self.net = libhardnet.DenseHardNet(self.config["stride"])
        self.net = hardnet.HardNet(pretrained=True)
        self.ws = self.config["window_size"]

    def _forward(self, data):
        return {}

    @torch.no_grad()
    def _describe(self, data, output):
        # Setup: Extract patches
        points = output['keypoints']
        B = len(points)
        images = data["image"]

        win = self.ws//2
        wx, wy = torch.meshgrid(torch.linspace(-win, self.ws-win, self.ws),
                                torch.linspace(-win, self.ws-win, self.ws))
        device = images.device
        wx = wx.to(device)
        wy = wy.to(device)

        def extract_patch(image, pt):
            # image: (1, C, H, W)
            # pt: (2) in row col coords
            # First, construct the points
            _, C, H, W = image.shape
            # these have shape (self.ws, self.ws)
            nx = (wy + pt[0])/W*2 - 1
            ny = (wx + pt[1])/H*2 - 1
            points = torch.stack((nx, ny), dim=-1).view(1, 1, -1, 2)

            patch = F.grid_sample(image, points, mode="bilinear", align_corners=False)
            patch = patch.view(1, C, self.ws, self.ws)
            # normalize patch
            return (patch - patch.mean())/patch.std()

        descs = []
        for i in range(B):
            # Run: Extract patches
            pts = points[i].float().to(device) # (N, 2), column, row
            patches = []
            for pt in pts:
                patches.append(extract_patch(images[i].unsqueeze(0), pt))
            if not patches:
                desc = torch.empty((0, 128))
            else:
                patches = torch.stack(patches)
                desc = self.net(patches.reshape(-1, 1, self.ws, self.ws))
            # B, 128

            descs.append(desc.T)
        output['descriptors'] = descs
        return output
