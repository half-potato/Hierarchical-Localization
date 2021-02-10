import sys
from pathlib import Path

from ..utils.base_model import BaseModel

sys.path.append(str(Path(__file__).parent / '../../third_party'))
import hardnet.examples.extract_DenseHardNet as libhardnet


class SuperPoint(BaseModel):
    default_conf = {
        "stride": 2,
        "upscale": False
    }
    required_inputs = ['image']

    def _init(self, conf):
        self.config = conf
        self.net = libhardnet.DenseHardNet(self.config["stride"])

    def _forward(self, data):
        return {}

    def _describe(self, data, output):
        # Compute descriptor
        _, descriptors = self.net._forward(data)

        # torch where over batch
        B = len(output['keypoints'])
        descs = []
        for i in range(B):
            pts = output['keypoints'][i].float()
            sampled = SP.sample_descriptors(pts.view(1, -1, 2), descriptors[i][None], 8)[0]

            descs.append(sampled)
        output['descriptors'] = descs
        return output
