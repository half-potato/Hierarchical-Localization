import sys
from pathlib import Path

from ..utils.base_model import BaseModel

sys.path.append(str(Path(__file__).parent / '../../third_party'))
import SuperGluePretrainedNetwork.models.superpoint as SP


class SuperPoint(BaseModel):
    default_conf = {
        'nms_radius': 4,
        'keypoint_threshold': 0.005,
        'max_keypoints': -1,
        'remove_borders': 4,
    }
    required_inputs = ['image']

    def _init(self, conf):
        self.net = SP.SuperPoint(conf)

    def _forward(self, data):
        a = self.net(data)
        print(a['keypoints'][0].shape)
        return a

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
        print(descs[0].shape)
        return output
