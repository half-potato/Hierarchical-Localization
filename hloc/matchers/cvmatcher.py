import cv2
import torch
import numpy as np

from ..utils.base_model import BaseModel

CV_NORM = {
    "NORM_L1": cv2.NORM_L1,
    "NORM_L2": cv2.NORM_L2,
    "NORM_HAMMING": cv2.NORM_HAMMING,
    "NORM_HAMMING2": cv2.NORM_HAMMING2,
}

class NearestNeighbor(BaseModel):
    default_conf = {
        'ratio_threshold': None,
        'distance_threshold': None,
        'do_mutual_check': True,
    }
    required_inputs = ['descriptors0', 'descriptors1']

    def _init(self, conf):
        self.matcher = cv2.BFMatcher(CV_NORM[conf["norm"]], crossCheck=conf["do_mutual_check"])

    def _forward(self, data):
        # (b, d, n)
        # (b, d, m)
        desc0 = data['descriptors0'].cpu().numpy()
        desc1 = data['descriptors1'].cpu().numpy()
        if self.conf["norm"] in ["NORM_HAMMING", "NORM_HAMMING2"]:
            desc0 = desc0.astype(np.uint8)
            desc1 = desc1.astype(np.uint8)
        else:
            desc0 = desc0.astype(np.float32)
            desc1 = desc1.astype(np.float32)
        b, _, n = desc0.shape
        _, _, m = desc1.shape

        matches0 = -torch.ones((b, n), dtype=torch.int64)
        scores0 = torch.zeros((b, n), dtype=torch.float)
        for i in range(b):
            if self.conf["do_mutual_check"]:
                good = self.matcher.match(desc0[i].T, desc1[i].T)
            else:
                matches = self.matcher.knnMatch(desc0[i].T, desc1[i].T, k=2)
                # Apply ratio test
                good = []
                for m,n in matches:
                    if m.distance < self.conf["ratio_threshold"]*n.distance:
                        good.append(m)
            for match in good:
                matches0[i, match.queryIdx] = match.trainIdx
                scores0[i, match.queryIdx] = match.distance
        return {
            'matches0': matches0, # shape (b, n) of inds into (b, m)
            'matching_scores0': scores0, # shape (b, n)
        }

