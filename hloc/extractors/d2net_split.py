import sys

from pathlib import Path
import subprocess
import logging
import torch

from ..utils.base_model import BaseModel, dynamic_load
from .. import extractors as base

d2net_path = Path(__file__).parent / '../../third_party/d2net'
sys.path.append(str(d2net_path))
from lib.model_test import D2Net as _D2Net

# To implement multiscale description
import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.exceptions import EmptyTensorError
from lib.utils import interpolate_dense_features, upscale_positions, downscale_positions


class D2Net_Split(BaseModel):
    default_conf = {
        'model_name': 'd2_tf.pth',
        'use_relu': True,
        'multiscale': True,
    }
    required_inputs = ['image']

    def _init(self, conf):
        model_file = d2net_path / 'models' / conf['model_name']
        if not model_file.exists():
            model_file.parent.mkdir(exist_ok=True)
            cmd = ['wget', 'https://dsmn.ml/files/d2-net/'+conf['model_name'],
                   '-O', str(model_file)]
            ret = subprocess.call(cmd)
            if ret != 0:
                logging.warning(
                    f'Cannot download the D2-Net model with `{cmd}`.')
                exit(ret)

        self.net = _D2Net(
            model_file=model_file,
            use_relu=conf['use_relu'],
            use_cuda=False)

        Detector = dynamic_load(base, conf["detector_name"])
        self.detector = Detector(conf)

    def _forward(self, data):
        image = data['image']
        image = image.flip(1)  # RGB -> BGR
        norm = image.new_tensor([103.939, 116.779, 123.68])
        image = (image * 255 - norm.view(1, 3, 1, 1))  # caffe normalization

        if self.conf['multiscale']:
            keypoints, scores, descriptors = process_multiscale(
                image, self.detector, self.net)
        else:
            keypoints, scores, descriptors = process_multiscale(
                image, self.detector, self.net, scales=[1])
        keypoints = keypoints[:, [1, 0]]  # (x, y) and remove the scale

        return {
            'keypoints': torch.from_numpy(keypoints)[None],
            'scores': torch.from_numpy(scores)[None],
            'descriptors': torch.from_numpy(descriptors).T[None],
        }


def process_multiscale(image, detector, model, scales=[.5, 1, 2]):
    # detector is of type BaseModel
    b, _, h_init, w_init = image.size()
    device = image.device
    assert(b == 1)

    all_keypoints = torch.zeros([3, 0])
    all_descriptors = torch.zeros([
        model.dense_feature_extraction.num_channels, 0
    ])
    all_scores = torch.zeros(0)

    previous_dense_features = None
    banned = None
    for idx, scale in enumerate(scales):
        current_image = F.interpolate(
            image, scale_factor=scale,
            mode='bilinear', align_corners=True
        )
        _, _, h_level, w_level = current_image.size()

        dense_features = model.dense_feature_extraction(current_image)

        data = {"image": current_image}
        output = detector(data)
        keypoints = output["keypoints"][0].T
        scores = output["scores"][0].cpu()
        fmap_keypoints = downscale_positions(keypoints, scaling_steps=2)

        del current_image

        _, _, h, w = dense_features.size()

        # Sum the feature maps.
        if previous_dense_features is not None:
            dense_features += F.interpolate(
                previous_dense_features, size=[h, w],
                mode='bilinear', align_corners=True
            )
            del previous_dense_features
        # ([x, y, score], N)

        try:
            raw_descriptors, _, ids = interpolate_dense_features(
                fmap_keypoints.to(device),
                dense_features[0]
            )
        except EmptyTensorError:
            continue
        fmap_keypoints = fmap_keypoints[:, ids]
        scores = scores[ids]
        del ids

        keypoints = upscale_positions(fmap_keypoints, scaling_steps=2)
        del fmap_keypoints

        descriptors = F.normalize(raw_descriptors, dim=0).cpu()
        del raw_descriptors

        keypoints[0, :] *= h_init / h_level
        keypoints[1, :] *= w_init / w_level

        keypoints = keypoints.cpu()

        keypoints = torch.cat([
            keypoints,
            torch.ones([1, keypoints.size(1)]) * 1 / scale,
        ], dim=0)

        all_keypoints = torch.cat([all_keypoints, keypoints], dim=1)
        all_descriptors = torch.cat([all_descriptors, descriptors], dim=1)
        all_scores = torch.cat([all_scores, scores], dim=0)
        del keypoints, descriptors

        previous_dense_features = dense_features
        del dense_features
    del previous_dense_features, banned

    keypoints = all_keypoints.t().numpy()
    del all_keypoints
    scores = all_scores.numpy()
    del all_scores
    descriptors = all_descriptors.t().numpy()
    del all_descriptors
    return keypoints, scores, descriptors
