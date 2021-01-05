import sys

from pathlib import Path
import subprocess
import logging
import torch

from ..utils.base_model import BaseModel

d2net_path = Path(__file__).parent / '../../third_party/d2net'
sys.path.append(str(d2net_path))
from lib.model_test import D2Net as _D2Net
from lib.pyramid import process_multiscale

# To implement multiscale description
import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.exceptions import EmptyTensorError
from lib.utils import interpolate_dense_features, upscale_positions


def describe_multiscale(image, model, keypoints, scales=[.5, 1, 2]):
    b, _, h_init, w_init = image.size()
    device = image.device
    assert(b == 1)

    previous_dense_features = None
    banned = None
    for idx, scale in enumerate(scales):
        current_image = F.interpolate(
            image, scale_factor=scale,
            mode='bilinear', align_corners=True
        )
        _, _, h_level, w_level = current_image.size()

        dense_features = model.dense_feature_extraction(current_image)
        del current_image

        _, _, h, w = dense_features.size()

        # Sum the feature maps.
        if previous_dense_features is not None:
            dense_features += F.interpolate(
                previous_dense_features, size=[h, w],
                mode='bilinear', align_corners=True
            )
            del previous_dense_features

        previous_dense_features = dense_features
        del dense_features


    all_descriptors = torch.zeros([
        model.dense_feature_extraction.num_channels, 0
    ])
    try:
        raw_descriptors, _, ids = interpolate_dense_features(
            keypoints.to(device),
            dense_features[0]
        )
    except EmptyTensorError:
        raw_descriptors = all_descriptors

    descriptors = F.normalize(raw_descriptors, dim=0).cpu()
    del raw_descriptors

    all_descriptors = torch.cat([all_descriptors, descriptors], dim=1)
    descriptors = all_descriptors.t().numpy()
    del all_descriptors

    return descriptors


class D2Net(BaseModel):
    default_conf = {
        'model_name': 'd2_tf.pth',
        'use_relu': True,
        'multiscale': False,
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

    def _forward(self, data):
        image = data['image']
        image = image.flip(1)  # RGB -> BGR
        norm = image.new_tensor([103.939, 116.779, 123.68])
        image = (image * 255 - norm.view(1, 3, 1, 1))  # caffe normalization

        if self.conf['multiscale']:
            keypoints, scores, descriptors = process_multiscale(
                image, self.net)
        else:
            keypoints, scores, descriptors = process_multiscale(
                image, self.net, scales=[1])
        keypoints = keypoints[:, [1, 0]]  # (x, y) and remove the scale

        return {
            'keypoints': torch.from_numpy(keypoints)[None],
            'scores': torch.from_numpy(scores)[None],
            'descriptors': torch.from_numpy(descriptors.T)[None],
        }
