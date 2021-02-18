import argparse
import torch
from pathlib import Path
import h5py
import logging
import numpy as np
from tqdm import tqdm
import pprint

from . import extractors
from .utils.base_model import dynamic_load
from .utils.tools import map_tensor
from .utils.image_dataset import ImageDataset


'''
A set of standard configurations that can be directly selected from the command
line using their name. Each is a dictionary with the following entries:
    - output: the name of the feature file that will be generated.
    - model: the model configuration, as passed to a feature extractor.
    - preprocessing: how to preprocess the images read from disk.
'''
confs = {
    'superpoint_aachen': {
        'output': 'feats-superpoint-n4096-r1024',
        'model': {
            'name': 'superpoint',
            'nms_radius': 3,
            'max_keypoints': 4096,
        },
        'preprocessing': {
            'grayscale': True,
            'resize_max': 1024,
        },
    },
    # Resize images to 1600px even if they are originally smaller.
    # Improves the keypoint localization if the images are of good quality.
    'superpoint_max': {
        'output': 'feats-superpoint-n4096-rmax1600',
        'model': {
            'name': 'superpoint',
            'nms_radius': 3,
            'max_keypoints': 4096,
        },
        'preprocessing': {
            'grayscale': True,
            'resize_max': 1600,
            'resize_force': True,
        },
    },

    'cpainted': {
        'output': 'feats-cpainted-n4096-r1024',
        'model': {
            'name': 'cpainted',
            'nms_radius': 3,
            'max_keypoints': 4096,
        },
        'preprocessing': {
            'grayscale': True,
            'resize_max': 1600,
            'resize_force': True,
        },
    },

    'cpainted_aachen': {
        'output': 'feats-cpainted-n4096-r1024',
        'model': {
            'name': 'cpainted',
            'nms_radius': 3,
            'max_keypoints': 4096,
        },
        'preprocessing': {
            'grayscale': True,
            'resize_max': 1600,
            'resize_force': True,
        },
    },

    'cpainted_inloc': {
        'output': 'feats-cpainted-n4096-r1024',
        'model': {
            'name': 'cpainted',
            'nms_radius': 4,
            'max_keypoints': 4096,
        },
        'preprocessing': {
            'grayscale': True,
            'resize_max': 1600,
            'resize_force': True,
        },
    },

    'superpoint_inloc': {
        'output': 'feats-superpoint-n4096-r1600',
        'model': {
            'name': 'superpoint',
            'nms_radius': 4,
            'max_keypoints': 4096,
        },
        'preprocessing': {
            'grayscale': True,
            'resize_max': 1600,
        },
    },
    'd2net-ss': {
        'output': 'feats-d2net-ss',
        'model': {
            'name': 'd2net',
            'multiscale': False,
        },
        'preprocessing': {
            'grayscale': False,
            'resize_max': 1600,
        },
    },
}

img_extensions = [".png", ".jpg", ".jpeg"]

@torch.no_grad()
def main(conf, image_dir, export_dir, as_half=False, return_num_points=False):
    logging.info('Extracting local features with configuration:'
                 f'\n{pprint.pformat(conf)}')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    Model = dynamic_load(extractors, conf['model']['name'])
    model = Model(conf['model']).eval().to(device)

    loader = ImageDataset(image_dir, conf['preprocessing'])
    loader = torch.utils.data.DataLoader(loader, num_workers=1)

    feature_path = Path(export_dir, conf['output']+'.h5')
    feature_path.parent.mkdir(exist_ok=True, parents=True)

    # First, check if the file has already been computed
    existing = feature_path.exists()
    feature_file = h5py.File(str(feature_path), 'a')
    main.total_num_points = 0
    if existing:
        main.i = 0
        def counter(name, obj):
            if Path(name).suffix.lower() in img_extensions:
                main.i += 1
            if Path(name).name == "keypoints":
                main.total_num_points += obj.shape[0]
        print("Found existing feature file, checking if we can skip computation")
        feature_file.visititems(counter)
        if len(loader) == main.i:
            print(f"Exact number {main.i} found, skipping")
            if return_num_points:
                feature_file.close()
                return feature_path, main.total_num_points / main.i
            else:
                feature_file.close()
                return feature_path
        if abs(len(loader) - main.i) < len(loader)//8:
            print(f"Missing {len(loader) - main.i} images, skipping anyways")
            if return_num_points:
                feature_file.close()
                return feature_path, main.total_num_points / main.i
            else:
                feature_file.close()
                return feature_path
        print(f"Missing {len(loader) - main.i} images, proceeding to computation")


    for data in tqdm(loader):
        if data['name'][0] in feature_file:
            continue

        pred = model(map_tensor(data, lambda x: x.to(device)))
        pred = {k: v[0].cpu().numpy() for k, v in pred.items()}

        pred['image_size'] = original_size = data['original_size'][0].numpy()
        if 'keypoints' in pred:
            size = np.array(data['image'].shape[-2:][::-1])
            scales = (original_size / size).astype(np.float32)
            pred['keypoints'] = (pred['keypoints'] + .5) * scales[None] - .5
            main.total_num_points += pred['keypoints'].shape[0]

        if as_half:
            for k in pred:
                dt = pred[k].dtype
                if (dt == np.float32) and (dt != np.float16):
                    pred[k] = pred[k].astype(np.float16)

        grp = feature_file.create_group(data['name'][0])
        for k, v in pred.items():
            grp.create_dataset(k, data=v)

        del pred

    feature_file.close()
    logging.info('Finished exporting features.')

    if return_num_points:
        return feature_path, main.total_num_points/len(loader)
    else:
        return feature_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=Path, required=True)
    parser.add_argument('--export_dir', type=Path, required=True)
    parser.add_argument('--conf', type=str, default='superpoint_aachen',
                        choices=list(confs.keys()))
    parser.add_argument('--as_half', action='store_true')
    args = parser.parse_args()
    main(confs[args.conf], args.image_dir, args.export_dir, args.as_half)
