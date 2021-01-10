from . import aachen
from . import FourSeasons
from . import RobotCar
from . import southbuilding
import json
from pathlib import Path
from hloc.match_features import confs as MATCHER_CONFS

PIPELINES = {
    "southbuilding": southbuilding,
    "aachen": aachen,
    "4Seasons": FourSeasons,
    "RobotCar": RobotCar,
}

METHODS = [
    #  {
    #      'model': {
    #          'name': 'superpoint',
    #          'nms_radius': 3,
    #          'max_keypoints': 4096,
    #      },
    #      'matcher_name': "superglue",
    #      'preprocessing': {
    #          'grayscale': True,
    #          'resize_max': 1024,
    #      },
    #  },
    {
        "name": "orb",
        'model': {
            'name': 'cvdetectors',
            'cvdetector_name': 'orb',
            'cvdescriptor_name': 'orb',
        },
        'matcher_name': "NN",
        'preprocessing': {
            'grayscale': True,
            'resize_max': 1024,
        },
    },
    {
        "name": "sift",
        'model': {
            'name': 'cvdetectors',
            'cvdetector_name': 'sift',
            'cvdescriptor_name': 'sift',
        },
        'matcher_name': "NN",
        'preprocessing': {
            'grayscale': True,
            'resize_max': 1024,
        },
    },
    {
        "name": "d2net",
        'model': {
            'name': 'd2net',
        },
        'matcher_name': "NN",
        'preprocessing': {
            'grayscale': False,
            'resize_max': 1024,
        },
    },
    {
        "name": "r2d2",
        'model': {
            'name': 'r2d2',
        },
        'matcher_name': "NN",
        'preprocessing': {
            'grayscale': False,
            'resize_max': 1024,
        },
    },
    {
        "name": "cpainted+r2d2",
        'model': {
            'name': 'r2d2',
            'detector_name': 'cpainted',
            'multiscale': False,
            'nms_radius': 3,
            'max_keypoints': 4096,
        },
        'matcher_name': "NN",
        'preprocessing': {
            'grayscale': True,
            'resize_max': 1024,
        },
    },
    {
        "name": "cpainted+d2net",
        'model': {
            'name': 'd2net_split',
            'detector_name': 'cpainted',
            'multiscale': False,
            'nms_radius': 3,
            'max_keypoints': 4096,
        },
        'matcher_name': "NN",
        'preprocessing': {
            'grayscale': True,
            'resize_max': 1024,
        },
    },
    {
        "name": "cpainted",
        'model': {
            'name': 'cpainted',
            'nms_radius': 3,
            'max_keypoints': 4096,
        },
        'matcher_name': "NN",
        'preprocessing': {
            'grayscale': True,
            'resize_max': 1024,
        },
    },
    {
        "name": "cpainted+superpoint",
        'model': {
            'name': 'split',
            'detector': 'cpainted',
            'descriptor': 'superpoint',
            'nms_radius': 3,
            'max_keypoints': 4096,
        },
        'matcher_name': "superglue",
        'preprocessing': {
            'grayscale': True,
            'resize_max': 1024,
        },
    },
    {
        'name': 'superpoint',
        'model': {
            'name': 'split',
            'detector': 'superpoint',
            'descriptor': 'superpoint',
            'nms_radius': 3,
            'max_keypoints': 4096,
        },
        'matcher_name': "superglue",
        'preprocessing': {
            'grayscale': True,
            'resize_max': 1024,
        },
    },
]

def run_pipeline(base_dir, output_dir, pipeline_name, config, run_localization):
    base_dir = Path(base_dir)
    output_dir = Path(output_dir)
    # Set names
    run_name = config["name"]
    #  config['output'] = f'feats-{run_name.replace("_", "-")}-n{config["model"]["max_keypoints"]}-r{config["model"]["preprocessing"]["resize_max"]}'
    config['output'] = f'feats-{run_name.replace("_", "-")}'

    # Run pipeline
    matcher_conf = MATCHER_CONFS[config["matcher_name"]]
    pipeline = PIPELINES[pipeline_name]
    stats = pipeline.run_test(base_dir, output_dir, config, matcher_conf, run_name, run_localization=run_localization)

    # Save results
    out_path = output_dir / pipeline_name / f"{run_name}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "stats": stats,
        "feature_conf": config,
        "matcher_conf": matcher_conf,
    }
    with out_path.open(mode="w") as f:
        json.dump(output, f, indent=4)

def get_config(method_name):
    for config in METHODS:
        if config["name"] == method_name:
            return config
