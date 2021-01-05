from . import aachen
from . import 4Seasons
from . import RobotCar
import json
from pathlib import Path
from hloc.match_features import confs as MATCHER_CONFS

PIPELINES = {
    "aachen": aachen,
    "4Seasons": 4Seasons,
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

def run_pipeline(base_dir, pipeline_name, config, run_localization):
    # Set names
    if "detector" in config["model"]:
        det_name = config["model"]["detector"]
        desc_name = config["model"]["descriptor"]
    else:
        det_name = config["model"]["name"]
        desc_name = config["model"]["name"]
    config['output'] = 'feats-{det_name}-{desc_name}-n{config["model"]["max_keypoints"]}-r{config["model"]["preprocessing"]["resize_max"]}'
    run_name = f'{det_name}_{desc_name}'

    # Run pipeline
    matcher_conf = MATCHER_CONFS[config["matcher_name"]]
    pipeline = PIPELINES[pipeline_name]
    stats = pipeline.run_test(base_dir, config, matcher_conf, run_name, run_localization=run_localization)

    # Save results
    out_path = Path("output") / pipeline_name / f"{run_name}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "stats": stats,
        "feature_conf": config,
        "matcher_conf": matcher_conf,
    }
    json.dump(output, str(out_path), indent=4)

def run_all(base_dir, run_localization):
    for pipeline_name in PIPELINES.keys():
        for config in METHODS:
            run_pipeline(base_dir, pipeline_name, config, run_localization=run_localization)
