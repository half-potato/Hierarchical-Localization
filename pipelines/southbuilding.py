#!/usr/bin/env python
# coding: utf-8

from pathlib import Path
from pprint import pformat

from hloc import extract_features, match_features
from hloc import triangulation, localize_sfm, visualization

# # Pipeline for outdoor day-night visual localization

def run_test(base_dir, output_dir, feature_conf, matcher_conf, run_name, run_localization=False):
    # ## Setup
    # Here we declare the paths to the dataset, the reconstruction and
    # localization outputs, and we choose the feature extractor and the
    # matcher. You only need to download the [Aachen Day-Night
    # dataset](https://www.visuallocalization.net/datasets/) and put it in
    # `datasets/aachen/`, or change the path.

    #  run_name = 'run_24_super_desc'
    dname = "southbuilding"
    dataset = Path(base_dir) / "South-Building"  # change this if your dataset is somewhere else
    images = dataset / 'images'

    pairs = Path('pairs/') / dname
    sfm_pairs = pairs / "pairs-retrieval.txt"

    run_dir = output_dir / f'{dname}/{run_name}'  # where everything will be saved
    run_dir.mkdir(exist_ok=True, parents=True)
    reference_sfm = run_dir / f'sfm'  # the SfM model we will build

    # ## Extract local features for database and query images
    feature_path = extract_features.main(feature_conf, images, run_dir)

    # The function returns the path of the file in which all the extracted features are stored.

    # ## Match the database images

    sfm_match_path = match_features.main(matcher_conf, sfm_pairs, feature_conf['output'], run_dir)

    # The function returns the path of the file in which all the computed matches are stored.

    # ## Triangulate a new SfM model from the given poses
    # We triangulate the sparse 3D pointcloud given the matches and the
    # reference poses stored in the SIFT COLMAP model.

    stats = triangulation.main(
        reference_sfm,
        output_dir / 'sfm_sift',
        images,
        sfm_pairs,
        feature_path,
        sfm_match_path,
        colmap_path='colmap')  # change if COLMAP is not in your PATH

    if run_localization:
        print(f"No localization for {dname} because there are no queries")
    return stats

