#!/usr/bin/env python
# coding: utf-8

from pathlib import Path
from pprint import pformat

from hloc import extract_features, match_features, pairs_from_covisibility
from hloc import colmap_from_nvm, triangulation, visualization
from . import localize

def run_test(base_dir, output_dir, feature_conf, matcher_conf, run_name, run_localization=False):
    # ## Setup
    # Here we declare the paths to the dataset, the reconstruction and
    # localization outputs, and we choose the feature extractor and the
    # matcher. 

    dataset = Path(base_dir) / 'RobotCar/'  # change this if your dataset is somewhere else
    images = dataset / 'images'

    pairs = Path('pairs/')
    sfm_pairs = pairs / 'robotcar-pairs-db-covis20.txt'  # top 20 most covisible in SIFT model
    loc_pairs = pairs / 'pairs-query-netvlad20-percam-perloc.txt'  # top 50 retrieved by NetVLAD

    run_dir = output_dir / f'robotcar/{run_name}'  # where everything will be saved
    run_dir.mkdir(exist_ok=True, parents=True)
    reference_sfm = run_dir / f'sfm'  # the SfM model we will build
    results_path = run_dir / f'RobotCar_hloc_netvlad50.txt'  # the result file

    # ## Extract local features for database and query images
    feature_path, avg_num_points = extract_features.main(feature_conf, images, run_dir, return_num_points=True)
    print(f"Avg num points: {avg_num_points}")
    sfm_match_path = match_features.main(matcher_conf, sfm_pairs, feature_conf['output'], run_dir)

    # The function returns the path of the file in which all the computed matches are stored.

    # ## Triangulate a new SfM model from the given poses
    # We triangulate the sparse 3D pointcloud given the matches and the
    # reference poses stored in the SIFT COLMAP model.

    stats = triangulation.main(
        reference_sfm,
        output_dir / "robotcar" / 'sfm_sift',
        images,
        sfm_pairs,
        feature_path,
        sfm_match_path,
        colmap_path='colmap')  # change if COLMAP is not in your PATH
    stats['avg_num_points'] = avg_num_points

    if run_localization:
        # ## Match the query images
        # Here we assume that the localization pairs are already computed using
        # image retrieval (NetVLAD). To generate new pairs from your own global
        # descriptors, have a look at `hloc/pairs_from_retrieval.py`. These pairs
        # are also used for the localization - see below.
        loc_match_path = match_features.main(matcher_conf, loc_pairs, feature_conf['output'], run_dir)

        # ## Localize!
        logs_path = localize.main(
            reference_sfm / "model",
            "outputs/*_queries_with_intrinsics.txt",
            "pairs/pairs-query-netvlad20-percam-perloc.txt",
            feature_path,
            loc_match_path,
            results_path,
            covisibility_clustering=matcher_conf["covisibility_clustering"])  # not required with SuperPoint+SuperGlue
        stats["logs_path"] = logs_path

    return stats

