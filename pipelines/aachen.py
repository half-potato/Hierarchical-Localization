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
    dname = "aachen"
    dataset = Path(base_dir) / "aachen"  # change this if your dataset is somewhere else
    images = dataset / 'images/images_upright/'

    pairs = Path('pairs/') / dname
    sfm_pairs = pairs / 'pairs-db-covis20.txt'  # top 20 most covisible in SIFT model

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
        results_path = run_dir / f'{dname}_hloc_netvlad50.txt'  # the result file
        loc_pairs = pairs / 'pairs-query-netvlad50.txt'  # top 50 retrieved by NetVLAD
        # ## Match the query images
        # Here we assume that the localization pairs are already computed using
        # image retrieval (NetVLAD). To generate new pairs from your own global
        # descriptors, have a look at `hloc/pairs_from_retrieval.py`. These pairs
        # are also used for the localization - see below.
        loc_match_path = match_features.main(matcher_conf, loc_pairs, feature_conf['output'], run_dir)


        # ## Localize!
        # Perform hierarchical localization using the precomputed retrieval and
        # matches. The file `Aachen_hloc_superpoint+superglue_netvlad50.txt` will
        # contain the estimated query poses. Have a look at
        # `Aachen_hloc_superpoint+superglue_netvlad50.txt_logs.pkl` to analyze some
        # statistics and find failure cases.
        logs_path = localize_sfm.main(
            reference_sfm / 'model',
            dataset / 'queries/*_time_queries_with_intrinsics.txt',
            loc_pairs,
            feature_path,
            loc_match_path,
            results_path,
            covisibility_clustering=matcher_conf["covisibility_clustering"])
        stats["logs_path"] = logs_path

    return stats
