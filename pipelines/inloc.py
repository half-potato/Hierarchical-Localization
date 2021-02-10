#!/usr/bin/env python
# coding: utf-8

from pathlib import Path
from pprint import pformat

from hloc import extract_features, match_features
from hloc import triangulation, localize_sfm, visualization, reconstruction, localize_inloc

# # Pipeline for outdoor day-night visual localization

'''
def run_test(base_dir, output_dir, feature_conf, matcher_conf, run_name, run_localization=False):
    # ## Setup
    # Here we declare the paths to the dataset, the reconstruction and
    # localization outputs, and we choose the feature extractor and the
    # matcher. You only need to download the [Aachen Day-Night
    # dataset](https://www.visuallocalization.net/datasets/) and put it in
    # `datasets/aachen/`, or change the path.

    #  run_name = 'run_24_super_desc'
    dname = "inloc"
    dataset = Path(base_dir) / "inloc"  # change this if your dataset is somewhere else
    images = dataset / "cutouts_imageonly"

    sfm_pairs = output_dir / "inloc" / "pairs-retrieval.txt"

    run_dir = output_dir / f'{dname}/{run_name}'  # where everything will be saved
    run_dir.mkdir(exist_ok=True, parents=True)
    reference_sfm = run_dir / f'sfm'  # the SfM model we will build

    # ## Extract local features for database and query images
    feature_path, avg_num_points = extract_features.main(feature_conf, images, run_dir, return_num_points=True)
    print(f"Avg num points: {avg_num_points}")

    # The function returns the path of the file in which all the extracted features are stored.

    # ## Match the database images

    sfm_match_path = match_features.main(matcher_conf, sfm_pairs, feature_conf['output'], run_dir)

    # The function returns the path of the file in which all the computed matches are stored.

    # ## Triangulate a new SfM model from the given poses
    # We triangulate the sparse 3D pointcloud given the matches and the
    # reference poses stored in the SIFT COLMAP model.

    stats = reconstruction.main(reference_sfm, images, sfm_pairs, feature_path, sfm_match_path)
    stats['avg_num_points'] = avg_num_points

    if run_localization:
        results_path = run_dir / f'{dname}_hloc_netvlad40.txt'  # the result file
        pairs = Path('pairs/') / dname
        loc_pairs = pairs / 'pairs-query-netvlad40.txt'  # top 40 retrieved by NetVLAD
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
            dataset / 'queries_with_intrinsics.txt',
            loc_pairs,
            feature_path,
            loc_match_path,
            results_path,
            covisibility_clustering=matcher_conf["covisibility_clustering"])  # not required with SuperPoint+SuperGlue
        stats["logs_path"] = logs_path

    return stats
'''

def run_test(base_dir, output_dir, feature_conf, matcher_conf, run_name, run_localization=False):
    # Fixed paths
    dname = "inloc"
    dataset = Path(base_dir) / "inloc"  # change this if your dataset is somewhere else
    sfm_pairs = output_dir / "inloc" / "pairs-retrieval.txt"
    loc_pairs = Path("pairs/inloc") / 'pairs-query-netvlad40.txt'  # top 40 retrieved by NetVLAD

    # Output specific
    run_dir = output_dir / f'{dname}/{run_name}'  # where everything will be saved
    run_dir.mkdir(exist_ok=True, parents=True)

    results_path = run_dir / f'{dname}_{run_name}_hloc_netvlad40.txt'  # the result file

    feature_path, avg_num_points = extract_features.main(feature_conf, dataset, run_dir, return_num_points=True)

    stats = {}
    print(f"Avg num points: {avg_num_points}")
    match_path = match_features.main(matcher_conf, loc_pairs, feature_conf['output'], run_dir)
    stats['avg_num_points'] = avg_num_points
    localize_inloc.main(dataset, loc_pairs, feature_path, match_path, results_path, skip_matches=20)  # skip database images with too few matches
    return stats
