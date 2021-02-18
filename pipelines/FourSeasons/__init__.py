
from pathlib import Path
import logging
import argparse

from hloc import extract_features, match_features, localize_sfm, triangulation

from .localize import relocalization_files
from . import prepare_reference

from .utils import get_timestamps, delete_unused_images
from .utils import generate_query_lists, generate_localization_pairs
from .utils import prepare_submission, evaluate_submission

def run_test(base_dir, output_dir, feature_conf, matcher_conf, run_name, run_localization=False, skip_reconstruction=False):

    # Setup path to dataset
    ref_dir = base_dir / "4Seasons" / 'reference'
    assert ref_dir.exists(), f'{ref_dir} does not exist'
    ref_images = ref_dir / 'undistorted_images'

    # Setup dirs for output sfm and localization files
    output_dir.mkdir(exist_ok=True, parents=True)
    ref_sfm_empty = output_dir / 'sfm_reference_empty'
    run_dir = Path(f'outputs/4Seasons/{run_name}')  # where everything will be saved
    run_dir.mkdir(exist_ok=True, parents=True)
    ref_sfm = run_dir / f'sfm'  # the SfM model we will build

    # Setup pair dirs
    num_ref_pairs = 20
    num_loc_pairs = 10
    ref_pairs = output_dir / f'pairs-db-dist{num_ref_pairs}.txt'

    # Extract, match, and triangulate the reference SfM model.
    ffile, avg_num_points = extract_features.main(feature_conf, ref_images, run_dir, return_num_points=True)
    print(f"Avg num points: {avg_num_points}")
    mfile = match_features.main(matcher_conf, ref_pairs, feature_conf['output'], run_dir)
    if not skip_reconstruction:
        stats = triangulation.main(ref_sfm, ref_sfm_empty, ref_images, ref_pairs, ffile, mfile)
    else:
        stats = {}
    stats['avg_num_points'] = avg_num_points

    if run_localization:
        logs_paths = []
        for sequence in ['training', 'validation', 'test0', 'test1'][2:]:
            seq_dir = base_dir / "4Seasons" / sequence
            assert seq_dir.exists(), f'{seq_dir} does not exist'
            seq_images = seq_dir / 'undistorted_images'

            results_path = run_dir / f'{sequence}_hloc.txt'  # the result file
            query_list = output_dir / f'{sequence}_queries_with_intrinsics.txt'
            loc_pairs = output_dir / f'pairs-query-{sequence}-dist{num_loc_pairs}.txt'

            # Extract, match, amd localize.
            ffile = extract_features.main(feature_conf, seq_images, run_dir)
            mfile = match_features.main(matcher_conf, loc_pairs, feature_conf['output'], run_dir)
            logs_path = localize_sfm.main(
                ref_sfm / 'model', query_list, loc_pairs, ffile, mfile, results_path,
                covisibility_clustering=matcher_conf["covisibility_clustering"])
            logs_paths.append(logs_path)

            """
            reloc = ref_dir / relocalization_files[sequence]
            submission_dir = output_dir / 'submission_hloc+superglue'
            # Convert the absolute poses to relative poses with the reference frames.
            submission_dir.mkdir(exist_ok=True)
            prepare_submission(results_path, reloc, ref_dir / 'poses.txt', submission_dir)

            # If not a test sequence: evaluation the localization accuracy
            if 'test' not in sequence:
                logging.info('Evaluating the relocalization submission...')
                evaluate_submission(submission_dir, reloc)
            """

        stats["logs_path"] = logs_paths

    return stats
