from pathlib import Path
import argparse

from hloc import extract_features, match_features
from hloc import pairs_from_poses, triangulation

from .localize import relocalization_files
from . import utils

def main(dataset_dir, output_dir):

    ref_dir = dataset_dir / '4Seasons' / 'reference'
    assert ref_dir.exists(), f'{ref_dir} does not exist'
    ref_images = ref_dir / 'undistorted_images'

    output_dir.mkdir(exist_ok=True, parents=True)
    ref_sfm_empty = output_dir / 'sfm_reference_empty'
    ref_sfm = output_dir / 'sfm_superpoint+superglue'

    # Setup pair dirs
    num_ref_pairs = 20
    num_loc_pairs = 10
    ref_pairs = output_dir / f'pairs-db-dist{num_ref_pairs}.txt'

    # Only reference images that have a pose are used in the pipeline.
    # To save time in feature extraction, we delete unsused images.
    utils.delete_unused_images(ref_images, utils.get_timestamps(ref_dir / 'poses.txt', 0))

    # Build an empty COLMAP model containing only camera and images
    # from the provided poses and intrinsics.
    utils.build_empty_colmap_model(ref_dir, ref_sfm_empty)

    # Match reference images that are spatially close.
    pairs_from_poses.main(ref_sfm_empty, ref_pairs, num_ref_pairs)

    for sequence in ['training', 'validation', 'test0', 'test1']:
        query_list = output_dir / f'{sequence}_queries_with_intrinsics.txt'
        loc_pairs = output_dir / f'pairs-query-{sequence}-dist{num_loc_pairs}.txt'
        # Generate a list of query images with their intrinsics.
        reloc = ref_dir / relocalization_files[sequence]

        seq_dir = dataset_dir / "4Seasons" / sequence
        assert seq_dir.exists(), f'{seq_dir} does not exist'
        seq_images = seq_dir / 'undistorted_images'

        timestamps = utils.get_timestamps(reloc, 1)
        utils.delete_unused_images(seq_images, timestamps)

        utils.generate_query_lists(timestamps, seq_dir, query_list)

        # Generate the localization pairs from the given reference frames.
        utils.generate_localization_pairs(
            sequence, reloc, num_loc_pairs, ref_pairs, loc_pairs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=Path, default='datasets',
                        help='Path to the dataset, default: %(default)s')
    parser.add_argument('--outputs', type=Path, default='outputs',
                        help='Path to the output directory, default: %(default)s')
    args = parser.parse_args()
    main(args.dataset, args.outputs)
