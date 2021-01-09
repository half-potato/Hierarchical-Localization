#!/usr/bin/env python
# coding: utf-8

from pathlib import Path
from hloc import colmap_from_nvm, pairs_from_covisibility
import argparse

def main(base_dir, output_dir):
    # ## Generate pairs for the SfM reconstruction
    # Instead of matching all database images exhaustively, we exploit the
    # existing SIFT model to find which image pairs are the most covisible. We
    # first convert the SIFT model from the NVM to the COLMAP format, and then
    # do a covisiblity search, selecting the top 20 most covisibile neighbors
    # for each image.
    dataset = Path(base_dir) / 'aachen/'  # change this if your dataset is somewhere else
    pairs = Path('pairs/aachen/')
    sfm_pairs = pairs / 'pairs-db-covis20.txt'  # top 20 most covisible in SIFT model

    colmap_from_nvm.main(
        dataset / '3D-models/aachen_cvpr2018_db.nvm',
        dataset / '3D-models/database_intrinsics.txt',
        dataset / 'aachen.db',
        output_dir / 'sfm_sift')

    pairs_from_covisibility.main(
        output_dir / 'sfm_sift', sfm_pairs, num_matched=20)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=Path, default='datasets',
                        help='Path to the dataset, default: %(default)s')
    parser.add_argument('outputs', type=Path, default='outputs',
                        help='Path to the output directory, default: %(default)s')
    args = parser.parse_args()
    main(args.dataset, args.outputs)
