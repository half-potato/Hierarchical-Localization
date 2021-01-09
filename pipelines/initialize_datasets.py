import aachen_generate_pairs
from FourSeasons import prepare_pairs
from RobotCar import robotcar_to_colmap, robotcar_generate_query_list
from hloc import pairs_from_covisibility, pairs_from_retrieval, retrieval
import argparse
from pathlib import Path
import subprocess

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', type=Path, default='datasets',
                        help='Path to the dataset, default: %(default)s')
    parser.add_argument('--outputs', type=Path, default='outputs',
                        help='Path to the output directory, default: %(default)s')
    parser.add_argument('--dataset', type=str, default='aachen',
                        help='The dataset to initialize')
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    output_dir = Path(args.dataset)
    if args.dataset == "aachen":
        print("Preparing Aachen")
        aachen_generate_pairs.main(base_dir, output_dir)
    elif args.dataset == "aachen":
        print("Preparing 4 Seasons")
        prepare_pairs.main(base_dir, output_dir)
    elif args.dataset == "aachen":
        print("Preparing RobotCar")
        robotcar_dir = base_dir / "RobotCar"
        robotcar_to_colmap.main(
            robotcar_dir / "3D-models/all-merged/all.nvm",
            robotcar_dir / "3D-models/overcast-reference.db",
            output_dir / "robotcar" / "sfm_sift")
        pairs_from_covisibility.main(output_dir / "robotcar" / "sfm_sift", "pairs/robotcar-pairs-db-covis20.txt", 20)
        robotcar_generate_query_list.main(robotcar_dir, output_dir / "robotcar")
    elif args.dataset == "inloc":
        print("Preparing InLoc")
    elif args.dataset == "SouthBuilding":
        print("Preparing SouthBuilding")
        # Download dataset if it does not exist
        if not (base_dir / "South-Building").exists():
            if not (base_dir / "South-Building.zip").exists():
                process = subprocess.Popen(
                        f"wget http://cvg.ethz.ch/research/local-feature-evaluation/South-Building.zip -P {base_dir}".split(),
                        stdout=subprocess.PIPE)
                output, error = process.communicate()
            process = subprocess.Popen(
                    f"unzip datasets/South-Building.zip -d {base_dir}".split(),
                    stdout=subprocess.PIPE)
            output, error = process.communicate()
        # Run image retrieval to generate pairs
        outdir = output_dir / "South-Building"
        sfm_pairs = outdir / "pairs-retrieval.txt"
        desc_path = retrieval.main(
                retrieval.confs["openibl"],
                base_dir / "South-Building" / "images",
                outdir,
                "openibl-gdesc-4096.h5")
        pairs_from_retrieval.main(desc_path, sfm_pairs, 40, db_prefix="P", query_prefix="P")
        print(f"Saved pairs to {sfm_pairs}")
    else:
        print("Dataset not found")
