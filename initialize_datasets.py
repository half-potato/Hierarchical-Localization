from pipelines import aachen_generate_pairs
from pipelines.FourSeasons import prepare_pairs
from pipelines.RobotCar import robotcar_to_colmap, robotcar_generate_query_list
from hloc import pairs_from_covisibility, pairs_from_retrieval, retrieval
import argparse
from pathlib import Path
import subprocess

def init_sfm(base_dir, dname, path_name=None):
    if path_name is None:
        path_name = dname
    print(f"Preparing {dname}")
    # Download dataset if it does not exist
    # Run image retrieval to generate pairs
    outdir = output_dir / path_name
    sfm_pairs = outdir / "pairs-retrieval.txt"
    desc_path = retrieval.main(
            retrieval.confs["openibl"],
            base_dir / path_name / "images",
            outdir,
            "openibl-gdesc-4096.h5")
    pairs_from_retrieval.main(desc_path, sfm_pairs, 40, db_prefix="", query_prefix="")
    print(f"Saved pairs to {sfm_pairs}")

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
    output_dir = Path(args.outputs)
    if args.dataset == "aachen":
        print("Preparing Aachen")
        aachen_generate_pairs.main(base_dir, output_dir)

    elif args.dataset == "4Seasons":
        print("Preparing 4Seasons")
        prepare_pairs.main(base_dir, output_dir)

    elif args.dataset == "RobotCar":
        print("Preparing RobotCar")
        name = "RobotCar"
        robotcar_dir = base_dir / name
        robotcar_to_colmap.main(
            robotcar_dir / "3D-models/all-merged/all.nvm",
            robotcar_dir / "3D-models/overcast-reference.db",
            output_dir / name / "sfm_sift")
        pairs_from_covisibility.main(output_dir / name / "sfm_sift", "pairs/robotcar-pairs-db-covis20.txt", 20)
        robotcar_generate_query_list.main(robotcar_dir, output_dir / name)

    elif args.dataset == "inloc":
        print("Preparing InLoc")
        # We need to generate the query list
        # The metadata was stripped from the images, but it seems that they were unedited from the iphone 7
        # the format for the metadata is:
        # camera_model, width, height, focal length, principal point (x, y), and extra stuff
        metadata = "SIMPLE_RADIAL 3024 4032 3225.60 3225.600000 1512.000000 2016.000000 0.000000"
        query_list_path = base_dir / "inloc" / "queries_with_intrinsics.txt"
        queries = list((base_dir / "inloc" / "iphone7").glob("*.JPG"))
        with query_list_path.open(mode='w') as f:
            for query in queries:
                f.write(f"{query.parent.name + '/' + query.name} {metadata}\n")
        # Run image retrieval to generate pairs
        outdir = output_dir / "inloc"
        sfm_pairs = outdir / "pairs-retrieval.txt"
        desc_path = retrieval.main(
                retrieval.confs["openibl"],
                base_dir / "inloc",
                outdir,
                "openibl-gdesc-4096.h5")
        pairs_from_retrieval.main(desc_path, sfm_pairs, 40, db_prefix="cutouts_imageonly/", query_prefix="cutouts_imageonly/")
        print(f"Saved pairs to {sfm_pairs}")

    elif args.dataset == "southbuilding":
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
        init_sfm(base_dir, args.dataset)
