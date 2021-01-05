import aachen_generate_pairs
from 4Seasons import prepare_pairs
from RobotCar import robotcar_to_colmap, robotcar_generate_query_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=Path, default='datasets/4Seasons',
                        help='Path to the dataset, default: %(default)s')
    parser.add_argument('--outputs', type=Path, default='outputs/4Seasons',
                        help='Path to the output directory, default: %(default)s')
    args = parser.parse_args()

    base_dir = Path(args.dataset)
    output_dir = Path(args.dataset)

    print("Preparing Aachen")
    aachen_generate_pairs.main(base_dir, output_dir)
    print("Preparing 4 Seasons")
    prepare_pairs.main(base_dir, output_dir)
    print("Preparing RobotCar")
    robotcar_dir = base_dir / "RobotCar"
    robotcar_to_colmap.main(
        robotcar_dir / "3D-models/all-merged/all.nvm",
        robotcar_dir / "3D-models/overcast-reference.db",
        output_dir / "robotcar" / "sfm_sift")
    pairs_from_covisibility.main(output_dir / "robotcar" / "sfm_sift", "pairs/robotcar-pairs-db-covis20.txt", 20)
    robotcar_generate_query_list.main(robotcar_dir, output_dir / "robotcar")
