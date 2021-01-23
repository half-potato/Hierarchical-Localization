import json
import csv
import argparse
import pipelines
from pathlib import Path

columns = [
    ["stats", "num_reg_images"],
    ["stats", "num_sparse_points"],
    ["stats", "num_observations"],
    ["stats", "mean_track_length"],
    ["stats", "num_observations_per_image"],
    ["stats", "mean_reproj_error"],
    ["stats", "num_input_images"],
    ["stats", "avg_num_points"],
]

parser = argparse.ArgumentParser()
parser.add_argument("output_dir", type=Path, help='Path to directory to store results')
parser.add_argument("pipeline_name", type=str, help='Name of pipeline to aggregate results on')
parser.add_argument("filename", type=Path, help='Name of output')

args = parser.parse_args()
args.output_dir
out_path = pipelines.gen_out_path(args.output_dir, args.pipeline_name)
results = []
# add column names
header = [column[-1] for column in columns]
header.insert(0, "pipeline")
header.insert(1, "method")
results.append(", ".join(header))

for path in out_path.iterdir():
    if path.suffix != ".json":
        continue
    with path.open(mode="r") as f:
        data = json.load(f)
    row = [args.pipeline_name, str(path.stem)]
    for column in columns:
        d = data
        for p in column:
            if p not in d:
                d = -1
                break
            d = d[p]
        row.append(str(d))
    results.append(", ".join(row))

with args.filename.open(mode="w") as f:
    f.write("\n".join(results))
