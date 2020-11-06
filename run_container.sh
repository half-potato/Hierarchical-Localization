#!/bin/bash

docker run -it --gpus all -v /data/output/:/app/outputs -v /data/2020VisualLocalization/Aachen-Day-Night:/app/datasets/aachen -v /data/2020VisualLocalization/InLoc:/app/datasets/inloc -p 8888:8888 hloc:latest /bin/bash -c "jupyter notebook --ip=0.0.0.0 --port=8888 --allow-root"

