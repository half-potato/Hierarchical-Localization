#!/bin/bash

docker run -it --gpus all -v /externd/datasets/2020VisualLocalization/RobotCar-Seasons:/app/datasets/robotcar -v /externd/output/:/app/outputs -v /externd/datasets/2020VisualLocalization/4Seasons:/app/datasets/4Seasons -v /externd/datasets/2020VisualLocalization/Aachen-Day-Night:/app/datasets/aachen -v /externd/datasets/2020VisualLocalization/InLoc:/app/datasets/inloc -p 8888:8888 hloc:latest /bin/bash -c "jupyter notebook --ip=0.0.0.0 --port=8888 --allow-root"
