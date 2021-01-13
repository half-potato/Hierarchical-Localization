FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04

# Prepare and empty machine for building
RUN apt-get update && apt-get install -y \
    git \
    cmake \
    vim \
    build-essential \
    libboost-program-options-dev \
    libboost-filesystem-dev \
    libboost-graph-dev \
    libboost-system-dev \
    libboost-test-dev \
    libeigen3-dev \
    libsuitesparse-dev \
    libfreeimage-dev \
    libgoogle-glog-dev \
    libgflags-dev \
    libglew-dev \
    qtbase5-dev \
    libqt5opengl5-dev \
    libcgal-dev \
    libcgal-qt5-dev \
    python3 python3-pip unzip wget

# Build and install ceres solver
RUN apt-get -y install \
    libatlas-base-dev \
    libsuitesparse-dev
RUN git clone https://github.com/ceres-solver/ceres-solver.git --branch 1.14.0
RUN cd ceres-solver && \
	mkdir build && \
	cd build && \
	cmake .. -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=OFF && \
	make -j4 && \
	make install

# Build and install COLMAP

# Note: This Dockerfile has been tested using COLMAP pre-release 3.6.
# Later versions of COLMAP (which will be automatically cloned as default) may
# have problems using the environment described thus far. If you encounter
# problems and want to install the tested release, then uncomment the branch
# specification in the line below
RUN git clone https://github.com/colmap/colmap.git #--branch 3.6

RUN cd colmap && \
	git checkout dev && \
	mkdir build && \
	cd build && \
	cmake .. && \
	make -j4 && \
	make install

# build opencv with contrib
# latest version currently has a segfault in sift
RUN pip3 install numpy
RUN cd / \
    && wget https://github.com/opencv/opencv/archive/4.5.1.tar.gz -O opencv.tar.gz \
    && wget https://github.com/opencv/opencv_contrib/archive/4.5.1.tar.gz -O opencv_contrib.tar.gz \
    && tar -xzf opencv.tar.gz && mv opencv-4.5.1 opencv \
    && tar -xzf opencv_contrib.tar.gz && mv opencv_contrib-4.5.1 opencv_contrib \
    && mkdir opencv/build && cd opencv/build \
    && cmake -DOPENCV_ENABLE_NONFREE=ON \
             -DOPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
             -DBUILD_opencv_alphamat=OFF \
             -DBUILD_opencv_aruco=OFF \
             -DBUILD_opencv_bgsegm=OFF \
             -DBUILD_opencv_bioinspired=OFF \
             -DBUILD_opencv_ccalib=OFF \
             -DBUILD_opencv_cnn_3dobj=OFF \
             -DBUILD_opencv_cvv=OFF \
             -DBUILD_opencv_datasets=OFF \
             -DBUILD_opencv_dnn_objdetect=OFF \
             -DBUILD_opencv_dnn_superres=OFF \
             -DBUILD_opencv_dnns_easily_fooled=OFF \
             -DBUILD_opencv_dpm=OFF \
             -DBUILD_opencv_face=OFF \
             -DBUILD_opencv_freetype=OFF \
             -DBUILD_opencv_fuzzy=OFF \
             -DBUILD_opencv_hdf=OFF \
             -DBUILD_opencv_julia=OFF \
             -DBUILD_opencv_line_descriptor=OFF \
             -DBUILD_opencv_matlab=OFF \
             -DBUILD_opencv_mcc=OFF \
             -DBUILD_opencv_optflow=OFF \
             -DBUILD_opencv_ovis=OFF \
             -DBUILD_opencv_plot=OFF \
             -DBUILD_opencv_reg=OFF \
             -DBUILD_opencv_rgbd=OFF \
             -DBUILD_opencv_saliency=OFF \
             -DBUILD_opencv_sfm=OFF \
             -DBUILD_opencv_stereo=OFF \
             -DBUILD_opencv_structured_light=OFF \
             -DBUILD_opencv_surface_matching=OFF \
             -DBUILD_opencv_text=OFF \
             -DBUILD_opencv_tracking=OFF \
             -DBUILD_opencv_xfeatures2d=ON \
             -DBUILD_opencv_ximgproc=OFF \
             -DBUILD_opencv_xobjdetect=OFF \
             -DBUILD_opencv_xphoto=OFF \
              .. \
    && make -j16 \
    && make install

RUN pip3 install scikit-build
RUN pip3 install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
# RUN pip3 install opencv-python
COPY . /app
WORKDIR app/
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt
RUN pip3 install jupyterlab notebook
RUN pip3 install git+https://github.com/mihaidusmanu/pycolmap
