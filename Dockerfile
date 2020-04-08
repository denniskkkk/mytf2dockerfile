ARG UBUNTU_VERSION=18.04

ARG ARCH=
ARG CUDA=10.1
FROM nvidia/cuda${ARCH:+-$ARCH}:${CUDA}-base-ubuntu${UBUNTU_VERSION} as base
# ARCH and CUDA are specified again because the FROM directive resets ARGs
# (but their default value is retained if set previously)
ARG ARCH
ARG CUDA
ARG CUDNN=7.6.4.38-1
ARG CUDNN_MAJOR_VERSION=7
ARG LIB_DIR_PREFIX=x86_64
ARG LIBNVINFER=6.0.1-1
ARG LIBNVINFER_MAJOR_VERSION=6

USER root

RUN groupadd --gid 1000 notebook && useradd -m --uid 1000 --gid notebook  notebook -d /tf

# Needed for string substitution
SHELL ["/bin/bash", "-c"]
# Pick up some TF dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cuda-command-line-tools-${CUDA/./-} \
        # There appears to be a regression in libcublas10=10.2.2.89-1 which
        # prevents cublas from initializing in TF. See
        # https://github.com/tensorflow/tensorflow/issues/9489#issuecomment-562394257
        libcublas10=10.2.1.243-1 \ 
        cuda-nvrtc-${CUDA/./-} \
        cuda-cufft-${CUDA/./-} \
        cuda-curand-${CUDA/./-} \
        cuda-cusolver-${CUDA/./-} \
        cuda-cusparse-${CUDA/./-} \
        curl \
        libcudnn7=${CUDNN}+cuda${CUDA} \
        libfreetype6-dev \
        libhdf5-serial-dev \
        libzmq3-dev \
        pkg-config \
        software-properties-common \
        unzip \
        fontconfig libaacs0 libasound2 libasound2-data libass9 \
        libasyncns0 libavcodec57 libavdevice57 libavfilter6 \
        libavformat57 libavresample3 libavutil55 libbdplus0 libbluray2 libbs2b0 \
        libcaca0 libcairo2 libcdio-cdda2 libcdio-paranoia2 libcdio17 libchromaprint1 \
        libcroco3 libdatrie1 libdc1394-22 \
        libfftw3-double3 libflac8 libflite1 \
        libfribidi0 libgdk-pixbuf2.0-0 libgdk-pixbuf2.0-bin libgdk-pixbuf2.0-common \
        libgl1 libgl1-mesa-dri libglx-mesa0 libglx0 libgme0 libgsm1 libiec61883-0 \
        libjbig0 libllvm6.0 libmp3lame0 libmpg123-0 libmysofa0 \
        libnuma1 libogg0 libopenal-data libopenal1 libopenjp2-7 libopenmpt0 libopus0 \
        libpango-1.0-0 libpangocairo-1.0-0 libpangoft2-1.0-0 libpciaccess0 \
        libpixman-1-0 libpostproc54 libpulse0 librsvg2-2 \
        librsvg2-common librubberband2 libsamplerate0 libsdl2-2.0-0 libsensors4 \
        libshine3 libslang2 libsnappy1v5 libsndfile1 libsndio6.1 libsoxr0 libspeex1 \
        libssh-gcrypt-4 libswresample2 libswscale4 libthai-data libthai0 libtheora0 \
        libtiff5 libtwolame0 libva-drm2 libva-x11-2 libva2 libvdpau1 \
        libvorbis0a libvorbisenc2 libvorbisfile3 libvpx5 libwavpack1 \
        libwayland-cursor0 libwayland-egl1-mesa libwebp6 libwebpmux3 libwrap0 \
        libx264-152 libx265-146 libxcb-glx0 libxcb-render0 libxcb-shape0 libxcb-shm0 \
        libxcursor1 libxdamage1 libxfixes3 libxi6 libxinerama1 libxkbcommon0 \
        libxrandr2 libxv1 libxvidcore4 libxxf86vm1 libzvbi-common libzvbi0 \
        mesa-va-drivers xkb-data \
        ffmpeg libasound2-plugins  \
        libbluray-bdj libfftw3-bin libfftw3-dev  \
        libportaudio2 librsvg2-bin \
        sndiod speex 

# Install TensorRT if not building for PowerPC
RUN [[ "${ARCH}" = "ppc64le" ]] || { apt-get update && \
        apt-get install -y --no-install-recommends libnvinfer${LIBNVINFER_MAJOR_VERSION}=${LIBNVINFER}+cuda${CUDA} \
        libnvinfer-plugin${LIBNVINFER_MAJOR_VERSION}=${LIBNVINFER}+cuda${CUDA} \
        && apt-get clean \
        && rm -rf /var/lib/apt/lists/*; }

# For CUDA profiling, TensorFlow requires CUPTI.
ENV LD_LIBRARY_PATH /usr/local/cuda/extras/CUPTI/lib64:/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Link the libcuda stub to the location where tensorflow is searching for it and reconfigure
# dynamic linker run-time bindings
RUN ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1 \
    && echo "/usr/local/cuda/lib64/stubs" > /etc/ld.so.conf.d/z-cuda-stubs.conf \
    && ldconfig

# See http://bugs.python.org/issue19846
ENV LANG C.UTF-8

RUN apt-get update && apt-get install -y \
    python3  \
    python3-pip

RUN python3 -m pip install --upgrade pip setuptools

# Some TF tools expect a "python" binary
RUN ln -s $(which python3) /usr/local/bin/python

# Options:
#   tensorflow
#   tensorflow-gpu
#   tf-nightly
#   tf-nightly-gpu
# Set --build-arg TF_PACKAGE_VERSION=1.11.0rc0 to install a specific version.
# Installs the latest version by default.
ARG TF_PACKAGE=tensorflow
ARG TF_PACKAGE_VERSION=
RUN pip install --upgrade pip && pip install tensorflow-gpu==2.1.0 keras pydot gensim theano word2vec pandas keras-word-char-embd word-embedder sklearn virtualenv eli5 keras.models cairocffi editdistance argparse matplotlib extra-keras-datasets librosa 

COPY bashrc /etc/bash.bashrc
RUN chmod a+rwx /etc/bash.bashrc

RUN python3 -m pip install jupyter matplotlib
# Pin ipykernel and nbformat; see https://github.com/ipython/ipykernel/issues/422
RUN python3 -m pip install jupyter_http_over_ws ipykernel==5.1.1 nbformat==4.4.0
RUN jupyter serverextension enable --py jupyter_http_over_ws

RUN mkdir -p /tf/tensorflow-tutorials && chmod -R a+rwx /tf/
RUN mkdir /.local && chmod a+rwx /.local
RUN apt-get install -y --no-install-recommends wget
# some examples require git to fetch dependencies
RUN apt-get install -y --no-install-recommends git
WORKDIR /tf/tensorflow-tutorials
RUN wget https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/keras/classification.ipynb
RUN wget https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/keras/overfit_and_underfit.ipynb
RUN wget https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/keras/regression.ipynb
RUN wget https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/keras/save_and_load.ipynb
RUN wget https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/keras/text_classification.ipynb
RUN wget https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/keras/text_classification_with_hub.ipynb
#COPY readme-for-jupyter.md README.md
RUN apt-get autoremove -y && apt-get remove -y wget
WORKDIR /tf

EXPOSE 8888
EXPOSE 6006

RUN python3 -m ipykernel.kernelspec

USER notebook
CMD ["bash", "-c", "source /etc/bash.bashrc && jupyter notebook --notebook-dir=/tf --ip 0.0.0.0 --no-browser --allow-root"]
