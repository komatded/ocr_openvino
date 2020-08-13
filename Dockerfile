FROM ubuntu:18.04

# ---------OPENVINO---------
#ARG DOWNLOAD_LINK=http://registrationcenter-download.intel.com/akdlm/irc_nas/16345/l_openvino_toolkit_p_2020.2.120.tgz
ARG DOWNLOAD_LINK=http://registrationcenter-download.intel.com/akdlm/irc_nas/16612/l_openvino_toolkit_p_2020.2.120.tgz
ARG INSTALL_DIR=/opt/intel/openvino
ARG TEMP_DIR=/tmp/openvino_installer
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    cpio \
    sudo \
    lsb-release && \
    rm -rf /var/lib/apt/lists/*
RUN mkdir -p $TEMP_DIR && cd $TEMP_DIR && \
    wget -c $DOWNLOAD_LINK && \
    tar xf l_openvino_toolkit*.tgz && \
    cd l_openvino_toolkit* && \
    sed -i 's/decline/accept/g' silent.cfg && \
    ./install.sh -s silent.cfg && \
    rm -rf $TEMP_DIR

RUN $INSTALL_DIR/install_dependencies/install_openvino_dependencies.sh
RUN /bin/bash -c "source $INSTALL_DIR/bin/setupvars.sh"
RUN /opt/intel/openvino_2020.2.120/deployment_tools/model_optimizer/install_prerequisites/install_prerequisites.sh
RUN /opt/intel/openvino_2020.2.120/deployment_tools/demo/demo_squeezenet_download_convert_run.sh

ENV PYTHONPATH "/opt/intel/openvino_2020.2.120/python/python3.6:/opt/intel/openvino_2020.2.120/python/python3:/opt/intel/openvino_2020.2.120/deployment_tools/open_model_zoo/tools/accuracy_checker:/opt/intel/openvino_2020.2.120/deployment_tools/model_optimizer"
ENV LD_LIBRARY_PATH "/opt/intel/openvino_2020.2.120/opencv/lib:/opt/intel/openvino_2020.2.120/deployment_tools/ngraph/lib:/opt/intel/opencl:/opt/intel/openvino_2020.2.120/deployment_tools/inference_engine/external/hddl/lib:/opt/intel/openvino_2020.2.120/deployment_tools/inference_engine/external/gna/lib:/opt/intel/openvino_2020.2.120/deployment_tools/inference_engine/external/mkltiny_lnx/lib:/opt/intel/openvino_2020.2.120/deployment_tools/inference_engine/external/tbb/lib:/opt/intel/openvino_2020.2.120/deployment_tools/inference_engine/lib/intel64:"

# ---------OPENVINO---------

ENV DEBIAN_FRONTEND noninteractive.

RUN apt-get update && apt-get install -y git swig
RUN apt-get upgrade -y && apt-get install -y language-pack-ru

ENV LANGUAGE ru_RU.UTF-8
ENV LANG ru_RU.UTF-8
ENV LC_ALL ru_RU.UTF-8
RUN locale-gen en_US en_US.UTF-8 && \
    locale-gen ru_RU ru_RU.UTF-8 && \
    dpkg-reconfigure locales && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
RUN apt-get update -y
RUN apt-get upgrade -y

RUN pip3 install --no-cache-dir pytest-runner \
    && pip3 install --no-cache-dir -r requirements.txt \
    && rm -rf /tmp/*

COPY ./src ./src
COPY ./resources ./resources

ENTRYPOINT python3 ./src/main.py 8000
#ENTRYPOINT ["/bin/bash"]
