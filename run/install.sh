function install_dependencies() {
    export NCCL_IB_DISABLE=0 
    export NCCL_IB_HCA=$ARNOLD_RDMA_DEVICE:1 
    export NCCL_IB_GID_INDEX=3 
    export NCCL_SOCKET_IFNAME=eth0

    basedir=$1

    pip3 show torch-imputer 1>/dev/null
    if [ $? != 0 ]; then
        pip install --editable $basedir 
        pip install tensorboardX tensorflow ninja spicy

        # thirdparty repo
        mkdir -p $basedir/thirdparty

        git clone https://github.com/rosinality/imputer-pytorch $basedir/thirdparty/imputer-pytorch
        pip install $basedir/thirdparty/imputer-pytorch 

        git clone --recursive https://github.com/parlance/ctcdecode.git $basedir/thirdparty/ctcdecode 
        pip install $basedir/thirdparty/ctcdecode/
    fi
}

install "`pwd`/.."
