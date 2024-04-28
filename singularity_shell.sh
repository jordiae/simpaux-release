#!/bin/bash

FULL_HOSTNAME=`hostname -f`

if [[ ${FULL_HOSTNAME} =~ blg.*\.int\.ets1\.calculquebec\.ca ]]; then
    module load singularity/3.4
elif [[ ${FULL_HOSTNAME} =~ .*\.server\.mila\.quebec ]]; then 
    module load singularity/3.4
fi

DATA_PATH=$SCRATCH/data/
mkdir -p $DATA_PATH

BINDS="-B `pwd`:/code \
    -B $SINGULARITY_HOMEDIR/simpaux:/home \
    -B $SCRATCH \
    -B $DATA_PATH:/data" 


if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    echo "shell without CUDA support"
    singularity shell $BINDS --pwd /code $SINGULARITY_IMGDIR/simpaux.simg
else
    echo "shell with CUDA support"
    singularity shell --nv $BINDS --pwd /code $SINGULARITY_IMGDIR/simpaux.simg
fi
