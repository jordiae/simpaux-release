#!/bin/bash
##SBATCH --output=logs/simpaux_hunt_%j.out
##SBATCH --error=logs/simpaux_hunt_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=0-12:00:00
#SBATCH --mem=30Gb
#SBATCH --mail-type=ALL
#SBATCH --mail-user=michals@mila.quebec


export PYTHONUNBUFFERED=1

FULL_HOSTNAME=`hostname -f`

if [[ ${FULL_HOSTNAME} =~ blg.*\.int\.ets1\.calculquebec\.ca ]]; then
    echo "running on beluga"
    GATEWAY_ADDRESS=beluga1
elif [[ ${FULL_HOSTNAME} =~ cdr.*\.int\.cedar\.computecanada\.ca ]]; then
    echo "running on cedar"
    GATEWAY_ADDRESS=cedar1
elif [[ ${FULL_HOSTNAME} =~ .*\.server\.mila\.quebec ]]; then 
    echo "running on mila cluster"
    module purge
    module load anaconda/3 python/3.7/cuda/10.2/cudnn/7.6/pytorch/1.5.0
elif [[ ${FULL_HOSTNAME} =~ damaghooo2 ]]; then
    echo "running on laptop"
    #SLURM_TMPDIR=/tmp/slurm_tmpdir
    #mkdir $SLURM_TMPDIR
fi


if [ "$#" -eq 3 ]; then
    DATASET=$1
    SEED=$2
    EXP_ROOT_DIR=$3
else
    echo "Usage: $0 <dataset> <seed> <exp-root-dir>"
    echo " dataset: dataset name ({cub,miniimagenet,sun}-attributes)"
    echo " seed: seed for the random number generator (int)"
    echo " exp-root-dir: root dir of this experiment"
    exit 1
fi

EXP_WORKING_DIR=${EXP_ROOT_DIR}/${DATASET}_${SEED}
DATA_FOLDER=$SLURM_TMPDIR/data
ENV_DIR=$SLURM_TMPDIR/conda/simpaux
CODE_DIR=$SLURM_TMPDIR/code

mkdir -p $DATA_FOLDER
mkdir -p $EXP_WORKING_DIR
mkdir -p logs
mkdir -p $CODE_DIR

# FIXME: this should be in a separate script:
if [ ! -f $SLURM_TMPDIR/.envdone ]; then
    # setup conda env
    conda create --yes --force -p $ENV_DIR
    source $CONDA_ACTIVATE
    conda activate $ENV_DIR

    conda install --yes numpy scipy opencv matplotlib scikit-learn scikit-image \
        pandas tqdm imageio pytables h5py gitpython tensorboard
    conda install --yes -c conda-forge ipdb
    conda install --yes -c pytorch torchvision=0.6.1
    yes | $CONDA_PREFIX/bin/pip install torchmeta future fasttext
    yes | $CONDA_PREFIX/bin/pip install --no-cache-dir git+git://github.com/epistimio/orion.git@develop

    # copy assets not contained in torchmeta repo
    cp -r $ENV_DIR/lib/python3.7/site-packages/torchmeta/datasets/assets/cub \
        $ENV_DIR/lib/python3.7/site-packages/torchmeta/datasets/assets/cub-attributes
    cp -r protonet/assets/sun-attributes \
        $ENV_DIR/lib/python3.7/site-packages/torchmeta/datasets/assets/
    cp -r protonet/assets/miniimagenet-attributes \
        $ENV_DIR/lib/python3.7/site-packages/torchmeta/datasets/assets/

    # mark env preparation as done
    touch $SLURM_TMPDIR/.envdone
else
    source $CONDA_ACTIVATE
    conda activate $ENV_DIR
fi
cp -r * $CODE_DIR/

# copy data
rsync -avz $SCRATCH/data/$DATASET $SLURM_TMPDIR/data/
$CONDA_PREFIX/bin/python -u ${CODE_DIR}/prepare_data.py --dataset $DATASET \
    --data-folder $DATA_FOLDER

KWARGS=" --dropout .1 --aux-dropout .1 --bridge-dropout .1 \
    --num-cases-val 600 --early-stop -1 --num-test-shots 32 \
    --queries-sampling uniform --dataset $DATASET \
    --mode protonet-auxiliar-joint\
    --num-workers 4 --init-learning-rate 0.01 --eval-batch-size 25 \
    --aux-num-layers-per-block 2 --use-diff-clipping --seed $SEED"

if [ $DATASET == 'cub-attributes' ]; then
    KWARGS+=" --num-cases-test 600"
elif [ $DATASET == 'miniimagenet-attributes' ]; then
    KWARGS+=" --num-cases-test 5000"
fi

echo "experiment preparation done. Launching orion..."

(set -x; 
$CONDA_PREFIX/bin/python -u ${CODE_DIR}/train.py \
    --data-folder $DATA_FOLDER \
    --output-folder ${EXP_WORKING_DIR} \
    --download \
    $KWARGS \
    simpaux_${DATASET}_${SEED}
)
    
if [ ! -z $SLURM_JOB_ID ]; then
	sstat  -j   $SLURM_JOB_ID.batch   --format=JobID,MaxVMSize
fi

if [ ! -z $GATEWAY_ADDRESS ]; then
    echo "closing ssh tunnel"
    ssh -S $SLURM_TMPDIR/.ssh_session -O exit $GATEWAY_ADDRESS 
fi


#stdbuf -i0 -e0 -o0 python train.py test --download
