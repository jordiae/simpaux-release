#!/bin/bash
#SBATCH --job-name=simpaux_hunt
#SBATCH --output=logs/simpaux_hunt_%j.out
#SBATCH --error=logs/simpaux_hunt_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
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
elif [[ ${FULL_HOSTNAME} =~ .*\.server\.mila\.quebec ]] || [[ ${FULL_HOSTNAME} =~ mila0.*  ]]; then 
    echo "running on mila cluster"
    module purge
    module load anaconda/3 python/3.7/cuda/10.2/cudnn/7.6/pytorch/1.5.0
    #module load anaconda/3 python/3.7/cuda/11.0/cudnn/8.0/pytorch
elif [[ ${FULL_HOSTNAME} =~ damaghooo2 ]]; then
    echo "running on laptop"
    #SLURM_TMPDIR=/tmp/slurm_tmpdir
    #mkdir $SLURM_TMPDIR
fi


if [ "$#" -eq 3 ]; then
    DATASET=$1
    EXP_NAME=$2
    MAX_TRIALS=$3
else
    echo "Usage: $0 <dataset> <exp-name> <exp-max-trials>"
    echo " dataset: dataset name (cub)"
    echo " exp-name: name of the experiment"
    echo " exp-max-trials: max number of trials to be run by the worker"
    exit 1
fi

EXP_WORKING_DIR=$SCRATCH/train_dirs/simpaux/hunt/$EXP_NAME
DATA_FOLDER=$SLURM_TMPDIR/data
ENV_DIR=$SLURM_TMPDIR/conda/simpaux

mkdir -p $DATA_FOLDER
mkdir -p $EXP_WORKING_DIR
mkdir -p logs

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
    yes | $CONDA_PREFIX/bin/pip install --no-cache-dir torchmeta future fasttext einops pytorch-warmup
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

# copy data
rsync -avz $SCRATCH/data/$DATASET $SLURM_TMPDIR/data/
python prepare_data.py --dataset $DATASET --data-folder $DATA_FOLDER


if [ $DATASET == 'cub-attributes' ]; then
    NUM_CASES_TEST=600
elif [ $DATASET == 'miniimagenet-attributes' ]; then
    NUM_CASES_TEST=5000
fi

KWARGS=" --dropout .1 --aux-dropout .1 --bridge-dropout .1 \
    --num-cases-val 600 --num-cases-test 600 \
    --early-stop -1 --num-test-shots 32 --queries-sampling uniform \
    --dataset $DATASET --mode protonet-auxiliar-frozen \
    --aux-net-checkpoint $SCRATCH/simpaux/pretrained_aux_net/protonet_cub-attributes_5shot_5way__best.pt \
    --num-workers 4"

SEARCHSPACE=" --bridge-num-hid-features~uniform(128,512,discrete=True)"
SEARCHSPACE=" --bridge-num-hid-layers~uniform(1,3,discrete=True)"
SEARCHSPACE+=" --weight-decay~loguniform(1e-5,1e-2)"
SEARCHSPACE+=" --init-learning-rate~uniform(.05,.1)"
#SEARCHSPACE+=" --pwc-decay-interval~uniform(1000,4000,discrete=True)"
SEARCHSPACE+=" --num-batches~fidelity(256,2048,base=2)"
KWARGS+=" --learning-rate-schedule none --eval-interval-steps=-1"

echo "experiment preparation done. Launching orion..."

(set -x; 
$CONDA_PREFIX/bin/orion -vv hunt \
    -n ${EXP_NAME} \
    -c orion_config.yaml \
    --working-dir $EXP_WORKING_DIR \
    --worker-max-trials $MAX_TRIALS \
    train.py \
    --data-folder $DATA_FOLDER \
    --output-folder ${EXP_WORKING_DIR}/{trial.hash_params} \
    --download \
    $KWARGS \
    simpaux_${DATASET}_hpopt \
    $SEARCHSPACE
)
    
if [ ! -z $SLURM_JOB_ID ]; then
	sstat  -j   $SLURM_JOB_ID.batch   --format=JobID,MaxVMSize
fi

if [ ! -z $GATEWAY_ADDRESS ]; then
    echo "closing ssh tunnel"
    ssh -S $SLURM_TMPDIR/.ssh_session -O exit $GATEWAY_ADDRESS 
fi


#stdbuf -i0 -e0 -o0 python train.py test --download
