#!/bin/bash

LOCALDIR=$1
HOST=$2
REMOTEDIR=$3

SSH_CONTROL_PATH="$HOME/.ssh/ctl/%L-%r@%h:%p"

while 
    sleep 2
    rsync -e "ssh -o \"ControlPath=${SSH_CONTROL_PATH}\"" -avz --del \
        --exclude='.git*/' --exclude='start_autosync.sh' --exclude='__pycache__/' \
        --exclude='output' --exclude='logs' --include='*/' \
        --include='*.py' --include='*.sh' --include='*.yml' \
        --include='*.dat' --include='*.json' --include='*.txt' \
        --include='*.yaml' --exclude='*' \
        $LOCALDIR/ $HOST:$REMOTEDIR;

    inotifywait -r -e modify,create,delete --exclude ".*(\.git|start_autosync\.sh|.*\.swp|l2m_dev\.sh|results|train_dir,__pycache__)" $LOCALDIR; do :;
done

#ssh -O exit -o "ControlPath=${SSH_CONTROL_PATH}" $HOST

#while inotifywait -r $LOCALDIR/*; do
#    rsync -avz --del --exclude='.git/' --exclude='start_autosync.sh*' \
#        --exclude='*.swp' \
#        $LOCALDIR/ $REMOTEDIR
#done
