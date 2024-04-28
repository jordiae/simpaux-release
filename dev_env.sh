#!/bin/bash


if [ "$#" -eq 1 ]; then
    HOST=$1
else
    echo "Usage: $0 <host>"
    exit 1
fi

SESSIONNAME="simpaux_dev"

CODENAME="simpaux"

LOCALDIR=$HOME/workspace/$CODENAME
REMOTEDIR=scratch/code_sync/$CODENAME

VIMWINDOW=$SESSIONNAME:VIM
GITWINDOW=$SESSIONNAME:git
SSHWINDOW=$SESSIONNAME:ssh
SYNCWINDOW=$SESSIONNAME:autosync

SSH_CONTROL_PATH="$HOME/.ssh/ctl/%L-%r@%h:%p"

tmux has-session -t $SESSIONNAME &> /dev/null

if [ $? != 0 ]
then
    mkdir -p ~/.ssh/ctl
    
    ssh_reconnect $HOST

    tmux new-session -s $SESSIONNAME -n "VIM" -d
    tmux send-keys -t $VIMWINDOW "cd $LOCALDIR" C-m
    tmux send-keys -t $VIMWINDOW "vim" C-m

    tmux new-window -t $SESSIONNAME -n "autosync"
    tmux send-keys -t $SYNCWINDOW "bash start_autosync.sh $LOCALDIR $HOST $REMOTEDIR" C-m

    # wait for 3 sec to allow syncing, then ssh and cd to code directory
    tmux new-window -t $SESSIONNAME -n "ssh"
    tmux send-keys -t $SSHWINDOW "sleep 3;ssh -o "ControlPath=${SSH_CONTROL_PATH}" -t $HOST 'cd $REMOTEDIR; bash --login'" C-m
    tmux send-keys -t $SSHWINDOW C-l

    # git folder
    tmux new-window -t $SESSIONNAME -n "git"
    tmux send-keys -t $GITWINDOW "cd $LOCALDIR" C-m C-l

    tmux swap-window -s $SYNCWINDOW -t $GITWINDOW
    tmux select-window -t $VIMWINDOW
fi

tmux attach -t $SESSIONNAME
