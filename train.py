#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
from orion.client import report_results
import torch
import time
import logging
from os.path import dirname as up
from protonet.trainer import ProtonetTrainer, AuxiliarTrainer, ProtonetWithAuxiliarNetworkJointTrainer,\
    ProtonetWithAuxiliarNetworkFrozenTrainer, AuxiliarEmbeddingsTrainer


def build_args(dir_):
    import argparse

    def parse_str_list(s):
        if s == '' or s is None:
            return []
        return s.split(',')

    def parse_int_list(s):
        if s == '' or s is None:
            return []
        return list(int(n) for n in s.split(','))

    parser = argparse.ArgumentParser(
        'SimpAux',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('name', type=str, help='Experiment name')
    parser.add_argument('--seed', type=int, default=1, help='Random seed')
    parser.add_argument('--data-folder', type=str, default=os.path.join(dir_, 'data'),
                        help='Path to the folder the data is downloaded to (default: data/)')
    parser.add_argument('--num-shots', type=int, default=5,
                        help='Number of examples per class (k in "k-shot", default: 5, SUPPORT).')
    parser.add_argument('--num-test-shots', type=int, default=5,
                        help='Number of QUERY samples (default: 5).')
    parser.add_argument('--num-ways', type=int, default=5,
                        help='Number of classes per task (N in "N-way", default: 5).')
    parser.add_argument('--output-folder', type=str, default=os.path.join(dir_, 'output'),
                        help='Path to the output folder for saving the model (default: output/).')
    parser.add_argument('--batch-size', type=int, default=2,
                        help='Number of tasks in a mini-batch of tasks (default: 2).')
    parser.add_argument('--num-batches', type=int, default=30000,
                        help='Number of batches the prototypical network is trained over (default: 100).')
    parser.add_argument('--num-workers', type=int, default=0,
                        help='Number of workers for data loading (default: 0).')
    parser.add_argument('--dataset', type=str, default="miniimagenet",
                        choices=('miniimagenet', 'omniglot', 'cub', 'cub-attributes', 'sun-attributes',
                                 'miniimagenet-attributes'),
                        help="Which dataset to train on")
    parser.add_argument('--download', action='store_true',
                        help='Download the dataset in the data folder.')
    parser.add_argument('--disable-cuda', dest='use_cuda', action='store_false',
                        help='Use CUDA if available.')
    parser.add_argument('--eval-interval-steps', type=int, default=1000,
                        help="steps between evaluations (set to -1 to eval at end only)")
    parser.add_argument('--num-cases-test', type=int, default=50000,
                        help="how many tasks to sample during evaluation")
    parser.add_argument('--num-cases-val', type=int, default=5000,
                        help="how many tasks to sample for early stopping")
    parser.add_argument('--num-layers-per-block', type=int, default=3,
                        help="number of conv layers per residual block")
    parser.add_argument('--num-blocks', type=int, default=4,
                        help="number of residual blocks")
    parser.add_argument('--num-channels', type=int, default=64,
                        help="number of channels in the first residual block")
    parser.add_argument('--num-channels-growth-factor', type=int, default=2,
                        help="factor by which to grow the number of channels from one residual block to the next.")
    parser.add_argument('--max-pool-kernel-size', type=int, default=2,
                        help="size max pooling kernel at the end of each residual block")
    parser.add_argument('--eval-batch-size', type=int, default=50,
                        help="batchsize during evaluation")
    parser.add_argument('--init-learning-rate', type=float, default=0.1,
                        help="initial learning rate")
    parser.add_argument('--learning-rate-schedule', type=str, default="pwc",
                        choices=['pwc', 'reduce-on-plateau', 'none'],
                        help="what kind of schedule to use for the learning rate")
    parser.add_argument('--pwc-decay-interval', type=int,
                        default=2500, help="how many steps between decay steps")
    parser.add_argument('--grad-clip-norm', type=float, default=1.,
                        help="gradient clipping norm threshold (protonet)")
    parser.add_argument('--weight-decay', type=float, default=0.0005,
                        help="weight decay regularization")
    parser.add_argument('--weight-init-variance', type=float, default=0.1,
                        help="variance of normally distributed random weight init, ignored if not --custom-init")
    parser.add_argument('--num-max-pools', type=int, default=3,
                        help="number of blocks having a max pooling layer")
    parser.add_argument('--debug-log-interval', type=int, default=100,
                        help="number of steps between debug log dumps")
    parser.add_argument('--no-bar', action='store_true', help='Disables progress bar')
    parser.add_argument('--early-stop', type=int,
                        help='Patience in early stop in validation set (-1 -> no early stop)', default=-1)
    parser.add_argument('--activation', type=str, default='swish-1', choices=['relu', 'selu', 'swish-1'],
                        help='Activation function')
    parser.add_argument('--custom-init', action='store_true',
                        help="If activated, use PyTorch's default weight initialization scheme")
    parser.add_argument('--train-mode', action='store_true',
                        help="If activated, train mode when evaluating")
    parser.add_argument('--queries-sampling', type=str, default='per-class', choices=['per-class', 'uniform'],
                        help='If --queries-sampling per-class (default), then --num-test-shots is the number of queries'
                             'per class. If --queries-sampling uniform, then --num-test-shots is the total number of'
                             'queries ')
    parser.add_argument('--momentum', type=float, help='Momentum', default=0.9)
    parser.add_argument('--no-batchnorm', dest='batchnorm', action='store_false',
                        help='Disable batch normalization.')
    parser.add_argument('--augment', action='store_true',
                        help='Use data augmentation.')
    parser.add_argument('--dropout', type=float, default=0.0,
                        help="Use dropout (notice that we only have convolutional layers)")
    parser.add_argument('--kernel-size', type=int, default=3,
                        help="Kernel size")
    parser.add_argument('--mode', type=str, default='protonet', choices=['protonet', 'auxiliar',
                                                                         'protonet-auxiliar-joint',
                                                                         'protonet-auxiliar-frozen',
                                                                         'auxiliar-embeddings'],
                        help='Whether to train just the main network (protonet, default option), only the auxiliar'
                             'network, the main network using a pretrained auxiliar network, or training both of them'
                             'jointly.')
    parser.add_argument('--n-extra-linear', type=int, default=1,
                        help="Extra linear layers (apart from the final one) in the auxiliary network (default 1)")
    parser.add_argument('--aux-dropout', type=float, default=0.0, help="dropout in the auxiliary network.")
    parser.add_argument('--bridge-num-hid-features', type=int, default=128,
                        help="Number of hidden units in each layer of the bridge network.")
    parser.add_argument('--bridge-num-hid-layers', type=int, default=1,
                        help="Number of hidden layers in the bridge network.")
    parser.add_argument('--bridge-dropout', type=float, default=0.0, help="dropout in the bridge network.")
    parser.add_argument('--bridge-input-aux-layers', type=parse_int_list, default=[], help="which aux net blocks' features to feed to the bridge")
    parser.add_argument('--aux-net-checkpoint', type=str, help="Checkpoint of a pretrained auxiliary network to load.")
    parser.add_argument('--aux-num-channels', type=int, default=64,
                        help="number of channels in the first residual block of the auxiliary network")
    parser.add_argument('--aux-loss-fn', type=str, default='mlsm', choices=['mlsm', 'bce', 'weighted-bce', 'macro-soft-f1'],
                        help='Loss function in auxiliary network (default mlsm, multi-label soft.argin, vs bce vs'
                             'weighted-bce)')
    parser.add_argument('--aux-loss-ignore', type=float, default=0.8,
                        help='% of attributes to ignore in auxiliary network loss function (default 0.8, ignored if'
                             'weighted-bce)')
    parser.add_argument('--aux-loss-coeff', type=float, default=1.0,
                        help="Coefficient for the auxiliary loss (used in joint training)")
    parser.add_argument('--debug-opts', type=parse_str_list, default=[],
                        help="comma-separated list of debug flags (valid flags are 'anomaly', 'grads', 'states',"
                             "'belief')")
    parser.add_argument('--aux-num-layers-per-block', type=int, default=3,
                        help="number of conv layers per residual block in the aux net")
    parser.add_argument('--aux-num-blocks', type=int, default=4,
                        help="number of residual blocks in the aux net")
    parser.add_argument('--aux-output', type=str, default='attributes', choices=['attributes', 'word-embeddings',
                                                                                 'protonet'],
                        help='Whether to make the auxiliary network to predict attributes (default) or their'
                             'respective word embeddings. Alternatively, if "protonet", the auxiliary network is'
                             'also a Protonet.')
    parser.add_argument('--word-embeddings-dim', type=int, default=100,
                        help='Word embeddings dimensionality, if used (default: 100)')
    parser.add_argument('--no-aux-backprop', action='store_false',
                        dest='aux_backprop',
                        help="If set, don't stop the gradients for auxnet.")
    parser.add_argument('--aux-init-learning-rate', type=float, default=0.05,
                        help="init learning rate for aux net")
    parser.add_argument('--bridge-init-learning-rate', type=float, default=0.05,
                        help="init learning rate for aux net")
    parser.add_argument('--scheduler-patience', type=int, default=10,
                        help="how many evaluations to wait for improvement \
                        before reducing the learning rate")
    parser.add_argument('--scheduler-factor', type=float, default=.1,
                        help="reduction factor for the learning rate")
    parser.add_argument('--optimizer', type=str, help='Default sgd, optional adam.', default='sgd',
                        choices=['sgd', 'adam'])
    parser.add_argument('--use-diff-clipping', action='store_true', help='Compute clipping separately for the 3 modules')
    parser.add_argument('--aux-grad-clip-norm', type=float, default=1.,
                        help="auxnet gradient clipping norm threshold")
    parser.add_argument('--bridge-grad-clip-norm', type=float, default=1.,
                        help="bridge gradient clipping norm threshold")
    parser.add_argument('--resume-dir', type=str,
                        help="if given, resume experiment in the specified folder")
    parser.add_argument('--no-decay-bias-bn', action='store_true', help='Use data augmentation.')
    parser.add_argument('--conv-bias', action='store_true', help='Enables bias in convolutional layers.')
    parser.add_argument('--warmup', type=int, default=-1,
                        help="Warmup steps (e.g. 2000). Default: no warmup (-1).")
    parser.add_argument('--bn-epsilon',  type=float, default=1e-05,
                        help="BN epsilon")
    parser.add_argument('--bn-momentum', type=float, default=0.1,
                        help="BN momentum")
    parser.add_argument('--debug-simpaux-aux-net', action='store_true',
                        help="switches off the metatrain loss")


    args = parser.parse_args()
    resume_dir = args.resume_dir
    if resume_dir is not None:
        data_folder = args.data_folder
        num_workers = args.num_workers
        download = args.download
        args = json.load(open(os.path.join(args.resume_dir, 'args.json')))
        args = argparse.Namespace(**args)
        args.resuming = True
        args.resume_dir = resume_dir
        args.data_folder = data_folder
        args.download = download
        args.num_workers = num_workers
    else:
        args.resuming = False
        if args.learning_rate_schedule == 'none':
            args.learning_rate_schedule = None
        if args.eval_interval_steps == -1:
            args.eval_interval_steps = args.num_batches

        assert all([o in ('anomaly', 'grads', 'outputs') for o in args.debug_opts])
    return args


if __name__ == '__main__':
    simpaux_dir = up(os.path.realpath(__file__))
    args = build_args(simpaux_dir)
    if 'anomaly' in args.debug_opts:
        torch.autograd.set_detect_anomaly(True)

    if args.resuming:
        exp_dir = args.resume_dir
        timestamp = exp_dir.split(sep='-')[-1]
    else:
        timestamp = time.strftime("%Y-%m-%d-%H%M")
        exp_dir = os.path.join(args.output_folder, f'{args.name}-{timestamp}')
        os.makedirs(exp_dir, exist_ok=True)

    log_path = os.path.join(exp_dir, 'train.log')
    logging.basicConfig(filename=log_path, level=logging.INFO)
    logging.getLogger('').addHandler(logging.StreamHandler())
    logging.info(f'Experiment {args.name} ({timestamp})')
    logging.info(args)

    args_dict = vars(args)
    # TODO: Implement resuming of experiment (args, checkpoints, seed)
    json.dump(args_dict, open(os.path.join(exp_dir, 'args.json'), 'w', encoding='utf-8'),
              indent=4, sort_keys=True)

    if torch.cuda.is_available() and args.use_cuda:
        logging.info('Training on GPU')
        args.device = torch.device('cuda')
        # torch.cuda.manual_seed(args.seed)
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
    else:
        logging.info('Training on CPU')
        args.device = torch.device('cpu')

    if args.mode == 'protonet':
        if args.resuming:
            raise NotImplementedError()
        trainer = ProtonetTrainer(args, exp_dir)
    elif args.mode == 'auxiliar':
        if args.resuming:
            raise NotImplementedError()
        trainer = AuxiliarTrainer(args, exp_dir)
    elif args.mode == 'auxiliar-embeddings':
        if args.resuming:
            raise NotImplementedError()
        trainer = AuxiliarEmbeddingsTrainer(args, exp_dir)
    elif args.mode == 'protonet-auxiliar-joint':
        trainer = ProtonetWithAuxiliarNetworkJointTrainer(args, exp_dir)
    else:
        if args.resuming:
            raise NotImplementedError()
        trainer = ProtonetWithAuxiliarNetworkFrozenTrainer(args, exp_dir)

    _, best_val_loss = trainer.train()

    print('reporting result to orion')
    report_results([dict(
        name='best val loss',
        type='objective',
        value=best_val_loss
    )])
