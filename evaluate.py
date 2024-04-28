from os.path import dirname as up
import os
import json
import argparse
import time
import logging
import torch
from protonet import AuxiliarTrainer, ProtonetTrainer, ProtonetWithAuxiliarNetworkFrozenTrainer,\
    AuxiliarEmbeddingsTrainer, ProtonetWithAuxiliarNetworkJointTrainer


def build_args():
    parser = argparse.ArgumentParser(
        'SimpAux evaluation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('model_path', type=str, help='Model directory')
    parser.add_argument('--checkpoint', type=str,
                        help='If not set, load the "best" checkpoint according to the filename')
    parser.add_argument('--seed', type=int, help='Random seed')
    parser.add_argument('--data-folder', type=str,
                        help='Path to the folder the data is downloaded to (default: data/)')
    parser.add_argument('--num-shots', type=int,
                        help='Number of examples per class (k in "k-shot", default: 5, SUPPORT).')
    parser.add_argument('--num-test-shots', type=int,
                        help='Number of QUERY samples (default: 5).')
    parser.add_argument('--num-ways', type=int,
                        help='Number of classes per task (N in "N-way", default: 5).')
    parser.add_argument('--output-folder', type=str,
                        help='Path to the output folder for saving the model (default: output/).')
    parser.add_argument('--batch-size', type=int,
                        help='Number of tasks in a mini-batch of tasks (default: 2).')
    parser.add_argument('--num-workers', type=int,
                        help='Number of workers for data loading (default: 0).')
    parser.add_argument('--dataset', type=str,
                        choices=('cub-attributes', 'sun-attributes', 'miniimagenet-attributes'),
                        help="Which dataset to train on")
    parser.add_argument('--download', action='store_true',
                        help='Download the dataset in the data folder.')
    parser.add_argument('--disable-cuda', dest='use_cuda', action='store_false',
                        help='Use CUDA if available.')
    parser.add_argument('--num-cases-test', type=int,
                        help="how many tasks to sample during evaluation")

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    simpaux_dir = up(os.path.realpath(__file__))
    eval_args = build_args()
    args = json.load(open(os.path.join(eval_args.model_path, 'args.json')))
    args = argparse.Namespace(**args)
    for key in vars(eval_args):
        if getattr(eval_args, key) is None and key not in ['checkpoint', 'dataset']:
            setattr(eval_args, key, getattr(args, key))

    args.mode_orig = args.mode
    args.dataset_orig = args.dataset
    args.num_shots_orig = args.num_shots
    args.num_ways_orig = args.num_ways
    args.dataset = eval_args.dataset
    args.num_shots = eval_args.num_shots
    args.num_ways = eval_args.num_ways

    timestamp = time.strftime("%Y-%m-%d-%H%M")
    log_path = os.path.join(eval_args.model_path, f'eval_{timestamp}.log')
    logging.basicConfig(filename=log_path, level=logging.INFO)
    logging.getLogger('').addHandler(logging.StreamHandler())
    logging.info(eval_args)

    args_dict = vars(eval_args)
    json.dump(args_dict, open(os.path.join(eval_args.model_path, f'eval_args_{timestamp}.json'), 'w', encoding='utf-8'),
              indent=4, sort_keys=True)

    if torch.cuda.is_available() and args.use_cuda:
        logging.info('Evaluating on GPU')
        args.device = torch.device('cuda')
    else:
        logging.info('Evaluating on CPU')
        args.device = torch.device('cpu')

    # Compatibility with old namespaces:
    if not hasattr(args, 'conv_bias'):
        args.conv_bias = False
    if not hasattr(args, 'bn_epsilon'):
        args.bn_epsilon = 1e-05
    if not hasattr(args, 'bn_momentum'):
        args.bn_momentum = 0.1

    if torch.cuda.is_available() and eval_args.use_cuda:
        logging.info('Evaluating on GPU')
        eval_args.device = torch.device('cuda')
    else:
        logging.info('Evaluating on CPU')
        eval_args.device = torch.device('cpu')

    if args.mode == 'protonet':
        Trainer = ProtonetTrainer
    elif args.mode == 'auxiliar':
        Trainer = AuxiliarTrainer
    elif args.mode == 'auxiliar-embeddings':
        Trainer = AuxiliarEmbeddingsTrainer
    elif args.mode == 'protonet-auxiliar-joint':
        Trainer = ProtonetWithAuxiliarNetworkJointTrainer
    else:
        Trainer = ProtonetWithAuxiliarNetworkFrozenTrainer

    trainer = Trainer(args, eval_args.model_path)
    trainer.evaluate(eval_args)
