#!/usr/bin/env python
# -*- coding: utf-8 -*-


from protonet.torchmeta_utils import omniglot, miniimagenet, cub, cub_attributes,\
    sun_attributes, miniimagenet_attributes


def build_args():
    import argparse

    parser = argparse.ArgumentParser(
        'SimpAux',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--data-folder', type=str,
                        help='Path to the folder the data is downloaded to')
    parser.add_argument('--dataset', type=str,
                        choices=('miniimagenet', 'omniglot', 'cub', 'cub-attributes', 'sun-attributes',
                                 'miniimagenet-attributes'),
                        help="Which dataset to train on")
    return parser.parse_args()


def download_dataset(args, shuffle=True):
    if args.dataset == 'omniglot':
        dataset_helper = omniglot
    elif args.dataset == 'miniimagenet':
        dataset_helper = miniimagenet
    elif args.dataset == 'cub':
        dataset_helper = cub
    elif args.dataset == 'cub-attributes':
        dataset_helper = cub_attributes
    elif args.dataset == 'sun-attributes':
        dataset_helper = sun_attributes
    elif args.dataset == 'miniimagenet-attributes':
        dataset_helper = miniimagenet_attributes
    else:
        raise NotImplementedError(args.dataset)

    for meta_split in ('train', 'val', 'test'):
        dataset_helper(args.data_folder, shots=5,
                       ways=5, shuffle=shuffle,
                       test_shots=32,
                       meta_split=meta_split, download=True)


if __name__ == '__main__':
    args = build_args()
    download_dataset(args)


# vim: set ts=4 sw=4 sts=4 expandtab:
