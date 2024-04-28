import torchmeta
from collections import OrderedDict
import numpy as np
from torchmeta.transforms.utils import apply_wrapper
import warnings
from torchmeta.datasets import Omniglot, MiniImagenet, CUB
from torchmeta.transforms import Categorical, Rotation
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor
from PIL import Image
import os
import io
import json
import h5py
from torchmeta.utils.data import Dataset, ClassDataset, CombinationMetaDataset, NonEpisodicWrapper
from torchvision.datasets.utils import download_url
from torchmeta.datasets.utils import get_asset
import torch
import scipy.io
from torch.utils.data.dataloader import default_collate

from torchmeta.utils.data.dataloader import MetaDataLoader, BatchMetaCollate
import contextlib
import torch.utils.data.dataset
from torchvision.datasets.utils import download_file_from_google_drive
import pickle
import sys
from subprocess import call
import nltk
from nltk.corpus import stopwords
from os.path import join
from scipy.io import loadmat
import random
import numpy as np
nltk.download('stopwords')


class NonEpisodicAttributeDatasetWrapper(NonEpisodicWrapper):
    def __init__(self, dataset, target_transform=None):
        super().__init__(dataset, target_transform)

    def __getitem__(self, index):
        class_dataset = self.dataset.dataset

        class_index = np.maximum(
            np.searchsorted(self._offsets,
                            index % self.num_samples, side='right') - 1, 0)
        offset = (index % self.num_samples) - self._offsets[class_index]
        label = self._labels[class_index]

        array = class_dataset.data[label][offset]
        # FIXME: apply transform!!!!!
        # FIXME: apply transform!!!!!
        # FIXME: apply transform!!!!!
        # FIXME: apply transform!!!!!
        image = (
            Image.open(io.BytesIO(array)) if array.ndim < 2
            else Image.fromarray(array)).convert('RGB')

        class_augmented_index = (
            class_dataset.num_classes * (index // self.num_samples) +
            class_index)
        transform = class_dataset.get_transform(class_augmented_index,
                                                class_dataset.transform)
        if transform is not None:
            image = transform(image)

        label = (label, index // self.num_samples)
        if self.target_transform is not None:
            label = self.target_transform(label)

        label = label[0]
        attrs = class_dataset.attributes[class_dataset.label_to_label_idx[label]]
        return image, label, attrs


def collate_attributes(batch):
    batch2 = default_collate(batch)
    if isinstance(batch2, dict):
        for key in batch2:
            batch2[key][2] = [[atts for atts in b[key][2]] for b in batch]
    else:
        batch2[2] = [element[2] for element in batch]
    return batch2


class AttributesBatchMetaDataLoader(MetaDataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=True, sampler=None, num_workers=0,
                 pin_memory=False, drop_last=False, timeout=0, worker_init_fn=None):
        collate_fn = BatchMetaCollate(collate_attributes)

        super(AttributesBatchMetaDataLoader, self).__init__(
              dataset, batch_size=batch_size, shuffle=shuffle, sampler=sampler,
              batch_sampler=None, num_workers=num_workers,
              collate_fn=collate_fn, pin_memory=pin_memory, drop_last=drop_last,
              timeout=timeout, worker_init_fn=worker_init_fn)


class UniformSplitter_(torchmeta.transforms.splitters.Splitter):
    def __init__(self, shuffle=True,
                 num_train_per_class=None, num_test_samples=None,
                 num_support_per_class=None, num_query_samples=None,
                 random_state_seed=0):
        if not shuffle:
            raise NotImplementedError("UniformSplitter doesn't support shuffle=False")
        num_samples = OrderedDict()
        if num_train_per_class is not None:
            num_samples['train'] = num_train_per_class
        elif num_support_per_class is not None:
            num_samples['support'] = num_support_per_class
        if num_test_samples is not None:
            num_samples['test'] = num_test_samples
        elif num_query_samples is not None:
            num_samples['query'] = num_query_samples
        assert len(num_samples) > 0
        self._min_samples_per_class_support = num_train_per_class if num_train_per_class is not None else \
            num_support_per_class
        super().__init__(num_samples, random_state_seed)

    def get_indices_task(self, task):
        all_class_indices = self._get_class_indices(task)
        indices = OrderedDict([(split, []) for split in self.splits])

        remaining_datasets_indices = set(list(range(len(task))))
        for name, class_indices in all_class_indices.items():
            num_samples = len(class_indices)
            if num_samples < self._min_samples_per_class_support:
                raise ValueError('The number of samples for class `{0}` ({1}) '
                                 'is smaller than the minimum number of samples per class '
                                 'required by `ClassSplitter` ({2}).'.format(name, num_samples,
                                                                             self._min_samples_per_class_support))

            seed = (hash(task) + self.random_state_seed) % (2 ** 32)
            dataset_indices = np.random.RandomState(seed).permutation(num_samples)

            ptr = 0
            for split, num_split in self.splits.items():  # This should only iterate once
                if split in ['train', 'support']:
                    split_indices = dataset_indices[ptr:ptr + num_split]
                    self.np_random.shuffle(split_indices)
                    indices[split].extend([class_indices[idx] for idx in split_indices])
                    remaining_datasets_indices = \
                        remaining_datasets_indices.difference([class_indices[idx] for idx in split_indices])
                    ptr += num_split
        if len(remaining_datasets_indices) != 0:
            raise ValueError("The dataset doesn't have enough instances to sample from.")
        remaining_datasets_indices = sorted(list(remaining_datasets_indices))
        n_queries = self.splits['query'] if 'query' in self.splits else self.splits['test']
        queries = self.np_random.choice(remaining_datasets_indices, n_queries, replace=False)
        if 'query' in self.splits:
            indices['query'].extend(queries)
        else:
            indices['test'].extend(queries)
        return indices

    def get_indices_concattask(self, task):
        indices = OrderedDict([(split, []) for split in self.splits])
        cum_size = 0

        remaining_datasets_indices = set(list(range(sum([len(dataset) for dataset in task.datasets]))))
        for dataset in task.datasets:
            num_samples = len(dataset)
            if num_samples < self._min_samples_per_class_support:
                raise ValueError('The number of samples for one class ({0}) '
                                 'is smaller than the minimum number of samples per class '
                                 'required by `ClassSplitter` ({1}).'.format(num_samples,
                                                                             self._min_samples_per_class_support))

            seed = (hash(task) + self.random_state_seed) % (2 ** 32)
            dataset_indices = np.random.RandomState(seed).permutation(num_samples)

            ptr = 0
            for split, num_split in self.splits.items():  # this should only iterate once
                if split in ['train', 'support']:
                    split_indices = dataset_indices[ptr:ptr + num_split]
                    self.np_random.shuffle(split_indices)
                    indices[split].extend(split_indices + cum_size)
                    remaining_datasets_indices = remaining_datasets_indices.difference(set(split_indices + cum_size))
                    ptr += num_split
            cum_size += num_samples

        if len(remaining_datasets_indices) == 0:
            raise ValueError("The dataset doesn't have enough instances to sample from.")
        remaining_datasets_indices = sorted(list(remaining_datasets_indices))
        n_queries = self.splits['query'] if 'query' in self.splits else self.splits['test']
        queries = self.np_random.choice(remaining_datasets_indices, n_queries, replace=False)
        if 'query' in self.splits:
            indices['query'].extend(queries)
        else:
            indices['test'].extend(queries)

        return indices


def UniformSplitter(task=None, *args, **kwargs):
    return apply_wrapper(UniformSplitter_(*args, **kwargs), task)


def helper_with_default_uniform_splitter(klass, folder, shots, ways, shuffle=True, test_shots=None, seed=None,
                                         defaults={}, **kwargs):
    if 'num_classes_per_task' in kwargs:
        warnings.warn('Both arguments `ways` and `num_classes_per_task` were '
                      'set in the helper function for the number of classes per task. '
                      'Ignoring the argument `ways`.', stacklevel=2)
        ways = kwargs['num_classes_per_task']
    if 'transform' not in kwargs:
        kwargs['transform'] = defaults.get('transform', ToTensor())
    if 'target_transform' not in kwargs:
        kwargs['target_transform'] = defaults.get('target_transform',
                                                  Categorical(ways))
    if 'class_augmentations' not in kwargs:
        kwargs['class_augmentations'] = defaults.get('class_augmentations', None)
    if test_shots is None:
        if kwargs['num_classes_per_task'] is None:
            test_shots = shots*ways
        else:
            test_shots = shots*kwargs['num_classes_per_task']
    dataset = klass(folder, num_classes_per_task=ways, **kwargs)
    dataset = UniformSplitter(dataset, shuffle=shuffle,
                              num_train_per_class=shots, num_test_samples=test_shots)
    dataset.seed(seed)
    return dataset


def omniglot(folder, shots, ways, shuffle=True, test_shots=None,
             seed=None, helper=torchmeta.datasets.helpers.helper_with_default, **kwargs):
    """Helper function to create a meta-dataset for the Omniglot dataset.
    Parameters
    ----------
    folder : string
        Root directory where the dataset folder `omniglot` exists.
    shots : int
        Number of (training) examples per class in each task. This corresponds
        to `k` in `k-shot` classification.
    ways : int
        Number of classes per task. This corresponds to `N` in `N-way`
        classification.
    shuffle : bool (default: `True`)
        Shuffle the examples when creating the tasks.
    test_shots : int, optional
        Number of test examples per class in each task. If `None`, then the
        number of test examples is equal to the number of training examples per
        class.
    seed : int, optional
        Random seed to be used in the meta-dataset.
    kwargs
        Additional arguments passed to the `Omniglot` class.
    See also
    --------
    `datasets.Omniglot` : Meta-dataset for the Omniglot dataset.
    """
    defaults = {
        'transform': Compose([Resize(28), ToTensor()]),
        'class_augmentations': [Rotation([90, 180, 270])]
    }

    return helper(Omniglot, folder, shots, ways, shuffle=shuffle, test_shots=test_shots, seed=seed, defaults=defaults,
                  **kwargs)


def miniimagenet(folder, shots, ways, shuffle=True, test_shots=None,
                 seed=None, helper=torchmeta.datasets.helpers.helper_with_default, **kwargs):
    """Helper function to create a meta-dataset for the Mini-Imagenet dataset.
    Parameters
    ----------
    folder : string
        Root directory where the dataset folder `miniimagenet` exists.
    shots : int
        Number of (training) examples per class in each task. This corresponds
        to `k` in `k-shot` classification.
    ways : int
        Number of classes per task. This corresponds to `N` in `N-way`
        classification.
    shuffle : bool (default: `True`)
        Shuffle the examples when creating the tasks.
    test_shots : int, optional
        Number of test examples per class in each task. If `None`, then the
        number of test examples is equal to the number of training examples per
        class.
    seed : int, optional
        Random seed to be used in the meta-dataset.
    kwargs
        Additional arguments passed to the `MiniImagenet` class.
    See also
    --------
    `datasets.MiniImagenet` : Meta-dataset for the Mini-Imagenet dataset.
    """
    defaults = {
        'transform': Compose([Resize(84), ToTensor()])
    }

    return helper(MiniImagenet, folder, shots, ways, shuffle=shuffle, test_shots=test_shots, seed=seed,
                  defaults=defaults, **kwargs)


def cub(folder, shots, ways, shuffle=True, test_shots=None, seed=None,
        helper=torchmeta.datasets.helpers.helper_with_default, **kwargs):
    """Helper function to create a meta-dataset for the Caltech-UCSD Birds dataset.
    Parameters
    ----------
    folder : string
        Root directory where the dataset folder `cub` exists.
    shots : int
        Number of (training) examples per class in each task. This corresponds
        to `k` in `k-shot` classification.
    ways : int
        Number of classes per task. This corresponds to `N` in `N-way`
        classification.
    shuffle : bool (default: `True`)
        Shuffle the examples when creating the tasks.
    test_shots : int, optional
        Number of test examples per class in each task. If `None`, then the
        number of test examples is equal to the number of training examples per
        class.
    seed : int, optional
        Random seed to be used in the meta-dataset.
    kwargs
        Additional arguments passed to the `CUB` class.
    See also
    --------
    `datasets.cub.CUB` : Meta-dataset for the Caltech-UCSD Birds dataset.
    """
    image_size = 84
    defaults = {
        'transform': Compose([
                        Resize(int(image_size * 1.5)),
                        CenterCrop(image_size),
                        ToTensor()
                    ])
    }

    return helper(CUB, folder, shots, ways, shuffle=shuffle, test_shots=test_shots, seed=seed, defaults=defaults,
                  **kwargs)


def cub_attributes(folder, shots, ways, shuffle=True, test_shots=None, seed=None,
                   helper=torchmeta.datasets.helpers.helper_with_default, **kwargs):
    """Helper function to create a meta-dataset for the Caltech-UCSD Birds dataset.
    Parameters
    ----------
    folder : string
        Root directory where the dataset folder `cub` exists.
    shots : int
        Number of (training) examples per class in each task. This corresponds
        to `k` in `k-shot` classification.
    ways : int
        Number of classes per task. This corresponds to `N` in `N-way`
        classification.
    shuffle : bool (default: `True`)
        Shuffle the examples when creating the tasks.
    test_shots : int, optional
        Number of test examples per class in each task. If `None`, then the
        number of test examples is equal to the number of training examples per
        class.
    seed : int, optional
        Random seed to be used in the meta-dataset.
    kwargs
        Additional arguments passed to the `CUB` class.
    See also
    --------
    `datasets.cub.CUB` : Meta-dataset for the Caltech-UCSD Birds dataset.
    """
    image_size = 84
    defaults = {
        'transform': Compose([
                        Resize(int(image_size * 1.5)),
                        CenterCrop(image_size),
                        ToTensor()
                    ])
    }

    return helper(CUBAttributes, folder, shots, ways, shuffle=shuffle, test_shots=test_shots, seed=seed,
                  defaults=defaults,
                  **kwargs)

# Note: We still need to:
# cp -r /home/mila/j/jordi.armengol-estape/.local/lib/python3.7/site-packages/torchmeta/datasets/assets/cub/ /home/mila/j/jordi.armengol-estape/.local/lib/python3.7/site-packages/torchmeta/datasets/assets/cub-attributes/
# Since it is not included in torchmeta assets


class CUBAttributes(CombinationMetaDataset):
    """
    The Caltech-UCSD Birds dataset, introduced in [1]. This dataset is based on
    images from 200 species of birds from the Caltech-UCSD Birds dataset [2].
    Parameters
    ----------
    root : string
        Root directory where the dataset folder `cub` exists.
    num_classes_per_task : int
        Number of classes per tasks. This corresponds to "N" in "N-way"
        classification.
    meta_train : bool (default: `False`)
        Use the meta-train split of the dataset. If set to `True`, then the
        arguments `meta_val` and `meta_test` must be set to `False`. Exactly one
        of these three arguments must be set to `True`.
    meta_val : bool (default: `False`)
        Use the meta-validation split of the dataset. If set to `True`, then the
        arguments `meta_train` and `meta_test` must be set to `False`. Exactly one
        of these three arguments must be set to `True`.
    meta_test : bool (default: `False`)
        Use the meta-test split of the dataset. If set to `True`, then the
        arguments `meta_train` and `meta_val` must be set to `False`. Exactly one
        of these three arguments must be set to `True`.
    meta_split : string in {'train', 'val', 'test'}, optional
        Name of the split to use. This overrides the arguments `meta_train`,
        `meta_val` and `meta_test` if all three are set to `False`.
    transform : callable, optional
        A function/transform that takes a `PIL` image, and returns a transformed
        version. See also `torchvision.transforms`.
    target_transform : callable, optional
        A function/transform that takes a target, and returns a transformed
        version. See also `torchvision.transforms`.
    dataset_transform : callable, optional
        A function/transform that takes a dataset (ie. a task), and returns a
        transformed version of it. E.g. `torchmeta.transforms.ClassSplitter()`.
    class_augmentations : list of callable, optional
        A list of functions that augment the dataset with new classes. These classes
        are transformations of existing classes. E.g.
        `torchmeta.transforms.HorizontalFlip()`.
    download : bool (default: `False`)
        If `True`, downloads the pickle files and processes the dataset in the root
        directory (under the `cub` folder). If the dataset is already
        available, this does not download/process the dataset again.
    Notes
    -----
    The dataset is downloaded from [2]. The dataset contains images from 200
    classes. The meta train/validation/test splits are over 100/50/50 classes.
    The splits are taken from [3] ([code](https://github.com/wyharveychen/CloserLookFewShot)
    for reproducibility).
    References
    ----------
    .. [1] Hilliard, N., Phillips, L., Howland, S., Yankov, A., Corley, C. D.,
           Hodas, N. O. (2018). Few-Shot Learning with Metric-Agnostic Conditional
           Embeddings. (https://arxiv.org/abs/1802.04376)
    .. [2] Wah, C., Branson, S., Welinder, P., Perona, P., Belongie, S. (2011).
           The Caltech-UCSD Birds-200-2011 Dataset
           (http://www.vision.caltech.edu/visipedia/CUB-200-2011.html)
    .. [3] Chen, W., Liu, Y. and Kira, Z. and Wang, Y. and  Huang, J. (2019).
           A Closer Look at Few-shot Classification. International Conference on
           Learning Representations (https://openreview.net/forum?id=HkxLXnAcFQ)
    """

    def __init__(self, root, num_classes_per_task=None, meta_train=False,
                 meta_val=False, meta_test=False, meta_split=None,
                 transform=None, target_transform=None, dataset_transform=None,
                 class_augmentations=None, download=False):
        dataset = CUBAttributesClassDataset(root, meta_train=meta_train, meta_val=meta_val,
                                            meta_test=meta_test, meta_split=meta_split, transform=transform,
                                            class_augmentations=class_augmentations, download=download)
        super(CUBAttributes, self).__init__(dataset, num_classes_per_task,
                                            target_transform=target_transform,
                                            dataset_transform=dataset_transform)


class CUBAttributesClassDataset(ClassDataset):
    folder = 'cub-attributes'
    download_url = 'http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz'
    tgz_md5 = '97eceeb196236b17998738112f37df78'
    image_folder = 'CUB_200_2011/images'

    filename = '{0}_data.hdf5'
    filename_labels = '{0}_labels.json'

    def __init__(self, root, meta_train=False, meta_val=False, meta_test=False,
                 meta_split=None, transform=None, class_augmentations=None,
                 download=False):
        super(CUBAttributesClassDataset, self).__init__(meta_train=meta_train,
                                                        meta_val=meta_val, meta_test=meta_test, meta_split=meta_split,
                                                        class_augmentations=class_augmentations)

        self.root = os.path.join(os.path.expanduser(root), self.folder)
        self.transform = transform

        self.split_filename = os.path.join(self.root,
                                           self.filename.format(self.meta_split))
        self.split_filename_labels = os.path.join(self.root,
                                                  self.filename_labels.format(self.meta_split))

        self._data_file = None
        self._data = None
        self._labels = None
        self._attributes = None
        self._attributes_file = os.path.join(self.root, 'attributes.csv')

        self._attribute_names_file = os.path.join(self.root, 'attributes.txt')

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('CUB Attributes integrity check failed')
        self._num_classes = len(self.labels)

        self.label_to_label_idx = OrderedDict()
        all_labels = []
        for split in ['train', 'val', 'test']:
            with open(os.path.join(self.root, self.filename_labels.format(split)), 'r') as f:
                labels_split = json.load(f)
                all_labels.extend(labels_split)
        all_labels = sorted(all_labels)
        self.label_idx_to_label = []
        for idx, label in enumerate(all_labels):
            self.label_to_label_idx[label] = idx
            self.label_idx_to_label.append(label)

        self._attribute_names_file = os.path.join(self.root, 'attributes.txt')
        sw = set(stopwords.words())
        self.attribute_words = []
        for line in open(self._attribute_names_file, 'r').readlines():
            line = line.split()[1].replace('::', '_').split('_')
            words = []
            for word in line:
                w = ''
                for c in word:
                    if c.isalpha():
                        w += c.lower()
                if w not in sw and len(w) > 0:
                    words.append(w)
            self.attribute_words.append(words)

    def __getitem__(self, index):
        label = self.labels[index % self.num_classes]
        data = self.data[label]
        transform = self.get_transform(index, self.transform)
        target_transform = self.get_target_transform(index)
        attributes = self.attributes[self.label_to_label_idx[label]]  # [index % self.num_classes]

        def bin_attributes2ints(atts):
            new_atts = []
            for idx, e in enumerate(atts):
                if e == 1:
                    new_atts.append(idx)
            return new_atts

        # attributes = [bin_attributes2ints(attributes)]
        # attributes = [bin_attributes2ints(a) for a in attributes]

        attributes = bin_attributes2ints(attributes)

        return CUBAttributesDataset(index, data, label, attributes, transform=transform,
                                    target_transform=target_transform)

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def data(self):
        if self._data is None:
            self._data_file = h5py.File(self.split_filename, 'r')
            self._data = self._data_file['datasets']
        return self._data

    @property
    def labels(self):
        if self._labels is None:
            with open(self.split_filename_labels, 'r') as f:
                self._labels = json.load(f)
        return self._labels

    @property
    def attributes(self):
        if self._attributes is None:
            self._attributes = np.genfromtxt(self._attributes_file, delimiter=' ')
        return self._attributes

    def _check_integrity(self):
        return (os.path.isfile(self.split_filename)
                and os.path.isfile(self.split_filename_labels) and os.path.isfile(self._attributes_file))

    def close(self):
        if self._data_file is not None:
            self._data_file.close()
            self._data_file = None
            self._data = None

    def download(self):
        import tarfile
        import shutil
        import glob
        from tqdm import tqdm

        if self._check_integrity():
            return

        skip = False
        if (os.path.isfile(self.split_filename) and os.path.isfile(self.split_filename_labels)) and not \
                os.path.isfile(self._attributes_file):
            skip = True

        filename = os.path.basename(self.download_url)
        tgz_filename = os.path.join(self.root, filename)

        if not skip:

            download_url(self.download_url, self.root, filename, self.tgz_md5)

            with tarfile.open(tgz_filename, 'r') as f:
                f.extractall(self.root)
            image_folder = os.path.join(self.root, self.image_folder)

            for split in ['train', 'val', 'test']:
                filename = os.path.join(self.root, self.filename.format(split))
                if os.path.isfile(filename):
                    continue

                labels = get_asset(self.folder, '{0}.json'.format(split))
                labels_filename = os.path.join(self.root, self.filename_labels.format(split))
                with open(labels_filename, 'w') as f:
                    json.dump(labels, f)

                with h5py.File(filename, 'w') as f:
                    group = f.create_group('datasets')
                    dtype = h5py.special_dtype(vlen=np.uint8)
                    for i, label in enumerate(tqdm(labels, desc=filename)):
                        images = glob.glob(os.path.join(image_folder, label, '*.jpg'))
                        images.sort()
                        dataset = group.create_dataset(label, (len(images),), dtype=dtype)
                        for i, image in enumerate(images):
                            with open(image, 'rb') as f:
                                array = bytearray(f.read())
                                dataset[i] = np.asarray(array, dtype=np.uint8)

        tar_folder, _ = os.path.splitext(tgz_filename)

        class_annotations = np.zeros((200, 312))
        count = 0
        with open(os.path.join(tar_folder, 'attributes', 'class_attribute_labels_continuous.txt'),
                  'r') as f:
            line = f.readline()
            while line:
                splt = line.split(' ')
                confs = np.array([float(item) > 50 for item in splt])
                class_annotations[count, :] = confs
                count += 1
                line = f.readline()

        np.savetxt(self._attributes_file, class_annotations, delimiter=' ')

        if os.path.isdir(tar_folder):
            shutil.rmtree(tar_folder)

        # attributes_filename = os.path.join(self.root, 'attributes.txt')
        # if os.path.isfile(attributes_filename):
        #    os.remove(attributes_filename)


class CUBAttributesDataset(Dataset):
    def __init__(self, index, data, label, attributes,
                 transform=None, target_transform=None):
        super(CUBAttributesDataset, self).__init__(index, transform=transform,
                                                   target_transform=target_transform)
        self.data = data
        self.label = label
        self.attributes = attributes

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image = Image.open(io.BytesIO(self.data[index])).convert('RGB')
        target = self.label

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            target = self.target_transform(target)

        attributes = self.attributes  # [target]

        return image, target, attributes


def sun_attributes(folder, shots, ways, shuffle=True, test_shots=None, seed=None,
                   helper=torchmeta.datasets.helpers.helper_with_default, **kwargs):
    """Helper function to create a meta-dataset for the SUN397 dataset.
    Parameters
    ----------
    folder : string
        Root directory where the dataset folder `sun` exists.
    shots : int
        Number of (training) examples per class in each task. This corresponds
        to `k` in `k-shot` classification.
    ways : int
        Number of classes per task. This corresponds to `N` in `N-way`
        classification.
    shuffle : bool (default: `True`)
        Shuffle the examples when creating the tasks.
    test_shots : int, optional
        Number of test examples per class in each task. If `None`, then the
        number of test examples is equal to the number of training examples per
        class.
    seed : int, optional
        Random seed to be used in the meta-dataset.
    kwargs
        Additional arguments passed to the `SUN` class.
    See also
    --------
    `datasets.sun.SUN` : Meta-dataset for the SUN397 dataset.
    """
    image_size = 84
    defaults = {
        'transform': Compose([
                        Resize(int(image_size * 1.5)),
                        CenterCrop(image_size),
                        ToTensor()
                    ])
    }

    return helper(SUNAttributes, folder, shots, ways, shuffle=shuffle, test_shots=test_shots, seed=seed,
                  defaults=defaults,
                  **kwargs)

# Note: We still need to:
# cp -r /home/mila/j/jordi.armengol-estape/.local/lib/python3.7/site-packages/torchmeta/datasets/assets/sun/ /home/mila/j/jordi.armengol-estape/.local/lib/python3.7/site-packages/torchmeta/datasets/assets/sun-attributes/
# Since it is not included in torchmeta assets


@contextlib.contextmanager
def temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


class SUNAttributes(CombinationMetaDataset):
    """
    The SUN397 dataset, introduced in [1]. This dataset is based on
    images from 200 species of birds from the SUN397 dataset [2].
    Parameters
    ----------
    root : string
        Root directory where the dataset folder `sun` exists.
    num_classes_per_task : int
        Number of classes per tasks. This corresponds to "N" in "N-way"
        classification.
    meta_train : bool (default: `False`)
        Use the meta-train split of the dataset. If set to `True`, then the
        arguments `meta_val` and `meta_test` must be set to `False`. Exactly one
        of these three arguments must be set to `True`.
    meta_val : bool (default: `False`)
        Use the meta-validation split of the dataset. If set to `True`, then the
        arguments `meta_train` and `meta_test` must be set to `False`. Exactly one
        of these three arguments must be set to `True`.
    meta_test : bool (default: `False`)
        Use the meta-test split of the dataset. If set to `True`, then the
        arguments `meta_train` and `meta_val` must be set to `False`. Exactly one
        of these three arguments must be set to `True`.
    meta_split : string in {'train', 'val', 'test'}, optional
        Name of the split to use. This overrides the arguments `meta_train`,
        `meta_val` and `meta_test` if all three are set to `False`.
    transform : callable, optional
        A function/transform that takes a `PIL` image, and returns a transformed
        version. See also `torchvision.transforms`.
    target_transform : callable, optional
        A function/transform that takes a target, and returns a transformed
        version. See also `torchvision.transforms`.
    dataset_transform : callable, optional
        A function/transform that takes a dataset (ie. a task), and returns a
        transformed version of it. E.g. `torchmeta.transforms.ClassSplitter()`.
    class_augmentations : list of callable, optional
        A list of functions that augment the dataset with new classes. These classes
        are transformations of existing classes. E.g.
        `torchmeta.transforms.HorizontalFlip()`.
    download : bool (default: `False`)
        If `True`, downloads the pickle files and processes the dataset in the root
        directory (under the `sun` folder). If the dataset is already
        available, this does not download/process the dataset again.
    Notes
    -----
    The dataset is downloaded from http://vision.princeton.edu/projects/2010/SUN/SUN397.tar.gz.
    397 classes and all 908 classes.
    """

    def __init__(self, root, num_classes_per_task=None, meta_train=False,
                 meta_val=False, meta_test=False, meta_split=None,
                 transform=None, target_transform=None, dataset_transform=None,
                 class_augmentations=None, download=False):
        dataset = SUNAttributesClassDataset(root, meta_train=meta_train, meta_val=meta_val,
                                            meta_test=meta_test, meta_split=meta_split, transform=transform,
                                            class_augmentations=class_augmentations, download=download)
        super(SUNAttributes, self).__init__(dataset, num_classes_per_task,
                                            target_transform=target_transform,
                                            dataset_transform=dataset_transform)


class OneClassDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, dir_):
        self.dir = dir_
        self.class_ = os.path.basename(os.path.normpath(self.dir))
        if os.path.isdir(os.path.join(self.dir, os.listdir(self.dir)[0])):
            self.dir = os.path.join(self.dir, os.listdir(self.dir)[0])
        self.images = sorted(os.listdir(self.dir))

    def __getitem__(self, item):
        return Image.open(os.path.join(self.dir, self.images[item]))

    def __len__(self):
        return len(self.images)


class SUNAttributesClassDataset(ClassDataset):
    folder = 'sun-attributes'
    download_images_url = 'http://vision.princeton.edu/projects/2010/SUN/SUN397.tar.gz'
    download_attributes_url = 'http://cs.brown.edu/~gmpatter/Attributes/SUNAttributeDB.tar.gz'
    images_tgz_md5 = '8ca2778205c41d23104230ba66911c7a'
    attributes_tgz_md5 = '883293e5b645822f6ae0046c6df54183'
    image_folder = 'SUN397'

    filename_labels = '{0}_labels.json'

    def __init__(self, root, meta_train=False, meta_val=False, meta_test=False,
                 meta_split=None, transform=None, class_augmentations=None,
                 download=False, seed=42, custom_split=True):
        super(SUNAttributesClassDataset, self).__init__(meta_train=meta_train,
                                                        meta_val=meta_val, meta_test=meta_test, meta_split=meta_split,
                                                        class_augmentations=class_augmentations)

        self.root = os.path.join(os.path.expanduser(root), self.folder)
        self.transform = transform

        self.split_filename_labels = os.path.join(self.root,
                                                  self.filename_labels.format(self.meta_split))

        assets = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'assets', 'sun-attributes')

        if not os.path.exists(self.split_filename_labels):

            if not custom_split:
                split = json.loads(open(os.path.join(assets, 'label_idx_sun.json'), 'r').read())
                labels = set(split['base_classes'])
                tr_labels = []
                for idx in labels:
                    tr_labels.append(split['label_names'][idx])
                with temp_seed(seed):
                    np.random.shuffle(tr_labels)
                train_labels = tr_labels[:8*len(tr_labels) // 10]
                valid_labels = tr_labels[8*len(tr_labels) // 10:]

                labels = set(split['novel_classes'])
                test_labels = []
                for idx in labels:
                    test_labels.append(split['label_names'][idx])
            else:
                split = json.loads(open(os.path.join(assets, 'label_idx_sun.json'), 'r').read())
                base_labels = set(split['base_classes'])
                novel_labels = set(split['novel_classes'])
                all_labels = base_labels.union(novel_labels)
                all_labels = list(all_labels)
                with temp_seed(seed):
                    np.random.shuffle(all_labels)
                train_labels = all_labels[:8*len(all_labels) // 10]
                train_label_names = []
                for idx in train_labels:
                    train_label_names.append(split['label_names'][idx])
                valid_labels = all_labels[8*len(all_labels) // 10:9*len(all_labels) // 10]
                valid_label_names = []
                for idx in valid_labels:
                    valid_label_names.append(split['label_names'][idx])
                test_label_names = []
                test_labels = all_labels[9 * len(all_labels) // 10:]
                for idx in test_labels:
                    test_label_names.append(split['label_names'][idx])
                train_labels = train_label_names
                valid_labels = valid_label_names
                test_labels = test_label_names

            with open(os.path.join(self.root, self.filename_labels.format('train')), 'w') as f:
                f.write(json.dumps(train_labels))
            with open(os.path.join(self.root, self.filename_labels.format('val')), 'w') as f:
                f.write(json.dumps(valid_labels))
            with open(os.path.join(self.root, self.filename_labels.format('test')), 'w') as f:
                f.write(json.dumps(test_labels))

        self._data = None
        self._labels = None
        self._attributes = None
        self._attributes_file = os.path.join(self.root, 'attributes.csv')

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('SUN Attributes integrity check failed')
        self._num_classes = len(self.labels)

        self._attribute_names_file = os.path.join(self.root, 'attributes.txt')
        self.label_to_label_idx = OrderedDict()
        all_labels = []
        for split in ['train', 'val', 'test']:
            with open(os.path.join(self.root, self.filename_labels.format(split)), 'r') as f:
                labels_split = json.load(f)
                all_labels.extend(labels_split)
        all_labels = sorted(all_labels)
        self.label_idx_to_label = []
        for idx, label in enumerate(all_labels):
            self.label_to_label_idx[label] = idx
            self.label_idx_to_label.append(label)

        self._attribute_names_file = os.path.join(self.root, 'SUNAttributeDB', 'attributes.mat')
        sw = set(stopwords.words())
        self.attribute_words = []
        for line in scipy.io.loadmat(self._attribute_names_file)['attributes']:
            line = line[0][0].split()
            words = []
            for word in line:
                w = ''
                for c in word:
                    if c.isalpha():
                        w += c.lower()
                if w not in sw and len(w) > 0:
                    words.append(w)
            self.attribute_words.append(words)

    def __getitem__(self, index):
        lengths = [len(data) for data in self.data]
        cumsum = np.cumsum(lengths)
        for idx, e in enumerate(cumsum):
            if index < e:
                break
        items = list(self.data.items())
        label = items[idx][0]
        data = self.data[label]
        transform = self.get_transform(index, self.transform)
        target_transform = self.get_target_transform(index)
        attributes = self.attributes[self.label_to_label_idx[label]]

        def bin_attributes2ints(atts):
            new_atts = []
            for idx, e in enumerate(atts):
                if e == 1:
                    new_atts.append(idx)
            return new_atts

        attributes = bin_attributes2ints(attributes)

        return SUNAttributesDataset(index, data, label, attributes, transform=transform,
                                    target_transform=target_transform)

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def data(self):
        image_folder = os.path.join(self.root, self.image_folder)
        if self._data is None:
            self._data = OrderedDict((class_, OneClassDataset(os.path.join(image_folder, self.meta_split, class_[0],
                                                              class_.split('/')[1]))) for class_ in self.labels)
        return self._data

    @property
    def labels(self):
        if self._labels is None:
            with open(self.split_filename_labels, 'r') as f:
                self._labels = json.load(f)
        return self._labels

    @property
    def attributes(self):
        if self._attributes is None:
            self._attributes = np.genfromtxt(self._attributes_file, delimiter=' ')
        return self._attributes

    def _check_integrity(self):
        image_folder = os.path.join(self.root, self.image_folder)
        return (os.path.isdir(os.path.join(image_folder, self.meta_split))
                and os.path.isfile(self.split_filename_labels) and os.path.isfile(self._attributes_file))

    def close(self):
        if self._data is not None:
            del self._data

    def download(self):
        import tarfile
        import shutil

        if self._check_integrity():
            return

        images_filename = os.path.basename(self.download_images_url)
        download_url(self.download_images_url, self.root, images_filename, self.images_tgz_md5)
        attributes_filename = os.path.basename(self.download_attributes_url)
        download_url(self.download_attributes_url, self.root, attributes_filename, self.attributes_tgz_md5)

        images_tgz_filename = os.path.join(self.root, images_filename)
        with tarfile.open(images_tgz_filename, 'r') as f:
            f.extractall(self.root)
        image_folder = os.path.join(self.root, self.image_folder)

        attributes_tgz_filename = os.path.join(self.root, attributes_filename)
        with tarfile.open(attributes_tgz_filename, 'r') as f:
            f.extractall(self.root)

        for split in ['train', 'val', 'test']:
            os.makedirs(os.path.join(image_folder, split))
            labels = json.loads(open(os.path.join(self.root, self.filename_labels.format(split)), 'r').read())
            for label in labels:
                os.makedirs(os.path.join(image_folder, split, label))
                for image in os.listdir(os.path.join(image_folder, label)):
                    image_path = os.path.join(image_folder, label, image)
                    shutil.move(image_path, os.path.join(image_folder, split, label, image))

        images = scipy.io.loadmat(os.path.join(attributes_tgz_filename[:-7], 'images.mat'))
        images = images['images']

        attributes = scipy.io.loadmat(os.path.join(attributes_tgz_filename[:-7], 'attributes.mat'))
        attributes = attributes['attributes']

        map = scipy.io.loadmat(os.path.join(attributes_tgz_filename[:-7], 'attributeLabels_continuous.mat'))
        map = torch.from_numpy(map['labels_cv']) > 0.5

        labels_395 = []
        with open(os.path.join(image_folder, 'ClassName.txt'), 'r') as f:
            temp = f.readline().rstrip()
            while temp:
                labels_395.append(temp[1:])
                temp = f.readline().rstrip()

        labels = {}
        counts = {}
        for i in range(len(images)):
            image = images[i][0][0]
            splt = image.split('/sun_')
            cat_name = splt[0]
            if cat_name not in labels_395:
                continue
            if cat_name not in labels:
                labels[cat_name] = torch.zeros(102)
                counts[cat_name] = 0
            labels[cat_name] += map[i, :].float()
            counts[cat_name] += 1

        annotation = torch.zeros(len(labels), 102)
        print(len(labels))
        label_list = []
        ind = 0
        for k in labels.keys():
            label_list.append(k)
            labels[k] /= float(counts[k])
            annotation[ind, :] = labels[k]
            ind += 1

        annotation = annotation >= 0.5

        ind = annotation.sum(0) > 1
        annotation = annotation[:, ind]

        print(annotation.size())

        np.savetxt(self._attributes_file, annotation, delimiter=' ')

        # if os.path.isdir(tar_folder):
        #     shutil.rmtree(tar_folder)


class SUNAttributesDataset(Dataset):
    def __init__(self, index, data, label, attributes,
                 transform=None, target_transform=None):
        super(SUNAttributesDataset, self).__init__(index, transform=transform,
                                                   target_transform=target_transform)
        self.data = data
        self.label = label
        self.attributes = attributes

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image = self.data[index].convert('RGB')  # Image.open(io.BytesIO(self.data[index])).convert('RGB')
        target = self.label

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            target = self.target_transform(target)

        attributes = self.attributes

        return image, target, attributes


def miniimagenet_attributes(folder, shots, ways, shuffle=True, test_shots=None,
                            seed=None, helper=torchmeta.datasets.helpers.helper_with_default, **kwargs):
    """Helper function to create a meta-dataset for the Mini-Imagenet dataset.
    Parameters
    ----------
    folder : string
        Root directory where the dataset folder `miniimagenet` exists.
    shots : int
        Number of (training) examples per class in each task. This corresponds
        to `k` in `k-shot` classification.
    ways : int
        Number of classes per task. This corresponds to `N` in `N-way`
        classification.
    shuffle : bool (default: `True`)
        Shuffle the examples when creating the tasks.
    test_shots : int, optional
        Number of test examples per class in each task. If `None`, then the
        number of test examples is equal to the number of training examples per
        class.
    seed : int, optional
        Random seed to be used in the meta-dataset.
    kwargs
        Additional arguments passed to the `MiniImagenet` class.
    See also
    --------
    `datasets.MiniImagenet` : Meta-dataset for the Mini-Imagenet dataset.
    """
    defaults = {
        'transform': Compose([Resize(84), ToTensor()])
    }

    return helper(MiniImagenetAttributes, folder, shots, ways, shuffle=shuffle, test_shots=test_shots, seed=seed,
                  defaults=defaults, **kwargs)


class MiniImagenetAttributes(CombinationMetaDataset):
    """
    The Mini-Imagenet dataset, introduced in [1]. This dataset contains images
    of 100 different classes from the ILSVRC-12 dataset (Imagenet challenge).
    The meta train/validation/test splits are taken from [2] for reproducibility.
    Parameters
    ----------
    root : string
        Root directory where the dataset folder `miniimagenet` exists.
    num_classes_per_task : int
        Number of classes per tasks. This corresponds to "N" in "N-way"
        classification.
    meta_train : bool (default: `False`)
        Use the meta-train split of the dataset. If set to `True`, then the
        arguments `meta_val` and `meta_test` must be set to `False`. Exactly one
        of these three arguments must be set to `True`.
    meta_val : bool (default: `False`)
        Use the meta-validation split of the dataset. If set to `True`, then the
        arguments `meta_train` and `meta_test` must be set to `False`. Exactly one
        of these three arguments must be set to `True`.
    meta_test : bool (default: `False`)
        Use the meta-test split of the dataset. If set to `True`, then the
        arguments `meta_train` and `meta_val` must be set to `False`. Exactly one
        of these three arguments must be set to `True`.
    meta_split : string in {'train', 'val', 'test'}, optional
        Name of the split to use. This overrides the arguments `meta_train`,
        `meta_val` and `meta_test` if all three are set to `False`.
    transform : callable, optional
        A function/transform that takes a `PIL` image, and returns a transformed
        version. See also `torchvision.transforms`.
    target_transform : callable, optional
        A function/transform that takes a target, and returns a transformed
        version. See also `torchvision.transforms`.
    dataset_transform : callable, optional
        A function/transform that takes a dataset (ie. a task), and returns a
        transformed version of it. E.g. `torchmeta.transforms.ClassSplitter()`.
    class_augmentations : list of callable, optional
        A list of functions that augment the dataset with new classes. These classes
        are transformations of existing classes. E.g.
        `torchmeta.transforms.HorizontalFlip()`.
    download : bool (default: `False`)
        If `True`, downloads the pickle files and processes the dataset in the root
        directory (under the `miniimagenet` folder). If the dataset is already
        available, this does not download/process the dataset again.
    Notes
    -----
    The dataset is downloaded from [this repository]
    (https://github.com/renmengye/few-shot-ssl-public/). The meta train/
    validation/test splits are over 64/16/20 classes.
    References
    ----------
    .. [1] Vinyals, O., Blundell, C., Lillicrap, T. and Wierstra, D. (2016).
           Matching Networks for One Shot Learning. In Advances in Neural
           Information Processing Systems (pp. 3630-3638) (https://arxiv.org/abs/1606.04080)
    .. [2] Ravi, S. and Larochelle, H. (2016). Optimization as a Model for
           Few-Shot Learning. (https://openreview.net/forum?id=rJY0-Kcll)
    """

    def __init__(self, root, num_classes_per_task=None, meta_train=False,
                 meta_val=False, meta_test=False, meta_split=None,
                 transform=None, target_transform=None, dataset_transform=None,
                 class_augmentations=None, download=False):
        dataset = MiniImagenetAttributesClassDataset(root, meta_train=meta_train,
                                                     meta_val=meta_val, meta_test=meta_test, meta_split=meta_split,
                                                     transform=transform, class_augmentations=class_augmentations,
                                                     download=download)
        super(MiniImagenetAttributes, self).__init__(dataset, num_classes_per_task,
                                                     target_transform=target_transform,
                                                     dataset_transform=dataset_transform)


class MiniImagenetAttributesClassDataset(ClassDataset):
    folder = 'miniimagenet'
    # Google Drive ID from https://github.com/renmengye/few-shot-ssl-public
    gdrive_id = '16V_ZlkW4SsnNDtnGmaBRq2OoPmUOc5mY'
    gz_filename = 'mini-imagenet.tar.gz'
    gz_md5 = 'b38f1eb4251fb9459ecc8e7febf9b2eb'
    pkl_filename = 'mini-imagenet-cache-{0}.pkl'

    filename = '{0}_data.hdf5'
    filename_labels = '{0}_labels.json'

    def __init__(self, root, meta_train=False, meta_val=False, meta_test=False,
                 meta_split=None, transform=None, class_augmentations=None,
                 download=False):
        super(MiniImagenetAttributesClassDataset, self).__init__(meta_train=meta_train,
                                                                 meta_val=meta_val, meta_test=meta_test,
                                                                 meta_split=meta_split,
                                                                 class_augmentations=class_augmentations)

        self.root = os.path.join(os.path.expanduser(root), self.folder)
        self.transform = transform

        self.split_filename = os.path.join(self.root,
                                           self.filename.format(self.meta_split))
        self.split_filename_labels = os.path.join(self.root,
                                                  self.filename_labels.format(self.meta_split))

        self._data = None
        self._labels = None
        self._attributes = None
        self._attributes_file = os.path.join(self.root, 'attributes.csv')

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('MiniImagenet attributes integrity check failed')
        self._num_classes = len(self.labels)

        self.label_to_label_idx = OrderedDict()
        all_labels = []
        for split in ['train', 'val', 'test']:
            with open(os.path.join(self.root, self.filename_labels.format(split)), 'r') as f:
                labels_split = json.load(f)
                all_labels.extend(labels_split)
        all_labels = sorted(all_labels)
        self.label_idx_to_label = []
        for idx, label in enumerate(all_labels):
            self.label_to_label_idx[label] = idx
            self.label_idx_to_label.append(label)

        assets = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'assets', 'miniimagenet-attributes')

        self._attribute_names_file = os.path.join(assets, 'attributes.txt')

        sw = set(stopwords.words())
        self.attribute_words = []
        for line in sorted(open(self._attribute_names_file, 'r').readlines()):
            line = line.replace(':', '_').split('_')
            words = []
            all_words = []
            for word in line:
                w = ''
                for c in word:
                    if c.isalpha():
                        w += c.lower()
                all_words.append(w)
                if w not in sw and len(w) > 0:
                    words.append(w)
            if len(words) == 0:
                self.attribute_words.append(all_words)
            else:
                self.attribute_words.append(words)

    def __getitem__(self, index):
        class_name = self.labels[index % self.num_classes]
        data = self.data[class_name]
        transform = self.get_transform(index, self.transform)
        target_transform = self.get_target_transform(index)
        attributes = self.attributes[self.label_to_label_idx[class_name]]

        def bin_attributes2ints(atts):
            new_atts = []
            for idx, e in enumerate(atts):
                if e == 1:
                    new_atts.append(idx)
            return new_atts

        attributes = bin_attributes2ints(attributes)

        return MiniImagenetAttributesDataset(index, data, class_name, attributes,
                                             transform=transform, target_transform=target_transform)

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def data(self):
        if self._data is None:
            self._data_file = h5py.File(self.split_filename, 'r')
            self._data = self._data_file['datasets']
        return self._data

    @property
    def labels(self):
        if self._labels is None:
            with open(self.split_filename_labels, 'r') as f:
                self._labels = json.load(f)
        return self._labels

    @property
    def attributes(self):
        if self._attributes is None:
            self._attributes = np.genfromtxt(self._attributes_file, delimiter=' ')
        return self._attributes

    def _check_integrity(self):
        return (os.path.isfile(self.split_filename)
                and os.path.isfile(self.split_filename_labels) and os.path.isfile(self._attributes_file))

    def close(self):
        if self._data_file is not None:
            self._data_file.close()
            self._data_file = None
            self._data = None

    def download(self):
        import tarfile

        if self._check_integrity():
            return

        download_file_from_google_drive(self.gdrive_id, self.root,
                                        self.gz_filename, md5=self.gz_md5)

        filename = os.path.join(self.root, self.gz_filename)
        with tarfile.open(filename, 'r') as f:
            f.extractall(self.root)

        for split in ['train', 'val', 'test']:
            filename = os.path.join(self.root, self.filename.format(split))
            if os.path.isfile(filename):
                continue

            pkl_filename = os.path.join(self.root, self.pkl_filename.format(split))
            if not os.path.isfile(pkl_filename):
                raise IOError()
            with open(pkl_filename, 'rb') as f:
                data = pickle.load(f)
                images, classes = data['image_data'], data['class_dict']

            with h5py.File(filename, 'w') as f:
                group = f.create_group('datasets')
                for name, indices in classes.items():
                    group.create_dataset(name, data=images[indices])

            labels_filename = os.path.join(self.root, self.filename_labels.format(split))
            with open(labels_filename, 'w') as f:
                labels = sorted(list(classes.keys()))
                json.dump(labels, f)

            if os.path.isfile(pkl_filename):
                os.remove(pkl_filename)

        tar_folder, _ = os.path.splitext(os.path.join(self.root, self.gz_filename))

        assets = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'assets', 'miniimagenet-attributes')

        anns = torch.load(os.path.join(assets, 'imnet_attributes_mapped.dat')).numpy()

        mini_labels = []
        for line in open(os.path.join(assets, 'miniimagenet_labels.txt'), 'r').readlines():
            if line.startswith('n'):
                mini_labels.append(line.split()[0])
        all_labels = []
        for line in open(os.path.join(assets, 'imagenet_labels.txt'), 'r').readlines():
            if line.startswith('n'):
                all_labels.append(line.split()[0])
        class_annotations = []
        for i, mini_label in zip(range(100), mini_labels):
            found = False
            for j, all_label in zip(range(1000), all_labels):
                if mini_label == all_label:
                    class_annotations.append(anns[j])
                    found = True
                    break
            if not found:
                raise RuntimeError('Missing label')

        class_annotations = np.array(class_annotations)

        np.savetxt(self._attributes_file, class_annotations, delimiter=' ')


class MiniImagenetAttributesDataset(Dataset):
    def __init__(self, index, data, class_name, attributes,
                 transform=None, target_transform=None):
        super(MiniImagenetAttributesDataset, self).__init__(index, transform=transform,
                                                            target_transform=target_transform)
        self.data = data
        self.class_name = class_name
        self.attributes = attributes

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        image = Image.fromarray(self.data[index])
        target = self.class_name

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            target = self.target_transform(target)

        attributes = self.attributes

        return image, target, attributes

def write_cars(filename_labels):
    cwd = os.getcwd()
    data_path = join(cwd,'cars_train')
    savedir = './'
    dataset_list = ['train','val','test']

    data_list = np.array(loadmat(os.path.join(cwd,'devkit/cars_train_annos.mat'))['annotations'][0])
    class_list = np.array([elem[0] for elem in loadmat(os.path.join(cwd,'devkit/cars_meta.mat'))['class_names'][0]])
    classfile_list_all = [[] for i in range(len(class_list))]


    for i in range(len(data_list)):
      folder_path = join(data_path, data_list[i][-1][0])
      classfile_list_all[data_list[i][-2][0][0] - 1].append(folder_path)

    for i in range(len(classfile_list_all)):
      random.shuffle(classfile_list_all[i])
    for dataset in dataset_list:
        file_list = []
        label_list = []
        for i, classfile_list in enumerate(classfile_list_all):
            if 'train' in dataset:
                if (i%2 == 0):
                    file_list = file_list + classfile_list
                    label_list = label_list + np.repeat(i, len(classfile_list)).tolist()
            if 'val' in dataset:
                if (i%4 == 1):
                    file_list = file_list + classfile_list
                    label_list = label_list + np.repeat(i, len(classfile_list)).tolist()
            if 'test' in dataset:
                if (i%4 == 3):
                    file_list = file_list + classfile_list
                    label_list = label_list + np.repeat(i, len(classfile_list)).tolist()

        fo = open(savedir + filename_labels.format(dataset), "w")
        fo.write('{"label_names": [')
        fo.writelines(['"%s",' % item for item in class_list])
        fo.seek(0, os.SEEK_END)
        fo.seek(fo.tell()-1, os.SEEK_SET)
        fo.write('],')

        fo.write('"image_names": [')
        fo.writelines(['"%s",' % item  for item in file_list])
        fo.seek(0, os.SEEK_END)
        fo.seek(fo.tell()-1, os.SEEK_SET)
        fo.write('],')

        fo.write('"image_labels": [')
        fo.writelines(['%d,' % item  for item in label_list])
        fo.seek(0, os.SEEK_END)
        fo.seek(fo.tell()-1, os.SEEK_SET)
        fo.write(']}')

        fo.close()
        print("%s -OK" %dataset)

def write_places(filename_labels):
    cwd = os.getcwd()
    data_path = join(cwd,'places365_standard/train')
    savedir = './'
    dataset_list = ['train','val','test']


    folder_list = [f for f in listdir(data_path) if isdir(join(data_path, f))]
    folder_list.sort()
    label_dict = dict(zip(folder_list,range(0,len(folder_list))))

    classfile_list_all = []

    for i, folder in enumerate(folder_list):
        folder_path = join(data_path, folder)
        cfs = [cf for cf in listdir(folder_path) if (isfile(join(folder_path,cf)) and cf[0] != '.')]
        cfs.sort()
        cfs = cfs[:200]
        classfile_list_all.append([ join(folder_path, cf) for cf in cfs])
        random.shuffle(classfile_list_all[i])
    print(len(classfile_list_all))

    for dataset in dataset_list:
        file_list = []
        label_list = []
        for i, classfile_list in enumerate(classfile_list_all):
            if 'train' in dataset:
                if (i%2 == 0):
                    file_list = file_list + classfile_list
                    label_list = label_list + np.repeat(i, len(classfile_list)).tolist()
            if 'val' in dataset:
                if (i%4 == 1):
                    file_list = file_list + classfile_list
                    label_list = label_list + np.repeat(i, len(classfile_list)).tolist()
            if 'test' in dataset:
                if (i%4 == 3):
                    file_list = file_list + classfile_list
                    label_list = label_list + np.repeat(i, len(classfile_list)).tolist()

        fo = open(savedir + filename_labels.format(dataset), "w")
        fo.write('{"label_names": [')
        fo.writelines(['"%s",' % item  for item in folder_list])
        fo.seek(0, os.SEEK_END)
        fo.seek(fo.tell()-1, os.SEEK_SET)
        fo.write('],')

        fo.write('"image_names": [')
        fo.writelines(['"%s",' % item  for item in file_list])
        fo.seek(0, os.SEEK_END)
        fo.seek(fo.tell()-1, os.SEEK_SET)
        fo.write('],')

        fo.write('"image_labels": [')
        fo.writelines(['%d,' % item  for item in label_list])
        fo.seek(0, os.SEEK_END)
        fo.seek(fo.tell()-1, os.SEEK_SET)
        fo.write(']}')

        fo.close()
        print("%s -OK" %dataset)

def write_plantae(filename_labels):
    cwd = os.getcwd()
    source_path = join(cwd,'Plantae')
    data_path = join(cwd,'images')
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    savedir = './'
    dataset_list = ['train','val','test']


    folder_list = [f for f in listdir(source_path) if isdir(join(source_path, f))]
    #folder_list.sort()
    folder_list_count = np.array([len(listdir(join(source_path, f))) for f in folder_list])
    folder_list_idx = np.argsort(folder_list_count)
    folder_list = np.array(folder_list)[folder_list_idx[-200:]].tolist()
    label_dict = dict(zip(folder_list,range(0,len(folder_list))))

    classfile_list_all = []

    for i, folder in enumerate(folder_list):
        source_folder_path = join(source_path, folder)
        folder_path = join(data_path, folder)
        classfile_list_all.append( [ cf for cf in listdir(source_folder_path) if (isfile(join(source_folder_path,cf)) and cf[0] != '.')])
        random.shuffle(classfile_list_all[i])
        classfile_list_all[i] = classfile_list_all[i][:min(len(classfile_list_all[i]), 600)]

        call('mkdir ' + folder_path, shell=True)
        for cf in classfile_list_all[i]:
          call('cp ' + join(source_folder_path, cf) + ' ' + join(folder_path, cf), shell=True)
        classfile_list_all[i] = [join(folder_path, cf) for cf in classfile_list_all[i]]

    for dataset in dataset_list:
        file_list = []
        label_list = []
        for i, classfile_list in enumerate(classfile_list_all):
            if 'base' in dataset:
                if (i%2 == 0):
                    file_list = file_list + classfile_list
                    label_list = label_list + np.repeat(i, len(classfile_list)).tolist()
            if 'val' in dataset:
                if (i%4 == 1):
                    file_list = file_list + classfile_list
                    label_list = label_list + np.repeat(i, len(classfile_list)).tolist()
            if 'novel' in dataset:
                if (i%4 == 3):
                    file_list = file_list + classfile_list
                    label_list = label_list + np.repeat(i, len(classfile_list)).tolist()

        fo = open(savedir + filename_labels.format(dataset), "w")
        fo.write('{"label_names": [')
        fo.writelines(['"%s",' % item  for item in folder_list])
        fo.seek(0, os.SEEK_END)
        fo.seek(fo.tell()-1, os.SEEK_SET)
        fo.write('],')

        fo.write('"image_names": [')
        fo.writelines(['"%s",' % item  for item in file_list])
        fo.seek(0, os.SEEK_END)
        fo.seek(fo.tell()-1, os.SEEK_SET)
        fo.write('],')

        fo.write('"image_labels": [')
        fo.writelines(['%d,' % item  for item in label_list])
        fo.seek(0, os.SEEK_END)
        fo.seek(fo.tell()-1, os.SEEK_SET)
        fo.write(']}')

        fo.close()
        print("%s -OK" %dataset)

def cpp_attributes(folder, shots, ways, dataset, shuffle=True, test_shots=None, seed=None,
                   helper=torchmeta.datasets.helpers.helper_with_default, **kwargs):
    image_size = 84
    defaults = {
        'transform': Compose([
                        Resize(int(image_size * 1.5)),
                        CenterCrop(image_size),
                        ToTensor()
                    ])
    }

    return helper(CPPAttributes, folder, shots, ways, dataset, shuffle=shuffle, test_shots=test_shots, seed=seed,
                  defaults=defaults,
                  **kwargs)

class CPPAttributes(CombinationMetaDataset):

    def __init__(self, root, dataset, num_classes_per_task=None, meta_train=False,
                 meta_val=False, meta_test=False, meta_split=None,
                 transform=None, target_transform=None, dataset_transform=None,
                 class_augmentations=None, download=False):
        dataset = CPPAttributesClassDataset(root, dataset=dataset, meta_train=meta_train, meta_val=meta_val,
                                            meta_test=meta_test, meta_split=meta_split, transform=transform,
                                            class_augmentations=class_augmentations, download=download)
        super(CPPAttributes, self).__init__(dataset, num_classes_per_task,
                                            target_transform=target_transform,
                                            dataset_transform=dataset_transform)

class CPPAttributesClassDataset(ClassDataset):
    folder = 'cpp-attributes'
    image_folder = 'CPP'
    filename_labels = '{0}_labels.json'

    def __init__(self, root, dataset, meta_train=False, meta_val=False, meta_test=False,
                 meta_split=None, transform=None, class_augmentations=None,
                 download=False, seed=42, custom_split=True):
        super(CPPAttributesClassDataset, self).__init__(meta_train=meta_train,
                                                        meta_val=meta_val, meta_test=meta_test, meta_split=meta_split,
                                                        class_augmentations=class_augmentations)

        self.root = os.path.join(os.path.expanduser(root), self.folder)
        self.transform = transform
        self.dataset = dataset
        self.split_filename_labels = os.path.join(self.root, self.image_folder, self.dataset,
                                                  self.filename_labels.format(self.meta_split))
        self.assets = os.path.join(os.path.abspath(os.path.dirname('')))
        self.transform = transform
        if download:
            self.download()

        self._data = None
        self._labels = None
        self._attributes = None

        self._num_classes = len(set(self.labels))
        self.data_grouped = {}
        for elem in set(self.labels):
            self.data_grouped[elem]=[]
        for i in range(len(self.data)):
            self.data_grouped[self.labels[i]].append(self.data[i])


    @property
    def num_classes(self):
        return self._num_classes

    @property
    def data(self):
        if self._data is None:
            with open("./"+self.split_filename_labels, 'r') as f:
                self._data = json.load(f)['image_names']
        return self._data

    @property
    def labels(self):
        if self._labels is None:
            with open("./"+self.split_filename_labels, 'r') as f:
                self._labels = json.load(f)['image_labels']
        return self._labels

    @property
    def attributes(self):
        if self._attributes is None:
            with open("./"+self.split_filename_labels, 'r') as f:
                self._attributes = json.load(f)['label_names']
        return self._attributes

    def close(self):
        if self._data is not None:
            del self._data

    def __getitem__(self, index):
        actual_label = list(set(self.labels))[index]
        image = self.data_grouped[actual_label]
        transform = self.get_transform(index, self.transform)
        target_transform = self.get_target_transform(index)
        return CPPAttributesDataset(index, image, actual_label, self.attributes, transform=transform,
                                    target_transform=target_transform)


    def download(self):
        """
        Download Dataset depending on self.dataset which needs to be passed
        """
        image_folder = os.path.join(self.root, self.image_folder, self.dataset)
        if not os.path.exists(image_folder):
                os.makedirs(image_folder)
        os.chdir(image_folder)
        if self.dataset == 'cars':
            if not (os.path.exists('cars_train') and os.path.exists('devkit')):
                print("Downloading dataset")
                call('wget http://imagenet.stanford.edu/internal/car196/cars_train.tgz', shell=True)
                call('tar -zxf cars_train.tgz', shell=True)
                call('wget https://ai.stanford.edu/~jkrause/cars/car_devkit.tgz', shell=True)
                call('tar -zxf car_devkit.tgz', shell=True)
                write_cars(self.filename_labels)
        elif self.dataset == 'places':
            if not os.path.exists('places365standard_easyformat'):
                call('wget http://data.csail.mit.edu/places/places365/places365standard_easyformat.tar', shell=True)
                call('tar -xf places365standard_easyformat.tar', shell=True)
                write_places(self.filename_labels)
        elif self.dataset == 'plantae':
            if not os.path.exists('plantae'):
                call('wget http://vllab.ucmerced.edu/ym41608/projects/CrossDomainFewShot/filelists/plantae.tar.gz', shell=True)
                call('tar -xzf plantae.tar.gz', shell=True)
                write_plantae(self.filename_labels)
        os.chdir(self.assets)

class CPPAttributesDataset(Dataset):
    def __init__(self, index, data, label, attributes,
                 transform=None, target_transform=None):
        super(CPPAttributesDataset, self).__init__(index, transform=transform,
                                                   target_transform=target_transform)
        self.data = data
        self.label = label
        self.attributes = attributes

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image = Image.open(self.data[index]).convert('RGB')
        target = self.label

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            target = self.target_transform(target)

        attributes = self.attributes

        return image, target, attributes


if __name__ == '__main__':
    def get_dataset_and_loader(meta_split='train', batch_size=2, shuffle=True):
        dataset_helper = sun_attributes

        class_augmentations = None

        dataset = dataset_helper('data', shots=5,
                                 ways=5, shuffle=shuffle,
                                 test_shots=32,
                                 meta_split=meta_split, download=True,
                                 helper=helper_with_default_uniform_splitter,
                                 class_augmentations=class_augmentations)
        loader = AttributesBatchMetaDataLoader
        dataloader = loader(
            dataset, batch_size=batch_size,
            shuffle=shuffle,
            num_workers=1
        )
        return dataset, dataloader
    dataset, dataloader = get_dataset_and_loader()
    for batch in dataloader:
        pass
