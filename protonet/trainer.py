import os
from warnings import warn
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
# from torchmeta.datasets.helpers import miniimagenet, omniglot, cub
from torchmeta.utils.data import BatchMetaDataLoader
from tqdm import tqdm
from protonet.losses import macro_soft_f1_loss
from protonet.model import AuxiliarNetwork, PrototypicalNetwork, SimpAux
from protonet.utils import (get_per_attribute_prototypes,
                            multilabel_prototypical_loss, get_prototypes,
                            get_accuracy, plot_grad_flow, prototypical_loss)
import logging
from protonet.torchmeta_utils import omniglot, miniimagenet, cub, cub_attributes, helper_with_default_uniform_splitter,\
    sun_attributes, miniimagenet_attributes
from protonet.torchmeta_utils import AttributesBatchMetaDataLoader, NonEpisodicAttributeDatasetWrapper
import random
from sklearn.metrics import (average_precision_score, classification_report,
                             accuracy_score, f1_score)
from collections import OrderedDict
import fasttext
import fasttext.util
import pytorch_warmup as warmup


class MultiLabelLoss(nn.Module):
    def __init__(self, n_attributes, loss_fn, device, ignore=0.8):
        assert loss_fn in ['bce', 'mlsm']
        super().__init__()
        self.n_attributes = n_attributes
        self.device = device
        self.ignore = ignore
        if ignore == 0:
            self.loss = nn.MultiLabelSoftMarginLoss() if loss_fn == 'mlsm' else nn.BCEWithLogitsLoss()
        else:
            self.loss = nn.MultiLabelSoftMarginLoss(reduction='none') \
                if loss_fn == 'mlsm' else nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, output, attributes):
        if self.ignore == 0:
            return self.loss(output, attributes)

        # Ignore N% of the negative attribute labels to balance the number of positive and negative examples
        # https://github.com/eeishaan/few_shot_recognition/blob/master/compositional_analyzer.py
        if attributes.dim() == 1:
            attributes = attributes.unsqueeze(0)
        loss_mask = torch.ones(output.shape[0], self.n_attributes).to(self.device)
        # ignore 80% of the attributes attribute labels to balance the number of positive and negative examples
        for i in range(attributes.shape[0]):
            zeros = (attributes[i, :] == 0).nonzero()
            indices = torch.randperm(len(zeros.squeeze()))[:int(round(len(zeros) * self.ignore))]
            indices = zeros.squeeze()[indices]
            #  indices = np.random.choice(zeros.squeeze(), int(round(len(zeros) * self.ignore)), False)
            att_non_zero = attributes[i].nonzero()
            if att_non_zero.shape[0] == 1:
                att_non_zero = att_non_zero[0]
            else:
                att_non_zero = att_non_zero.squeeze()
            assert len(set(att_non_zero).intersection(set(indices))) == 0
            loss_mask[i, indices] = 0

        # ignore samples without attribute annotations
        index_empty = attributes.sum(1).squeeze() == 0

        # output[index_empty] = 0
        # attributes[index_empty] = 0
        # output *= loss_mask
        # attributes *= loss_mask
        empty_mask = torch.ones(output.shape[0], self.n_attributes).to(self.device)
        empty_mask[index_empty] = 0
        output = output * empty_mask
        output = output * loss_mask
        attributes = attributes * empty_mask
        attributes = attributes * loss_mask
        loss = self.loss(output, attributes)
        # loss[index_empty] = 0
        # loss *= loss_mask

        return loss.mean()

# TODO: Check F1 metrics aggregations


class Trainer(object):
    def __init__(self, args, exp_dir):
        self.args = args
        self.exp_dir = exp_dir

    @staticmethod
    def group_weight(module):
        group_decay = []
        group_no_decay = []
        for m in module.modules():
            if isinstance(m, nn.Linear):
                group_decay.append(m.weight)
                if m.bias is not None:
                    group_no_decay.append(m.bias)
            elif isinstance(m, torch.nn.modules.conv._ConvNd):
                group_decay.append(m.weight)
                if m.bias is not None:
                    group_no_decay.append(m.bias)
            elif isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                if m.bias is not None:
                    group_no_decay.append(m.weight)
                if m.bias is not None:
                    group_no_decay.append(m.bias)

        assert len(list(module.parameters())) == len(group_decay) + len(group_no_decay)
        groups = [dict(params=group_decay), dict(params=group_no_decay, weight_decay=.0)]
        return groups

    @staticmethod
    def aux_metrics(predicted, target):
        predicted = torch.sigmoid(predicted).round().long()
        target = target.round()
        accuracy = (predicted == target).sum().float() / predicted.numel()
        all_match_ratio = ((predicted == target).all(1).float()).sum() / predicted.shape[0]
        acc_score = accuracy_score(y_pred=predicted.cpu(), y_true=target.cpu())  # should be same as all match ratio
        f1_micro = f1_score(y_pred=predicted.cpu(), y_true=target.cpu(), average='micro')
        f1_macro = f1_score(y_pred=predicted.cpu(), y_true=target.cpu(), average='macro')
        class_rep = classification_report(y_pred=predicted.cpu(), y_true=target.cpu())
        return dict(accuracy=accuracy.cpu(), all_match_ratio=all_match_ratio.cpu(), acc_score=acc_score,
                    f1_micro=f1_micro,
                    f1_macro=f1_macro, class_rep=class_rep)

    def get_dataset_and_loader(self, meta_split, batch_size, shuffle=True):
        if self.args.dataset == 'omniglot':
            dataset_helper = omniglot
        elif self.args.dataset == 'miniimagenet':
            dataset_helper = miniimagenet
        elif self.args.dataset == 'cub':
            dataset_helper = cub
        elif self.args.dataset == 'cub-attributes':
            dataset_helper = cub_attributes
        elif self.args.dataset == 'sun-attributes':
            dataset_helper = sun_attributes
        elif self.args.dataset == 'miniimagenet-attributes':
            dataset_helper = miniimagenet_attributes
        else:
            raise NotImplementedError(self.args.dataset)

        class_augmentations = None  # if not args.augment else [RandomResizedCrop(84), ColorJitter(brightness=0.4,
        #                                                                                        contrast=0.4, saturation=0.4, hue=0),
        #                                                     RandomHorizontalFlip()]
        if self.args.queries_sampling == 'per-class':
            dataset = dataset_helper(self.args.data_folder, shots=self.args.num_shots,
                                     ways=self.args.num_ways, shuffle=shuffle,
                                     test_shots=self.args.num_test_shots,
                                     meta_split=meta_split, download=self.args.download,
                                     class_augmentations=class_augmentations)
        elif self.args.queries_sampling == 'uniform':
            dataset = dataset_helper(self.args.data_folder, shots=self.args.num_shots,
                                     ways=self.args.num_ways, shuffle=shuffle,
                                     test_shots=self.args.num_test_shots,
                                     meta_split=meta_split, download=self.args.download,
                                     helper=helper_with_default_uniform_splitter,
                                     class_augmentations=class_augmentations)
        else:
            raise NotImplementedError('args.queries_sampling', self.args.queries_sampling)

        if self.args.dataset in ['cub-attributes', 'sun-attributes', 'miniimagenet-attributes']:
            loader = AttributesBatchMetaDataLoader
        else:
            loader = BatchMetaDataLoader
        dataloader = loader(
            dataset, batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.args.num_workers
        )
        return dataset, dataloader

    def init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            #nn.init.normal_(m.weight, std=self.args.weight_init_variance)
            #nn.init.constant_(m.bias, 0.0)

            #nn.init.xavier_normal_(m.weight)

            #in_channels = 3
            #height = 84
            #width = 8
            ## TODO: don't hardcode these values
            #n = in_channels * height * width
            #std = np.sqrt(self.args.weight_init_variance * 2.0 / n)
            #mean = 0.0

            #nn.init.normal_(m.weight, std=std, mean=mean)
            #if hasattr(m, 'bias') and m.bias is not None:
            #    nn.init.zeros_(m.bias)
            pass
        elif hasattr(m, 'zero_init') and isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            nn.init.constant_(m.weight, 0)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    @staticmethod
    def print_metrics(step, metrics):
        val_acc = metrics['val_acc'][-1]
        train_acc = metrics['train_acc'][-1]
        val_loss = metrics['val_loss'][-1]
        train_loss = metrics['train_loss'][-1]
        best_val_loss = metrics['best_val_loss']
        best_val_step = metrics['best_val_step']
        test_acc_at_best_val_step = metrics['test_acc_at_best_val_step']

        logging.info(f'step = {step} | val_acc = {val_acc} | train_acc = {train_acc} | val_loss = {val_loss} |'
                     f' train_loss = {train_loss} | best_val_loss = {best_val_loss} | best_val_step = {best_val_step}'
                     f' test_acc_at_best_val_step = {test_acc_at_best_val_step}\n')

    def train(self):
        seed = self.args.seed
        os.environ['PYTHONHASHSEED'] = str(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
        np.random.seed(seed)  # Numpy module.
        random.seed(seed)  # Python random module.
        torch.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        return self._train()

    def evaluate(self, eval_args):
        seed = eval_args.seed
        os.environ['PYTHONHASHSEED'] = str(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
        np.random.seed(seed)  # Numpy module.
        random.seed(seed)  # Python random module.
        torch.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        if self.args.dataset in ['cub', 'miniimagenet', 'sun']:
            in_channels = 3
        elif self.args.dataset == 'cub-attributes':
            in_channels = 3
            self.n_atts = 312
        elif self.args.dataset == 'sun-attributes':
            in_channels = 3
            self.n_atts = 102
        elif self.args.dataset == 'miniimagenet-attributes':
            in_channels = 3
            self.n_atts = 160
        else:
            raise NotImplementedError(self.args.dataset)
        model = self.load_arch(in_channels)
        model.to(device=eval_args.device)
        if eval_args.checkpoint is None:
            model = self.load_model_for_evaluation(model)
        else:
            model = self.load_model_for_evaluation(model, os.path.join(eval_args.model_path, eval_args.checkpoint))
        with torch.no_grad():
            test_metrics, _ = self.eval_one_split(
                model, meta_split='test',
                num_cases=eval_args.num_cases_test
            )
            if isinstance(test_metrics, float):
                logging.info(f'Test accuracy {test_metrics}')
            else:
                self.print_metrics('EVAL', test_metrics)

    def load_arch(self, in_channels):
        model = PrototypicalNetwork(
            in_channels,
            num_layers_per_block=self.args.num_layers_per_block,
            num_blocks=self.args.num_blocks,
            max_pool_kernel_size=self.args.max_pool_kernel_size,
            num_channels=self.args.num_channels,
            num_channels_growth_factor=self.args.num_channels_growth_factor,
            num_max_pools=self.args.num_max_pools,
            activation=self.args.activation,
            batchnorm=self.args.batchnorm,
            dropout=self.args.dropout,
            kernel_size=self.args.kernel_size,
            conv_bias=self.args.conv_bias,
            bn_epsilon=self.args.bn_epsilon,
            bn_momentum=self.args.bn_momentum
        )
        return model

    def _train(self):
        dataset, dataloader = self.get_dataset_and_loader(
            meta_split='train', batch_size=self.args.batch_size
        )

        if self.args.dataset == 'omniglot':
            in_channels = 1
        elif self.args.dataset in ['miniimagenet', 'cub', 'cub-attributes', 'sun-attributes']:
            in_channels = 3
        else:
            raise NotImplementedError(self.args.dataset)
        model = self.load_arch(in_channels)
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logging.info(f'Training {params} parameters')

        model.to(device=self.args.device)

        # init weights
        if self.args.custom_init:
            model.apply(self.init_weights)

        model.train()

        writer = SummaryWriter(os.path.join(self.exp_dir, 'tensorboard'))

        if self.args.no_decay_bias_bn:
            weight_decay_groups = self.group_weight(model)
            weight_decay_groups[0]['weight_decay'] = self.args.weight_decay
            optimizer = torch.optim.SGD(weight_decay_groups, lr=self.args.init_learning_rate,
                                        momentum=self.args.momentum)
        else:
            optimizer = torch.optim.SGD(model.parameters(), weight_decay=self.args.weight_decay,
                                        lr=self.args.init_learning_rate,
                                        momentum=self.args.momentum)
        if self.args.learning_rate_schedule is not None:
            if self.args.learning_rate_schedule == 'pwc':
                scheduler = torch.optim.lr_scheduler.MultiStepLR(
                    optimizer,
                    milestones=[self.args.num_batches / 2,
                                self.args.num_batches / 2 + self.args.pwc_decay_interval,
                                self.args.num_batches / 2 + 2 * self.args.pwc_decay_interval],
                    gamma=.1
                )
            else:
                raise ValueError('Unsupported learning rate schedule')
        else:
            scheduler = None

        if self.args.warmup != -1:
            warmup_scheduler = warmup.LinearWarmup(optimizer, warmup_period=self.args.warmup)

        metrics = {
            'steps': [],
            'val_acc': [],
            'train_acc': [],
            'val_loss': [],
            'train_loss': [],
            'best_val_loss': np.inf,
            'best_val_step': -1,
            'test_acc_at_best_val_step': 0.
        }

        fig = plt.figure()

        # Training loop
        eval_steps_without_improvement = 0
        with tqdm(dataloader, total=self.args.num_batches, disable=self.args.no_bar) as pbar:
            for step, batch in enumerate(pbar):
                model.zero_grad()

                if len(batch['train']) == 2:
                    train_inputs, train_targets = batch['train']
                else:
                    train_inputs, train_targets, _ = batch['train']
                train_inputs = train_inputs.to(device=self.args.device)
                train_targets = train_targets.to(device=self.args.device)
                train_embeddings = model(train_inputs)

                if len(batch['test']) == 2:
                    test_inputs, test_targets = batch['test']
                else:
                    test_inputs, test_targets, _ = batch['test']
                test_inputs = test_inputs.to(device=self.args.device)
                test_targets = test_targets.to(device=self.args.device)
                test_embeddings = model(test_inputs)

                prototypes = get_prototypes(train_embeddings, train_targets, dataset.num_classes_per_task)
                loss = prototypical_loss(prototypes, test_embeddings, test_targets)

                loss.backward()
                if (step + 1) % self.args.debug_log_interval == 0:
                    logging.info('Debug step')

                    fig.clf()
                    plot_grad_flow(model.named_parameters())
                    writer.add_figure('grad_flow', fig, global_step=step,
                                      close=False)

                _ = nn.utils.clip_grad_norm_(
                    model.parameters(), self.args.grad_clip_norm, norm_type=2)
                optimizer.step()

                # adapt learning rate
                if self.args.learning_rate_schedule is not None:
                    scheduler.step()

                if self.args.warmup != -1:
                    warmup_scheduler.dampen()

                with torch.no_grad():
                    accuracy = get_accuracy(prototypes, test_embeddings,
                                            test_targets).item()
                    pbar.set_postfix(accuracy=f'{accuracy:.4f}')

                # add summaries
                writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'],
                                  global_step=step)
                writer.add_scalar('metatrain_loss', loss.item(),
                                  global_step=step)
                writer.add_scalar('metatrain_accuracy', accuracy,
                                  global_step=step)

                del train_inputs, train_targets, train_embeddings, \
                    test_inputs, test_targets, test_embeddings, prototypes, loss, \
                    accuracy

                if (step + 1) % self.args.eval_interval_steps == 0:
                    logging.info('Eval step')
                    best = self.eval(model, writer, step, metrics)
                    self.print_metrics(step, metrics)
                    self.save_model(model, optimizer, metrics, scheduler,
                                    suffix='_last')
                    if best:
                        self.save_model(model, optimizer, metrics, scheduler,
                                        suffix='_best')
                        eval_steps_without_improvement = 0
                        logging.info(f'{eval_steps_without_improvement} eval steps without improvement')
                    else:
                        eval_steps_without_improvement += 1
                        logging.info(f'{eval_steps_without_improvement} eval steps without improvement')
                        if self.args.early_stop != -1 and eval_steps_without_improvement >= self.args.early_stop:
                            logging.info('Early stop')
                            break

                if step >= self.args.num_batches:
                    break

        # Save model
        self.save_model(model, optimizer, metrics, scheduler,
                        suffix='_end')

        return best, metrics['best_val_loss']

    def save_metrics(self, metrics):
        torch.save(metrics, os.path.join(self.exp_dir, 'metrics.pt'))

    def eval_one_split(self, model, meta_split, num_cases):
        num_batches = int(np.ceil(num_cases / self.args.eval_batch_size))
        dataset, dataloader = self.get_dataset_and_loader(
            meta_split=meta_split, batch_size=self.args.eval_batch_size
        )

        accuracies = []
        losses = []
        ns = []

        done_batches = 0
        done = False
        while not done:
            with tqdm(dataloader, total=min(num_batches - done_batches, len(dataloader)),
                      disable=self.args.no_bar) as pbar:
                for step, batch in enumerate(pbar):
                    if len(batch['train']) == 2:
                        train_inputs, train_targets = batch['train']
                    else:
                        train_inputs, train_targets, _ = batch['train']
                    train_inputs = train_inputs.to(device=self.args.device)
                    train_targets = train_targets.to(device=self.args.device)
                    train_embeddings = model(train_inputs)

                    if len(batch['test']) == 2:
                        test_inputs, test_targets = batch['test']
                    else:
                        test_inputs, test_targets, _ = batch['test']
                    test_inputs = test_inputs.to(device=self.args.device)
                    test_targets = test_targets.to(device=self.args.device)
                    test_embeddings = model(test_inputs)

                    prototypes = get_prototypes(train_embeddings, train_targets, dataset.num_classes_per_task)

                    batch_loss = prototypical_loss(
                        prototypes, test_embeddings, test_targets).item()
                    batch_accuracy = get_accuracy(
                        prototypes, test_embeddings, test_targets).item()

                    accuracies.append(batch_accuracy)
                    losses.append(batch_loss)
                    ns.append(len(test_inputs))
                    done_batches += 1
                    if done_batches >= num_batches:
                        done = True
                        break
            if not done:
                logging.info('Reloading exhausted eval dataloader...')

        total_n = sum(ns)
        accuracy = sum(acc * n / total_n for acc, n in zip(accuracies, ns))
        loss = sum(l * n / total_n for l, n in zip(losses, ns))
        return accuracy, loss

    def save_model(self, model, optimizer, metrics, scheduler=None, suffix=''):
        filename = os.path.join(
            self.exp_dir,
            f'{self.args.mode}_{self.args.dataset}_{self.args.num_shots}shot_'
            f'{self.args.num_ways}way{suffix}.pt'
        )

        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'metrics': metrics
        }
        if scheduler is not None:
            checkpoint['scheduler'] = scheduler.state_dict()

        # FIXME: rng states
        checkpoint['torch_rng_state'] = torch.get_rng_state()
        checkpoint['numpy_rng_state'] = np.random.get_state()
        checkpoint['python_rng_state'] = random.getstate()

        with open(filename, 'wb') as f:
            torch.save(checkpoint, f)
        return filename

    def load_model(self, model, optimizer, metrics,
                   scheduler=None, filename=None):
        if filename is None:
            # default to loading last checkpoint
            filename = os.path.join(
                self.exp_dir,
                f'{self.args.mode}_{self.args.dataset}_{self.args.num_shots}shot_'
                f'{self.args.num_ways}way_last.pt'
            )
            checkpoint = torch.load(filename)
        else:
            checkpoint = torch.load(os.path.join(self.exp_dir, filename))

        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        if scheduler is not None:
            scheduler.load_state_dict(checkpoint['scheduler'])
        metrics.clear()
        metrics.update(checkpoint['metrics'])
        torch.set_rng_state(checkpoint['torch_rng_state'])
        np.random.set_state(checkpoint['numpy_rng_state'])
        random.setstate(checkpoint['python_rng_state'])
        return model, optimizer, metrics, scheduler

    def load_model_for_evaluation(self, model, filename=None):
        if filename is None:
            # default to loading best checkpoint
            filename = os.path.join(
                self.exp_dir,
                f'{self.args.mode_orig}_{self.args.dataset_orig}_{self.args.num_shots_orig}shot_'
                f'{self.args.num_ways_orig}way_best.pt'
            )
            try:
                checkpoint = torch.load(filename)
            except RuntimeError:
                map_location = torch.device('cpu')
                checkpoint = torch.load(filename, map_location=map_location)
        else:
            try:
                checkpoint = torch.load(os.path.join(self.exp_dir, filename))
            except RuntimeError:
                map_location = torch.device('cpu')
                checkpoint = torch.load(os.path.join(self.exp_dir, filename), map_location=map_location)

        model.load_state_dict(checkpoint['model'])
        return model

    def eval(self, model, writer, step, metrics):
        best = False
        if not self.args.train_mode:
            model.eval()
        with torch.no_grad():
            train_acc, train_loss = self.eval_one_split(
                model, meta_split='train',
                num_cases=self.args.num_cases_val
            )
            val_acc, val_loss = self.eval_one_split(
                model, meta_split='val',
                num_cases=self.args.num_cases_val
            )
            metrics['steps'].append(step)
            metrics['val_acc'].append(val_acc)
            writer.add_scalar('accuracy/val', val_acc, global_step=step)
            metrics['train_acc'].append(train_acc)
            writer.add_scalar('accuracy/train', train_acc, global_step=step)
            metrics['val_loss'].append(val_loss)
            writer.add_scalar('loss/val', val_loss, global_step=step)
            metrics['train_loss'].append(train_loss)
            writer.add_scalar('loss/train', train_loss, global_step=step)
            if val_loss < metrics['best_val_loss']:
                best = True
                logging.info("found new best validation loss, running test eval")
                metrics['best_val_loss'] = val_loss
                metrics['best_val_step'] = step
                test_acc, _ = self.eval_one_split(
                    model, meta_split='test',
                    num_cases=self.args.num_cases_test
                )
                metrics['test_acc_at_best_val_step'] = test_acc
                writer.add_scalar('accuracy/test', test_acc, global_step=step)
                logging.info(f"test accuracy: {test_acc}")
        self.save_metrics(metrics)
        writer.flush()

        model.train()

        return best


class ProtonetTrainer(Trainer):
    pass


class AuxiliarTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_atts = None
        self.criterion = None

    def get_dataset_and_loader(self, meta_split, batch_size, shuffle=True):
        if self.args.dataset == 'omniglot':
            dataset_helper = omniglot
        elif self.args.dataset == 'miniimagenet':
            dataset_helper = miniimagenet
        elif self.args.dataset == 'cub':
            dataset_helper = cub
        elif self.args.dataset == 'cub-attributes':
            dataset_helper = cub_attributes
        elif self.args.dataset == 'sun-attributes':
            dataset_helper = sun_attributes
        elif self.args.dataset == 'miniimagenet-attributes':
            dataset_helper = miniimagenet_attributes
        else:
            raise NotImplementedError(self.args.dataset)

        dataset = dataset_helper(
            self.args.data_folder, shots=self.args.num_shots,
            ways=self.args.num_ways, shuffle=shuffle,
            test_shots=self.args.num_test_shots,
            meta_split=meta_split, download=self.args.download,
            class_augmentations=None)

        dataset = NonEpisodicAttributeDatasetWrapper(dataset)

        dataloader = DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            shuffle=shuffle,
            num_workers=self.args.num_workers,
            pin_memory=True
        )
        return dataset, dataloader

    def eval(self, model, writer, step, metrics):
        best = False
        if not self.args.train_mode:
            model.eval()
        with torch.no_grad():
            train_metrics, train_loss = self.eval_one_split(
                model, meta_split='train',
                num_cases=self.args.num_cases_val
            )
            val_metrics, val_loss = self.eval_one_split(
                model, meta_split='val',
                num_cases=self.args.num_cases_val
            )

            metrics['steps'].append(step)
            metrics['val_acc'].append(val_metrics['accuracy'])
            metrics['val_acc_score'].append(val_metrics['acc_score'])
            metrics['val_f1_micro'].append(val_metrics['f1_micro'])
            metrics['val_f1_macro'].append(val_metrics['f1_macro'])
            metrics['val_amr'].append(val_metrics['all_match_ratio'])
            metrics['train_acc'].append(train_metrics['accuracy'])
            metrics['train_acc_score'].append(train_metrics['acc_score'])
            metrics['train_f1_micro'].append(train_metrics['f1_micro'])
            metrics['train_f1_macro'].append(train_metrics['f1_macro'])
            metrics['train_amr'].append(train_metrics['all_match_ratio'])

            writer.add_scalar('accuracy/val', val_metrics['accuracy'], global_step=step)
            writer.add_scalar('accuracy_score/val', val_metrics['acc_score'], global_step=step)
            writer.add_scalar('f1_micro/val', val_metrics['f1_micro'], global_step=step)
            writer.add_scalar('f1_macro/val', val_metrics['f1_macro'], global_step=step)
            writer.add_scalar('all_match_ratio/val', val_metrics['all_match_ratio'], global_step=step)
            writer.add_scalar('accuracy/train', train_metrics['accuracy'], global_step=step)
            writer.add_scalar('accuracy_score/train', train_metrics['acc_score'], global_step=step)
            writer.add_scalar('f1_micro/train', train_metrics['f1_micro'], global_step=step)
            writer.add_scalar('f1_macro/train', train_metrics['f1_macro'], global_step=step)
            writer.add_scalar('all_match_ratio/train', train_metrics['all_match_ratio'], global_step=step)

            metrics['val_loss'].append(val_loss)
            writer.add_scalar('loss/val', val_loss, global_step=step)
            metrics['train_loss'].append(train_loss)
            writer.add_scalar('loss/train', train_loss, global_step=step)
            if val_loss < metrics['best_val_loss']:
                best = True
                logging.info("found new best validation loss, running test eval")
                metrics['best_val_loss'] = val_loss
                metrics['best_val_step'] = step
                test_metrics, _ = self.eval_one_split(
                    model, meta_split='test',
                    num_cases=self.args.num_cases_test
                )
                metrics['test_acc_at_best_val_step'] = test_metrics['accuracy']
                metrics['test_acc_score_at_best_val_step'] = test_metrics['acc_score']
                metrics['test_f1_micro_at_best_val_step'] = test_metrics['f1_micro']
                metrics['test_f1_macro_at_best_val_step'] = test_metrics['f1_macro']
                metrics['test_amr_at_best_val_step'] = test_metrics['all_match_ratio']

                writer.add_scalar('accuracy/test', test_metrics['accuracy'], global_step=step)
                writer.add_scalar('accuracy_score/test', test_metrics['acc_score'], global_step=step)
                writer.add_scalar('f1_micro/test', test_metrics['f1_micro'], global_step=step)
                writer.add_scalar('f1_macro/test', test_metrics['f1_macro'], global_step=step)
                writer.add_scalar('all_match_ratio/test', test_metrics['all_match_ratio'], global_step=step)
                # logging.info(f"test all_match_ratio: {test_amr}")
        self.save_metrics(metrics)
        writer.flush()

        model.train()

        return best

    def load_arch(self, in_channels):
        model = AuxiliarNetwork(
            in_channels,
            n_outputs=self.n_atts,
            num_layers_per_block=self.args.num_layers_per_block,
            num_blocks=self.args.num_blocks,
            max_pool_kernel_size=self.args.max_pool_kernel_size,
            num_channels=self.args.num_channels,
            num_channels_growth_factor=self.args.num_channels_growth_factor,
            num_max_pools=self.args.num_max_pools,
            activation=self.args.activation,
            batchnorm=self.args.batchnorm,
            dropout=self.args.dropout,
            kernel_size=self.args.kernel_size,
            n_extra_linear=self.args.n_extra_linear,
            conv_bias=self.args.conv_bias,
            bn_epsilon=self.args.bn_epsilon,
            bn_momentum=self.args.bn_momentum
        )
        return model

    def _train(self):
        dataset, dataloader = self.get_dataset_and_loader(
            meta_split='train', batch_size=self.args.batch_size
        )

        if self.args.dataset == 'cub-attributes':
            in_channels = 3
            self.n_atts = 312
        elif self.args.dataset == 'sun-attributes':
            in_channels = 3
            self.n_atts = 102
        elif self.args.dataset == 'miniimagenet-attributes':
            in_channels = 3
            self.n_atts = 160
        else:
            raise NotImplementedError(self.args.dataset)
        model = self.load_arch(in_channels)
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logging.info(f'Training {params} parameters')

        model.to(device=self.args.device)

        # init weights
        if self.args.custom_init:
            model.apply(self.init_weights)
        model.train()

        writer = SummaryWriter(os.path.join(self.exp_dir, 'tensorboard'))

        opt_kwargs = {}
        if self.args.optimizer == 'sgd':
            Opt = torch.optim.SGD
            opt_kwargs['momentum'] = self.args.momentum
        else:
            Opt = torch.optim.Adam
        if self.args.no_decay_bias_bn:
            weight_decay_groups = self.group_weight(model)
            weight_decay_groups[0]['weight_decay'] = self.args.weight_decay
            optimizer = Opt(weight_decay_groups, lr=self.args.init_learning_rate,
                                            **opt_kwargs)
        else:
            optimizer = Opt(model.parameters(), lr=self.args.init_learning_rate,
                                        weight_decay=self.args.weight_decay, **opt_kwargs)

        if self.args.learning_rate_schedule is not None:
            if self.args.learning_rate_schedule == 'pwc':
                scheduler = torch.optim.lr_scheduler.MultiStepLR(
                    optimizer,
                    milestones=[self.args.num_batches / 2,
                                self.args.num_batches / 2 + self.args.pwc_decay_interval,
                                self.args.num_batches / 2 + 2 * self.args.pwc_decay_interval],
                    gamma=.1
                )
            else:
                raise ValueError('Unsupported learning rate schedule')
        else:
            scheduler = None

        if self.args.warmup != -1:
            warmup_scheduler = warmup.LinearWarmup(optimizer, warmup_period=self.args.warmup)

        metrics = {
            'steps': [],
            'val_acc': [],
            'val_amr': [],
            'val_acc_score': [],
            'val_f1_micro': [],
            'val_f1_macro': [],
            'train_acc': [],
            'train_amr': [],
            'train_acc_score': [],
            'train_f1_micro': [],
            'train_f1_macro': [],
            'val_loss': [],
            'train_loss': [],
            'best_val_loss': np.inf,
            'best_val_step': -1,
            'test_acc_at_best_val_step': 0.,
            'test_amr_at_best_val_step': 0.,
            'test_acc_score_at_best_val_step': 0.,
            'test_f1_micro_at_best_val_step': 0.,
            'test_f1_macro_at_best_val_step': 0.
        }

        fig = plt.figure()

        if self.args.aux_loss_fn == 'weighted-bce':
            # Compute pos_weight
            # Assuming all classes have same probability
            attributes = dataset.dataset.dataset.attributes

            def bin_attributes2ints(atts):
                new_atts = []
                for idx, e in enumerate(atts):
                    if e == 1:
                        new_atts.append(idx)
                return new_atts

            # attributes = [bin_attributes2ints(attributes)]
            total = attributes.shape[0]
            attributes = [bin_attributes2ints(a) for a in attributes]

            atts_freq = [0] * self.n_atts
            for class_ in attributes:
                for att in class_:
                    if atts_freq[att] is None:
                        atts_freq[att] = 1
                    else:
                        atts_freq[att] += 1

            atts_freq = list(map(lambda x: 1 if x == 0 else (total - x) / x, atts_freq))
            pos_weight = torch.tensor(atts_freq).to(self.args.device)
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        elif self.args.aux_loss_fn == 'macro-soft-f1':
            self.criterion = macro_soft_f1_loss
        else:
            self.criterion = MultiLabelLoss(self.n_atts, self.args.aux_loss_fn, self.args.device,
                                            self.args.aux_loss_ignore)

        # Training loop
        eval_steps_without_improvement = 0
        first = True
        done = False
        step = 0
        while not done:
            with tqdm(dataloader, initial=step, total=self.args.num_batches, disable=self.args.no_bar) as pbar:
                for batch in pbar:
                    step += 1
                    model.zero_grad()

                    inputs, _, attributes = batch
                    inputs = inputs.to(device=self.args.device)
                    attributes = attributes.to(device=self.args.device, dtype=torch.float32)
                    output = model(inputs)
                    loss = self.criterion(output, attributes)
                    loss.backward()

                    if (step + 1) % self.args.debug_log_interval == 0:
                        logging.info('Debug step')

                        fig.clf()
                        plot_grad_flow(model.named_parameters())
                        writer.add_figure('grad_flow', fig, global_step=step,
                                          close=False)

                        if 'grads' in self.args.debug_opts:
                            for tag, value in model.named_parameters():
                                writer.add_histogram(
                                    f'debug/param_{tag}', value.data,
                                    global_step=step
                                )
                                writer.add_histogram(
                                    f'debug/grad_{tag}', value.grad,
                                    global_step=step
                                )
                        if 'outputs' in self.args.debug_opts:
                            writer.add_histogram(
                                f'debug/outputs',
                                torch.sigmoid(output),
                                global_step=step
                            )
                        if first:
                            writer.add_graph(model, inputs)
                            #  writer.add_graph(criterion, (output, attributes))
                            first = False

                    total_grad_norm = nn.utils.clip_grad_norm_(
                        model.parameters(), self.args.grad_clip_norm, norm_type=2)
                    if 'grads' in self.args.debug_opts:
                        writer.add_scalar('debug/total_grad_norm',
                                          total_grad_norm,
                                          global_step=step)
                    optimizer.step()

                    # adapt learning rate
                    if self.args.learning_rate_schedule is not None:
                        scheduler.step()

                    if self.args.warmup != -1:
                        warmup_scheduler.dampen()

                    with torch.no_grad():
                        # all_match_ratio = ((output.round().long() == attributes).all(1).float()).sum()/output.shape[0]
                        met = self.aux_metrics(output.clone().detach(), attributes.clone().detach())
                        pbar.set_postfix(all_match_ratio=f'{aux_met["all_match_ratio"]:.4f}')

                    # add summaries
                    writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'],
                                      global_step=step)
                    writer.add_scalar('loss', loss.item(),
                                      global_step=step)
                    writer.add_scalar('accuracy', met['accuracy'],
                                      global_step=step)
                    writer.add_scalar('all_match_ratio', met['all_match_ratio'],
                                      global_step=step)

                    del output, inputs, attributes, loss

                    if (step + 1) % self.args.eval_interval_steps == 0:
                        logging.info('Eval step')
                        best = self.eval(model, writer, step, metrics)
                        self.print_metrics(step, metrics)
                        self.save_model(model, optimizer, metrics, scheduler,
                                        suffix='_last')
                        if best:
                            self.save_model(model, optimizer, metrics, scheduler,
                                            suffix='_best')
                            eval_steps_without_improvement = 0
                            logging.info(f'{eval_steps_without_improvement} eval steps without improvement')
                        else:
                            eval_steps_without_improvement += 1
                            logging.info(f'{eval_steps_without_improvement} eval steps without improvement')
                            if self.args.early_stop != -1 and eval_steps_without_improvement == self.args.early_stop:
                                logging.info('Early stop')
                                break

                    if step >= self.args.num_batches:
                        done = True
                        break

        # Save model
        self.save_model(model, optimizer, metrics, scheduler,
                        suffix='_end')

        return best, metrics['best_val_loss']

    @staticmethod
    def print_metrics(step, metrics):
        m = OrderedDict(
            step=step,
            val_acc=metrics['val_acc'][-1],
            val_amr=metrics['val_amr'][-1],
            val_acc_score=metrics['val_acc_score'][-1],
            val_f1_micro=metrics['val_f1_micro'][-1],
            val_f1_macro=metrics['val_f1_macro'][-1],

            train_acc=metrics['train_acc'][-1],
            train_amr=metrics['train_amr'][-1],
            train_acc_score=metrics['train_acc_score'][-1],
            train_f1_micro=metrics['train_f1_micro'][-1],
            train_f1_macro=metrics['train_f1_macro'][-1],

            val_loss=metrics['val_loss'][-1],
            train_loss=metrics['train_loss'][-1],
            best_val_loss=metrics['best_val_loss'],
            best_val_step=metrics['best_val_step'],

            test_acc_at_best_val_step=metrics['test_acc_at_best_val_step'],
            test_amr_at_best_val_step=metrics['test_amr_at_best_val_step'],
            test_acc_score_at_best_val_step=metrics['test_acc_score_at_best_val_step'],
            test_f1_micro_at_best_val_step=metrics['test_f1_micro_at_best_val_step'],
            test_f1_macro_at_best_val_step=metrics['test_f1_macro_at_best_val_step']
        )
        res = ''
        for key in m:
            res += f'{key} = {m[key]} | '
        res += '\n'
        # FIXME: KeyError: 'train_class_rep'
        # res += metrics['train_class_rep'][-1] + '\n'
        # res += metrics['val_class_rep'][-1] + '\n'
        # res += '\n'

        logging.info(res)

    def eval_one_split(self, model, meta_split, num_cases):
        num_batches = int(np.ceil(num_cases / self.args.eval_batch_size))
        dataset, dataloader = self.get_dataset_and_loader(
            meta_split=meta_split, batch_size=self.args.eval_batch_size,
            shuffle=False
        )

        ms = []
        losses = []
        ns = []

        done_batches = 0
        done = False
        while not done:
            with tqdm(dataloader, total=min(num_batches - done_batches, len(dataloader)),
                      disable=self.args.no_bar) as pbar:
                for step, batch in enumerate(pbar):
                    inputs, _, attributes = batch
                    attributes = attributes.to(self.args.device)
                    output = model(inputs.to(self.args.device))
                    batch_loss = self.criterion(output, attributes)

                    batch_ms = self.aux_metrics(output, attributes)
                    del output

                    ms.append(batch_ms)
                    losses.append(batch_loss)
                    ns.append(len(inputs))
                    done_batches += 1
                    if done_batches >= num_batches:
                        done = True
                        break
            if not done:
                logging.info('Reloading exhausted eval dataloader...')

        total_n = sum(ns)
        reduced_metrics = self.reduce_aux_metrics(ms, total_n, ns)
        # sum(acc * n/total_n for acc, n in zip(ms, ns))
        loss = sum(l * n / total_n for l, n in zip(losses, ns))
        return reduced_metrics, loss

    @staticmethod
    def reduce_aux_metrics(ms, total_n, ns):
        met_dict = OrderedDict()
        for key in ms[0]:
            if key not in ['class_rep']:
                met_dict[key] = sum(m * n / total_n for m, n in zip([values[key] for values in ms], ns))
        return met_dict


class AuxiliarEmbeddingsTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_atts = None
        self.criterion = None
        self.word_emb_dim = self.args.word_embeddings_dim
        fasttext.util.download_model('en', if_exists='ignore')
        self.ft = fasttext.load_model('cc.en.300.bin')
        if self.ft.get_dimension() > self.word_emb_dim:
            fasttext.util.reduce_model(self.ft, self.word_emb_dim)

    def eval(self, model, writer, step, metrics):
        best = False
        if not self.args.train_mode:
            model.eval()
        with torch.no_grad():
            _, train_loss = self.eval_one_split(
                model, meta_split='train',
                num_cases=self.args.num_cases_val
            )
            _, val_loss = self.eval_one_split(
                model, meta_split='val',
                num_cases=self.args.num_cases_val
            )
            metrics['steps'].append(step)
            metrics['val_loss'].append(val_loss)
            writer.add_scalar('loss/val', val_loss, global_step=step)
            metrics['train_loss'].append(train_loss)
            writer.add_scalar('loss/train', train_loss, global_step=step)
            if val_loss < metrics['best_val_loss']:
                best = True
                logging.info("found new best validation loss, running test eval")
                metrics['best_val_loss'] = val_loss
                metrics['best_val_step'] = step
                _, test_loss = self.eval_one_split(
                    model, meta_split='test',
                    num_cases=self.args.num_cases_test
                )
                metrics['test_loss_at_best_val_step'] = test_loss
                writer.add_scalar('loss/test', test_loss, global_step=step)
                logging.info(f"test loss: {test_loss}")
        self.save_metrics(metrics)
        writer.flush()

        model.train()

        return best

    def load_arch(self, in_channels):
        return AuxiliarNetwork(
            in_channels,
            n_outputs=self.word_emb_dim,
            num_layers_per_block=self.args.num_layers_per_block,
            num_blocks=self.args.num_blocks,
            max_pool_kernel_size=self.args.max_pool_kernel_size,
            num_channels=self.args.num_channels,
            num_channels_growth_factor=self.args.num_channels_growth_factor,
            num_max_pools=self.args.num_max_pools,
            activation=self.args.activation,
            batchnorm=self.args.batchnorm,
            dropout=self.args.dropout,
            kernel_size=self.args.kernel_size,
            n_extra_linear=self.args.n_extra_linear,
            conv_bias=self.args.conv_bias,
            bn_epsilon=self.args.bn_epsilon,
            bn_momentum=self.args.bn_momentum
        )

    def _train(self):
        dataset, dataloader = self.get_dataset_and_loader(
            meta_split='train', batch_size=self.args.batch_size
        )

        if self.args.dataset == 'cub-attributes':
            in_channels = 3
            self.n_atts = 312
        elif self.args.dataset == 'sun-attributes':
            in_channels = 3
            self.n_atts = 102
        elif self.args.dataset == 'miniimagenet-attributes':
            in_channels = 3
            self.n_atts = 160
        else:
            raise NotImplementedError(self.args.dataset)
        model = self.load_arch(in_channels)
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logging.info(f'Training {params} parameters')

        model.to(device=self.args.device)

        # init weights
        if self.args.custom_init:
            model.apply(self.init_weights)
        model.train()

        writer = SummaryWriter(os.path.join(self.exp_dir, 'tensorboard'))

        opt_kwargs = {}
        if self.args.optimizer == 'sgd':
            Opt = torch.optim.SGD
            opt_kwargs['momentum'] = self.args.momentum
        else:
            Opt = torch.optim.Adam
        if self.args.no_decay_bias_bn:
            weight_decay_groups = self.group_weight(model)
            weight_decay_groups[0]['weight_decay'] = self.args.weight_decay
            optimizer = Opt(weight_decay_groups, lr=self.args.init_learning_rate, **opt_kwargs)
        else:
            optimizer = Opt(model.parameters(), lr=self.args.init_learning_rate, weight_decay=self.args.weight_decay,
                            **opt_kwargs)

        if self.args.learning_rate_schedule is not None:
            if self.args.learning_rate_schedule == 'pwc':
                scheduler = torch.optim.lr_scheduler.MultiStepLR(
                    optimizer,
                    milestones=[self.args.num_batches / 2,
                                self.args.num_batches / 2 + self.args.pwc_decay_interval,
                                self.args.num_batches / 2 + 2 * self.args.pwc_decay_interval],
                    gamma=.1
                )
            else:
                raise ValueError('Unsupported learning rate schedule')
        else:
            scheduler = None

        if self.args.warmup != -1:
            warmup_scheduler = warmup.LinearWarmup(optimizer, warmup_period=self.args.warmup)

        metrics = {
            'steps': [],
            'val_loss': [],
            'train_loss': [],
            'best_val_loss': np.inf,
            'best_val_step': -1,
            'test_loss_sim_at_best_val_step': 0.
        }

        fig = plt.figure()

        self.criterion = nn.CosineEmbeddingLoss()

        # Training loop
        eval_steps_without_improvement = 0
        first = True
        y = None  # target ones: cosine sim -> similar
        with tqdm(dataloader, total=self.args.num_batches, disable=self.args.no_bar) as pbar:
            for step, batch in enumerate(pbar):
                model.zero_grad()

                train_inputs, _, train_attributes = batch['train']
                train_word_embeddings = []
                for i, att_task in enumerate(train_attributes):
                    embeddings = []
                    for j, atts in enumerate(att_task):
                        words = []
                        for att in atts:
                            '''
                            att_name = dataset.dataset.label_idx_to_label[att]
                            att_names = att_name.split('_')
                            for att_name in att_names:
                                att_name = ''.join([c.lower() for c in att_name if c.isalpha])
                                if len(att_name) > 0:
                                    words.append(self.ft.get_word_vector(att_name))
                            '''
                            att_words = dataset.dataset.attribute_words[att]
                            att_words = list(map(self.ft.get_word_vector, att_words))
                            words.extend(att_words)
                        average_embeddings = np.mean(words, axis=0)
                        embeddings.append(average_embeddings)
                    train_word_embeddings.append(embeddings)
                train_word_embeddings = torch.tensor(train_word_embeddings)

                train_word_embeddings = train_word_embeddings.to(device=self.args.device)
                train_inputs = train_inputs.to(device=self.args.device)

                test_inputs, _, test_attributes = batch['test']
                test_word_embeddings = []
                for i, att_task in enumerate(test_attributes):
                    embeddings = []
                    for j, atts in enumerate(att_task):
                        words = []
                        for att in atts:
                            '''
                            att_name = dataset.dataset.label_idx_to_label[att]
                            att_names = att_name.split('_')
                            for att_name in att_names:
                                att_name = ''.join([c.lower() for c in att_name if c.isalpha])
                                if len(att_name) > 0:
                                    words.append(self.ft.get_word_vector(att_name))
                            '''
                            att_words = dataset.dataset.attribute_words[att]
                            att_words = list(map(self.ft.get_word_vector, att_words))
                            words.extend(att_words)
                        average_embeddings = np.mean(words, axis=0)
                        embeddings.append(average_embeddings)
                    test_word_embeddings.append(embeddings)
                test_word_embeddings = torch.tensor(test_word_embeddings)

                test_word_embeddings = test_word_embeddings.to(device=self.args.device)
                test_inputs = test_inputs.to(device=self.args.device)

                inputs = torch.cat((train_inputs, test_inputs), dim=1)
                # inputs = inputs.view(inputs.shape[0]*inputs.shape[1], inputs.shape[2], inputs.shape[3], inputs.shape[4])
                word_embeddings = torch.cat((train_word_embeddings, test_word_embeddings), dim=1)
                word_embeddings = word_embeddings.view(word_embeddings.shape[0] * word_embeddings.shape[1],
                                                       *word_embeddings.shape[2:])

                output = model(inputs)
                if y is None:
                    y = torch.ones(output.shape[0]).to(device=self.args.device)
                loss = self.criterion(output, word_embeddings, y)
                loss.backward()

                if (step + 1) % self.args.debug_log_interval == 0:
                    logging.info('Debug step')

                    fig.clf()
                    plot_grad_flow(model.named_parameters())
                    writer.add_figure('grad_flow', fig, global_step=step,
                                      close=False)
                    if first:
                        writer.add_graph(model, inputs)
                        #  writer.add_graph(criterion, (output, attributes))
                        first = False

                _ = nn.utils.clip_grad_norm_(
                    model.parameters(), self.args.grad_clip_norm, norm_type=2)
                optimizer.step()

                # adapt learning rate
                if self.args.learning_rate_schedule is not None:
                    scheduler.step()

                if self.args.warmup != -1:
                    warmup_scheduler.dampen()

                with torch.no_grad():
                    cosine_sim = loss.item()
                    pbar.set_postfix(cosine_sim=f'{cosine_sim:.4f}')

                # add summaries
                writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'],
                                  global_step=step)
                writer.add_scalar('loss', loss.item(),
                                  global_step=step)

                if (step + 1) % self.args.eval_interval_steps == 0:
                    logging.info('Eval step')
                    best = self.eval(model, writer, step, metrics)
                    self.print_metrics(step, metrics)
                    self.save_model(model, optimizer, metrics, scheduler,
                                    suffix='_last')
                    if best:
                        self.save_model(model, optimizer, metrics, scheduler,
                                        suffix='_best')
                        eval_steps_without_improvement = 0
                        logging.info(f'{eval_steps_without_improvement} eval steps without improvement')
                    else:
                        eval_steps_without_improvement += 1
                        logging.info(f'{eval_steps_without_improvement} eval steps without improvement')
                        if self.args.early_stop != -1 and eval_steps_without_improvement >= self.args.early_stop:
                            logging.info('Early stop')
                            break

                if step >= self.args.num_batches:
                    break

        # Save model
        self.save_model(model, optimizer, metrics, scheduler,
                        suffix='_end')

        return best, metrics['best_val_loss']

    @staticmethod
    def print_metrics(step, metrics):
        val_loss = metrics['val_loss'][-1]
        train_loss = metrics['train_loss'][-1]
        best_val_loss = metrics['best_val_loss']
        best_val_step = metrics['best_val_step']
        test_loss_at_best_val_step = metrics['test_loss_at_best_val_step']

        logging.info(f'step = {step} | val_loss = {val_loss} |'
                     f' train_loss = {train_loss} | best_val_loss = {best_val_loss} | best_val_step = {best_val_step}'
                     f' test_loss_at_best_val_step = {test_loss_at_best_val_step}\n')

    def eval_one_split(self, model, meta_split, num_cases):
        num_batches = int(np.ceil(num_cases / self.args.eval_batch_size))
        dataset, dataloader = self.get_dataset_and_loader(
            meta_split=meta_split, batch_size=self.args.eval_batch_size
        )

        losses = []
        ns = []

        done_batches = 0
        done = False
        y = None
        while not done:
            with tqdm(dataloader, total=min(num_batches - done_batches, len(dataloader)),
                      disable=self.args.no_bar) as pbar:
                for step, batch in enumerate(pbar):
                    train_inputs, _, train_attributes = batch['train']
                    train_word_embeddings = []
                    for i, att_task in enumerate(train_attributes):
                        embeddings = []
                        for j, atts in enumerate(att_task):
                            words = []
                            for att in atts:
                                '''
                                att_name = dataset.dataset.label_idx_to_label[att]
                                att_names = att_name.split('_')
                                for att_name in att_names:
                                    att_name = ''.join([c.lower() for c in att_name if c.isalpha])
                                    if len(att_name) > 0:
                                        words.append(self.ft.get_word_vector(att_name))
                                '''
                                att_words = dataset.dataset.attribute_words[att]
                                att_words = list(map(self.ft.get_word_vector, att_words))
                                words.extend(att_words)
                            average_embeddings = np.mean(words, axis=0)
                            embeddings.append(average_embeddings)
                        train_word_embeddings.append(embeddings)
                    train_word_embeddings = torch.tensor(train_word_embeddings)

                    train_word_embeddings = train_word_embeddings.to(device=self.args.device)
                    train_inputs = train_inputs.to(device=self.args.device)

                    test_inputs, _, test_attributes = batch['test']
                    test_word_embeddings = []
                    for i, att_task in enumerate(test_attributes):
                        embeddings = []
                        for j, atts in enumerate(att_task):
                            words = []
                            for att in atts:
                                '''
                                att_name = dataset.dataset.label_idx_to_label[att]
                                att_names = att_name.split('_')
                                for att_name in att_names:
                                    att_name = ''.join([c.lower() for c in att_name if c.isalpha])
                                    if len(att_name) > 0:
                                        words.append(self.ft.get_word_vector(att_name))
                                '''
                                att_words = dataset.dataset.attribute_words[att]
                                att_words = list(map(self.ft.get_word_vector, att_words))
                                words.extend(att_words)
                            average_embeddings = np.mean(words, axis=0)
                            embeddings.append(average_embeddings)
                        test_word_embeddings.append(embeddings)
                    test_word_embeddings = torch.tensor(test_word_embeddings)

                    test_word_embeddings = test_word_embeddings.to(device=self.args.device)
                    test_inputs = test_inputs.to(device=self.args.device)

                    inputs = torch.cat((train_inputs, test_inputs), dim=1)
                    # inputs = inputs.view(inputs.shape[0]*inputs.shape[1], inputs.shape[2], inputs.shape[3], inputs.shape[4])
                    word_embeddings = torch.cat((train_word_embeddings, test_word_embeddings), dim=1)
                    word_embeddings = word_embeddings.view(word_embeddings.shape[0] * word_embeddings.shape[1],
                                                           *word_embeddings.shape[2:])

                    output = model(inputs)
                    if y is None:
                        y = torch.ones(output.shape[0]).to(device=self.args.device)
                    batch_loss = self.criterion(output, word_embeddings, y)

                    losses.append(batch_loss)
                    ns.append(len(inputs))
                    done_batches += 1
                    if done_batches >= num_batches:
                        done = True
                        break
            if not done:
                logging.info('Reloading exhausted eval dataloader...')

        total_n = sum(ns)
        loss = sum(l * n / total_n for l, n in zip(losses, ns))
        return None, loss


class ProtonetWithAuxiliarNetworkJointTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_atts = None
        self.aux_criterion = None
        if self.args.aux_output == 'word-embeddings':
            self.word_emb_dim = self.args.word_embeddings_dim
            fasttext.util.download_model('en', if_exists='ignore')
            self.ft = fasttext.load_model('cc.en.300.bin')
            if self.ft.get_dimension() > self.word_emb_dim:
                fasttext.util.reduce_model(self.ft, self.word_emb_dim)

    def eval(self, model, writer, step, metrics):
        best = False
        if not self.args.train_mode:
            model.eval()
        with torch.no_grad():
            train_metrics, train_loss = self.eval_one_split(
                model, meta_split='train',
                num_cases=self.args.num_cases_val
            )
            val_metrics, val_loss = self.eval_one_split(
                model, meta_split='val',
                num_cases=self.args.num_cases_val
            )
            metrics['steps'].append(step)

            metrics['val_loss'].append(val_loss)
            metrics['val_acc'].append(val_metrics['accuracy'])
            metrics['val_aux_loss'].append(val_metrics['aux_loss'])

            metrics['train_loss'].append(train_loss)
            metrics['train_acc'].append(train_metrics['accuracy'])
            metrics['train_aux_loss'].append(train_metrics['aux_loss'])

            if self.args.aux_output == 'attributes':
                metrics['val_aux_acc'].append(val_metrics['aux_accuracy'])
                metrics['val_aux_acc_score'].append(val_metrics['aux_acc_score'])
                metrics['val_aux_f1_micro'].append(val_metrics['aux_f1_micro'])
                metrics['val_aux_f1_macro'].append(val_metrics['aux_f1_macro'])
                metrics['val_aux_amr'].append(val_metrics['aux_all_match_ratio'])

                metrics['train_aux_acc'].append(train_metrics['aux_accuracy'])
                metrics['train_aux_acc_score'].append(train_metrics['aux_acc_score'])
                metrics['train_aux_f1_micro'].append(train_metrics['aux_f1_micro'])
                metrics['train_aux_f1_macro'].append(train_metrics['aux_f1_macro'])
                metrics['train_aux_amr'].append(train_metrics['aux_all_match_ratio'])

                writer.add_scalar('aux_accuracy/val', val_metrics['aux_accuracy'],
                                  global_step=step)
                writer.add_scalar('aux_accuracy_score/val',
                                  val_metrics['aux_acc_score'], global_step=step)
                writer.add_scalar('aux_f1_micro/val',
                                  val_metrics['aux_f1_micro'], global_step=step)
                writer.add_scalar('aux_f1_macro/val',
                                  val_metrics['aux_f1_macro'], global_step=step)
                writer.add_scalar('aux_all_match_ratio/val',
                                  val_metrics['aux_all_match_ratio'],
                                  global_step=step)

                writer.add_scalar('aux_accuracy/train', train_metrics['aux_accuracy'],
                                  global_step=step)
                writer.add_scalar('aux_accuracy_score/train',
                                  train_metrics['aux_acc_score'], global_step=step)
                writer.add_scalar('aux_f1_micro/train',
                                  train_metrics['aux_f1_micro'], global_step=step)
                writer.add_scalar('aux_f1_macro/train',
                                  train_metrics['aux_f1_macro'], global_step=step)
                writer.add_scalar('aux_all_match_ratio/train',
                                  train_metrics['aux_all_match_ratio'],
                                  global_step=step)
            elif self.args.aux_output == 'protonet':
                metrics['val_aux_acc'].append(val_metrics['aux_accuracy'])
                metrics['val_aux_f1_score'].append(val_metrics['aux_f1_score'])
                metrics['val_aux_ap_score'].append(val_metrics['aux_ap_score'])
                metrics['train_aux_acc'].append(train_metrics['aux_accuracy'])
                metrics['train_aux_f1_score'].append(
                    train_metrics['aux_f1_score'])
                metrics['train_aux_ap_score'].append(
                    train_metrics['aux_ap_score'])
                writer.add_scalar('aux_accuracy/val', val_metrics['aux_accuracy'],
                                  global_step=step)
                writer.add_scalar('aux_f1_score/val', val_metrics['aux_f1_score'],
                                  global_step=step)
                writer.add_scalar('aux_ap_score/val', val_metrics['aux_ap_score'],
                                  global_step=step)
                writer.add_scalar('aux_accuracy/train', train_metrics['aux_accuracy'],
                                  global_step=step)
                writer.add_scalar('aux_f1_score/train', train_metrics['aux_f1_score'],
                                  global_step=step)
                writer.add_scalar('aux_ap_score/train', train_metrics['aux_ap_score'],
                                  global_step=step)

            writer.add_scalar('accuracy/val', val_metrics['accuracy'],
                              global_step=step)

            writer.add_scalar('accuracy/train', train_metrics['accuracy'],
                              global_step=step)

            writer.add_scalar('aux-loss/val', train_metrics['aux_loss'], global_step=step)
            writer.add_scalar('aux-loss/train', val_metrics['aux_loss'], global_step=step)
            writer.add_scalar('loss/val', val_loss, global_step=step)
            writer.add_scalar('loss/train', train_loss, global_step=step)

            if val_loss < metrics['best_val_loss']:
                best = True
                logging.info("found new best validation loss, running test eval")
                metrics['best_val_loss'] = val_loss
                metrics['best_val_step'] = step
                test_metrics, _ = self.eval_one_split(
                    model, meta_split='test',
                    num_cases=self.args.num_cases_test
                )
                metrics['test_acc_at_best_val_step'] = \
                    test_metrics['accuracy']

                metrics['test_aux_loss_at_best_val_step'] = \
                    test_metrics['aux_loss']
                if self.args.aux_output == 'attributes':
                    metrics['test_aux_acc_at_best_val_step'] = \
                        test_metrics['aux_accuracy']
                    metrics['test_acc_score_at_best_val_step'] = \
                        test_metrics['aux_acc_score']
                    metrics['test_aux_f1_micro_at_best_val_step'] = \
                        test_metrics['aux_f1_micro']
                    metrics['test_aux_f1_macro_at_best_val_step'] = \
                        test_metrics['aux_f1_macro']
                    metrics['test_aux_amr_at_best_val_step'] = \
                        test_metrics['aux_all_match_ratio']

                    writer.add_scalar('aux_accuracy/test', metrics['test_aux_acc_at_best_val_step'],
                                      global_step=step)
                    writer.add_scalar('aux_accuracy_score/test',
                                      metrics['test_aux_acc_score_at_best_val_step'],
                                      global_step=step)
                    writer.add_scalar('aux_f1_micro/test',
                                      metrics['test_aux_f1_micro_at_best_val_step'],
                                      global_step=step)
                    writer.add_scalar('aux_f1_macro/test',
                                      metrics['test_aux_f1_macro_at_best_val_step'],
                                      global_step=step)
                    writer.add_scalar('aux_all_match_ratio/test',
                                      metrics['test_aux_amr_at_best_val_step'], global_step=step)

                writer.add_scalar('accuracy/test', metrics['test_acc_at_best_val_step'],
                                  global_step=step)
                writer.add_scalar('aux-loss/test', test_metrics['aux_loss'],
                                  global_step=step)

        self.save_metrics(metrics)
        writer.flush()

        model.train()

        return best

    def load_arch(self, in_channels):
        return SimpAux(
            in_channels,
            aux_n_outputs=self.n_atts if self.args.aux_output == 'attributes' else self.args.word_embeddings_dim,
            num_layers_per_block=self.args.num_layers_per_block,
            num_blocks=self.args.num_blocks,
            aux_num_layers_per_block=self.args.aux_num_layers_per_block,
            aux_num_blocks=self.args.aux_num_blocks,
            max_pool_kernel_size=self.args.max_pool_kernel_size,
            num_channels=self.args.num_channels,
            num_channels_growth_factor=self.args.num_channels_growth_factor,
            num_max_pools=self.args.num_max_pools,
            activation=self.args.activation,
            dropout=self.args.dropout,
            kernel_size=self.args.kernel_size,
            aux_n_extra_linear=self.args.n_extra_linear,
            aux_dropout=self.args.aux_dropout,
            bridge_num_hid_features=self.args.bridge_num_hid_features,
            bridge_num_hid_layers=self.args.bridge_num_hid_layers,
            bridge_dropout=self.args.bridge_dropout,
            aux_num_channels=self.args.aux_num_channels,
            aux_backprop=self.args.aux_backprop,
            bridge_input_aux_layers=self.args.bridge_input_aux_layers,
            conv_bias=self.args.conv_bias,
            bn_epsilon=self.args.bn_epsilon,
            bn_momentum=self.args.bn_momentum
        )

    def _train(self):
        dataset, dataloader = self.get_dataset_and_loader(
            meta_split='train', batch_size=self.args.batch_size
        )
        if self.args.dataset == 'cub-attributes':
            in_channels = 3
            self.n_atts = 312
        elif self.args.dataset == 'sun-attributes':
            in_channels = 3
            self.n_atts = 102
        elif self.args.dataset == 'miniimagenet-attributes':
            in_channels = 3
            self.n_atts = 160
        else:
            raise NotImplementedError(self.args.dataset)

        model = self.load_arch(in_channels)
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logging.info(f'Training {params} parameters')

        # init weights
        if self.args.custom_init:
            model.apply(self.init_weights)

        # load weights of aux net
        if self.args.aux_net_checkpoint is not None:
            raise ValueError('are we sure we want to use pretraining if we do joint training?')
            # pretrain_checkpoint = torch.load(self.args.aux_net_checkpoint)
            # model.aux_net.load_state_dict(pretrain_checkpoint['model'])
        model.train()
        model.to(device=self.args.device)

        if self.args.aux_output == 'attributes':
            if self.args.aux_loss_fn == 'weighted-bce':
                # Compute pos_weight
                # Assuming all classes have same probability
                attributes = dataset.dataset.attributes

                def bin_attributes2ints(atts):
                    new_atts = []
                    for idx, e in enumerate(atts):
                        if e == 1:
                            new_atts.append(idx)
                    return new_atts

                # attributes = [bin_attributes2ints(attributes)]
                total = attributes.shape[0]
                attributes = [bin_attributes2ints(a) for a in attributes]

                atts_freq = [0] * self.n_atts
                for class_ in attributes:
                    for att in class_:
                        if atts_freq[att] is None:
                            atts_freq[att] = 1
                        else:
                            atts_freq[att] += 1

                atts_freq = list(map(lambda x: 1 if x == 0 else (total - x) / x, atts_freq))
                pos_weight = torch.tensor(atts_freq).to(self.args.device)
                self.aux_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            else:
                self.aux_criterion = MultiLabelLoss(self.n_atts, self.args.aux_loss_fn, self.args.device,
                                                    self.args.aux_loss_ignore)
        else:
            self.aux_criterion = nn.CosineEmbeddingLoss()

        scheduler = None

        writer = SummaryWriter(os.path.join(self.exp_dir, 'tensorboard'))
        opt_kwargs = {}

        if self.args.optimizer == 'sgd':
            Opt = torch.optim.SGD
            opt_kwargs['momentum'] = self.args.momentum
        else:
            Opt = torch.optim.Adam
        if self.args.no_decay_bias_bn:
            aux_net_weight_decay_groups = self.group_weight(model.aux_net)
            aux_net_weight_decay_groups[0]['weight_decay'] = self.args.weight_decay
            for idx, _ in enumerate(aux_net_weight_decay_groups):
                aux_net_weight_decay_groups[idx]['lr'] = self.args.aux_init_learning_rate
            bridge_weight_decay_groups = self.group_weight(model.bridge)
            bridge_weight_decay_groups[0]['weight_decay'] = self.args.weight_decay
            for idx, _ in enumerate(bridge_weight_decay_groups):
                bridge_weight_decay_groups[idx]['lr'] = self.args.bridge_init_learning_rate
            protonet_weight_decay_groups = self.group_weight(model.protonet)
            protonet_weight_decay_groups[0]['weight_decay'] = self.args.weight_decay
            for idx, _ in enumerate(protonet_weight_decay_groups):
                protonet_weight_decay_groups[idx]['lr'] = self.args.init_learning_rate

            optimizer = Opt(
                aux_net_weight_decay_groups + bridge_weight_decay_groups + protonet_weight_decay_groups,
                lr=self.args.init_learning_rate,
                **opt_kwargs)
        else:
            optimizer = Opt(
                [
                    {'params': model.aux_net.parameters(),
                     'lr': self.args.aux_init_learning_rate},
                    {'params': model.bridge.parameters(),
                     'lr': self.args.bridge_init_learning_rate},
                    {'params': model.protonet.parameters(),
                     'lr': self.args.init_learning_rate}
                 ],
                lr=self.args.init_learning_rate,
                weight_decay=self.args.weight_decay,
                **opt_kwargs)
        if self.args.learning_rate_schedule is not None:
            if self.args.learning_rate_schedule == 'pwc':
                scheduler = torch.optim.lr_scheduler.MultiStepLR(
                    optimizer,
                    milestones=[self.args.num_batches / 2,
                                self.args.num_batches / 2 + self.args.pwc_decay_interval,
                                self.args.num_batches / 2 + 2 * self.args.pwc_decay_interval],
                    gamma=.1
                )
            elif self.args.learning_rate_schedule == 'reduce-on-plateau':
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, factor=self.args.scheduler_factor,
                    patience=self.args.scheduler_patience
                )
            else:
                raise ValueError('Unsupported learning rate schedule')
        else:
            scheduler = None

        if self.args.warmup != -1:
            warmup_scheduler = warmup.LinearWarmup(optimizer, warmup_period=self.args.warmup)

        if self.args.aux_output == 'attributes':
            metrics = {
                'steps': [],

                'val_loss': [],
                'val_acc': [],
                'val_aux_loss': [],
                'val_aux_acc': [],
                'val_aux_amr': [],
                'val_aux_acc_score': [],
                'val_aux_f1_micro': [],
                'val_aux_f1_macro': [],

                'train_loss': [],
                'train_acc': [],
                'train_aux_loss': [],
                'train_aux_acc': [],
                'train_aux_amr': [],
                'train_aux_acc_score': [],
                'train_aux_f1_micro': [],
                'train_aux_f1_macro': [],
                'train_classification_loss': [],

                'best_val_loss': np.inf,
                'best_val_step': -1,

                'test_acc_at_best_val_step': 0.,
                'test_aux_loss_at_best_val_step': 0.,
                'test_aux_acc_at_best_val_step': 0.,
                'test_aux_amr_at_best_val_step': 0.,
                'test_aux_acc_score_at_best_val_step': 0.,
                'test_aux_f1_micro_at_best_val_step': 0.,
                'test_aux_f1_macro_at_best_val_step': 0.
            }
        else:
            metrics = {
                'steps': [],

                'val_loss': [],
                'val_acc': [],
                'val_aux_loss': [],

                'train_loss': [],
                'train_acc': [],
                'train_aux_loss': [],
                'train_classification_loss': [],

                'best_val_loss': np.inf,
                'best_val_step': -1,

                'test_acc_at_best_val_step': 0.,
                'test_aux_loss_at_best_val_step': 0.
            }
            if self.args.aux_output == 'protonet':
                metrics['val_aux_acc'] = []
                metrics['val_aux_f1_score'] = []
                metrics['val_aux_ap_score'] = []
                metrics['train_aux_acc'] = []
                metrics['train_aux_f1_score'] = []
                metrics['train_aux_ap_score'] = []
        metrics['prev_step'] = 0
        metrics['eval_steps_without_improvement'] = 0

        if self.args.resuming:
            try:
                model, optimizer, metrics, scheduler = self.load_model(model, optimizer, metrics, scheduler)
                logging.info('successfully loaded checkpoint. Resuming from '
                             f'step {metrics["steps"][-1]}.')
            except IOError:
                logging.warn('no checkpoint found. start from scratch')
                self.args.resuming = False

        # Training loop
        y = None  # Needed for cosine embedding loss
        metrics['eval_steps_without_improvement'] = 0
        step = metrics['prev_step']

        writer = SummaryWriter(os.path.join(self.exp_dir, 'tensorboard'),
                               purge_step=metrics['prev_step'])

        with tqdm(dataloader, total=self.args.num_batches,
                  initial=metrics['prev_step'] + 1,
                  disable=self.args.no_bar) as pbar:
            for batch in pbar:
                step += 1
                metrics['prev_step'] = step
                model.zero_grad()

                train_inputs, train_targets, train_attributes = batch['train']
                train_inputs = train_inputs.to(device=self.args.device)
                if self.args.aux_output in ('attributes', 'protonet'):
                    train_atts = torch.zeros(train_inputs.shape[0], train_inputs.shape[1], self.n_atts)
                    for i, att_task in enumerate(train_attributes):
                        for j, atts in enumerate(att_task):
                            train_atts[i, j][atts] = 1
                    train_attributes = train_atts.to(device=self.args.device)
                elif self.args.aux_output == 'word-embeddings':
                    train_word_embeddings = []
                    for i, att_task in enumerate(train_attributes):
                        embeddings = []
                        for j, atts in enumerate(att_task):
                            words = []
                            for att in atts:
                                att_words = dataset.dataset.attribute_words[att]
                                att_words = list(map(self.ft.get_word_vector, att_words))
                                words.extend(att_words)
                            if len(words) == 0:  # policy if no attributes. TODO: check
                                words = np.random.rand(1, self.word_emb_dim)
                            average_embeddings = np.mean(words, axis=0)
                            embeddings.append(average_embeddings)
                        train_word_embeddings.append(embeddings)
                    train_word_embeddings = torch.tensor(train_word_embeddings)
                    train_attributes = train_word_embeddings.to(device=self.args.device)
                else:
                    raise ValueError('unsupported aux_output type')

                train_targets = train_targets.to(device=self.args.device)

                test_inputs, test_targets, test_attributes = batch['test']
                test_inputs = test_inputs.to(device=self.args.device)
                if self.args.aux_output in ('attributes', 'protonet'):
                    test_atts = torch.zeros(test_inputs.shape[0], test_inputs.shape[1], self.n_atts)
                    for i, att_task in enumerate(test_attributes):
                        for j, atts in enumerate(att_task):
                            test_atts[i, j][atts] = 1
                    test_attributes = test_atts.to(device=self.args.device)
                elif self.args.aux_output == 'word-embeddings':
                    test_word_embeddings = []
                    for i, att_task in enumerate(test_attributes):
                        embeddings = []
                        for j, atts in enumerate(att_task):
                            words = []
                            for att in atts:
                                att_words = dataset.dataset.attribute_words[att]
                                att_words = list(map(self.ft.get_word_vector, att_words))
                                words.extend(att_words)
                            if len(words) == 0:  # policy if no attributes. TODO: check
                                words = np.random.rand(1, self.word_emb_dim)
                            average_embeddings = np.mean(words, axis=0)
                            embeddings.append(average_embeddings)
                        test_word_embeddings.append(embeddings)
                    test_word_embeddings = torch.tensor(test_word_embeddings)
                    test_attributes = test_word_embeddings.to(device=self.args.device)
                else:
                    raise ValueError('unsupported aux_output type')

                test_targets = test_targets.to(device=self.args.device)

                inputs = torch.cat((train_inputs, test_inputs), dim=1)

                if (step + 1) % self.args.debug_log_interval == 0:
                    embeddings, aux_outputs = model(
                        inputs,
                        summary_writer=writer,
                        step=step,
                        return_aux_outputs=True
                    )
                else:
                    embeddings, aux_outputs = model(
                        inputs,
                        return_aux_outputs=True
                    )
                aux_outputs = aux_outputs.view(self.args.batch_size,
                                               -1,
                                               aux_outputs.shape[1])

                train_embeddings, test_embeddings = torch.split(
                    embeddings, (train_inputs.shape[1], test_inputs.shape[1]),
                    dim=1)

                prototypes = get_prototypes(train_embeddings, train_targets,
                                            dataset.num_classes_per_task)
                classification_loss = prototypical_loss(
                    prototypes, test_embeddings, test_targets)

                attributes = torch.cat((train_attributes, test_attributes), dim=1)
                attributes = attributes.view(attributes.shape[0] * attributes.shape[1], *attributes.shape[2:])

                train_aux_outputs, test_aux_outputs = torch.split(
                    aux_outputs, (train_inputs.shape[1], test_inputs.shape[1]),
                    dim=1)
                if self.args.aux_output == 'attributes':
                    # aux_loss = self.args.aux_loss_coeff * self.aux_criterion(
                    #     aux_outputs, attributes)
                    aux_loss = self.args.aux_loss_coeff * self.aux_criterion(
                        train_aux_outputs.reshape(-1, self.n_atts),
                        train_attributes.view(-1, self.n_atts))
                elif self.args.aux_output == 'word-embeddings':
                    if y is None:
                        y = torch.ones(train_aux_outputs.shape[0] *
                                       train_aux_outputs.shape[1]).to(
                                           device=self.args.device)
                    aux_loss = self.args.aux_loss_coeff * self.aux_criterion(
                        train_aux_outputs.detach().clone().view(
                            -1, self.args.word_embeddings_dim),
                        train_attributes.detach().clone().view(
                            -1, self.args.word_embeddings_dim), y)
                elif self.args.aux_output == 'protonet':
                    if train_attributes.nonzero().numel() > 0 and \
                            test_attributes.nonzero().numel() > 0:
                        attribute_prototypes, batch_attributes = \
                            get_per_attribute_prototypes(
                                embeddings=train_aux_outputs,
                                attribute_targets=train_attributes)

                        aux_loss, aux_accuracy, aux_preds, _ = \
                            multilabel_prototypical_loss(
                                prototypes=attribute_prototypes,
                                batch_attributes=batch_attributes,
                                embeddings=test_aux_outputs,
                                targets=test_attributes, return_accuracy=True,
                                return_preds_and_targets=True)
                        aux_loss *= self.args.aux_loss_coeff
                    else:
                        warn('the whole batch has 0 attributes for support '
                             'and or query samples. using aux_loss = 0')
                        aux_loss = torch.scalar_tensor(0)

                else:
                    raise ValueError('unsupported aux_output')
                if self.args.debug_simpaux_aux_net:
                    loss = 0. * classification_loss + aux_loss
                    if aux_loss > 0:
                        loss.backward()
                else:
                    loss = classification_loss + aux_loss
                    loss.backward()

                if self.args.use_diff_clipping:
                    protonet_grad_norm = nn.utils.clip_grad_norm_(
                        model.protonet.parameters(), self.args.grad_clip_norm, norm_type=2)
                    auxnet_grad_norm = nn.utils.clip_grad_norm_(
                        model.aux_net.parameters(), self.args.aux_grad_clip_norm, norm_type=2)
                    bridge_grad_norm = nn.utils.clip_grad_norm_(
                        model.bridge.parameters(), self.args.bridge_grad_clip_norm, norm_type=2)
                    if 'grads' in self.args.debug_opts:
                        writer.add_scalar('debug/protonet_grad_norm',
                                          protonet_grad_norm,
                                          global_step=step)
                        writer.add_scalar('debug/auxnet_grad_norm',
                                          auxnet_grad_norm,
                                          global_step=step)
                        writer.add_scalar('debug/bridge_grad_norm',
                                          bridge_grad_norm,
                                          global_step=step)
                else:
                    total_grad_norm = nn.utils.clip_grad_norm_(
                        model.parameters(), self.args.grad_clip_norm, norm_type=2)
                    if 'grads' in self.args.debug_opts:
                        writer.add_scalar('debug/total_grad_norm',
                                          total_grad_norm,
                                          global_step=step)
                if not self.args.debug_simpaux_aux_net or loss.item() > 0:
                    optimizer.step()

                    if 'grads' in self.args.debug_opts and \
                            (step + 1) % self.args.debug_log_interval == 0:
                        for tag, value in model.named_parameters():
                            writer.add_histogram(
                                f'debug/param_{tag}', value.data,
                                global_step=step
                            )
                            writer.add_histogram(
                                f'debug/grad_{tag}', value.grad,
                                global_step=step
                            )
                if 'outputs' in self.args.debug_opts:
                    if self.args.aux_output == 'attributes':
                        writer.add_histogram(
                            f'debug/aux_outputs',
                            torch.sigmoid(aux_outputs),
                            global_step=step
                        )
                    elif self.args.aux_output == 'protonet':
                        if not torch.any(torch.isnan(aux_preds)):
                            writer.add_histogram(
                                f'debug/aux_predictions',
                                aux_preds.flatten(),
                                global_step=step)
                    else:
                        warn(f'no output visualization for '
                             f'--aux-output {self.args.aux_output}')

                # adapt learning rate
                if self.args.learning_rate_schedule is not None and self.args.learning_rate_schedule != 'reduce-on-plateau':
                    scheduler.step()

                if self.args.warmup != -1:
                    warmup_scheduler.dampen()

                with torch.no_grad():
                    if self.args.aux_output == 'protonet':
                        # FIXME: dummy accuracy
                        aux_met = {  # 'accuracy': np.inf,
                               'accuracy': aux_accuracy.item()}
                        pass

                    else:
                        aux_met = self.aux_metrics(
                            train_aux_outputs.detach().clone().view(
                                train_aux_outputs.shape[0] *
                                train_aux_outputs.shape[1],
                                -1),
                            train_attributes.detach().clone().view(
                                train_attributes.shape[0] *
                                train_attributes.shape[1],
                                -1))
                    accuracy = get_accuracy(prototypes, test_embeddings,
                                            test_targets).item()
                    try:
                        pbar.set_postfix(loss=f'{loss:.4f}',
                                         aux_loss=f'{aux_loss:.4f}',
                                         classification_loss=f'{classification_loss:.4f}',
                                         aux_accuracy=f'{aux_met["accuracy"]:.4F}',
                                         accuracy=f'{accuracy:.4f}'
                                         )
                    except (NameError, KeyError) as e:
                        pbar.set_postfix(loss=f'{loss:.4f}',
                                         aux_loss=f'{aux_loss:.4f}',
                                         classification_loss=f'{classification_loss:.4f}',
                                         accuracy=f'{accuracy:.4f}'
                                         # accuracy=f'{met["accuracy"]:.4F}'
                                         )

                # add summaries
                writer.add_scalar('learning_rate/aux_net',
                                  optimizer.param_groups[0]['lr'],
                                  global_step=step)
                writer.add_scalar('learning_rate/bridge',
                                  optimizer.param_groups[1]['lr'],
                                  global_step=step)
                writer.add_scalar('learning_rate/protonet',
                                  optimizer.param_groups[2]['lr'],
                                  global_step=step)
                writer.add_scalar('loss', loss.item(),
                                  global_step=step)
                writer.add_scalar('metatrain_loss', classification_loss.item(),
                                  global_step=step)
                writer.add_scalar('metatrain_accuracy', accuracy,
                                  global_step=step)
                if torch.isfinite(torch.tensor(aux_met["accuracy"])):
                    writer.add_scalar('aux_loss', aux_loss.item(),
                                      global_step=step)
                    writer.add_scalar('aux_accuracy', aux_met["accuracy"],
                                      global_step=step)

                del train_inputs, train_targets, train_embeddings, \
                    test_inputs, test_targets, test_embeddings, prototypes

                if (step + 1) % self.args.eval_interval_steps == 0:
                    logging.info('Eval step')
                    best = self.eval(model, writer, step, metrics)
                    self.print_metrics(step, metrics)
                    self.save_model(model, optimizer, metrics, scheduler,
                                    suffix='_last')
                    if self.args.learning_rate_schedule == 'reduce-on-plateau':
                        scheduler.step(metrics['best_val_loss'])
                    if best:
                        self.save_model(model, optimizer, metrics, scheduler,
                                        suffix='_best')
                        metrics['eval_steps_without_improvement'] = 0
                        logging.info(f'{metrics["eval_steps_without_improvement"]} eval steps without improvement')
                    else:
                        metrics['eval_steps_without_improvement'] += 1
                        logging.info(f'{metrics["eval_steps_without_improvement"]} eval steps without improvement')
                        if self.args.early_stop != -1 and metrics["eval_steps_without_improvement"] >= self.args.early_stop:
                            logging.info('Early stop')
                            break

                if step >= self.args.num_batches:
                    break

        # Save model
        self.save_model(model, optimizer, metrics, scheduler,
                        suffix='_end')

        return best, metrics['best_val_loss']

    def eval_one_split(self, model, meta_split, num_cases):
        num_batches = int(np.ceil(num_cases / self.args.eval_batch_size))
        dataset, dataloader = self.get_dataset_and_loader(
            meta_split=meta_split, batch_size=self.args.eval_batch_size
        )

        ms = []
        losses = []
        ns = []

        if self.args.aux_output == 'protonet':
            aux_predictions = []
            aux_targets = []

        done_batches = 0
        done = False
        y = None  # Need for cosine loss
        while not done:
            with tqdm(dataloader, total=min(num_batches - done_batches, len(dataloader)),
                      disable=self.args.no_bar) as pbar:
                for step, batch in enumerate(pbar):
                    train_inputs, train_targets, train_attributes = batch['train']
                    train_inputs = train_inputs.to(device=self.args.device)
                    if self.args.aux_output in ('attributes', 'protonet'):
                        train_atts = torch.zeros(train_inputs.shape[0], train_inputs.shape[1], self.n_atts)
                        for i, att_task in enumerate(train_attributes):
                            for j, atts in enumerate(att_task):
                                train_atts[i, j][atts] = 1
                        train_attributes = train_atts.to(device=self.args.device)
                    elif self.args.aux_output == 'word-embeddings':
                        train_word_embeddings = []
                        for i, att_task in enumerate(train_attributes):
                            embeddings = []
                            for j, atts in enumerate(att_task):
                                words = []
                                for att in atts:
                                    att_words = dataset.dataset.attribute_words[att]
                                    att_words = list(map(self.ft.get_word_vector, att_words))
                                    words.extend(att_words)
                                if len(words) == 0:  # policy if no attributes. TODO: check
                                    words = np.random.rand(1, self.word_emb_dim)
                                average_embeddings = np.mean(words, axis=0)
                                embeddings.append(average_embeddings)
                            train_word_embeddings.append(embeddings)
                        train_word_embeddings = torch.tensor(train_word_embeddings)
                        train_attributes = train_word_embeddings.to(device=self.args.device)
                    else:
                        raise ValueError('unsupported aux_output')

                    train_targets = train_targets.to(device=self.args.device)

                    test_inputs, test_targets, test_attributes = batch['test']
                    test_inputs = test_inputs.to(device=self.args.device)
                    if self.args.aux_output in ('attributes', 'protonet'):
                        test_atts = torch.zeros(test_inputs.shape[0], test_inputs.shape[1], self.n_atts)
                        for i, att_task in enumerate(test_attributes):
                            for j, atts in enumerate(att_task):
                                test_atts[i, j][atts] = 1
                        test_attributes = test_atts.to(device=self.args.device)
                    elif self.args.aux_output == 'word-embeddings':
                        test_word_embeddings = []
                        for i, att_task in enumerate(test_attributes):
                            embeddings = []
                            for j, atts in enumerate(att_task):
                                words = []
                                for att in atts:
                                    att_words = dataset.dataset.attribute_words[att]
                                    att_words = list(map(self.ft.get_word_vector, att_words))
                                    words.extend(att_words)
                                if len(words) == 0:  # policy if no attributes. TODO: check
                                    words = np.random.rand(1, self.word_emb_dim)
                                average_embeddings = np.mean(words, axis=0)
                                embeddings.append(average_embeddings)
                            test_word_embeddings.append(embeddings)
                        test_word_embeddings = torch.tensor(test_word_embeddings)
                        test_attributes = test_word_embeddings.to(device=self.args.device)
                    else:
                        raise ValueError('unsupported aux_output')
                    test_targets = test_targets.to(device=self.args.device)

                    inputs = torch.cat((train_inputs, test_inputs), dim=1)

                    attributes = torch.cat((train_attributes, test_attributes), dim=1)
                    attributes = attributes.view(attributes.shape[0] * attributes.shape[1], *attributes.shape[2:])

                    embeddings, aux_outputs = model(
                        inputs,
                        return_aux_outputs=True
                    )

                    aux_outputs = aux_outputs.view(inputs.shape[0],
                                                   -1,
                                                   aux_outputs.shape[1])

                    train_aux_outputs, test_aux_outputs = torch.split(
                        aux_outputs, (train_inputs.shape[1], test_inputs.shape[1]),
                        dim=1)
                    train_embeddings, test_embeddings = torch.split(
                        embeddings, (train_inputs.shape[1], test_inputs.shape[1]),
                        dim=1)

                    prototypes = get_prototypes(train_embeddings, train_targets,
                                                dataset.num_classes_per_task)

                    batch_loss = prototypical_loss(
                        prototypes, test_embeddings, test_targets).item()

                    if self.args.aux_output == 'attributes':
                        batch_ms = self.aux_metrics(aux_outputs.detach().clone().view(-1, self.n_atts),
                                                    attributes.detach().clone().view(-1, self.n_atts))
                        batch_ms['loss'] = self.args.aux_loss_coeff * \
                            self.aux_criterion(
                                aux_outputs.view(-1, self.n_atts),
                                attributes.view(-1, self.n_atts))
                    elif self.args.aux_output == 'protonet':

                        if train_attributes.nonzero().numel() > 0 and \
                                test_attributes.nonzero().numel() > 0:
                            attribute_prototypes, batch_attributes = \
                                get_per_attribute_prototypes(
                                    embeddings=train_aux_outputs,
                                    attribute_targets=train_attributes)

                            # FIXME: need to collect preds or the number of
                            # samples considered in accuracy batch
                            aux_loss, aux_preds_batch, aux_targets_batch = \
                                multilabel_prototypical_loss(
                                    prototypes=attribute_prototypes,
                                    batch_attributes=batch_attributes,
                                    embeddings=test_aux_outputs,
                                    targets=test_attributes,
                                    return_preds_and_targets=True)
                            aux_loss *= self.args.aux_loss_coeff
                            aux_predictions.extend(
                                aux_preds_batch.detach().cpu().flatten())
                            aux_targets.extend(
                                aux_targets_batch.detach().cpu().flatten())
                        else:
                            warn('the whole batch has 0 attributes for support '
                                 'and or query samples. using aux_loss = 0')
                            aux_loss = torch.scalar_tensor(0)
                        batch_ms = {'loss': aux_loss}

                    elif self.args.aux_output == 'word-embeddings':
                        if y is None:
                            y = torch.ones(aux_outputs.shape[0] *
                                           aux_outputs.shape[1]).to(
                                               device=self.args.device)
                        aux_loss = self.args.aux_loss_coeff * self.aux_criterion(
                            aux_outputs.detach().clone().view(
                                -1, self.args.word_embeddings_dim),
                            attributes.detach().clone().view(
                                -1, self.args.word_embeddings_dim),
                            y)
                        batch_ms = {'loss': aux_loss}
                    else:
                        raise ValueError('unsupported aux_output')

                    for k, v in list(batch_ms.items()):
                        batch_ms[f'aux_{k}'] = v
                        del batch_ms[k]
                    batch_ms['accuracy'] = get_accuracy(
                        prototypes, test_embeddings, test_targets).item()

                    ms.append(batch_ms)
                    losses.append(batch_loss)
                    ns.append(len(test_inputs))
                    done_batches += 1
                    if done_batches >= num_batches:
                        done = True
                        break
            if not done:
                logging.info('Reloading exhausted eval dataloader...')

        total_n = sum(ns)
        reduced_metrics = self.reduce_aux_metrics(ms, total_n, ns)
        if self.args.aux_output == 'protonet':
            reduced_metrics['aux_accuracy'] = accuracy_score(aux_predictions,
                                                             aux_targets)
            reduced_metrics['aux_f1_score'] = f1_score(aux_predictions,
                                                       aux_targets)
            reduced_metrics['aux_ap_score'] = average_precision_score(
                aux_predictions, aux_targets)

            # TODO: compute some metrics for evaluation
        loss = sum(l * n / total_n for l, n in zip(losses, ns))
        return reduced_metrics, loss

    @staticmethod
    def reduce_aux_metrics(ms, total_n, ns):
        met_dict = OrderedDict()
        for key in ms[0]:
            if key not in ['aux_class_rep']:
                met_dict[key] = sum(m * n / total_n for m, n in zip([values[key] for values in ms], ns))
        return met_dict


class ProtonetWithAuxiliarNetworkFrozenTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def load_arch(self, in_channels):
        return SimpAux(
            in_channels,
            aux_n_outputs=self.n_atts,
            num_layers_per_block=self.args.num_layers_per_block,
            num_blocks=self.args.num_blocks,
            aux_num_layers_per_block=self.args.aux_num_layers_per_block,
            aux_num_blocks=self.args.aux_num_blocks,
            max_pool_kernel_size=self.args.max_pool_kernel_size,
            num_channels=self.args.num_channels,
            num_channels_growth_factor=self.args.num_channels_growth_factor,
            num_max_pools=self.args.num_max_pools,
            activation=self.args.activation,
            dropout=self.args.dropout,
            kernel_size=self.args.kernel_size,
            aux_n_extra_linear=self.args.n_extra_linear,
            aux_dropout=self.args.aux_dropout,
            bridge_num_hid_features=self.args.bridge_num_hid_features,
            bridge_num_hid_layers=self.args.bridge_num_hid_layers,
            bridge_dropout=self.args.bridge_dropout,
            aux_num_channels=self.args.aux_num_channels,
            bridge_input_aux_layers=self.args.bridge_input_aux_layers,
            conv_bias=self.args.conv_bias,
            bn_epsilon=self.args.bn_epsilon,
            bn_momentum=self.args.bn_momentum
        )

    def _train(self):
        dataset, dataloader = self.get_dataset_and_loader(
            meta_split='train', batch_size=self.args.batch_size
        )
        if self.args.dataset in ['cub', 'cub-attributes']:
            in_channels = 3
            self.n_atts = 312
        elif self.args.dataset == 'sun-attributes':
            in_channels = 3
            self.n_atts = 102
        else:
            raise NotImplementedError(self.args.dataset)

        model = self.load_arch(in_channels)
        params = sum(p.numel() for p in model.parameters_wo_aux()
                     if p.requires_grad)
        logging.info(f'Training {params} parameters')

        # init weights
        if self.args.custom_init:
            model.apply(self.init_weights)

        # load weights of aux net
        if self.args.aux_net_checkpoint is not None:
            pretrain_checkpoint = torch.load(self.args.aux_net_checkpoint)
            model.aux_net.load_state_dict(pretrain_checkpoint['model'])
        model.train()
        model.to(device=self.args.device)

        writer = SummaryWriter(os.path.join(self.exp_dir, 'tensorboard'))

        opt_kwargs = {}
        if self.args.optimizer == 'sgd':
            Opt = torch.optim.SGD
            opt_kwargs['momentum'] = self.args.momentum
        else:
            Opt = torch.optim.Adam

        if self.args.no_decay_bias_bn:
            bridge_weight_decay_groups = self.group_weight(model.bridge)
            bridge_weight_decay_groups[0]['weight_decay'] = self.args.weight_decay
            for idx, _ in enumerate(bridge_weight_decay_groups):
                bridge_weight_decay_groups[idx]['lr'] = self.args.bridge_init_learning_rate
            protonet_weight_decay_groups = self.group_weight(model.protonet)
            protonet_weight_decay_groups[0]['weight_decay'] = self.args.weight_decay
            for idx, _ in enumerate(protonet_weight_decay_groups):
                protonet_weight_decay_groups[idx]['lr'] = self.args.init_learning_rate
            optimizer = Opt(
                bridge_weight_decay_groups + protonet_weight_decay_groups,
                lr=self.args.lr,
                **opt_kwargs)
        else:
            optimizer = Opt(
                [
                    {'params': model.bridge.parameters(),
                     'lr': self.args.bridge_init_learning_rate},
                    {'params': model.protonet.parameters(),
                     'lr': self.args.init_learning_rate}
                 ],
                lr=self.args.init_learning_rate,
                weight_decay=self.args.weight_decay,
                **opt_kwargs)
        if self.args.learning_rate_schedule is not None:
            if self.args.learning_rate_schedule == 'pwc':
                scheduler = torch.optim.lr_scheduler.MultiStepLR(
                    optimizer,
                    milestones=[self.args.num_batches / 2,
                                self.args.num_batches / 2 + self.args.pwc_decay_interval,
                                self.args.num_batches / 2 + 2 * self.args.pwc_decay_interval],
                    gamma=.1
                )
            elif self.args.learning_rate_schedule == 'reduce-on-plateau':
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, factor=self.args.scheduler_factor,
                    patience=self.args.scheduler_patience
                )
            else:
                raise ValueError('Unsupported learning rate schedule')
        else:
            scheduler = None

        if self.args.warmup != -1:
            warmup_scheduler = warmup.LinearWarmup(optimizer, warmup_period=self.args.warmup)

        metrics = {
            'steps': [],
            'val_acc': [],
            'train_acc': [],
            'val_loss': [],
            'train_loss': [],
            'best_val_loss': np.inf,
            'best_val_step': -1,
            'test_acc_at_best_val_step': 0.
        }

        # Training loop
        eval_steps_without_improvement = 0
        with tqdm(dataloader, total=self.args.num_batches, disable=self.args.no_bar) as pbar:
            for step, batch in enumerate(pbar):
                model.zero_grad()

                train_inputs, train_targets = batch['train']
                train_inputs = train_inputs.to(device=self.args.device)
                train_targets = train_targets.to(device=self.args.device)
                if (step + 1) % self.args.debug_log_interval == 0:
                    train_embeddings = model(
                        train_inputs,
                        summary_writer=writer,
                        step=step
                    )
                else:
                    train_embeddings = model(
                        train_inputs
                    )

                test_inputs, test_targets = batch['test']
                test_inputs = test_inputs.to(device=self.args.device)
                test_targets = test_targets.to(device=self.args.device)
                test_embeddings = model(test_inputs)

                prototypes = get_prototypes(train_embeddings, train_targets, dataset.num_classes_per_task)
                loss = prototypical_loss(prototypes, test_embeddings, test_targets)
                loss.backward()

                _ = nn.utils.clip_grad_norm_(
                    model.parameters(), self.args.grad_clip_norm, norm_type=2)
                optimizer.step()

                if 'grads' in self.args.debug_opts and \
                        (step + 1) % self.args.debug_log_interval == 0:
                    for tag, value in model.named_parameters_wo_aux():
                        writer.add_histogram(
                            f'debug/param_{tag}', value.data,
                            global_step=step
                        )
                        writer.add_histogram(
                            f'debug/grad_{tag}', value.grad,
                            global_step=step
                        )

                # adapt learning rate
                if self.args.learning_rate_schedule is not None:
                    scheduler.step()

                if self.args.warmup != -1:
                    warmup_scheduler.dampen()

                with torch.no_grad():
                    accuracy = get_accuracy(prototypes, test_embeddings,
                                            test_targets).item()
                    pbar.set_postfix(accuracy=f'{accuracy:.4f}')

                # add summaries
                writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'],
                                  global_step=step)
                writer.add_scalar('metatrain_loss', loss.item(),
                                  global_step=step)
                writer.add_scalar('metatrain_accuracy', accuracy,
                                  global_step=step)

                del train_inputs, train_targets, train_embeddings, \
                    test_inputs, test_targets, test_embeddings, prototypes, loss, \
                    accuracy

                if (step + 1) % self.args.eval_interval_steps == 0:
                    logging.info('Eval step')
                    best = self.eval(model, writer, step, metrics)
                    self.print_metrics(step, metrics)
                    self.save_model(model, optimizer, metrics, scheduler,
                                    suffix='_last')
                    if best:
                        self.save_model(model, optimizer, metrics, scheduler,
                                        suffix='_best')
                        eval_steps_without_improvement = 0
                        logging.info(f'{eval_steps_without_improvement} eval steps without improvement')
                    else:
                        eval_steps_without_improvement += 1
                        logging.info(f'{eval_steps_without_improvement} eval steps without improvement')
                        if self.args.early_stop != -1 and eval_steps_without_improvement >= self.args.early_stop:
                            logging.info('Early stop')
                            break

                if step >= self.args.num_batches:
                    break

        # Save model
        self.save_model(model, optimizer, metrics, scheduler,
                        suffix='_end')

        return best, metrics['best_val_loss']

    def eval_one_split(self, model, meta_split, num_cases):
        num_batches = int(np.ceil(num_cases / self.args.eval_batch_size))
        dataset, dataloader = self.get_dataset_and_loader(
            meta_split=meta_split, batch_size=self.args.eval_batch_size
        )

        accuracies = []
        losses = []
        ns = []

        done_batches = 0
        done = False
        while not done:
            with tqdm(dataloader, total=min(num_batches - done_batches, len(dataloader)),
                      disable=self.args.no_bar) as pbar:
                for step, batch in enumerate(pbar):
                    train_inputs, train_targets = batch['train']
                    train_inputs = train_inputs.to(device=self.args.device)
                    train_targets = train_targets.to(device=self.args.device)
                    train_embeddings = model(train_inputs)

                    test_inputs, test_targets = batch['test']
                    test_inputs = test_inputs.to(device=self.args.device)
                    test_targets = test_targets.to(device=self.args.device)
                    test_embeddings = model(test_inputs)

                    prototypes = get_prototypes(train_embeddings, train_targets, dataset.num_classes_per_task)

                    batch_loss = prototypical_loss(
                        prototypes, test_embeddings, test_targets).item()
                    batch_accuracy = get_accuracy(
                        prototypes, test_embeddings, test_targets).item()

                    accuracies.append(batch_accuracy)
                    losses.append(batch_loss)
                    ns.append(len(test_inputs))
                    done_batches += 1
                    if done_batches >= num_batches:
                        done = True
                        break
            if not done:
                logging.info('Reloading exhausted eval dataloader...')

        total_n = sum(ns)
        accuracy = sum(acc * n / total_n for acc, n in zip(accuracies, ns))
        loss = sum(l * n / total_n for l, n in zip(losses, ns))
        return accuracy, loss
