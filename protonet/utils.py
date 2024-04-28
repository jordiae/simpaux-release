from warnings import warn
from einops import rearrange, reduce
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


def plot_grad_flow(named_parameters):
    """Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow"""
    ave_grads = []
    max_grads = []
    layers = []
    for n, p in named_parameters:
        if p.requires_grad and "bias" not in n:
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom=-0.001, top=0.05)  # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])


def get_prototypes(embeddings, targets, num_classes):
    """Compute the prototypes (the mean vector of the embedded training/support
    points belonging to its class) for each classes in the task.

    Parameters
    ----------
    embeddings : `torch.FloatTensor` instance
        A tensor containing the embeddings of the support points. This tensor
        has shape `(meta_batch_size, num_examples, embedding_size)`.
    targets : `torch.LongTensor` instance
        A tensor containing the targets of the support points. This tensor has
        shape `(meta_batch_size, num_examples)`.
    num_classes : int
        Number of classes in the task.

    Returns
    -------
    prototypes : `torch.FloatTensor` instance
        A tensor containing the prototypes for each class. This tensor has shape
        `(meta_batch_size, num_classes, embedding_size)`.
    """
    batch_size, embedding_size = embeddings.size(0), embeddings.size(-1)
    with torch.no_grad():
        ones = torch.ones_like(targets, dtype=embeddings.dtype)
        num_samples = embeddings.new_zeros((batch_size, num_classes))
        num_samples.scatter_add_(1, targets, ones)
        num_samples = num_samples.unsqueeze(-1)
        num_samples = torch.max(num_samples, torch.ones_like(num_samples))

    prototypes = embeddings.new_zeros((batch_size, num_classes, embedding_size))
    indices = targets.unsqueeze(-1).expand_as(embeddings)
    prototypes.scatter_add_(1, indices, embeddings).div_(num_samples)

    return prototypes


def prototypical_loss(prototypes, embeddings, targets, return_accuracy=False):
    """Compute the loss (i.e. negative log-likelihood) for the prototypical
    network, on the test/query points.

    Parameters
    ----------
    prototypes : `torch.FloatTensor` instance
        A tensor containing the prototypes for each class. This tensor has shape
        `(meta_batch_size, num_classes, embedding_size)`.
    embeddings : `torch.FloatTensor` instance
        A tensor containing the embeddings of the query points. This tensor has
        shape `(meta_batch_size, num_examples, embedding_size)`.
    targets : `torch.LongTensor` instance
        A tensor containing the targets of the query points. This tensor has
        shape `(meta_batch_size, num_examples)`.

    Returns
    -------
    loss : `torch.FloatTensor` instance
        The negative log-likelihood on the query points.
    """
    sq_distances = torch.sum((prototypes.unsqueeze(2) - embeddings.unsqueeze(1)) ** 2, dim=-1)

    loss = F.cross_entropy(-sq_distances, targets)
    if return_accuracy:
        accuracy = reduce((sq_distances.argmin(1) == targets).to(
            torch.float), 'b p -> 1', 'mean')
        return loss, accuracy
    return loss


def get_per_attribute_prototypes(embeddings, attribute_targets, max_n_support=5):
    """Computes prototypes for each attribute occuring in the batch

    Parameters
    ----------
    embeddings : `torch.FloatTensor` instance
        A tensor containing the embeddings of support samples
    attribute_targets : `torch.Tensor` instance
        A binary tensor containing ones indicating present attributes of each
        sample
    max_n_support: `int` instance
        Maximum number of support samples for each positive and negative
        attribute prototype

    Returns
    -------
    prototypes : `torch.FloatTensor` instance
        A tensor containing the negative/positive prototypes for each present
        attribute. Shape Ax2xE, where A is the number of present attributes,
        the 2 axis corresponds to negative and positive prototypes (in that
        order), and E is the embedding size.
    batch_attributes: `torch.LongTensor` instance
        A tensor containing the indices of present attributes in the same order
        as the corresponding prototypes.
    """
    # embeddings BxD, targets BxA

    # mask to only consider samples with at least one present attribute
    has_attrib_mask = reduce(attribute_targets,
                             'b n p->b n', 'sum') != 0
    attribute_targets_ = attribute_targets[has_attrib_mask]
    # only consider attributes with at least one positive and one negative
    # sample
    batch_attribute_count = reduce(attribute_targets_, 'b p -> p', 'sum')
    batch_attributes_ = (
        (batch_attribute_count > 0) *
        (batch_attribute_count < attribute_targets_.shape[0])
    ).nonzero().flatten()

    pos_attribute_masks = attribute_targets_[:, batch_attributes_]
    neg_attribute_masks = torch.logical_not(
        attribute_targets_[:, batch_attributes_]).to(dtype=torch.float)

    # we want to consider at most max_n_support many samples to compute each
    # prototype so we mask out all beyond the first max_n_support
    pos_cutoff_indices = torch.where(pos_attribute_masks.cumsum(0) >
                                     max_n_support)
    neg_cutoff_indices = torch.where(neg_attribute_masks.cumsum(0) >
                                     max_n_support)

    for (pos_i, pos_j), (neg_i, neg_j) in zip(zip(*pos_cutoff_indices),
                                              zip(*neg_cutoff_indices)):
        pos_attribute_masks[pos_i:, pos_j].fill_(False)
        neg_attribute_masks[neg_i:, neg_j].fill_(False)

    embeddings_ = embeddings[has_attrib_mask]
    pos_prototypes = torch.einsum('be,bp->pe', (embeddings_,
                                                pos_attribute_masks)) / \
        pos_attribute_masks.sum(0).unsqueeze(1)
    neg_prototypes = torch.einsum('be,bp->pe', (embeddings_,
                                                neg_attribute_masks)) / \
        neg_attribute_masks.sum(0).unsqueeze(1)

    prototypes = torch.stack((neg_prototypes, pos_prototypes), dim=1)
    return prototypes, batch_attributes_


def multilabel_prototypical_loss(prototypes, batch_attributes,
                                 embeddings, targets, return_accuracy=False,
                                 return_preds_and_targets=False):
    attribute_targets = targets[..., batch_attributes].to(torch.long)

    # mask to only consider samples with at least one present attribute
    has_attrib_mask = reduce(attribute_targets,
                             'b n p->b n', 'sum') != 0

    prototypes_ = rearrange(prototypes, 'p c e -> 1 e c p')
    embeddings_ = rearrange(embeddings[has_attrib_mask], 'b e -> b e 1 1')

    sq_dist_ = reduce((embeddings_ - prototypes_)**2,
                      'b e c p -> b c p', 'sum')

    attribute_targets_ = attribute_targets[has_attrib_mask]
    if sq_dist_.numel() == 0:
        warn('no annotated attributes in support and/or query samples')
        loss = torch.tensor(0.0)
        accuracy = torch.tensor(float('NaN'))
        preds = torch.tensor(float('NaN'))
    else:
        loss = F.cross_entropy(-sq_dist_,
                               attribute_targets_)
        preds = sq_dist_.detach().argmin(1)
        accuracy = reduce((preds == attribute_targets_).to(
            torch.float), 'b p -> 1', 'mean')

    rvals = [loss]
    if return_accuracy:
        rvals.append(accuracy)
    if return_preds_and_targets:
        rvals.extend([preds, attribute_targets_])
    return rvals if len(rvals) > 1 else rvals[0]


def get_accuracy(prototypes, embeddings, targets):
    """Compute the accuracy of the prototypical network on the test/query points.

    Parameters
    ----------
    prototypes : `torch.FloatTensor` instance
        A tensor containing the prototypes for each class. This tensor has shape
        `(meta_batch_size, num_classes, embedding_size)`.
    embeddings : `torch.FloatTensor` instance
        A tensor containing the embeddings of the query points. This tensor has
        shape `(meta_batch_size, num_examples, embedding_size)`.
    targets : `torch.LongTensor` instance
        A tensor containing the targets of the query points. This tensor has
        shape `(meta_batch_size, num_examples)`.

    Returns
    -------
    accuracy : `torch.FloatTensor` instance
        Mean accuracy on the query points.
    """
    predictions = get_predictions(prototypes, embeddings)
    return torch.mean(predictions.eq(targets).float())


def get_predictions(prototypes, embeddings):
    """Compute the predictions of the prototypical network on test/query points.

    Parameters
    ----------
    prototypes : `torch.FloatTensor` instance
        A tensor containing the prototypes for each class. This tensor has shape
        `(meta_batch_size, num_classes, embedding_size)`.
    embeddings : `torch.FloatTensor` instance
        A tensor containing the embeddings of the query points. This tensor has
        shape `(meta_batch_size, num_examples, embedding_size)`.

    Returns
    -------
    predictions: `torch.FloatTensor` instance
        predictions of the query points.
    """
    sq_distances = torch.sum((prototypes.unsqueeze(1) - embeddings.unsqueeze(2)) ** 2, dim=-1)
    _, predictions = torch.min(sq_distances, dim=-1)
    return predictions
