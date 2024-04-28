#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch


def macro_soft_f1_loss(pred_logits, targets, eps=1e-15):
    preds = torch.sigmoid(pred_logits)
    targets = targets.to(dtype=torch.float32)
    tp = torch.sum(preds * targets, axis=0)
    fp = torch.sum(preds * (1 - targets), axis=0)
    fn = torch.sum((1 - preds) * targets, axis=0)
    soft_f1 = 2 * tp / (2 * tp + fn + fp + eps)
    soft_f1_loss = 1 - soft_f1
    return torch.mean(soft_f1_loss)


def macro_double_soft_f1_loss(pred_logits, targets, eps=1e-15):
    preds = torch.sigmoid(pred_logits)
    targets = targets.to(dtype=torch.float32)
    tp = torch.sum(preds * targets, axis=0)
    fp = torch.sum(preds * (1 - targets), axis=0)
    fn = torch.sum((1 - preds) * targets, axis=0)
    tn = torch.sum((1 - preds) * (1 - targets), axis=0)
    soft_f1_class_1 = 2 * tp / (2 * tp + fn + fp + eps)
    soft_f1_class_0 = 2 * tn / (2 * tn + fn + fp + eps)
    soft_f1_loss_class_1 = 1 - soft_f1_class_1
    soft_f1_loss_class_0 = 1 - soft_f1_class_0
    return torch.mean(.5 * (soft_f1_loss_class_1 + soft_f1_loss_class_0))


def macro_f1_score(pred_logits, targets, threshold, eps=1e-15):
    preds = torch.gt(torch.sigmoid(pred_logits), threshold)
    targets = targets.to(dtype=torch.float32)
    tp = ((preds * targets) != 0).sum(0, dtype=torch.float32)
    fp = ((preds * (1 - targets)) != 0).sum(0, dtype=torch.float32)
    fn = ((~preds * targets) != 0).sum(0, dtype=torch.float32)
    f1_score = 2 * tp / (2 * tp + fn + fp + eps)
    return torch.mean(f1_score)


def precision_recall(y, y_hat, thresh=.5):
    raise NotImplementedError()


def f1_score(y, y_hat):
    precision, recall = precision_recall(y, y_hat)


# vim: set ts=4 sw=4 sts=4 expandtab:
