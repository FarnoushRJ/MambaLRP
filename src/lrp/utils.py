import torch
import copy
from typing import List
import numpy as np
import torch.nn.functional as F


def stabilize(z):
    return z + ((z == 0.).to(z) + z.sign()) * 1e-6


def relevance_propagation(
        model,
        embeddings,
        targets,
        n_classes
):
    embeddings = (embeddings * 1).data
    embeddings.requires_grad_(True)
    logits = model(embeddings)[:, -1, :]
    predictions = torch.argmax(logits, dim=-1)[:, None]

    one_hot_targets = torch.nn.functional.one_hot(targets, n_classes)
    R_out = (logits * one_hot_targets.cuda()).sum()
    R_out.backward()

    R = (embeddings * embeddings.grad).sum(2).detach().cpu().numpy()

    return R, predictions
