import torch
import copy
from typing import List
import numpy as np
import torch.nn.functional as F


def stabilize(z):
    return z + ((z == 0.).to(z) + z.sign()) * 1e-6


def modified_layer(
        layer,
        transform
):
    """
    This function creates a copy of a layer and modify
    its parameters based on a transformation function 'transform'.
    -------------------

    :param layer: a layer which its parameters are going to be transformed.
    :param transform: a transformation function.
    :return: a new layer with modified parameters.
    """
    new_layer = copy.deepcopy(layer)

    try:
        new_layer.weight = torch.nn.Parameter(transform(layer.weight.float(), name='weight'))
    except AttributeError as e:
        print(e)

    try:
        new_layer.bias = torch.nn.Parameter(transform(layer.bias.float(), name='bias'))
    except AttributeError as e:
        print(e)

    return new_layer


def relevance_propagation(
        model,
        embeddings,
        targets,
        n_classes
):
    embeddings = (embeddings * 1).data
    embeddings.requires_grad_(True)
    logits = model(inputs_embeds=embeddings)[:, -1, :]
    predictions = torch.argmax(logits, dim=-1)[:, None]

    one_hot_targets = torch.nn.functional.one_hot(targets, n_classes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    R_out = (logits * one_hot_targets.to(device)).sum()
    R_out.backward()

    R = (embeddings * embeddings.grad).sum(2).detach().cpu().numpy()

    return R, predictions


def vision_relevance_propagation(
        model,
        embeddings,
        targets,
        n_classes,
        params_to_detach=['A', 'B', 'C', 'z']
):
    """
    This function is only tested for 'vim_small_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2'.
    """
    B, M, _ = embeddings.shape
    cls_token = model.cls_token.expand(B, -1, -1)
    token_position = M // 2
    # Add cls token in the middle.
    embeddings = torch.cat((embeddings[:, :token_position, :], cls_token, embeddings[:, token_position:, :]), dim=1)
    M = embeddings.shape[1]

    embeddings = embeddings + model.pos_embed
    embeddings = model.pos_drop(embeddings)
                
    embeddings = (embeddings * 1).data
    embeddings.requires_grad_(True)
    logits = model(embeddings, token_position, params_to_detach=params_to_detach)
    predictions = torch.argmax(logits, dim=-1)[:, None]

    one_hot_targets = torch.nn.functional.one_hot(targets, n_classes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    R_out = (logits * one_hot_targets.to(device)).sum()
    R_out.backward()

    R = (embeddings * embeddings.grad).sum(2).detach().cpu().numpy()

    return R, predictions, logits
