# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import torch
import torch.nn as nn
from functools import partial
from torch import Tensor
from typing import Optional, List

from timm.models.vision_transformer import VisionTransformer
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_, lecun_normal_

from timm.models.layers import DropPath, to_2tuple
import math
from collections import namedtuple
from mamba_ssm.modules.mamba_simple import Mamba
from mamba_ssm.utils.generation import GenerationMixin
from mamba_ssm.utils.hf import load_config_hf, load_state_dict_hf

from rope import *
import random

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None
from einops import rearrange, repeat, einsum, repeat
from mamba_lrp.lrp.utils import *
from mamba_lrp.lrp.core import *
from mamba_lrp.lrp.rule import *


def selective_scan_fn(
        u,
        delta,
        A,
        B,
        C,
        D,
        params_to_detach: List = []
):
    """
    An explainable version of the `Selective Scan` function.

    This is the discrete state space model formula:
        x(t + 1) = Ax(t) + Bu(t)
        y(t) = Cx(t) + Du(t)
    -------------------

    :param D: shape = (d_in, )
    :param C: shape = (b, l, n)
    :param B: shape = (b, l, n)
    :param A: shape = (d_in, n)
    :param u: (b, l, d_in)
    :param delta: discretization parameter.
    :param params_to_detach: parameters that should be detached.
    """
    (b, l, d_in) = u.shape
    n = A.shape[1]

    # Discretize continuous parameters (A, B).
    deltaA = torch.exp(einsum(delta, A, 'b l d_in, d_in n -> b l d_in n'))
    deltaB = einsum(delta, B, 'b l d_in, b l n -> b l d_in n')

    # Detach the parameters A, B, and C.
    if 'A' in params_to_detach:
        deltaA = deltaA.detach()

    if 'B' in params_to_detach:
        deltaB = deltaB.detach()

    if 'C' in params_to_detach:
        C = C.detach()

    deltaB_u = einsum(deltaB, u, 'b l d_in n, b l d_in -> b l d_in n')

    # Perform selective scan.
    x = torch.zeros((b, d_in, n), device=deltaA.device)

    ys = []
    for i in range(l):
        x = deltaA[:, i] * x + deltaB_u[:, i]
        y = einsum(x, C[:, i, :], 'b d_in n, b n -> b d_in')
        ys.append(y)
    y = torch.stack(ys, dim=1)
    y = y + u * D

    return y


def mamba_inner_fn_no_out_proj(
        xz,
        conv1d,
        act,
        x_proj,
        dt_proj,
        softplus,
        A,
        D,
        seqlen,
        params_to_detach
):
    dt_rank = dt_proj.fc.weight.shape[1]
    d_state = A.shape[-1] * (1 if not A.is_complex() else 2)

    x, z = xz.chunk(2, dim=1)
    z = rearrange(z, 'b d l -> b l d')

    # Conv1d.
    x = conv1d(x)[..., :seqlen]
    x = rearrange(x, 'b d l -> b l d')
    x = act(x)

    x_dbl = rearrange(x_proj(x), "b l d -> (b l) d")
    dt, B, C = torch.split(x_dbl, [dt_rank, d_state, d_state], dim=-1)

    dt = rearrange(dt, "(b l) d -> b l d", l=seqlen)
    dt = softplus(dt_proj(dt))

    B = rearrange(B, "(b l) dstate -> b l dstate", l=seqlen).contiguous()
    C = rearrange(C, "(b l) dstate -> b l dstate", l=seqlen).contiguous()

    # ------------------------
    y = selective_scan_fn(
        u=x.float(),
        delta=dt.float(),
        A=A.float(),
        B=B.float(),
        C=C.float(),
        D=D.float(),
        params_to_detach=params_to_detach
    )
    # ------------------------

    # Half-propagation rule.
    if z is not None:
        if 'z' in params_to_detach:
            y = y * act(z)
            y = (y / 2.) + (y / 2.).detach()
        else:
            y = y * act(z)
    return y


class ModifiedMamba(nn.Module):
    def __init__(
            self,
            mamba,
            zero_bias: bool = False,
            layer_transforms: List = []
    ):
        """
        A wrapper to make model.backbone.layer[i].mixer explainable.
        -------------------

        :param mamba: model.backbone.layer[i].mixer
        """
        super(ModifiedMamba, self).__init__()
        self.d_model = mamba.d_model
        self.d_state = mamba.d_state
        self.d_conv = mamba.d_conv
        self.expand = mamba.expand
        self.d_inner = mamba.d_inner
        self.dt_rank = mamba.dt_rank
        self.use_fast_path = False
        self.layer_idx = mamba.layer_idx
        self.bimamba_type = mamba.bimamba_type
        self.if_devide_out = mamba.if_devide_out
        self.init_layer_scale = mamba.init_layer_scale
        if self.init_layer_scale is not None:
            self.gamma = mamba.gamma

        # Input projection layer.
        if 'in_proj' in layer_transforms:
            self.in_proj = ModifiedLinear(fc=mamba.in_proj,
                                          transform='gamma',
                                          zero_bias=zero_bias)
        else:
            self.in_proj = ModifiedLinear(fc=mamba.in_proj,
                                          transform=None,
                                          zero_bias=zero_bias)

        # Conv1d layer.
        if 'conv1d' in layer_transforms:
            self.conv1d = ModifiedConv(conv=mamba.conv1d,
                                       transform='gamma',
                                       zero_bias=zero_bias)
        else:
            self.conv1d = ModifiedConv(conv=mamba.conv1d,
                                       transform=None,
                                       zero_bias=zero_bias)

        # SiLU.
        self.activation = mamba.activation
        if 'act' in layer_transforms:
            self.act = ModifiedAct(mamba.act, transform='identity')
        else:
            self.act = ModifiedAct(mamba.act, transform=None)

        # Softplus.
        if 'softplus' in layer_transforms:
            self.softplus = ModifiedSoftPlus(nn.Softplus(), transform='identity')
        else:
            self.softplus = ModifiedSoftPlus(nn.Softplus(), transform=None)

        # X projection layer.
        if 'x_proj' in layer_transforms:
            self.x_proj = ModifiedLinear(fc=mamba.x_proj,
                                         transform='gamma',
                                         zero_bias=zero_bias)
        else:
            self.x_proj = ModifiedLinear(fc=mamba.x_proj,
                                         transform=None,
                                         zero_bias=zero_bias)

        # Delta projection layer.
        if 'dt_proj' in layer_transforms:
            self.dt_proj = ModifiedLinear(fc=mamba.dt_proj,
                                          transform='gamma',
                                          zero_bias=zero_bias)
        else:
            self.dt_proj = ModifiedLinear(fc=mamba.dt_proj,
                                          transform=None,
                                          zero_bias=zero_bias)

        # S4 parameters.
        self.A_log = mamba.A_log
        self.D = mamba.D  # D "skip" parameter.

        # bidirectional
        if self.bimamba_type == "v1":
            self.A_b_log = mamba.A_b_log
        elif self.bimamba_type == "v2":
            self.A_b_log = mamba.A_b_log
            if 'conv1d' in layer_transforms:
                self.conv1d_b = ModifiedConv(conv=mamba.conv1d_b,
                                             transform='gamma',
                                             zero_bias=zero_bias)
            else:
                self.conv1d_b = ModifiedConv(conv=mamba.conv1d_b,
                                             transform=None,
                                             zero_bias=zero_bias)
            if 'x_proj' in layer_transforms:
                self.x_proj_b = ModifiedLinear(fc=mamba.x_proj_b,
                                               transform='gamma',
                                               zero_bias=zero_bias)
            else:
                self.x_proj_b = ModifiedLinear(fc=mamba.x_proj_b,
                                               transform=None,
                                               zero_bias=zero_bias)
            if 'dt_proj' in layer_transforms:
                self.dt_proj_b = ModifiedLinear(fc=mamba.dt_proj_b,
                                                transform='gamma',
                                                zero_bias=zero_bias)
            else:
                self.dt_proj_b = ModifiedLinear(fc=mamba.dt_proj_b,
                                                transform=None,
                                                zero_bias=zero_bias)
            self.D_b = mamba.D_b

        # Output projection layer.
        if 'out_proj' in layer_transforms:
            self.out_proj = ModifiedLinear(fc=mamba.out_proj,
                                           transform='gamma',
                                           zero_bias=zero_bias)
        else:
            self.out_proj = ModifiedLinear(fc=mamba.out_proj,
                                           transform=None,
                                           zero_bias=zero_bias)

    def forward(
            self,
            hidden_states,
            params_to_detach=[]
    ):
        """
        This function do not support the use_fast_path=True mode.
        -------------------

        :param hidden_states: (B, L, D)
        :param params_to_detach: parameters that should be detached in the selective_scan_fn.

        return same shape as hidden_states
        """
        batch, seqlen, dim = hidden_states.shape

        # We do matmul and transpose BLD -> DBL at the same time.
        xz = rearrange(
            self.in_proj(hidden_states),
            "b l d -> b d l",
            l=seqlen,
        )

        A = -torch.exp(self.A_log.float())
        out = mamba_inner_fn_no_out_proj(
            xz,
            self.conv1d,
            self.act,
            self.x_proj,
            self.dt_proj,
            self.softplus,
            A,
            self.D,
            seqlen,
            params_to_detach
        )

        A_b = -torch.exp(self.A_b_log.float())
        out_b = mamba_inner_fn_no_out_proj(
            xz.flip([-1]),
            self.conv1d_b,
            self.act,
            self.x_proj_b,
            self.dt_proj_b,
            self.softplus,
            A_b,
            self.D_b,
            seqlen,
            params_to_detach
        )

        if not self.if_devide_out:
            out = self.out_proj(out + out_b.flip([1]))
        else:
            out = self.out_proj((out + out_b.flip([1])) / 2)

        if self.init_layer_scale is not None:
            out = out * self.gamma
        return out


class ModifiedPatchEmbed(nn.Module):
    """
    2D Image to Patch Embedding
    """

    def __init__(self, patch_embed):
        super(ModifiedPatchEmbed, self).__init__()
        self.img_size = patch_embed.img_size
        self.patch_size = patch_embed.patch_size
        self.grid_size = patch_embed.grid_size
        self.num_patches = patch_embed.num_patches
        self.flatten = patch_embed.flatten

        self.proj = ModifiedConv(patch_embed.proj, transform='gamma')
        self.norm = patch_embed.norm

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x


class ModifiedBlock(nn.Module):
    def __init__(
            self,
            block,
            zero_bias: bool = False,
            layer_transforms=[]
    ):
        """
        A wrapper to make model.backbone.layer[i] explainable.
        -------------------

        :param block: model.backbone.layer[i]
        """
        super(ModifiedBlock, self).__init__()

        self.residual_in_fp32 = block.residual_in_fp32
        self.fused_add_norm = False
        self.mixer = ModifiedMamba(block.mixer, zero_bias=zero_bias, layer_transforms=layer_transforms)

        if 'norm' in layer_transforms:
            self.norm = ModifiedRMSNorm(block.norm, zero_bias=zero_bias, transform='identity') if \
                isinstance(block.norm, RMSNorm) else ModifiedLayerNorm(block.norm, zero_bias=zero_bias)
        else:
            self.norm = ModifiedRMSNorm(block.norm, zero_bias=zero_bias, transform=None) if \
                isinstance(block.norm, RMSNorm) else ModifiedLayerNorm(block.norm, zero_bias=zero_bias)

        self.drop_path = block.drop_path

    def forward(
            self,
            hidden_states: Tensor,
            residual: Optional[Tensor] = None,
            params_to_detach=[]
    ):

        if not self.fused_add_norm:
            # A residual connection.
            residual = (self.drop_path(hidden_states) + residual) if residual is not None else hidden_states
            hidden_states, residual = self.norm(
                residual.to(dtype=self.norm.weight.dtype),
                prenorm=True
            )
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)

        hidden_states = self.mixer(
            hidden_states,
            params_to_detach=params_to_detach
        )

        return hidden_states, residual


class ModifiedVisionMamba(nn.Module):
    def __init__(self, vim, zero_bias, layer_transforms=[]):
        super(ModifiedVisionMamba, self).__init__()

        self.residual_in_fp32 = vim.residual_in_fp32
        self.fused_add_norm = False
        self.if_bidirectional = vim.if_bidirectional
        self.final_pool_type = vim.final_pool_type
        self.if_abs_pos_embed = vim.if_abs_pos_embed
        self.if_rope = vim.if_rope
        self.if_rope_residual = vim.if_rope_residual
        self.flip_img_sequences_ratio = vim.flip_img_sequences_ratio
        self.if_cls_token = vim.if_cls_token
        self.use_double_cls_token = vim.use_double_cls_token
        self.use_middle_cls_token = vim.use_middle_cls_token
        self.num_tokens = 1 if vim.if_cls_token else 0
        self.num_classes = vim.num_classes
        self.d_model = vim.d_model
        self.patch_embed = ModifiedPatchEmbed(vim.patch_embed)
        if self.if_cls_token:
            if self.use_double_cls_token:
                self.cls_token_head = vim.cls_token_head
                self.cls_token_tail = vim.cls_token_tail
                self.num_tokens = vim.num_tokens
            else:
                self.cls_token = vim.cls_token

        if self.if_abs_pos_embed:
            self.pos_embed = vim.pos_embed
            self.pos_drop = vim.pos_drop

        if self.if_rope:
            self.rope = vim.rope

        self.head = ModifiedLinear(fc=vim.head,
                                   transform=None,
                                   zero_bias=zero_bias)
        self.drop_path = vim.drop_path
        self.layers = nn.ModuleList(
            [
                ModifiedBlock(
                    block,
                    zero_bias=zero_bias,
                    layer_transforms=layer_transforms
                )
                for layer_idx, block in enumerate(vim.layers)
            ]
        )

        if 'norm' in layer_transforms:
            self.norm_f = ModifiedRMSNorm(vim.norm_f, zero_bias=zero_bias, transform="identity")
        else:
            self.norm_f = ModifiedRMSNorm(vim.norm_f, zero_bias=zero_bias, transform=None)

    def forward_features(
            self,
            x,
            token_position,
            if_random_cls_token_position=False,
            if_random_token_rank=False,
            params_to_detach=[]
    ):

        B, M, _ = x.shape
        if if_random_token_rank:
            shuffle_indices = torch.randperm(M)

            if isinstance(token_position, list):
                print("original value: ", x[0, token_position[0], 0], x[0, token_position[1], 0])
            else:
                print("original value: ", x[0, token_position, 0])
            print("original token_position: ", token_position)

            x = x[:, shuffle_indices, :]

            if isinstance(token_position, list):

                new_token_position = [torch.where(shuffle_indices == token_position[i])[0].item() for i in
                                      range(len(token_position))]
                token_position = new_token_position
            else:
                token_position = torch.where(shuffle_indices == token_position)[0].item()

            if isinstance(token_position, list):
                print("new value: ", x[0, token_position[0], 0], x[0, token_position[1], 0])
            else:
                print("new value: ", x[0, token_position, 0])
            print("new token_position: ", token_position)

        if_flip_img_sequences = False
        if self.flip_img_sequences_ratio > 0 and (self.flip_img_sequences_ratio - random.random()) > 1e-5:
            x = x.flip([1])
            if_flip_img_sequences = True

        # Mamba.
        residual = None
        hidden_states = x
        if not self.if_bidirectional:
            for layer in self.layers:
                if if_flip_img_sequences and self.if_rope:
                    hidden_states = hidden_states.flip([1])
                    if residual is not None:
                        residual = residual.flip([1])

                if self.if_rope:
                    hidden_states = self.rope(hidden_states)
                    if residual is not None and self.if_rope_residual:
                        residual = self.rope(residual)

                if if_flip_img_sequences and self.if_rope:
                    hidden_states = hidden_states.flip([1])
                    if residual is not None:
                        residual = residual.flip([1])

                hidden_states, residual = layer(
                    hidden_states, residual, params_to_detach=params_to_detach
                )
        else:
            for i in range(len(self.layers) // 2):
                if self.if_rope:
                    hidden_states = self.rope(hidden_states)
                    if residual is not None and self.if_rope_residual:
                        residual = self.rope(residual)

                hidden_states_f, residual_f = self.layers[i * 2](
                    hidden_states, residual,
                    params_to_detach=params_to_detach
                )
                hidden_states_b, residual_b = self.layers[i * 2 + 1](
                    hidden_states.flip([1]), None if residual is None else residual.flip([1]),
                    params_to_detach=params_to_detach
                )
                hidden_states = hidden_states_f + hidden_states_b.flip([1])
                residual = residual_f + residual_b.flip([1])

        if not self.fused_add_norm:
            if residual is None:
                residual = hidden_states
            else:
                residual = residual + self.drop_path(hidden_states)
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            raise NotImplementedError

        # Return only cls token if it exists.
        if self.if_cls_token:
            if self.use_double_cls_token:
                return (hidden_states[:, token_position[0], :] + hidden_states[:, token_position[1], :]) / 2
            else:
                if self.use_middle_cls_token:
                    return hidden_states[:, token_position, :]
                elif if_random_cls_token_position:
                    return hidden_states[:, token_position, :]
                else:
                    return hidden_states[:, token_position, :]

        if self.final_pool_type == 'none':
            return hidden_states[:, -1, :]
        elif self.final_pool_type == 'mean':
            return hidden_states.mean(dim=1)
        elif self.final_pool_type == 'max':
            return hidden_states
        elif self.final_pool_type == 'all':
            return hidden_states
        else:
            raise NotImplementedError

    def forward(
            self,
            x,
            token_position,
            return_features=False,
            if_random_cls_token_position=False,
            if_random_token_rank=False,
            params_to_detach=[]
    ):
        x = self.forward_features(
            x,
            token_position,
            if_random_cls_token_position=if_random_cls_token_position,
            if_random_token_rank=if_random_token_rank,
            params_to_detach=params_to_detach
        )
        if return_features:
            return x
        x = self.head(x)
        if self.final_pool_type == 'max':
            x = x.max(dim=1)[0]
        return x
