from typing import Any, Tuple
from torch.nn.modules import Module
import torch
from torch import nn as nn
from mamba_lrp.lrp.utils import *
from mamba_lrp.lrp.rule import *


class ModifiedLinear(Module):
    def __init__(
            self,
            fc: torch.nn.Linear,
            transform: Any,
            zero_bias: bool = False
    ):
        """
        A wrapper to make torch.nn.Linear explainable.
        -------------------

        :param fc: a fully-connected layer (torch.nn.Linear).
        :param transform: a transformation function to modify the layer's parameters.
        :param zero_bias: set the layer's bias to zero. It is useful when checking the conservation property.
        """
        super(ModifiedLinear, self).__init__()
        self.fc = fc

        if zero_bias:
            if fc.bias is not None:
                self.fc.bias = torch.nn.Parameter(
                    torch.zeros(self.fc.bias.shape, dtype=torch.float, device="cuda")
                )

        self.transform = transform

        if self.transform is None:
            self.modifiers = None
        elif self.transform == 'gamma':
            self.modifiers = [
                modified_layer(layer=self.fc, transform=gamma(gam=0.25, minimum=0)),  # Pos
                modified_layer(layer=self.fc, transform=gamma(gam=0.25, maximum=0, modify_bias=False)),  # Neg
                modified_layer(layer=self.fc, transform=gamma(gam=0.25, maximum=0)),  # Neg
                modified_layer(layer=self.fc, transform=gamma(gam=0.25, minimum=0, modify_bias=False))  # Pos
            ]

    def forward(
            self,
            x: torch.Tensor
    ) -> torch.Tensor:

        z = self.fc(x)

        if self.transform is None:
            return z
        elif transform == 'gamma':
            inputs = [
                x.clamp(min=0),  # Pos
                x.clamp(max=0),  # Neg
                x.clamp(min=0),  # Pos
                x.clamp(max=0)   # Neg
            ]

            outputs = [
                self.modifiers[0](inputs[0]),
                self.modifiers[1](inputs[1]),
                self.modifiers[2](inputs[2]),
                self.modifiers[3](inputs[3])
            ]

            zp_pos = outputs[0] + outputs[1]
            zp_neg = outputs[2] + outputs[3]

            zp_pos *= (z > 1e-6)
            zp_neg *= (z < 1e-6)

            zp = zp_pos + zp_neg
            return zp * (z / zp).data


class ModifiedConv(Module):
    def __init__(
            self,
            conv,
            transform: Any,
            zero_bias: bool = False
    ):
        """
        A wrapper to make torch.nn.Conv1d explainable.
        -------------------

        :param conv: a Convolution layer (e.g. torch.nn.Conv1d).
        :param transform: a transformation function to modify the layer's parameters.
        :param zero_bias: set the layer's bias to zero. It is useful when checking the conservation property.
        """
        super(ModifiedConv, self).__init__()
        self.conv = conv

        if zero_bias:
            if conv.bias is not None:
                self.conv.bias = torch.nn.Parameter(
                    torch.zeros(self.conv.bias.shape, dtype=torch.float, device="cuda")
                )

        self.transform = transform

        if self.transform is None:
            self.modifiers = None
        elif self.transform == 'gamma':
            self.modifiers = [
                modified_layer(layer=self.conv, transform=gamma(gam=0.25, minimum=0)),
                modified_layer(layer=self.conv, transform=gamma(gam=0.25, maximum=0, modify_bias=False)),
                modified_layer(layer=self.conv, transform=gamma(gam=0.25, maximum=0)),
                modified_layer(layer=self.conv, transform=gamma(gam=0.25, minimum=0, modify_bias=False))
            ]

    def forward(
            self,
            x: torch.Tensor
    ) -> torch.Tensor:

        z = self.conv(x)

        if self.transform is None:
            return z
        elif self.transform == 'gamma':
            inputs = [
                x.clamp(min=0),
                x.clamp(max=0),
                x.clamp(min=0),
                x.clamp(max=0)
            ]

            outputs = [
                self.modifiers[0](inputs[0]),
                self.modifiers[1](inputs[1]),
                self.modifiers[2](inputs[2]),
                self.modifiers[3](inputs[3])
            ]

            zp_pos = outputs[0] + outputs[1]
            zp_neg = outputs[2] + outputs[3]

            zp_pos *= (z > 1e-6)
            zp_neg *= (z < 1e-6)

            zp = zp_pos + zp_neg

        return zp * (z / zp).data


class ModifiedAct(Module):
    def __init__(
            self,
            act: Any,
            transform: Any
    ):
        """
       A wrapper to make activation layers such as torch.nn.SiLU or torch.nn.GELU explainable.
       -------------------

       :param act: an activation layer (torch.nn.SiLU or torch.nn.GELU).
       """
        super(ModifiedAct, self).__init__()
        self.modified_act = nn.Identity()
        self.act = act
        self.transform = transform

    def forward(
            self,
            x
    ):
        z = self.act(x)

        if self.transform is None:
            return z
        elif self.transform == 'identity':
            zp = self.modified_act(x)
            zp = stabilize(zp)
            return zp * (z / zp).data
        else:
            raise NotImplementedError


class ModifiedSoftPlus(Module):
    def __init__(
            self,
            act: Any,
            transform: Any
    ):
        """
       A wrapper to make torch.nn.Softplus explainable.
       -------------------

       :param act: an activation layer (torch.nn.Softplus).
       """
        super(ModifiedSoftPlus, self).__init__()
        self.modified_act = nn.Identity()
        self.act = act
        self.transform = transform

    def forward(
            self,
            x
    ):
        z = self.act(x)
        if self.transform is None:
            return z
        elif self.transform == 'identity':
            zp = self.modified_act(x)
            zp = stabilize(zp)
            return zp * (z / zp).data


class ModifiedRMSNorm(torch.nn.Module):
    def __init__(
            self,
            norm,
            zero_bias: bool = False,
            transform: Any = None
    ):
        """
        A wrapper to make RMSNorm explainable.
        -------------------

        :param norm: a norm layer (RMSNorm).
        :param zero_bias: set the layer's bias to zero. It is useful when checking the conservation property.
        """
        super(ModifiedRMSNorm, self).__init__()

        self.eps = norm.eps
        self.weight = norm.weight
        self.bias = norm.bias
        self.transform = transform

        if zero_bias:
            if norm.bias is not None:
                norm.bias = torch.nn.Parameter(
                    torch.zeros(norm.bias.shape, dtype=torch.float, device="cuda")
                )

    def forward(
            self,
            x: torch.Tensor,
            residual=None,
            prenorm: bool = False,
            residual_in_fp32: bool = False
    ) -> torch.Tensor:
        if residual is not None:
            if residual_in_fp32:
                residual = residual.to(torch.float32)
            x = (x + residual).to(x.dtype)

        denominator = 1 / torch.sqrt((x.square().mean(dim=-1, keepdim=True) + self.eps))

        if self.transform is None:
            z = (x * denominator) * self.weight + self.bias if self.bias is not None else (x * denominator) * self.weight
        elif self.transform == 'identity':
            denominator = denominator.detach()
            z = (x * denominator) * self.weight + self.bias if self.bias is not None else (x * denominator) * self.weight

        return z if not prenorm else (z, x)
