import torch.nn as nn
import torch
import torch.nn.functional as F
from torch import Tensor
from einops import rearrange, repeat, einsum, repeat
from typing import Optional, List
from transformers import MambaConfig

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
    from mamba_ssm.ops.selective_scan_interface import mamba_inner_fn, selective_scan_fn
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    RMSNorm = None

from mamba_lrp.lrp.core import ModifiedAct


class MambaCache:
    """
    Arguments:
        config: MambaConfig
        batch_size: int
        dtype: torch.dtype
        device: torch.device

    Attributes:
        seqlen_offset: int
        dtype: torch.dtype
        conv_states: Dict[int, torch.Tensor] # layer_idx -> [batch_size, intermediate_size, conv_kernel_size]
        ssm_states: Dict[int, torch.Tensor] # layer_idx -> [batch_size, intermediate_size, ssm_state_size]
    """

    def __init__(
            self, config: MambaConfig, batch_size: int, dtype: torch.dtype = torch.float16, device: Optional[str] = None
    ):
        self.seqlen_offset = 0
        self.dtype = dtype
        intermediate_size = config.intermediate_size
        ssm_state_size = config.state_size
        conv_kernel_size = config.conv_kernel

        self.conv_states = {
            i: torch.zeros(batch_size, intermediate_size, conv_kernel_size, device=device, dtype=dtype)
            for i in range(config.num_hidden_layers)
        }
        self.ssm_states = {
            i: torch.zeros(batch_size, intermediate_size, ssm_state_size, device=device, dtype=dtype)
            for i in range(config.num_hidden_layers)
        }


class ModifiedMambaMixer(nn.Module):
    def __init__(
            self,
            mixer,
            is_fast_forward_available=True
    ):
        """
        A wrapper to make Mamba mixer layer explainable.
        -------------------

        :param mixer: Mamba mixer layer
        :param is_fast_forward_available: if True, fast CUDA kernels will be used
        """
        super(ModifiedMambaMixer, self).__init__()
        self.hidden_size = mixer.hidden_size
        self.ssm_state_size = mixer.ssm_state_size
        self.conv_kernel_size = mixer.conv_kernel_size
        self.intermediate_size = mixer.intermediate_size
        self.time_step_rank = int(mixer.time_step_rank)
        self.layer_idx = mixer.layer_idx
        self.use_conv_bias = mixer.use_conv_bias
        self.use_bias = mixer.use_bias

        # Layers.
        self.conv1d = mixer.conv1d
        self.activation = mixer.activation
        self.act = ModifiedAct(mixer.act, transform='identity')
        self.in_proj = mixer.in_proj
        self.x_proj = mixer.x_proj
        self.dt_proj = mixer.dt_proj
        self.out_proj = mixer.out_proj

        # SSM parameters.
        self.A_log = mixer.A_log
        self.D = mixer.D

        self.is_fast_forward_available = is_fast_forward_available

    def cuda_kernels_forward(
            self,
            hidden_states: torch.Tensor,
            cache_params=None
    ):
        # 1. Gated MLP's linear projection
        projected_states = self.in_proj(hidden_states).transpose(1, 2)
        hidden_states, gate = projected_states.chunk(2, dim=1)

        # 2. Convolution sequence transformation
        conv_weights = self.conv1d.weight.view(self.conv1d.weight.size(0), self.conv1d.weight.size(2))
        if cache_params is not None and cache_params.seqlen_offset > 0:
            hidden_states = causal_conv1d_update(
                hidden_states.squeeze(-1),
                cache_params.conv_states[self.layer_idx],
                conv_weights,
                self.conv1d.bias,
                self.activation,
            )
            hidden_states = hidden_states.unsqueeze(-1)
        else:
            if cache_params is not None:
                conv_states = nn.functional.pad(
                    hidden_states, (self.conv_kernel_size - hidden_states.shape[-1], 0)
                )
                cache_params.conv_states[self.layer_idx].copy_(conv_states)
            hidden_states = causal_conv1d_fn(
                hidden_states, conv_weights, self.conv1d.bias, activation=self.activation
            )

        # 3. State Space Model sequence transformation
        # 3.a. input varying initialization of time_step, B and C
        # Detach parameters A, B, C and discrete_time_step.
        ssm_parameters = self.x_proj(hidden_states.transpose(1, 2))
        time_step, B, C = torch.split(
            ssm_parameters, [self.time_step_rank, self.ssm_state_size, self.ssm_state_size], dim=-1
        )
        discrete_time_step = self.dt_proj.weight @ time_step.transpose(1, 2)

        A = -torch.exp(self.A_log.float())
        # 3.c perform the recurrence y ← SSM(A, B, C)(x)
        time_proj_bias = self.dt_proj.bias.float() if hasattr(self.dt_proj, "bias") else None
        if cache_params is not None and cache_params.seqlen_offset > 0:
            scan_outputs = selective_state_update(
                cache_params.ssm_states[self.layer_idx],
                hidden_states[..., 0],
                discrete_time_step[..., 0].detach(),
                A.detach(),
                B[:, 0].detach(),
                C[:, 0].detach(),
                self.D,
                gate[..., 0],
                time_proj_bias,
                dt_softplus=True,
            ).unsqueeze(-1)
        else:
            scan_outputs, ssm_state = selective_scan_fn(
                hidden_states,
                discrete_time_step.detach(),
                A.detach(),
                B.transpose(1, 2).detach(),
                C.transpose(1, 2).detach(),
                self.D.float(),
                gate,
                time_proj_bias,
                delta_softplus=True,
                return_last_state=True,
            )
            if ssm_state is not None and cache_params is not None:
                cache_params.ssm_states[self.layer_idx].copy_(ssm_state)

        # 4. Final linear projection
        scan_outputs = (scan_outputs / 2.) + (scan_outputs / 2.).detach()
        contextualized_states = self.out_proj(scan_outputs.transpose(1, 2))
        return contextualized_states

    # fmt: off
    def slow_forward(
            self,
            input_states,
            cache_params: Optional[MambaCache] = None
    ):
        batch_size, seq_len, _ = input_states.shape
        dtype = input_states.dtype

        # 1. Gated MLP's linear projection
        projected_states = self.in_proj(input_states).transpose(1, 2)
        hidden_states, gate = projected_states.chunk(2, dim=1)

        # 2. Convolution sequence transformation
        if cache_params is not None:
            ssm_state = cache_params.ssm_states[self.layer_idx].clone()
            if cache_params.seqlen_offset > 0:
                conv_state = cache_params.conv_states[self.layer_idx]
                conv_state = torch.roll(conv_state, shifts=-1, dims=-1)
                conv_state[:, :, -1] = hidden_states[:, :, 0]
                cache_params.conv_states[self.layer_idx].copy_(conv_state)
                hidden_states = torch.sum(conv_state * self.conv1d.weight[:, 0, :], dim=-1)
                if self.use_conv_bias:
                    hidden_states += self.conv1d.bias
                hidden_states = self.act(hidden_states).to(dtype).unsqueeze(-1)
            else:
                conv_state = nn.functional.pad(
                    hidden_states,
                    (self.conv_kernel_size - hidden_states.shape[-1], 0)
                )
                cache_params.conv_states[self.layer_idx].copy_(conv_state)
                hidden_states = self.act(self.conv1d(hidden_states)[..., :seq_len])
        else:
            ssm_state = torch.zeros(
                (batch_size, self.intermediate_size, self.ssm_state_size),
                device=hidden_states.device, dtype=dtype
            )
            hidden_states = self.act(self.conv1d(hidden_states)[..., :seq_len])

        # 3. State Space Model sequence transformation
        # 3.a. Selection:  [batch, seq_len, self.time_step_rank + self.ssm_state_size * 2]
        ssm_parameters = self.x_proj(hidden_states.transpose(1, 2))
        time_step, B, C = torch.split(
            ssm_parameters, [self.time_step_rank, self.ssm_state_size, self.ssm_state_size], dim=-1
        )

        discrete_time_step = self.dt_proj(time_step)
        discrete_time_step = nn.functional.softplus(discrete_time_step).transpose(1, 2)

        # 3.b. Discretization: B and C to [batch, seq_len, intermediate_size, ssm_state_size] (SRAM)
        A = -torch.exp(self.A_log.float())
        discrete_A = torch.exp(A[None, :, None, :] * discrete_time_step[:, :, :, None])
        discrete_B = discrete_time_step[:, :, :, None] * B[:, None, :, :].float()

        # Detach parameters discrete_A, discrete_B, and C.
        discrete_A = discrete_A.detach()
        discrete_B = discrete_B.detach()
        C = C.detach()
        deltaB_u = discrete_B * hidden_states[:, :, :, None].float()

        # 3.c perform the recurrence y ← SSM(A, B, C)(x)
        scan_outputs = []
        for i in range(seq_len):
            ssm_state = discrete_A[:, :, i, :] * ssm_state + deltaB_u[:, :, i, :]
            scan_output = torch.matmul(ssm_state.to(dtype), C[:, i, :].unsqueeze(-1))
            scan_outputs.append(scan_output[:, :, 0])
        scan_output = torch.stack(scan_outputs, dim=-1)
        scan_output = scan_output + (hidden_states * self.D[None, :, None])
        scan_output = (scan_output * self.act(gate))

        if cache_params is not None:
            cache_params.ssm_states[self.layer_idx].copy_(ssm_state)

        # 4. Final linear projection
        scan_output = (scan_output / 2.) + (scan_output / 2.).detach()  # Half-relevance propagation.
        contextualized_states = self.out_proj(scan_output.transpose(1, 2))
        return contextualized_states

    # fmt: on
    def forward(self, hidden_states, cache_params: Optional[MambaCache] = None):
        if self.is_fast_forward_available and "cuda" in self.x_proj.weight.device.type:
            return self.cuda_kernels_forward(hidden_states, cache_params)
        return self.slow_forward(hidden_states, cache_params)


class ModifiedMambaRMSNorm(nn.Module):
    def __init__(
            self,
            norm
    ):
        """
        A wrapper to make Mamba norm layer explainable.
        -------------------

        :param norm: Mamba norm layer
        """
        super(ModifiedMambaRMSNorm, self).__init__()
        self.weight = norm.weight
        self.variance_epsilon = norm.variance_epsilon

    def forward(
            self,
            hidden_states
    ):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon).detach()
        return self.weight * hidden_states.to(input_dtype)


class ModifiedMambaBlock(nn.Module):
    def __init__(
            self,
            block,
            is_fast_forward_available
    ):
        """
        A wrapper to make Mamba block explainable.
        -------------------

        :param block: Mamba block
        :param is_fast_forward_available: if True, fast CUDA kernels will be used
        """
        super(ModifiedMambaBlock, self).__init__()
        self.config = block.config
        self.layer_idx = block.layer_idx
        self.residual_in_fp32 = block.residual_in_fp32
        self.norm = ModifiedMambaRMSNorm(block.norm)
        self.mixer = ModifiedMambaMixer(block.mixer, is_fast_forward_available)

    def forward(
            self,
            hidden_states,
            cache_params=None
    ):
        residual = hidden_states
        hidden_states = self.norm(hidden_states.to(dtype=self.norm.weight.dtype))
        if self.residual_in_fp32:
            residual = residual.to(torch.float32)

        hidden_states = self.mixer(hidden_states, cache_params=cache_params)
        hidden_states = residual + hidden_states
        return hidden_states


class ModifiedMambaModel(nn.Module):
    def __init__(
            self,
            model,
            is_fast_forward_available=True
    ):
        """
        A wrapper to make Mamba model explainable.
        -------------------

        :param model: Mamba model
        :param is_fast_forward_available: if True, fast CUDA kernels will be used
        """
        super(ModifiedMambaModel, self).__init__()

        self.embeddings = model.embeddings
        self.layers = nn.ModuleList([ModifiedMambaBlock(block, is_fast_forward_available) for block in model.layers])

        self.gradient_checkpointing = model.gradient_checkpointing
        self.norm_f = ModifiedMambaRMSNorm(model.norm_f)
        self.config = model.config

    def forward(
            self,
            input_ids=None,
            inputs_embeds=None,
            cache_params=None,
            use_cache=None,
            output_hidden_states=None,
            return_dict=None,
            **kwargs,  # `attention_mask` is passed by the tokenizer and we don't want it
    ):
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else (self.config.use_cache if not self.training else False)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):  # ^ is python for xor
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)

        if self.gradient_checkpointing and self.training and use_cache:
            use_cache = False

        if cache_params is None and use_cache:
            cache_params = MambaCache(
                self.config, inputs_embeds.size(0), device=inputs_embeds.device, dtype=inputs_embeds.dtype
            )

        hidden_states = inputs_embeds
        all_hidden_states = () if output_hidden_states else None
        for mixer_block in self.layers:
            if self.gradient_checkpointing and self.training:
                hidden_states = self._gradient_checkpointing_func(mixer_block.__call__, hidden_states, cache_params)
            else:
                hidden_states = mixer_block(hidden_states, cache_params=cache_params)

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

        if use_cache:
            cache_params.seqlen_offset += inputs_embeds.shape[1]

        hidden_states = self.norm_f(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, cache_params, all_hidden_states] if v is not None)

        return {'last_hidden_state': hidden_states,
                'cache_params': cache_params if use_cache else None,
                'hidden_states': all_hidden_states
                }


class ModifiedMambaForCausalLM(nn.Module):
    def __init__(
            self,
            model,
            is_fast_forward_available=True
    ):
        super(ModifiedMambaForCausalLM, self).__init__()

        self.backbone = ModifiedMambaModel(model.backbone, is_fast_forward_available)
        self.lm_head = model.lm_head
        self.config = model.config

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            cache_params: Optional[MambaCache] = None,
            labels: Optional[torch.LongTensor] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            use_cache: Optional[bool] = None
    ):
        mamba_outputs = self.backbone(
            input_ids,
            cache_params=cache_params,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            use_cache=use_cache,
        )
        hidden_states = mamba_outputs['last_hidden_state']

        logits = self.lm_head(hidden_states.to(self.lm_head.weight.dtype)).float()
        return logits
