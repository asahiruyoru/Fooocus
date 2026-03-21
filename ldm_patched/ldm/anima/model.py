"""
Anima Preview2 DiT model - ported from ComfyUI
Based on MiniTrainDIT (cosmos predict2) + LLM Adapter for Qwen text encoder integration.
"""
import math
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
from typing import Optional, Tuple

from ldm_patched.ldm.modules.attention import optimized_attention
import ldm_patched.modules.model_management


# ---- Rotary Position Embedding for LLM Adapter ----
def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_adapter(x, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return (x * cos) + (rotate_half(x) * sin)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim):
        super().__init__()
        self.rope_theta = 10000
        inv_freq = 1.0 / (self.rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.int64).to(dtype=torch.float) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()
        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


# ---- Adapter Attention ----
class AdapterAttention(nn.Module):
    def __init__(self, query_dim, context_dim, n_heads, head_dim, device=None, dtype=None, operations=None):
        super().__init__()
        inner_dim = head_dim * n_heads
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.q_proj = operations.Linear(query_dim, inner_dim, bias=False, device=device, dtype=dtype)
        self.q_norm = operations.RMSNorm(head_dim, eps=1e-6, device=device, dtype=dtype)
        self.k_proj = operations.Linear(context_dim, inner_dim, bias=False, device=device, dtype=dtype)
        self.k_norm = operations.RMSNorm(head_dim, eps=1e-6, device=device, dtype=dtype)
        self.v_proj = operations.Linear(context_dim, inner_dim, bias=False, device=device, dtype=dtype)
        self.o_proj = operations.Linear(inner_dim, query_dim, bias=False, device=device, dtype=dtype)

    def forward(self, x, mask=None, context=None, position_embeddings=None, position_embeddings_context=None):
        context = x if context is None else context
        input_shape = x.shape[:-1]
        q_shape = (*input_shape, self.n_heads, self.head_dim)
        context_shape = context.shape[:-1]
        kv_shape = (*context_shape, self.n_heads, self.head_dim)

        query_states = self.q_norm(self.q_proj(x).view(q_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(context).view(kv_shape)).transpose(1, 2)
        value_states = self.v_proj(context).view(kv_shape).transpose(1, 2)

        if position_embeddings is not None:
            assert position_embeddings_context is not None
            cos, sin = position_embeddings
            query_states = apply_rotary_pos_emb_adapter(query_states, cos, sin)
            cos, sin = position_embeddings_context
            key_states = apply_rotary_pos_emb_adapter(key_states, cos, sin)

        attn_output = F.scaled_dot_product_attention(query_states, key_states, value_states, attn_mask=mask)
        attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
        return self.o_proj(attn_output)


# ---- Adapter Transformer Block ----
class TransformerBlock(nn.Module):
    def __init__(self, source_dim, model_dim, num_heads=16, mlp_ratio=4.0, use_self_attn=False, layer_norm=False, device=None, dtype=None, operations=None):
        super().__init__()
        self.use_self_attn = use_self_attn
        if self.use_self_attn:
            self.norm_self_attn = operations.LayerNorm(model_dim, device=device, dtype=dtype) if layer_norm else operations.RMSNorm(model_dim, eps=1e-6, device=device, dtype=dtype)
            self.self_attn = AdapterAttention(model_dim, model_dim, num_heads, model_dim // num_heads, device=device, dtype=dtype, operations=operations)
        self.norm_cross_attn = operations.LayerNorm(model_dim, device=device, dtype=dtype) if layer_norm else operations.RMSNorm(model_dim, eps=1e-6, device=device, dtype=dtype)
        self.cross_attn = AdapterAttention(model_dim, source_dim, num_heads, model_dim // num_heads, device=device, dtype=dtype, operations=operations)
        self.norm_mlp = operations.LayerNorm(model_dim, device=device, dtype=dtype) if layer_norm else operations.RMSNorm(model_dim, eps=1e-6, device=device, dtype=dtype)
        self.mlp = nn.Sequential(
            operations.Linear(model_dim, int(model_dim * mlp_ratio), device=device, dtype=dtype),
            nn.GELU(),
            operations.Linear(int(model_dim * mlp_ratio), model_dim, device=device, dtype=dtype)
        )

    def forward(self, x, context, target_attention_mask=None, source_attention_mask=None, position_embeddings=None, position_embeddings_context=None):
        if self.use_self_attn:
            normed = self.norm_self_attn(x)
            attn_out = self.self_attn(normed, mask=target_attention_mask, position_embeddings=position_embeddings, position_embeddings_context=position_embeddings)
            x = x + attn_out
        normed = self.norm_cross_attn(x)
        attn_out = self.cross_attn(normed, mask=source_attention_mask, context=context, position_embeddings=position_embeddings, position_embeddings_context=position_embeddings_context)
        x = x + attn_out
        x = x + self.mlp(self.norm_mlp(x))
        return x


# ---- LLM Adapter: Converts Qwen3 embeddings to T5-compatible cross-attention space ----
class LLMAdapter(nn.Module):
    def __init__(self, source_dim=1024, target_dim=1024, model_dim=1024, num_layers=6, num_heads=16, use_self_attn=True, layer_norm=False, device=None, dtype=None, operations=None):
        super().__init__()
        self.embed = operations.Embedding(32128, target_dim, device=device, dtype=dtype)
        if model_dim != target_dim:
            self.in_proj = operations.Linear(target_dim, model_dim, device=device, dtype=dtype)
        else:
            self.in_proj = nn.Identity()
        self.rotary_emb = RotaryEmbedding(model_dim // num_heads)
        self.blocks = nn.ModuleList([
            TransformerBlock(source_dim, model_dim, num_heads=num_heads, use_self_attn=use_self_attn, layer_norm=layer_norm, device=device, dtype=dtype, operations=operations) for _ in range(num_layers)
        ])
        self.out_proj = operations.Linear(model_dim, target_dim, device=device, dtype=dtype)
        self.norm = operations.RMSNorm(target_dim, eps=1e-6, device=device, dtype=dtype)

    def forward(self, source_hidden_states, target_input_ids, target_attention_mask=None, source_attention_mask=None):
        if target_attention_mask is not None:
            target_attention_mask = target_attention_mask.to(torch.bool)
            if target_attention_mask.ndim == 2:
                target_attention_mask = target_attention_mask.unsqueeze(1).unsqueeze(1)
        if source_attention_mask is not None:
            source_attention_mask = source_attention_mask.to(torch.bool)
            if source_attention_mask.ndim == 2:
                source_attention_mask = source_attention_mask.unsqueeze(1).unsqueeze(1)

        context = source_hidden_states
        x = self.in_proj(self.embed(target_input_ids, out_dtype=context.dtype))
        position_ids = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
        position_ids_context = torch.arange(context.shape[1], device=x.device).unsqueeze(0)
        position_embeddings = self.rotary_emb(x, position_ids)
        position_embeddings_context = self.rotary_emb(x, position_ids_context)
        for block in self.blocks:
            x = block(x, context, target_attention_mask=target_attention_mask, source_attention_mask=source_attention_mask, position_embeddings=position_embeddings, position_embeddings_context=position_embeddings_context)
        return self.norm(self.out_proj(x))


# ---- Cosmos Predict2 DiT components ----
def apply_rotary_pos_emb_dit(t, freqs):
    t_ = t.reshape(*t.shape[:-1], 2, -1).movedim(-2, -1).unsqueeze(-2).float()
    t_out = freqs[..., 0] * t_[..., 0] + freqs[..., 1] * t_[..., 1]
    t_out = t_out.movedim(-1, -2).reshape(*t.shape).type_as(t)
    return t_out


def torch_attention_op(q, k, v, transformer_options={}):
    in_q_shape = q.shape
    in_k_shape = k.shape
    q_r = rearrange(q, "b ... h k -> b h ... k").view(in_q_shape[0], in_q_shape[-2], -1, in_q_shape[-1])
    k_r = rearrange(k, "b ... h v -> b h ... v").view(in_k_shape[0], in_k_shape[-2], -1, in_k_shape[-1])
    v_r = rearrange(v, "b ... h v -> b h ... v").view(in_k_shape[0], in_k_shape[-2], -1, in_k_shape[-1])
    return optimized_attention(q_r, k_r, v_r, in_q_shape[-2], skip_reshape=True)


class GPT2FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, device=None, dtype=None, operations=None):
        super().__init__()
        self.activation = nn.GELU()
        self.layer1 = operations.Linear(d_model, d_ff, bias=False, device=device, dtype=dtype)
        self.layer2 = operations.Linear(d_ff, d_model, bias=False, device=device, dtype=dtype)

    def forward(self, x):
        return self.layer2(self.activation(self.layer1(x)))


class DITAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, n_heads=8, head_dim=64, dropout=0.0, device=None, dtype=None, operations=None):
        super().__init__()
        self.is_selfattn = context_dim is None
        context_dim = query_dim if context_dim is None else context_dim
        inner_dim = head_dim * n_heads
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.q_proj = operations.Linear(query_dim, inner_dim, bias=False, device=device, dtype=dtype)
        self.q_norm = operations.RMSNorm(head_dim, eps=1e-6, device=device, dtype=dtype)
        self.k_proj = operations.Linear(context_dim, inner_dim, bias=False, device=device, dtype=dtype)
        self.k_norm = operations.RMSNorm(head_dim, eps=1e-6, device=device, dtype=dtype)
        self.v_proj = operations.Linear(context_dim, inner_dim, bias=False, device=device, dtype=dtype)
        self.v_norm = nn.Identity()
        self.output_proj = operations.Linear(inner_dim, query_dim, bias=False, device=device, dtype=dtype)

    def compute_qkv(self, x, context=None, rope_emb=None):
        q = self.q_proj(x)
        context = x if context is None else context
        k = self.k_proj(context)
        v = self.v_proj(context)
        q, k, v = map(lambda t: rearrange(t, "b ... (h d) -> b ... h d", h=self.n_heads, d=self.head_dim), (q, k, v))
        q = self.q_norm(q)
        k = self.k_norm(k)
        v = self.v_norm(v)
        if self.is_selfattn and rope_emb is not None:
            q = apply_rotary_pos_emb_dit(q, rope_emb)
            k = apply_rotary_pos_emb_dit(k, rope_emb)
        return q, k, v

    def forward(self, x, context=None, rope_emb=None, transformer_options={}):
        q, k, v = self.compute_qkv(x, context, rope_emb=rope_emb)
        result = torch_attention_op(q, k, v, transformer_options=transformer_options)
        return self.output_proj(result)


class Timesteps(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.num_channels = num_channels

    def forward(self, timesteps_B_T):
        assert timesteps_B_T.ndim == 2
        timesteps = timesteps_B_T.flatten().float()
        half_dim = self.num_channels // 2
        exponent = -math.log(10000) * torch.arange(half_dim, dtype=torch.float32, device=timesteps.device)
        exponent = exponent / (half_dim - 0.0)
        emb = torch.exp(exponent)
        emb = timesteps[:, None].float() * emb[None, :]
        sin_emb = torch.sin(emb)
        cos_emb = torch.cos(emb)
        emb = torch.cat([cos_emb, sin_emb], dim=-1)
        return rearrange(emb, "(b t) d -> b t d", b=timesteps_B_T.shape[0], t=timesteps_B_T.shape[1])


class TimestepEmbedding(nn.Module):
    def __init__(self, in_features, out_features, use_adaln_lora=False, device=None, dtype=None, operations=None):
        super().__init__()
        self.linear_1 = operations.Linear(in_features, out_features, bias=not use_adaln_lora, device=device, dtype=dtype)
        self.activation = nn.SiLU()
        self.use_adaln_lora = use_adaln_lora
        if use_adaln_lora:
            self.linear_2 = operations.Linear(out_features, 3 * out_features, bias=False, device=device, dtype=dtype)
        else:
            self.linear_2 = operations.Linear(out_features, out_features, bias=False, device=device, dtype=dtype)

    def forward(self, sample):
        emb = self.linear_2(self.activation(self.linear_1(sample)))
        if self.use_adaln_lora:
            return sample, emb
        else:
            return emb, None


class PatchEmbed(nn.Module):
    def __init__(self, spatial_patch_size, temporal_patch_size, in_channels=3, out_channels=768, device=None, dtype=None, operations=None):
        super().__init__()
        self.spatial_patch_size = spatial_patch_size
        self.temporal_patch_size = temporal_patch_size
        self.proj = nn.Sequential(
            Rearrange("b c (t r) (h m) (w n) -> b t h w (c r m n)", r=temporal_patch_size, m=spatial_patch_size, n=spatial_patch_size),
            operations.Linear(in_channels * spatial_patch_size * spatial_patch_size * temporal_patch_size, out_channels, bias=False, device=device, dtype=dtype),
        )

    def forward(self, x):
        assert x.dim() == 5
        return self.proj(x)


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, spatial_patch_size, temporal_patch_size, out_channels, use_adaln_lora=False, adaln_lora_dim=256, device=None, dtype=None, operations=None):
        super().__init__()
        self.layer_norm = operations.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, device=device, dtype=dtype)
        self.linear = operations.Linear(hidden_size, spatial_patch_size * spatial_patch_size * temporal_patch_size * out_channels, bias=False, device=device, dtype=dtype)
        self.hidden_size = hidden_size
        self.n_adaln_chunks = 2
        self.use_adaln_lora = use_adaln_lora
        if use_adaln_lora:
            self.adaln_modulation = nn.Sequential(
                nn.SiLU(),
                operations.Linear(hidden_size, adaln_lora_dim, bias=False, device=device, dtype=dtype),
                operations.Linear(adaln_lora_dim, self.n_adaln_chunks * hidden_size, bias=False, device=device, dtype=dtype),
            )
        else:
            self.adaln_modulation = nn.Sequential(
                nn.SiLU(), operations.Linear(hidden_size, self.n_adaln_chunks * hidden_size, bias=False, device=device, dtype=dtype)
            )

    def forward(self, x_B_T_H_W_D, emb_B_T_D, adaln_lora_B_T_3D=None):
        if self.use_adaln_lora:
            assert adaln_lora_B_T_3D is not None
            shift, scale = (self.adaln_modulation(emb_B_T_D) + adaln_lora_B_T_3D[:, :, :2 * self.hidden_size]).chunk(2, dim=-1)
        else:
            shift, scale = self.adaln_modulation(emb_B_T_D).chunk(2, dim=-1)
        shift = rearrange(shift, "b t d -> b t 1 1 d")
        scale = rearrange(scale, "b t d -> b t 1 1 d")
        x = self.layer_norm(x_B_T_H_W_D) * (1 + scale) + shift
        return self.linear(x)


class Block(nn.Module):
    def __init__(self, x_dim, context_dim, num_heads, mlp_ratio=4.0, use_adaln_lora=False, adaln_lora_dim=256, device=None, dtype=None, operations=None):
        super().__init__()
        self.x_dim = x_dim
        self.layer_norm_self_attn = operations.LayerNorm(x_dim, elementwise_affine=False, eps=1e-6, device=device, dtype=dtype)
        self.self_attn = DITAttention(x_dim, None, num_heads, x_dim // num_heads, device=device, dtype=dtype, operations=operations)
        self.layer_norm_cross_attn = operations.LayerNorm(x_dim, elementwise_affine=False, eps=1e-6, device=device, dtype=dtype)
        self.cross_attn = DITAttention(x_dim, context_dim, num_heads, x_dim // num_heads, device=device, dtype=dtype, operations=operations)
        self.layer_norm_mlp = operations.LayerNorm(x_dim, elementwise_affine=False, eps=1e-6, device=device, dtype=dtype)
        self.mlp = GPT2FeedForward(x_dim, int(x_dim * mlp_ratio), device=device, dtype=dtype, operations=operations)
        self.use_adaln_lora = use_adaln_lora
        if self.use_adaln_lora:
            self.adaln_modulation_self_attn = nn.Sequential(nn.SiLU(), operations.Linear(x_dim, adaln_lora_dim, bias=False, device=device, dtype=dtype), operations.Linear(adaln_lora_dim, 3 * x_dim, bias=False, device=device, dtype=dtype))
            self.adaln_modulation_cross_attn = nn.Sequential(nn.SiLU(), operations.Linear(x_dim, adaln_lora_dim, bias=False, device=device, dtype=dtype), operations.Linear(adaln_lora_dim, 3 * x_dim, bias=False, device=device, dtype=dtype))
            self.adaln_modulation_mlp = nn.Sequential(nn.SiLU(), operations.Linear(x_dim, adaln_lora_dim, bias=False, device=device, dtype=dtype), operations.Linear(adaln_lora_dim, 3 * x_dim, bias=False, device=device, dtype=dtype))
        else:
            self.adaln_modulation_self_attn = nn.Sequential(nn.SiLU(), operations.Linear(x_dim, 3 * x_dim, bias=False, device=device, dtype=dtype))
            self.adaln_modulation_cross_attn = nn.Sequential(nn.SiLU(), operations.Linear(x_dim, 3 * x_dim, bias=False, device=device, dtype=dtype))
            self.adaln_modulation_mlp = nn.Sequential(nn.SiLU(), operations.Linear(x_dim, 3 * x_dim, bias=False, device=device, dtype=dtype))

    def forward(self, x_B_T_H_W_D, emb_B_T_D, crossattn_emb, rope_emb_L_1_1_D=None, adaln_lora_B_T_3D=None, extra_per_block_pos_emb=None, transformer_options={}):
        if extra_per_block_pos_emb is not None:
            x_B_T_H_W_D = x_B_T_H_W_D + extra_per_block_pos_emb

        residual_dtype = x_B_T_H_W_D.dtype
        compute_dtype = emb_B_T_D.dtype

        def _get_mod(mod_fn, emb, adaln_lora=None):
            if self.use_adaln_lora and adaln_lora is not None:
                return (mod_fn(emb) + adaln_lora).chunk(3, dim=-1)
            return mod_fn(emb).chunk(3, dim=-1)

        shift_sa, scale_sa, gate_sa = _get_mod(self.adaln_modulation_self_attn, emb_B_T_D, adaln_lora_B_T_3D)
        shift_ca, scale_ca, gate_ca = _get_mod(self.adaln_modulation_cross_attn, emb_B_T_D, adaln_lora_B_T_3D)
        shift_mlp, scale_mlp, gate_mlp = _get_mod(self.adaln_modulation_mlp, emb_B_T_D, adaln_lora_B_T_3D)

        def _expand(t):
            return rearrange(t, "b t d -> b t 1 1 d")

        B, T, H, W, D = x_B_T_H_W_D.shape

        def _adaln(x, norm, scale, shift):
            return norm(x) * (1 + _expand(scale)) + _expand(shift)

        # Self-attention
        normalized = _adaln(x_B_T_H_W_D, self.layer_norm_self_attn, scale_sa, shift_sa)
        result = rearrange(self.self_attn(rearrange(normalized.to(compute_dtype), "b t h w d -> b (t h w) d"), None, rope_emb=rope_emb_L_1_1_D, transformer_options=transformer_options), "b (t h w) d -> b t h w d", t=T, h=H, w=W)
        x_B_T_H_W_D = x_B_T_H_W_D + _expand(gate_sa).to(residual_dtype) * result.to(residual_dtype)

        # Cross-attention
        normalized = _adaln(x_B_T_H_W_D, self.layer_norm_cross_attn, scale_ca, shift_ca)
        result = rearrange(self.cross_attn(rearrange(normalized.to(compute_dtype), "b t h w d -> b (t h w) d"), crossattn_emb, rope_emb=rope_emb_L_1_1_D, transformer_options=transformer_options), "b (t h w) d -> b t h w d", t=T, h=H, w=W)
        x_B_T_H_W_D = result.to(residual_dtype) * _expand(gate_ca).to(residual_dtype) + x_B_T_H_W_D

        # MLP
        normalized = _adaln(x_B_T_H_W_D, self.layer_norm_mlp, scale_mlp, shift_mlp)
        result = self.mlp(normalized.to(compute_dtype))
        x_B_T_H_W_D = x_B_T_H_W_D + _expand(gate_mlp).to(residual_dtype) * result.to(residual_dtype)
        return x_B_T_H_W_D


# ---- VideoRoPE Position Embedding (simplified for image-only mode) ----
class VideoRopePosition3DEmb(nn.Module):
    def __init__(self, model_channels, len_h, len_w, len_t, head_dim, is_learnable=False, interpolation="crop",
                 h_extrapolation_ratio=1.0, w_extrapolation_ratio=1.0, t_extrapolation_ratio=1.0,
                 max_fps=30, min_fps=1, enable_fps_modulation=True, **kwargs):
        super().__init__()
        self.head_dim = head_dim
        self.len_h = len_h
        self.len_w = len_w
        self.len_t = len_t
        self.h_extrapolation_ratio = h_extrapolation_ratio
        self.w_extrapolation_ratio = w_extrapolation_ratio
        self.t_extrapolation_ratio = t_extrapolation_ratio

    def _get_freqs(self, seq_len, dim, theta=10000.0, device=None, dtype=None):
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32)[: (dim // 2)] / dim))
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(t, freqs)
        freqs_cos = torch.cos(freqs)
        freqs_sin = torch.sin(freqs)
        return torch.stack([freqs_cos, -freqs_sin, freqs_sin, freqs_cos], dim=-1).view(*freqs.shape, 2, 2)

    def forward(self, x_B_T_H_W_D, fps=None, device=None, dtype=None):
        if device is None:
            device = x_B_T_H_W_D.device
        if dtype is None:
            dtype = x_B_T_H_W_D.dtype

        B, T, H, W, D = x_B_T_H_W_D.shape
        half_dim = self.head_dim // 2
        dim_t = half_dim // 3
        dim_h = half_dim // 3
        dim_w = half_dim - dim_t - dim_h

        # Generate RoPE frequencies for each axis
        freqs_t = self._get_freqs(T, dim_t * 2, theta=10000.0 * self.t_extrapolation_ratio, device=device)
        freqs_h = self._get_freqs(H, dim_h * 2, theta=10000.0 * self.h_extrapolation_ratio, device=device)
        freqs_w = self._get_freqs(W, dim_w * 2, theta=10000.0 * self.w_extrapolation_ratio, device=device)

        # Expand and combine: T×H×W grid
        freqs_t = freqs_t.view(T, 1, 1, dim_t, 2, 2).expand(-1, H, W, -1, -1, -1)
        freqs_h = freqs_h.view(1, H, 1, dim_h, 2, 2).expand(T, -1, W, -1, -1, -1)
        freqs_w = freqs_w.view(1, 1, W, dim_w, 2, 2).expand(T, H, -1, -1, -1, -1)

        freqs = torch.cat([freqs_t, freqs_h, freqs_w], dim=3)  # T, H, W, half_dim, 2, 2
        freqs = freqs.reshape(T * H * W, half_dim, 2, 2).unsqueeze(0).unsqueeze(0)  # 1, 1, L, half_dim, 2, 2
        return freqs.to(dtype)


class LearnablePosEmbAxis(nn.Module):
    def __init__(self, model_channels, len_h, len_w, len_t, head_dim,
                 h_extrapolation_ratio=1.0, w_extrapolation_ratio=1.0, t_extrapolation_ratio=1.0,
                 device=None, dtype=None, **kwargs):
        super().__init__()
        self.pos_emb_h = nn.Parameter(torch.randn(1, 1, len_h, 1, model_channels, device=device, dtype=dtype) * 0.02)
        self.pos_emb_w = nn.Parameter(torch.randn(1, 1, 1, len_w, model_channels, device=device, dtype=dtype) * 0.02)
        self.pos_emb_t = nn.Parameter(torch.randn(1, len_t, 1, 1, model_channels, device=device, dtype=dtype) * 0.02)

    def forward(self, x_B_T_H_W_D, fps=None, device=None, dtype=None):
        B, T, H, W, D = x_B_T_H_W_D.shape
        pos_h = self.pos_emb_h[:, :, :H]
        pos_w = self.pos_emb_w[:, :, :, :W]
        pos_t = self.pos_emb_t[:, :T]
        return (pos_h + pos_w + pos_t).to(x_B_T_H_W_D.dtype)


# ---- Padding utility ----
def pad_to_patch_size(x, patch_size):
    # x: B, C, T, H, W
    t_pad = (patch_size[0] - x.shape[2] % patch_size[0]) % patch_size[0]
    h_pad = (patch_size[1] - x.shape[3] % patch_size[1]) % patch_size[1]
    w_pad = (patch_size[2] - x.shape[4] % patch_size[2]) % patch_size[2]
    if t_pad > 0 or h_pad > 0 or w_pad > 0:
        x = F.pad(x, (0, w_pad, 0, h_pad, 0, t_pad))
    return x


# ---- MiniTrainDIT: The core DiT backbone ----
class MiniTrainDIT(nn.Module):
    def __init__(self, max_img_h=240, max_img_w=240, max_frames=128,
                 in_channels=16, out_channels=16, patch_spatial=2, patch_temporal=1,
                 concat_padding_mask=True, model_channels=2048, num_blocks=28, num_heads=16,
                 mlp_ratio=4.0, crossattn_emb_channels=1024, pos_emb_cls="rope3d",
                 pos_emb_learnable=True, pos_emb_interpolation="crop",
                 min_fps=1, max_fps=30, use_adaln_lora=True, adaln_lora_dim=256,
                 rope_h_extrapolation_ratio=1.0, rope_w_extrapolation_ratio=1.0,
                 rope_t_extrapolation_ratio=1.0, extra_per_block_abs_pos_emb=False,
                 extra_h_extrapolation_ratio=1.0, extra_w_extrapolation_ratio=1.0,
                 extra_t_extrapolation_ratio=1.0, rope_enable_fps_modulation=True,
                 image_model=None, device=None, dtype=None, operations=None):
        super().__init__()
        self.dtype = dtype
        self.max_img_h = max_img_h
        self.max_img_w = max_img_w
        self.max_frames = max_frames
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.patch_spatial = patch_spatial
        self.patch_temporal = patch_temporal
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.model_channels = model_channels
        self.concat_padding_mask = concat_padding_mask
        self.pos_emb_cls = pos_emb_cls
        self.pos_emb_learnable = pos_emb_learnable
        self.pos_emb_interpolation = pos_emb_interpolation
        self.rope_h_extrapolation_ratio = rope_h_extrapolation_ratio
        self.rope_w_extrapolation_ratio = rope_w_extrapolation_ratio
        self.rope_t_extrapolation_ratio = rope_t_extrapolation_ratio
        self.extra_per_block_abs_pos_emb = extra_per_block_abs_pos_emb
        self.extra_h_extrapolation_ratio = extra_h_extrapolation_ratio
        self.extra_w_extrapolation_ratio = extra_w_extrapolation_ratio
        self.extra_t_extrapolation_ratio = extra_t_extrapolation_ratio
        self.min_fps = min_fps
        self.max_fps = max_fps
        self.rope_enable_fps_modulation = rope_enable_fps_modulation

        self.use_adaln_lora = use_adaln_lora
        self.adaln_lora_dim = adaln_lora_dim

        # Position embedding
        self._build_pos_embed(device=device, dtype=dtype)

        # Timestep Embedding
        self.t_embedder = nn.Sequential(
            Timesteps(model_channels),
            TimestepEmbedding(model_channels, model_channels, use_adaln_lora=use_adaln_lora, device=device, dtype=dtype, operations=operations),
        )

        in_ch = in_channels + 1 if concat_padding_mask else in_channels
        self.x_embedder = PatchEmbed(
            spatial_patch_size=patch_spatial, temporal_patch_size=patch_temporal,
            in_channels=in_ch, out_channels=model_channels,
            device=device, dtype=dtype, operations=operations,
        )

        self.blocks = nn.ModuleList([
            Block(x_dim=model_channels, context_dim=crossattn_emb_channels, num_heads=num_heads,
                  mlp_ratio=mlp_ratio, use_adaln_lora=use_adaln_lora, adaln_lora_dim=adaln_lora_dim,
                  device=device, dtype=dtype, operations=operations)
            for _ in range(num_blocks)
        ])

        self.final_layer = FinalLayer(
            hidden_size=model_channels, spatial_patch_size=patch_spatial,
            temporal_patch_size=patch_temporal, out_channels=out_channels,
            use_adaln_lora=use_adaln_lora, adaln_lora_dim=adaln_lora_dim,
            device=device, dtype=dtype, operations=operations,
        )
        self.t_embedding_norm = operations.RMSNorm(model_channels, eps=1e-6, device=device, dtype=dtype)

    def _build_pos_embed(self, device=None, dtype=None):
        kwargs = dict(
            model_channels=self.model_channels,
            len_h=self.max_img_h // self.patch_spatial,
            len_w=self.max_img_w // self.patch_spatial,
            len_t=self.max_frames // self.patch_temporal,
            max_fps=self.max_fps, min_fps=self.min_fps,
            is_learnable=self.pos_emb_learnable,
            interpolation=self.pos_emb_interpolation,
            head_dim=self.model_channels // self.num_heads,
            h_extrapolation_ratio=self.rope_h_extrapolation_ratio,
            w_extrapolation_ratio=self.rope_w_extrapolation_ratio,
            t_extrapolation_ratio=self.rope_t_extrapolation_ratio,
            enable_fps_modulation=self.rope_enable_fps_modulation,
            device=device,
        )
        self.pos_embedder = VideoRopePosition3DEmb(**kwargs)
        if self.extra_per_block_abs_pos_emb:
            kwargs["h_extrapolation_ratio"] = self.extra_h_extrapolation_ratio
            kwargs["w_extrapolation_ratio"] = self.extra_w_extrapolation_ratio
            kwargs["t_extrapolation_ratio"] = self.extra_t_extrapolation_ratio
            kwargs["dtype"] = dtype
            self.extra_pos_embedder = LearnablePosEmbAxis(**kwargs)

    def prepare_embedded_sequence(self, x_B_C_T_H_W, fps=None, padding_mask=None):
        from torchvision import transforms
        if self.concat_padding_mask:
            if padding_mask is None:
                padding_mask = torch.zeros(x_B_C_T_H_W.shape[0], 1, x_B_C_T_H_W.shape[3], x_B_C_T_H_W.shape[4], dtype=x_B_C_T_H_W.dtype, device=x_B_C_T_H_W.device)
            else:
                padding_mask = transforms.functional.resize(padding_mask, list(x_B_C_T_H_W.shape[-2:]), interpolation=transforms.InterpolationMode.NEAREST)
            x_B_C_T_H_W = torch.cat([x_B_C_T_H_W, padding_mask.unsqueeze(1).repeat(1, 1, x_B_C_T_H_W.shape[2], 1, 1)], dim=1)
        x_B_T_H_W_D = self.x_embedder(x_B_C_T_H_W)

        extra_pos_emb = None
        if self.extra_per_block_abs_pos_emb:
            extra_pos_emb = self.extra_pos_embedder(x_B_T_H_W_D, fps=fps, device=x_B_C_T_H_W.device, dtype=x_B_C_T_H_W.dtype)

        if "rope" in self.pos_emb_cls.lower():
            return x_B_T_H_W_D, self.pos_embedder(x_B_T_H_W_D, fps=fps, device=x_B_C_T_H_W.device), extra_pos_emb
        x_B_T_H_W_D = x_B_T_H_W_D + self.pos_embedder(x_B_T_H_W_D, device=x_B_C_T_H_W.device)
        return x_B_T_H_W_D, None, extra_pos_emb

    def unpatchify(self, x_B_T_H_W_M):
        return rearrange(x_B_T_H_W_M, "B T H W (p1 p2 t C) -> B C (T t) (H p1) (W p2)",
                         p1=self.patch_spatial, p2=self.patch_spatial, t=self.patch_temporal)

    def forward(self, x, timesteps, context, fps=None, padding_mask=None, **kwargs):
        orig_shape = list(x.shape)
        x = pad_to_patch_size(x, (self.patch_temporal, self.patch_spatial, self.patch_spatial))

        x_B_T_H_W_D, rope_emb, extra_pos_emb = self.prepare_embedded_sequence(x, fps=fps, padding_mask=padding_mask)

        if timesteps.ndim == 1:
            timesteps = timesteps.unsqueeze(1)
        t_emb, adaln_lora = self.t_embedder[1](self.t_embedder[0](timesteps).to(x_B_T_H_W_D.dtype))
        t_emb = self.t_embedding_norm(t_emb)

        block_kwargs = {
            "rope_emb_L_1_1_D": rope_emb.unsqueeze(1).unsqueeze(0) if rope_emb is not None else None,
            "adaln_lora_B_T_3D": adaln_lora,
            "extra_per_block_pos_emb": extra_pos_emb,
            "transformer_options": kwargs.get("transformer_options", {}),
        }

        if x_B_T_H_W_D.dtype == torch.float16:
            x_B_T_H_W_D = x_B_T_H_W_D.float()

        for block in self.blocks:
            x_B_T_H_W_D = block(x_B_T_H_W_D, t_emb, context, **block_kwargs)

        x_out = self.final_layer(x_B_T_H_W_D.to(context.dtype), t_emb, adaln_lora_B_T_3D=adaln_lora)
        x_out = self.unpatchify(x_out)[:, :, :orig_shape[-3], :orig_shape[-2], :orig_shape[-1]]
        return x_out


# ---- Anima Model: MiniTrainDIT + LLMAdapter ----
class Anima(MiniTrainDIT):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.llm_adapter = LLMAdapter(
            device=kwargs.get("device"), dtype=kwargs.get("dtype"),
            operations=kwargs.get("operations"))

    def preprocess_text_embeds(self, text_embeds, text_ids, t5xxl_weights=None):
        if text_ids is not None:
            out = self.llm_adapter(text_embeds, text_ids)
            if t5xxl_weights is not None:
                out = out * t5xxl_weights
            if out.shape[1] < 512:
                out = torch.nn.functional.pad(out, (0, 0, 0, 512 - out.shape[1]))
            return out
        else:
            return text_embeds

    def forward(self, x, timesteps, context, **kwargs):
        t5xxl_ids = kwargs.pop("t5xxl_ids", None)
        if t5xxl_ids is not None:
            context = self.preprocess_text_embeds(context, t5xxl_ids, t5xxl_weights=kwargs.pop("t5xxl_weights", None))
        return super().forward(x, timesteps, context, **kwargs)
