import torch
import ldm_patched.modules.model_management

def cast_bias_weight(s, input):
    bias = None
    non_blocking = ldm_patched.modules.model_management.device_supports_non_blocking(input.device)
    if s.bias is not None:
        bias = s.bias.to(device=input.device, dtype=input.dtype, non_blocking=non_blocking)
    weight = s.weight.to(device=input.device, dtype=input.dtype, non_blocking=non_blocking)
    return weight, bias


class disable_weight_init:
    class Linear(torch.nn.Linear):
        ldm_patched_cast_weights = False
        def reset_parameters(self):
            return None

        def forward_ldm_patched_cast_weights(self, input):
            weight, bias = cast_bias_weight(self, input)
            return torch.nn.functional.linear(input, weight, bias)

        def forward(self, *args, **kwargs):
            if self.ldm_patched_cast_weights:
                return self.forward_ldm_patched_cast_weights(*args, **kwargs)
            else:
                return super().forward(*args, **kwargs)

    class Conv2d(torch.nn.Conv2d):
        ldm_patched_cast_weights = False
        def reset_parameters(self):
            return None

        def forward_ldm_patched_cast_weights(self, input):
            weight, bias = cast_bias_weight(self, input)
            return self._conv_forward(input, weight, bias)

        def forward(self, *args, **kwargs):
            if self.ldm_patched_cast_weights:
                return self.forward_ldm_patched_cast_weights(*args, **kwargs)
            else:
                return super().forward(*args, **kwargs)

    class Conv3d(torch.nn.Conv3d):
        ldm_patched_cast_weights = False
        def reset_parameters(self):
            return None

        def forward_ldm_patched_cast_weights(self, input):
            weight, bias = cast_bias_weight(self, input)
            return self._conv_forward(input, weight, bias)

        def forward(self, *args, **kwargs):
            if self.ldm_patched_cast_weights:
                return self.forward_ldm_patched_cast_weights(*args, **kwargs)
            else:
                return super().forward(*args, **kwargs)

    class GroupNorm(torch.nn.GroupNorm):
        ldm_patched_cast_weights = False
        def reset_parameters(self):
            return None

        def forward_ldm_patched_cast_weights(self, input):
            weight, bias = cast_bias_weight(self, input)
            return torch.nn.functional.group_norm(input, self.num_groups, weight, bias, self.eps)

        def forward(self, *args, **kwargs):
            if self.ldm_patched_cast_weights:
                return self.forward_ldm_patched_cast_weights(*args, **kwargs)
            else:
                return super().forward(*args, **kwargs)


    class LayerNorm(torch.nn.LayerNorm):
        ldm_patched_cast_weights = False
        def reset_parameters(self):
            return None

        def forward_ldm_patched_cast_weights(self, input):
            weight, bias = cast_bias_weight(self, input)
            return torch.nn.functional.layer_norm(input, self.normalized_shape, weight, bias, self.eps)

        def forward(self, *args, **kwargs):
            if self.ldm_patched_cast_weights:
                return self.forward_ldm_patched_cast_weights(*args, **kwargs)
            else:
                return super().forward(*args, **kwargs)

    @classmethod
    def conv_nd(s, dims, *args, **kwargs):
        if dims == 2:
            return s.Conv2d(*args, **kwargs)
        elif dims == 3:
            return s.Conv3d(*args, **kwargs)
        else:
            raise ValueError(f"unsupported dimensions: {dims}")

    RMSNorm = None  # Placeholder, will be set after class definition
    Embedding = None  # Placeholder, will be set after class definition


class manual_cast(disable_weight_init):
    class Linear(disable_weight_init.Linear):
        ldm_patched_cast_weights = True

    class Conv2d(disable_weight_init.Conv2d):
        ldm_patched_cast_weights = True

    class Conv3d(disable_weight_init.Conv3d):
        ldm_patched_cast_weights = True

    class GroupNorm(disable_weight_init.GroupNorm):
        ldm_patched_cast_weights = True

    class LayerNorm(disable_weight_init.LayerNorm):
        ldm_patched_cast_weights = True


# Embedding and RMSNorm for DiT models (Anima, etc.)
class Embedding(torch.nn.Embedding):
    def reset_parameters(self):
        return None

    def forward(self, input, out_dtype=None):
        output = super().forward(input)
        if out_dtype is not None:
            output = output.to(out_dtype)
        return output


class RMSNorm(torch.nn.Module):
    def __init__(self, dim, eps=1e-6, elementwise_affine=True, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.dim = dim
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = torch.nn.Parameter(torch.ones(dim, device=device, dtype=dtype))
        else:
            self.weight = None

    def reset_parameters(self):
        return None

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        if self.weight is not None:
            weight = self.weight.to(device=x.device, dtype=x.dtype)
            output = output * weight
        return output


# Assign to namespaces
disable_weight_init.RMSNorm = RMSNorm
disable_weight_init.Embedding = Embedding
manual_cast.RMSNorm = RMSNorm
manual_cast.Embedding = Embedding
