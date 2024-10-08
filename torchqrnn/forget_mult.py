# torchqrnn/forget_mult.py
import torch
from torch import nn
from torch.autograd import Function
from torch.utils.cpp_extension import load
import os

# Load the custom CUDA kernel
module_path = os.path.dirname(__file__)
forget_mult_cuda = load(
    name='forget_mult_cuda',
    sources=[
        os.path.join(module_path, 'cuda', 'forget_mult_cuda.cpp'),
        os.path.join(module_path, 'cuda', 'forget_mult_cuda_kernel.cu')
    ],
    verbose=True
)

class GPUForgetMult(Function):
    @staticmethod
    def forward(ctx, f, x, hidden_init=None):
        assert f.is_cuda and x.is_cuda, 'f and x must be CUDA tensors'

        if hidden_init is None:
            hidden_init = torch.zeros(x.size(1), x.size(2), device=x.device, dtype=x.dtype)
        else:
            hidden_init = hidden_init.contiguous()

        # The CUDA extension returns a tuple of tensors (h, h_full)
        h, h_full = forget_mult_cuda.forward(f.contiguous(), x.contiguous(), hidden_init)
        ctx.save_for_backward(f, x, h_full)
        ctx.hidden_init = hidden_init
        # Remove the line that assigns to ctx.needs_input_grad
        # ctx.needs_input_grad = (f.requires_grad, x.requires_grad, hidden_init.requires_grad)
        return h

    @staticmethod
    def backward(ctx, grad_h):
        f, x, h_full = ctx.saved_tensors
        hidden_init = ctx.hidden_init

        # Call the backward function of the CUDA extension
        grad_f, grad_x, grad_h_init = forget_mult_cuda.backward(
            grad_h.contiguous(), f, x, h_full, hidden_init
        )

        # Use ctx.needs_input_grad to determine which gradients to return
        grad_f = grad_f if ctx.needs_input_grad[0] else None
        grad_x = grad_x if ctx.needs_input_grad[1] else None
        grad_hidden_init = grad_h_init if ctx.needs_input_grad[2] else None

        return grad_f, grad_x, grad_hidden_init

class ForgetMult(nn.Module):
    def __init__(self):
        super(ForgetMult, self).__init__()

    def forward(self, f, x, hidden_init=None):
        if f.is_cuda and x.is_cuda:
            return GPUForgetMult.apply(f, x, hidden_init)
        else:
            # CPU implementation (if needed)
            raise NotImplementedError("CPU version is not implemented.")
