// torchqrnn/cuda/forget_mult_cuda.cpp
#include <torch/extension.h>
#include <vector>

// CUDA forward declarations
void recurrent_forget_mult_forward_cuda(
    torch::Tensor f,
    torch::Tensor x,
    torch::Tensor h);

void recurrent_forget_mult_backward_cuda(
    torch::Tensor grad_h,
    torch::Tensor f,
    torch::Tensor x,
    torch::Tensor h,
    torch::Tensor hidden_init,
    torch::Tensor grad_f,
    torch::Tensor grad_x,
    torch::Tensor grad_h_init);

std::vector<torch::Tensor> recurrent_forget_mult_forward(
    torch::Tensor f,
    torch::Tensor x,
    torch::Tensor hidden_init)
{
    int64_t seq_size = f.size(0);
    int64_t batch_size = f.size(1);
    int64_t hidden_size = f.size(2);

    auto h = torch::zeros({seq_size + 1, batch_size, hidden_size}, f.options());

    if (hidden_init.defined()) {
        h[0] = hidden_init;
    }

    recurrent_forget_mult_forward_cuda(f, x, h);

    // Return h[1:] as the output, and h as the full hidden states for backward
    auto h_output = h.narrow(0, 1, seq_size);
    return {h_output, h};
}

std::vector<torch::Tensor> recurrent_forget_mult_backward(
    torch::Tensor grad_h,
    torch::Tensor f,
    torch::Tensor x,
    torch::Tensor h,
    torch::Tensor hidden_init)
{
    auto grad_f = torch::zeros_like(f);
    auto grad_x = torch::zeros_like(x);
    auto grad_h_init = torch::zeros_like(hidden_init);

    recurrent_forget_mult_backward_cuda(grad_h, f, x, h, hidden_init, grad_f, grad_x, grad_h_init);

    return {grad_f, grad_x, grad_h_init};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &recurrent_forget_mult_forward, "ForgetMult forward");
  m.def("backward", &recurrent_forget_mult_backward, "ForgetMult backward");
}
