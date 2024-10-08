// torchqrnn/cuda/forget_mult_cuda_kernel.cu
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

template <typename scalar_t>
__global__ void recurrent_forget_mult_forward_kernel(
    scalar_t* __restrict__ h,
    const scalar_t* __restrict__ f,
    const scalar_t* __restrict__ x,
    int64_t SEQ,
    int64_t BATCH,
    int64_t HIDDEN)
{
    int hid = blockIdx.x * blockDim.x + threadIdx.x;
    int bid = blockIdx.y * blockDim.y + threadIdx.y;
    if (hid >= HIDDEN || bid >= BATCH)
        return;
    for (int ts = 1; ts < SEQ + 1; ts++) {
        int i           = (ts - 1) * HIDDEN * BATCH + bid * HIDDEN + hid;
        int h_i         = ts * HIDDEN * BATCH + bid * HIDDEN + hid;
        int h_iminus1   = (ts - 1) * HIDDEN * BATCH + bid * HIDDEN + hid;
        h[h_i]          = f[i] * x[i];
        h[h_i]         += (1 - f[i]) * h[h_iminus1];
    }
}

template <typename scalar_t>
__global__ void recurrent_forget_mult_backward_kernel(
    const scalar_t* __restrict__ grad_h,
    const scalar_t* __restrict__ f,
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ h,
    scalar_t* __restrict__ grad_f,
    scalar_t* __restrict__ grad_x,
    scalar_t* __restrict__ grad_h_init,
    int64_t SEQ,
    int64_t BATCH,
    int64_t HIDDEN)
{
    int hid = blockIdx.x * blockDim.x + threadIdx.x;
    int bid = blockIdx.y * blockDim.y + threadIdx.y;
    if (hid >= HIDDEN || bid >= BATCH)
        return;

    scalar_t running_f = 0;
    for (int ts = SEQ; ts >= 1; ts--) {
        int i           = (ts - 1) * HIDDEN * BATCH + bid * HIDDEN + hid;
        int h_i         = ts * HIDDEN * BATCH + bid * HIDDEN + hid;
        int h_iminus1   = (ts - 1) * HIDDEN * BATCH + bid * HIDDEN + hid;

        running_f      += grad_h[h_i];
        grad_x[i]       = f[i] * running_f;
        grad_f[i]       = (x[i] - h[h_iminus1]) * running_f;
        running_f       = running_f - f[i] * running_f;
    }
    grad_h_init[bid * HIDDEN + hid] = running_f;
}

void recurrent_forget_mult_forward_cuda(
    torch::Tensor f,
    torch::Tensor x,
    torch::Tensor h)
{
    const auto seq_size = f.size(0);
    const auto batch_size = f.size(1);
    const auto hidden_size = f.size(2);

    const int threads = 512;
    const dim3 blocks((hidden_size + threads - 1) / threads, batch_size);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(f.scalar_type(), "recurrent_forget_mult_forward_cuda", ([&] {
        recurrent_forget_mult_forward_kernel<scalar_t><<<blocks, threads>>>(
            h.data_ptr<scalar_t>(),
            f.data_ptr<scalar_t>(),
            x.data_ptr<scalar_t>(),
            seq_size,
            batch_size,
            hidden_size);
    }));
}

void recurrent_forget_mult_backward_cuda(
    torch::Tensor grad_h,
    torch::Tensor f,
    torch::Tensor x,
    torch::Tensor h,
    torch::Tensor hidden_init,
    torch::Tensor grad_f,
    torch::Tensor grad_x,
    torch::Tensor grad_h_init)
{
    const auto seq_size = f.size(0);
    const auto batch_size = f.size(1);
    const auto hidden_size = f.size(2);

    const int threads = 512;
    const dim3 blocks((hidden_size + threads - 1) / threads, batch_size);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(f.scalar_type(), "recurrent_forget_mult_backward_cuda", ([&] {
        recurrent_forget_mult_backward_kernel<scalar_t><<<blocks, threads>>>(
            grad_h.data_ptr<scalar_t>(),
            f.data_ptr<scalar_t>(),
            x.data_ptr<scalar_t>(),
            h.data_ptr<scalar_t>(),
            grad_f.data_ptr<scalar_t>(),
            grad_x.data_ptr<scalar_t>(),
            grad_h_init.data_ptr<scalar_t>(),
            seq_size,
            batch_size,
            hidden_size);
    }));
}
