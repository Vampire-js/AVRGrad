#include "matmul_gpu.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
void check_cuda(cudaError_t error, const char *operation)
{
    if (error != cudaSuccess)
    {
        throw std::runtime_error(
            std::string(operation) + ": " + cudaGetErrorString(error));
    }
}
}

__global__ void matmul_kernel(
    const double *a,
    const double *b,
    double *c,
    std::size_t l,
    std::size_t m,
    std::size_t n)
{
    if (blockIdx.x != 0 || threadIdx.x != 0)
    {
        return;
    }

    for (std::size_t i = 0; i < l; ++i)
    {
        for (std::size_t j = 0; j < n; ++j)
        {
            double sum = 0.0;
            for (std::size_t k = 0; k < m; ++k)
            {
                sum += a[i * m + k] * b[k * n + j];
            }
            c[i * n + j] = sum;
        }
    }
}

void matmul_gpu(const Tensor &A, const Tensor &B, Tensor &C)
{
    if (A.rank() != 2 || B.rank() != 2 || C.rank() != 2)
    {
        throw std::invalid_argument("matmul_gpu expects rank-2 tensors");
    }

    if (A.shape()[1] != B.shape()[0])
    {
        throw std::invalid_argument("input tensor shapes cannot be multiplied");
    }

    if (C.shape()[0] != A.shape()[0] || C.shape()[1] != B.shape()[1])
    {
        throw std::invalid_argument("output tensor has an incorrect shape");
    }

    std::size_t l = A.shape()[0];
    std::size_t m = A.shape()[1];
    std::size_t n = B.shape()[1];

    if (C.numel() == 0)
    {
        return;
    }

    double *device_a = nullptr;
    double *device_b = nullptr;
    double *device_c = nullptr;

    const std::size_t a_bytes = A.numel() * sizeof(double);
    const std::size_t b_bytes = B.numel() * sizeof(double);
    const std::size_t c_bytes = C.numel() * sizeof(double);

    try
    {
        if (a_bytes != 0)
        {
            check_cuda(cudaMalloc(&device_a, a_bytes), "cudaMalloc(A)");
            check_cuda(
                cudaMemcpy(device_a, A.data().data(), a_bytes, cudaMemcpyHostToDevice),
                "cudaMemcpy(A)");
        }

        if (b_bytes != 0)
        {
            check_cuda(cudaMalloc(&device_b, b_bytes), "cudaMalloc(B)");
            check_cuda(
                cudaMemcpy(device_b, B.data().data(), b_bytes, cudaMemcpyHostToDevice),
                "cudaMemcpy(B)");
        }

        check_cuda(cudaMalloc(&device_c, c_bytes), "cudaMalloc(C)");

        matmul_kernel<<<1, 1>>>(device_a, device_b, device_c, l, m, n);
        check_cuda(cudaGetLastError(), "matmul_kernel launch");
        check_cuda(cudaDeviceSynchronize(), "matmul_kernel execution");

        std::vector<double> result(C.numel());
        check_cuda(
            cudaMemcpy(result.data(), device_c, c_bytes, cudaMemcpyDeviceToHost),
            "cudaMemcpy(C)");

        for (std::size_t i = 0; i < l; ++i)
        {
            for (std::size_t j = 0; j < n; ++j)
            {
                C.at({i, j}) = result[i * n + j];
            }
        }
    }
    catch (...)
    {
        cudaFree(device_a);
        cudaFree(device_b);
        cudaFree(device_c);
        throw;
    }

    check_cuda(cudaFree(device_a), "cudaFree(A)");
    check_cuda(cudaFree(device_b), "cudaFree(B)");
    check_cuda(cudaFree(device_c), "cudaFree(C)");
}
