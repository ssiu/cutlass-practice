#include "cuda_transpose.cu"
#include "naive_transpose.cu"
#include "../utils/utils.cu"
#include <iostream>
#include <cstdio>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>


int main(int argc, char *argv[]) {
    // transpose an randon N x N matrix
    int M = std::stoi(argv[1]);
    int N = std::stoi(argv[2]);

    thrust::host_vector<float> h_S = generateRandomMatrix(M, N);
    thrust::host_vector<float> h_D(M * N);
    thrust::host_vector<float> h_D_naive(M * N);

    thrust::device_vector<float> d_S = h_S;
    thrust::device_vector<float> d_D = h_D;
    thrust::device_vector<float> d_D_naive = h_D_naive;

    dim3 dimGrid(M/32, N/32); // You can adjust this based on your GPU's capability
    dim3 dimBlock(32, 32);

    // Launch the matrix multiplication kernel
    cuda_transpose<<<dimGrid, dimBlock>>>(thrust::raw_pointer_cast(d_S.data()), thrust::raw_pointer_cast(d_D.data()), M, N);

    naive_transpose(thrust::raw_pointer_cast(d_S.data()), thrust::raw_pointer_cast(d_D_naive.data()), M, N);

    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
    }

    h_D = d_D;
    h_D_naive = d_D_naive;
    if (areSameMatrices(h_D.data(), h_D_naive.data(), M, N) == 0) {
        printf("Wrong answer\n");
    }

    return 0;
}