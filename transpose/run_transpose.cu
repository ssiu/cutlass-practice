#include "cuda_transpose.cu"
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

    thrust::device_vector<float> d_S = h_S;
    thrust::device_vector<float> d_D = h_D;


    dim3 dimGrid(M/32, N/32); // You can adjust this based on your GPU's capability
    dim3 dimBlock(32, 32);


    // Launch the matrix multiplication kernel
    cuda_transpose<<<dimGrid, dimBlock>>>(d_S.data().get(), d_D.data().get(), M, N);

    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
    }

    h_D = d_D;

//    if (isSameMatrix(h_out, h_in_t, N) == 0) {
//        printf("Wrong answer\n");
//    }

    return 0;
}