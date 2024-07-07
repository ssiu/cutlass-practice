__global__ void cuda_transpose(float* d_S, float* d_D, int M, int N) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // d_S is M * N row major
    // d_D is N * M row major
    d_D[col * N + row] = d_S[row * N + col];

}