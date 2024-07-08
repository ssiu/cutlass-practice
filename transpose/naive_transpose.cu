#include <cute/tensor.hpp>

using namespace cute;

template <class TensorS, class TensorD>
__global__ void naive_transpose_kernel(TensorS tiled_tensor_S, TensorD tiled_tensor_D) {

    Tensor tile_S = tiled_tensor_S(make_coord(_, _), blockIdx.x, blockIdx.y);
    Tensor tile_D = tiled_tensor_D(make_coord(_, _), blockIdx.x, blockIdx.y);

    auto thr_layout = make_layout(make_shape(Int<8>{}, Int<32>{}), GenRowMajor{});
    Tensor thr_tile_S = local_partition(tile_S, thr_layout, threadIdx.x);
    Tensor thr_tile_D = local_partition(tile_D, thr_layout, threadIdx.x);

    Tensor rmem = make_tensor_like(thr_tile_S);
    copy(thr_tile_S, rmem);
    copy(rmem, thr_tile_D);

}


void naive_transpose(float* d_S, float* d_D, int M, int N) {
    auto gShape = make_shape(M,N);
    auto gLayout_S = make_layout(gShape, GenRowMajor{});
    //transpose is M * N in column major
    auto gLayout_D = make_layout(gShape, GenColMajor{});

    Tensor tensor_S = make_tensor(make_gmem_ptr(d_S), gLayout_S);
    Tensor tensor_D = make_tensor(make_gmem_ptr(d_D), gLayout_D);

    using block_size = Int<32>;
    auto block_shape = make_shape(block_size{}, block_size{});
    Tensor tiled_tensor_S = tiled_divide(tensor_S, block_shape);
    Tensor tiled_tensor_D = tiled_divide(tensor_D, block_shape);

    dim3 dimGrid(M/32, N/32);
    dim3 dimBlock(256);

    naive_transpose_kernel<<<dimGrid, dimBlock>>>(tiled_tensor_S, tiled_tensor_D);

}





