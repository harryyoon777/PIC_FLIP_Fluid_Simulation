#include "PressureSolver.cuh"
#include "../MaterialType.h"
#include "../NeighborDirection.h"
#include <thrust/device_vector.h>
#include <thrust/inner_product.h>


namespace {
    const float kFloatZero = 1.0e-6;

    __device__ __host__ inline int index3D(int x, int y, int z, int ny_, int nz_) {
        return x * ny_ * nz_ + y * nz_ + z;
    }

    inline float dot(float* a, float* b, int size) {
        thrust::device_ptr<float> a_ptr = thrust::device_pointer_cast(a);
        thrust::device_ptr<float> b_ptr = thrust::device_pointer_cast(b);
        return thrust::inner_product(a_ptr, a_ptr + size, b_ptr, 0.0f);
    }
}

__global__ void ZeroOutKernel(StaggeredGrid* grid, float* r, float* d, float* q, int nx_, int ny_, int nz_) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= nx_ || y >= ny_ || z >= nz_) return;

    grid->p()[index3D(x, y, z, ny_, nz_)] = 0.0f;
    r[index3D(x, y, z, ny_, nz_)] = 0.0f;
    d[index3D(x, y, z, ny_, nz_)] = 0.0f;
    q[index3D(x, y, z, ny_, nz_)] = 0.0f;
}

__global__ void ATimesKernel(float* d, StaggeredGrid* grid, int nx_, int ny_, int nz_, float* q_) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i < 1 || i >= nx_ - 1 || j < 1 || j >= ny_ - 1 || k < 1 || k >= nz_ - 1) {
        return;
    }

    const unsigned short CENTER = 7;
    const unsigned short LEFT = 8;
    const unsigned short DOWN = 16;
    const unsigned short BACK = 32;
    const unsigned short RIGHT = 64;
    const unsigned short UP = 128;
    const unsigned short FORWARD = 256;

    unsigned short nbrs = grid->neighbors()[index3D(i, j, k, ny_, nz_)];
    if (!nbrs) {
        q_[index3D(i, j, k, ny_, nz_)] = 0.0f;
        return;
    }

    q_[index3D(i, j, k, ny_, nz_)] = 
        ((nbrs & CENTER) * d[index3D(i, j, k, ny_, nz_)]) -
        ((nbrs & LEFT) ? d[index3D(i - 1, j, k, ny_, nz_)] : 0) -
        ((nbrs & DOWN) ? d[index3D(i, j - 1, k, ny_, nz_)] : 0) -
        ((nbrs & BACK) ? d[index3D(i, j, k - 1, ny_, nz_)] : 0) -
        ((nbrs & RIGHT) ? d[index3D(i + 1, j, k, ny_, nz_)] : 0) -
        ((nbrs & UP) ? d[index3D(i, j + 1, k, ny_, nz_)] : 0) -
        ((nbrs & FORWARD) ? d[index3D(i, j, k + 1, ny_, nz_)] : 0);
}

__global__ void MakeResidualFromVelocityDivergenceKernel(StaggeredGrid* grid, int nx_, int ny_, int nz_, float* r_) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i < 1 || i >= nx_ - 1 || j < 1 || j >= ny_ - 1 || k < 1 || k >= nz_ - 1) {
        return;
    }

    if (grid->cell_labels()[index3D(i, j, k, ny_, nz_)] != MaterialType::FLUID) {
        r_[index3D(i, j, k, ny_, nz_)] = 0.0f;
        return;
    }

    float du_dx = grid->u()[index3D(i + 1, j, k, ny_, nz_)] - grid->u()[index3D(i, j, k, ny_, nz_)];
    float dv_dy = grid->v()[index3D(i, j + 1, k, ny_ + 1, nz_)] - grid->v()[index3D(i, j, k, ny_ + 1, nz_)];
    float dw_dz = grid->w()[index3D(i, j, k + 1, ny_, nz_ + 1)] - grid->w()[index3D(i, j, k, ny_, nz_ + 1)];

    float velocity_divergence_of_cell = du_dx + dv_dy + dw_dz;
    r_[index3D(i, j, k, ny_, nz_)] = -velocity_divergence_of_cell;
}

__global__ void PlusEqualsKernel(StaggeredGrid* grid, float alpha, float* d, float* r, float* q, int nx_, int ny_, int nz_) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= nx_ || j >= ny_ || k >= nz_) return;

    grid->p()[index3D(i, j, k, ny_, nz_)] += alpha * d[index3D(i, j, k, ny_, nz_)];
    r[index3D(i, j, k, ny_, nz_)] -= alpha * q[index3D(i, j, k, ny_, nz_)];
}

__global__ void EqualsPlusTimesKernel(float* r, float beta, float* d, int nx_, int ny_, int nz_) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= nx_ || j >= ny_ || k >= nz_) return;

    float temp = d[index3D(i, j, k, ny_, nz_)];

    d[index3D(i, j, k, ny_, nz_)] = r[index3D(i, j, k, ny_, nz_)] + beta * temp;
}

PressureSolver::PressureSolver(std::size_t nx, std::size_t ny, std::size_t nz)
  : nx_(nx), 
    ny_(ny), 
    nz_(nz) {
    size_t size_p = nx_ * ny_ * nz_;
    cudaMalloc(&r_, size_p * sizeof(float));
    cudaMalloc(&d_, size_p * sizeof(float));
    cudaMalloc(&q_, size_p * sizeof(float));

    cudaGridSize = dim3(
        (nx_ + cudaBlockSize.x - 1) / cudaBlockSize.x,
        (ny_ + cudaBlockSize.y - 1) / cudaBlockSize.y,
        (nz_ + cudaBlockSize.z - 1) / cudaBlockSize.z
    );
}

void PressureSolver::ProjectPressure(StaggeredGrid* grid) {
    ZeroOutKernel<<<cudaGridSize, cudaBlockSize>>>(grid, r_, d_, q_, nx_, ny_, nz_);
    cudaDeviceSynchronize();

    MakeResidualFromVelocityDivergenceKernel<<<cudaGridSize, cudaBlockSize>>>(grid, nx_, ny_, nz_, r_);
    cudaDeviceSynchronize();

    cudaMemcpy(d_, r_, nx_ * ny_ * nz_ * sizeof(float), cudaMemcpyDeviceToDevice);

    float sigma = dot(r_, r_, nx_ * ny_ * nz_);
    float tolerance = kFloatZero * sigma;
    const std::size_t kMaxIters = 1000u;

    for (std::size_t iter = 0; iter < kMaxIters && sigma > tolerance; iter++) {
        ATimesKernel<<<cudaGridSize, cudaBlockSize>>>(d_, grid, nx_, ny_, nz_, q_);
        cudaDeviceSynchronize();

        float alpha = sigma / dot(d_, q_, nx_ * ny_ * nz_);

        PlusEqualsKernel<<<cudaGridSize, cudaBlockSize>>>(grid, alpha, d_, r_, q_, nx_, ny_, nz_);
        cudaDeviceSynchronize();

        float sigma_old = sigma;
        sigma = dot(r_, r_, nx_ * ny_ * nz_);
        float beta = sigma / sigma_old;
        
        EqualsPlusTimesKernel<<<cudaGridSize, cudaBlockSize>>>(r_, beta, d_, nx_, ny_, nz_);
        cudaDeviceSynchronize();
    }
}                                 

PressureSolver::~PressureSolver() {
    cudaFree(r_);
    cudaFree(d_);
    cudaFree(q_);
}