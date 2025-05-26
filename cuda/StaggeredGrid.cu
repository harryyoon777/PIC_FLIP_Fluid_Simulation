#include <cuda_runtime.h>
#include <cmath>
#include <iostream>
#include "StaggeredGrid.cuh"

namespace {
    const float kFloatZero = 1.0e-6f;
    const float kGravAccMetersPerSecond = 9.81f;


    // Cell reference coordinates: left corner --> based on cell faces
    __device__ __host__ float3 HalfShiftYZ(float dx) {
    float3 half_shift;
    float dx_2 = dx / 2.0;
    half_shift.x = 0.0;
    half_shift.y = dx_2;
    half_shift.z = dx_2;
    return half_shift;
    }

    __device__ __host__ float3 HalfShiftXZ(float dx) {
    float3 half_shift;
    float dx_2 = dx / 2.0;
    half_shift.x = dx_2;
    half_shift.y = 0.0;
    half_shift.z = dx_2;
    return half_shift;
    }

    __device__ __host__ float3 HalfShiftXY(float dx) {
    float3 half_shift;
    float dx_2 = dx / 2.0;
    half_shift.x = dx_2;
    half_shift.y = dx_2;
    half_shift.z = 0.0;
    return half_shift;
    }

    __device__ __host__ float3 operator+(const float3& a, const float3& b) {
        return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
    }

    __device__ const NeighborDirection NeighborDirections[] = {
        LEFT, RIGHT, DOWN, UP, BACK, FORWARD
    };

    __device__ __host__ inline int index3D(int x, int y, int z, int ny_, int nz_) {
        return x * ny_ * nz_ + y * nz_ + z;
    }
} // namespace

__device__ MaterialType GetNeighborMaterial(MaterialType* cell_labels, int i, int j, int k, NeighborDirection dir, int ny_, int nz_) {
    switch (dir) {
        case LEFT:
            return cell_labels[index3D(i - 1, j, k, ny_, nz_)];
        case RIGHT:
            return cell_labels[index3D(i + 1, j, k, ny_, nz_)];
        case DOWN:
            return cell_labels[index3D(i, j - 1, k, ny_, nz_)];
        case UP:
            return cell_labels[index3D(i, j + 1, k, ny_, nz_)];
        case BACK:
            return cell_labels[index3D(i, j, k - 1, ny_, nz_)];
        case FORWARD:
            return cell_labels[index3D(i, j, k + 1, ny_, nz_)];
    }
    assert(false);
    return SOLID;
}

__device__ unsigned short UpdateFromNeighbor(unsigned short nbr_info,
                                  MaterialType nbr_material,
                                  NeighborDirection dir) {
  unsigned short new_nbr_info = nbr_info;

  if (nbr_material != SOLID) {
    new_nbr_info++;
  }

  if (nbr_material != FLUID) {
    return new_nbr_info;
  }

  return new_nbr_info | dir;
}

__global__ void ZeroOutVelocitiesKernel_u(StaggeredGrid* grid) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= grid->nx() + 1 || y >= grid->ny() || z >= grid->nz()) return;

    int index = index3D(x, y, z, grid->ny(), grid->nz());

    grid->u()[index] = 0.0;
    grid->fu()[index] = 0.0;
}

__global__ void ZeroOutVelocitiesKernel_v(StaggeredGrid* grid) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= grid->nx() || y >= grid->ny() + 1 || z >= grid->nz()) return;

    int index = index3D(x, y, z, grid->ny() + 1, grid->nz());

    grid->v()[index] = 0.0;
    grid->fv()[index] = 0.0;
}

__global__ void ZeroOutVelocitiesKernel_w(StaggeredGrid* grid) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= grid->nx() || y >= grid->ny() || z >= grid->nz() + 1) return;

    int index = index3D(x, y, z, grid->ny(), grid->nz() + 1);

    grid->w()[index] = 0.0;
    grid->fw()[index] = 0.0;
}

__global__ void ClearCellLabelsKernel(StaggeredGrid* grid) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= grid->nx() || y >= grid->ny() || z >= grid->nz()) return;

    int index = index3D(x, y, z, grid->ny(), grid->nz());

    // init cell labels
    grid->cell_labels()[index] = MaterialType::EMPTY;

    // set outer cell labels to solid
    if (x == 0 || x == grid->nx() - 1 || y == 0 || y == grid->ny() - 1 || z == 0 || z == grid->nz() - 1) {
        grid->cell_labels()[index] = MaterialType::SOLID;
    }
}

__global__ void NormalizeAndStoreVelocitiesKernel_u(StaggeredGrid* grid) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= grid->nx() + 1 || j >= grid->ny() || k >= grid->nz()) return;
    
    int index = index3D(i, j, k, grid->ny(), grid->nz());

    // initialize boundary and adjacent cell velocities to zero
    if (i == 0 || i == 1 || i == grid->nx() - 1 || i == grid->nx()) {
        grid->u()[index] = 0.0f;
    }

    // normalize velocities in interior cells
    if (i >= 2 && i < grid->nx() - 1) {
        float weight = grid->fu()[index];
        if (weight < kFloatZero) {
            grid->u()[index] = 0.0f;
        } else {
            grid->u()[index] /= weight;
        }
    }
    grid->fu()[index] = grid->u()[index];
}

__global__ void NormalizeAndStoreVelocitiesKernel_v(StaggeredGrid* grid) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= grid->nx() || j >= grid->ny() + 1 || k >= grid->nz()) return;

    int index = index3D(i, j, k, grid->ny() + 1, grid->nz());

    // initialize boundary and adjacent cell velocities to zero
    if (j == 0 || j == 1 || j == grid->ny() - 1 || j == grid->ny()) {
        grid->v()[index] = 0.0f;
    }

    // normalize velocities in interior cells
    if (j >= 2 && j < grid->ny() - 1) {
        float weight = grid->fv()[index];
        if (weight < kFloatZero) {
            grid->v()[index] = 0.0f;
        } else {
            grid->v()[index] /= weight;
        }
    }
    grid->fv()[index] = grid->v()[index];
}

__global__ void NormalizeAndStoreVelocitiesKernel_w(StaggeredGrid* grid) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= grid->nx() || j >= grid->ny() || k >= grid->nz() + 1) return;

    int index = index3D(i, j, k, grid->ny(), grid->nz() + 1);
    
    // initialize boundary and adjacent cell velocities to zero
    if (k == 0 || k == 1 || k == grid->nz() - 1 || k == grid->nz()) {
        grid->w()[index] = 0.0f;
    }

    // normalize velocities in interior cells
    if (k >= 2 && k < grid->nz() - 1) {
        float weight = grid->fw()[index];
        if (weight < kFloatZero) {
            grid->w()[index] = 0.0f;
        } else {
            grid->w()[index] /= weight;
        }
    }
    grid->fw()[index] = grid->w()[index];
}

__global__ void SetLRBoundaryVelocitiesKernel(StaggeredGrid* grid) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    int nx = grid->nx();
    int ny = grid->ny();
    int nz = grid->nz();

    if (i >= nx + 1 || j >= ny + 1 || k >= nz + 1) return;

    if (i == 0) {
        // u: 0, [0, ny-1], [0, nz-1]
        if (j < ny && k < nz)
            grid->u()[index3D(0, j, k, ny, nz)] = 0.0f;

        // v: 0, [0, ny], [0, nz-1]
        if (j < ny + 1 && k < nz)
            grid->v()[index3D(0, j, k, ny + 1, nz)] = grid->v()[index3D(1, j, k, ny + 1, nz)];

        // w: 0, [0, ny-1], [0, nz]
        if (j < ny && k < nz + 1)
            grid->w()[index3D(0, j, k, ny, nz + 1)] = grid->w()[index3D(1, j, k, ny, nz + 1)];
    }

    if (i == 1 && j < ny && k < nz)
        grid->u()[index3D(1, j, k, ny, nz)] = 0.0f;

    if (i == nx - 1) {
        if (j < ny && k < nz)
            grid->u()[index3D(i, j, k, ny, nz)] = 0.0f;

        if (j < ny + 1 && k < nz)
            grid->v()[index3D(i, j, k, ny + 1, nz)] = grid->v()[index3D(i - 1, j, k, ny + 1, nz)];

        if (j < ny && k < nz + 1)
            grid->w()[index3D(i, j, k, ny, nz + 1)] = grid->w()[index3D(i - 1, j, k, ny, nz + 1)];
    }

    if (i == nx && j < ny && k < nz)
        grid->u()[index3D(i, j, k, ny, nz)] = 0.0f;
}

__global__ void SetABBoundaryVelocitiesKernel(StaggeredGrid* grid) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    int nx = grid->nx();
    int ny = grid->ny();
    int nz = grid->nz();

    if (i >= nx + 1 || j >= ny + 1 || k >= nz + 1) return;

    if (j == 0) {
        // u: [0, nx], 0, [0, nz-1]
        if (i < nx + 1 && k < nz)
            grid->u()[index3D(i, 0, k, ny, nz)] = grid->u()[index3D(i, 1, k, ny, nz)];

        // v: [0, nx-1], 0, [0, nz-1]
        if (i < nx && k < nz)
            grid->v()[index3D(i, 0, k, ny + 1, nz)] = 0.0f;

        // w: [0, nx-1], 0, [0, nz]
        if (i < nx && k < nz + 1)
            grid->w()[index3D(i, 0, k, ny, nz + 1)] = grid->w()[index3D(i, 1, k, ny, nz + 1)];
    }

    if (j == 1 && i < nx && k < nz) 
        grid->v()[index3D(i, 1, k, ny + 1, nz)] = 0.0f;

    if (j == ny - 1) {
        if (i < nx + 1 && k < nz)
            grid->u()[index3D(i, j, k, ny, nz)] = grid->u()[index3D(i, j - 1, k, ny, nz)];

        if (i < nx && k < nz)
            grid->v()[index3D(i, j, k, ny + 1, nz)] = 0.0f;
        
        if (i < nx && k < nz + 1)
            grid->w()[index3D(i, j, k, ny, nz + 1)] = grid->w()[index3D(i, j - 1, k, ny, nz + 1)];
    }

    if (j == ny && i < nx && k < nz)
        grid->v()[index3D(i, j, k, ny + 1, nz)] = 0.0f;
}   

__global__ void SetFBBoundaryVelocitiesKernel(StaggeredGrid* grid) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    int nx = grid->nx();
    int ny = grid->ny();
    int nz = grid->nz();

    if (i >= nx + 1 || j >= ny + 1|| k >= nz + 1) return; 

    if (k == 0) {
        // u: [0, nx], [0, ny-1], 0
        if (i < nx + 1 && j < ny)
            grid->u()[index3D(i, j, 0, ny, nz)] = grid->u()[index3D(i, j, 1, ny, nz)];

        // v: [0, nx-1], [0, ny], 0
        if (i < nx && j < ny + 1)
            grid->v()[index3D(i, j, 0, ny + 1, nz)] = grid->v()[index3D(i, j, 1, ny + 1, nz)];

        // w: [0, nx-1], [0, ny-1], 0
        if (i < nx && j < ny)
            grid->w()[index3D(i, j, 0, ny, nz + 1)] = 0.0f;
    }

    if (k == 1 && i < nx && j < ny)
        grid->w()[index3D(i, j, 1, ny, nz + 1)] = 0.0f;

    if (k == nz - 1) {
        if (i < nx + 1 && j < ny)
            grid->u()[index3D(i, j, k, ny, nz)] = grid->u()[index3D(i, j, k - 1, ny, nz)];

        if (i < nx && j < ny + 1)
            grid->v()[index3D(i, j, k, ny + 1, nz)] = grid->v()[index3D(i, j, k - 1, ny + 1, nz)];

        if (i < nx && j < ny)
            grid->w()[index3D(i, j, k, ny, nz + 1)] = 0.0f;
    }

    if (k == nz && i < nx && j < ny)
        grid->w()[index3D(i, j, nz, ny, nz + 1)] = 0.0f;
}   


__global__ void ApplyGravityKernel(StaggeredGrid* grid, float dt) {
    float vertical_velocity_change = -dt * kGravAccMetersPerSecond;
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    int nx = grid->nx();
    int ny = grid->ny();
    int nz = grid->nz();

    if (i >= nx || j >= ny || k >= nz + 1) return;

    grid->w()[index3D(i, j, k, ny, nz + 1)] += vertical_velocity_change;
}

__global__ void MakeNeighborMaterialInfoKernel(StaggeredGrid* grid) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    int nx = grid->nx();
    int ny = grid->ny();
    int nz = grid->nz();

    if (i < 1 || i >= nx - 1 || j < 1 || j >= ny - 1 || k < 1 || k >= nz - 1) {
        return;
    }

    grid->neighbors()[index3D(i, j, k, ny, nz)] = 0u;

    if (grid->cell_labels()[index3D(i, j, k, ny, nz)] != FLUID) {
        return;
    }

    unsigned short nbr_info = 0u;
    for (NeighborDirection dir : NeighborDirections) {
        MaterialType nbr_material = GetNeighborMaterial(grid->cell_labels(), i, j, k, dir, ny, nz);
        nbr_info = UpdateFromNeighbor(nbr_info, nbr_material, dir);
    }
    grid->neighbors()[index3D(i, j, k, ny, nz)] = nbr_info;
}

__global__ void SubtractPressureGradientFromVelocityKernel(StaggeredGrid* grid) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    int nx = grid->nx();
    int ny = grid->ny();
    int nz = grid->nz();

    if (i < 1 || i >= nx - 1 || j < 1 || j >= ny - 1 || k < 1 || k >= nz - 1) {
        return;
    }

    if (grid->cell_labels()[index3D(i, j, k, ny, nz)] == SOLID) {
        return;
    }

    float pijk = grid->p()[index3D(i, j, k, ny, nz)];
    
    if (grid->cell_labels()[index3D(i - 1, j, k, ny, nz)] != SOLID) {
        grid->u()[index3D(i, j, k, ny, nz)] -= pijk - grid->p()[index3D(i - 1, j, k, ny, nz)];
    }
    if (grid->cell_labels()[index3D(i, j - 1, k, ny, nz)] != SOLID) {
        grid->v()[index3D(i, j, k, ny + 1, nz)] -= pijk - grid->p()[index3D(i, j - 1, k, ny, nz)];
    }
    if (grid->cell_labels()[index3D(i, j, k - 1, ny, nz)] != SOLID) {
        grid->w()[index3D(i, j, k, ny, nz + 1)] -= pijk - grid->p()[index3D(i, j, k - 1, ny, nz)];
    }
}



StaggeredGrid::StaggeredGrid(int nx, int ny, int nz, float3 lc, float dx)
    : nx_(nx),
    ny_(ny),
    nz_(nz),
    ny_nz_(ny * nz),
    lc_(lc),
    uc_(lc + make_float3(nx * dx, ny * dx, nz * dx)),
    dx_(dx),
    half_shift_yz_(HalfShiftYZ(dx)),
    half_shift_xz_(HalfShiftXZ(dx)),
    half_shift_xy_(HalfShiftXY(dx)) {

    size_t size_p = nx * ny * nz;
    size_t size_u = (nx + 1) * ny * nz;
    size_t size_v = nx * (ny + 1) * nz;
    size_t size_w = nx * ny * (nz + 1);

    cudaMalloc(&p_, size_p * sizeof(float));
    cudaMalloc(&u_, size_u * sizeof(float));
    cudaMalloc(&v_, size_v * sizeof(float));
    cudaMalloc(&w_, size_w * sizeof(float));
    cudaMalloc(&cell_labels_, nx * ny * nz * sizeof(MaterialType));
    cudaMalloc(&neighbors_, nx * ny * nz * sizeof(unsigned short));
    cudaMalloc(&fu_, size_u * sizeof(float));
    cudaMalloc(&fv_, size_v * sizeof(float));
    cudaMalloc(&fw_, size_w * sizeof(float));
}

__device__ int3 StaggeredGrid::getCellIndex(float3 p_lc) {
    int x = (int)(p_lc.x / dx_);
    int y = (int)(p_lc.y / dx_);
    int z = (int)(p_lc.z / dx_);

    x = max(0, min(x, nx_ - 1));
    y = max(0, min(y, ny_ - 1));
    z = max(0, min(z, nz_ - 1));

    return make_int3(x, y, z);
}

__device__ void StaggeredGrid::setCellLabel(int3 cell_idx, MaterialType cell_label) {
    cell_labels_[index3D(cell_idx.x, cell_idx.y, cell_idx.z, ny_, nz_)] = cell_label;
}

StaggeredGrid::~StaggeredGrid() {
    cudaFree(p_);
    cudaFree(u_);
    cudaFree(v_);
    cudaFree(w_);
    cudaFree(fu_);
    cudaFree(fv_);
    cudaFree(fw_);
}