#ifndef STAGGERED_GRID_CUH
#define STAGGERED_GRID_CUH

#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#ifndef EIGEN_NO_CUDA
#define EIGEN_NO_CUDA
#endif

#include <vector>
#include "../MaterialType.h"
#include "../NeighborDirection.h"

class StaggeredGrid {
public:
    StaggeredGrid(int nx, int ny, int nz, float3 lc, float dx);
    ~StaggeredGrid();

    void ZeroOutVelocities();
    void ClearCellLabels();
    void NormalizeAllVelocities();
    void StoreNormalizedVelocities();

    // Sets boundary conditions on the grid velocities.
    // For example, at the x-direction boundary:
    // Setting u velocity to zero:
    // This means that fluid cannot pass through the boundary in the x-direction.
    // This models a physical wall that blocks fluid flow in the x-direction.
    // Copying v and w velocities from adjacent cells:
    // This means that fluid can flow along the boundary in the y and z directions.
    // This models the physical phenomenon where fluid can slip along the wall.
    void SetBoundaryVelocities();
    void ApplyGravity(float dt);
    void MakeNeighborMaterialInfo();
    void SubtractPressureGradientFromVelocity();

    __device__ int3 getCellIndex(float3 p_lc);
    __device__ void setCellLabel(int3 cell_idx, MaterialType cell_label);
   
    // Get the index of the cell in the staggered grid
    // __host__ __device__ inline int index3D(int x, int y, int z, int ny_, int nz_);

    __host__ __device__ int nx() const { return nx_; }
    __host__ __device__ int ny() const { return ny_; }
    __host__ __device__ int nz() const { return nz_; }
    __host__ __device__ int ny_nz() const { return ny_nz_; }
    __host__ __device__ float3 lc() const { return lc_; }
    __host__ __device__ float3 uc() const { return uc_; }
    __host__ __device__ float dx() const { return dx_; }
    __host__ __device__ float3 half_shift_yz() const { return half_shift_yz_; }
    __host__ __device__ float3 half_shift_xz() const { return half_shift_xz_; }
    __host__ __device__ float3 half_shift_xy() const { return half_shift_xy_; }
    
    __host__ __device__ float* p() { return p_; }
    __host__ __device__ float* u() { return u_; }
    __host__ __device__ float* v() { return v_; }
    __host__ __device__ float* w() { return w_; }
    __host__ __device__ float* fu() { return fu_; }
    __host__ __device__ float* fv() { return fv_; }
    __host__ __device__ float* fw() { return fw_; }
    __host__ __device__ MaterialType* cell_labels() { return cell_labels_; }
    __host__ __device__ unsigned short* neighbors() { return neighbors_; }  

private:
    const int nx_;
    const int ny_;
    const int nz_;
    const int ny_nz_;
    const float3 lc_;
    const float3 uc_;
    const float dx_;
    const float3 half_shift_yz_;
    const float3 half_shift_xz_;
    const float3 half_shift_xy_;

    // GridParameters grid_params_;
    float* p_;
    float* u_;
    float* v_;
    float* w_;
    float* fu_;
    float* fv_;
    float* fw_;
    MaterialType* cell_labels_;
    unsigned short* neighbors_;  
};


__global__ void ZeroOutVelocitiesKernel_u(StaggeredGrid* grid);
__global__ void ZeroOutVelocitiesKernel_v(StaggeredGrid* grid);
__global__ void ZeroOutVelocitiesKernel_w(StaggeredGrid* grid);

__global__ void NormalizeAndStoreVelocitiesKernel_u(StaggeredGrid* grid);
__global__ void NormalizeAndStoreVelocitiesKernel_v(StaggeredGrid* grid);
__global__ void NormalizeAndStoreVelocitiesKernel_w(StaggeredGrid* grid);

__global__ void ClearCellLabelsKernel(StaggeredGrid* grid);

// Left and Right Boundary
__global__ void SetLRBoundaryVelocitiesKernel(StaggeredGrid* grid);
// Above and Below Boundary
__global__ void SetABBoundaryVelocitiesKernel(StaggeredGrid* grid);
// Front and Back Boundary
__global__ void SetFBBoundaryVelocitiesKernel(StaggeredGrid* grid);

__global__ void ApplyGravityKernel(StaggeredGrid* grid, float dt);
__global__ void MakeNeighborMaterialInfoKernel(StaggeredGrid* grid);

__global__ void SubtractPressureGradientFromVelocityKernel(StaggeredGrid* grid);

#endif // STAGGERED_GRID_CUH