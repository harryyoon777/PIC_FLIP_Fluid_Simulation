#include <cuda_runtime.h>
#include <stdio.h>
#include "PIC_FLIP.cuh"


namespace {
    const float kClampCushion = 1.0e-4f;
    const float kGravAccMetersPerSecond = 9.80665;

    __device__ inline int index3D(int x, int y, int z, int ny_, int nz_) {
        return x * ny_ * nz_ + y * nz_ + z;
    }

    __device__ inline float3 operator-(float3 a, float3 b) {
        return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
    }

    __device__ inline float3 operator*(float a, float3 b) {
        return make_float3(a * b.x, a * b.y, a * b.z);
    }

    __device__ inline float3 operator+(float3 a, float3 b) {
        return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
    }

    // dim3 cudaBlockSize = dim3(8,8,8);
    // // TODO: Make this dynamic
    // dim3 cudaGridSize = dim3(
    //     (25 + cudaBlockSize.x - 1) / cudaBlockSize.x,
    //     (50 + cudaBlockSize.y - 1) / cudaBlockSize.y,
    //     (25 + cudaBlockSize.z - 1) / cudaBlockSize.z
    // );
}

__device__ void contribute(float weight, float particle_velocity, float* grid_vels, float* grid_vel_weights, int index3D) {
    atomicAdd(&grid_vels[index3D], weight * particle_velocity);
    atomicAdd(&grid_vel_weights[index3D], weight);
}


__device__ void splat(float3 shifted_particle_position_lc, float dx, 
                        float particle_velocity, float* grid_vels, float* grid_vel_weights,
                        StaggeredGrid* grid, int ny, int nz) {
    // Determine the grid cell containing the shifted particle position.
    int3 shifted_cell_idx = grid->getCellIndex(shifted_particle_position_lc);

    // Determine the barycentric weights of the shifted particle position inside
    // that grid cell.
    float fx = shifted_particle_position_lc.x / dx - shifted_cell_idx.x;
    float fy = shifted_particle_position_lc.y / dx - shifted_cell_idx.y;
    float fz = shifted_particle_position_lc.z / dx - shifted_cell_idx.z;

    float w0 = fx;
    float om_w0 = 1.0 - w0;
    float w1 = fy;
    float om_w1 = 1.0 - w1;
    float w2 = fz;
    float om_w2 = 1.0 - w2;

    contribute(om_w0 * om_w1 * om_w2, particle_velocity, grid_vels, grid_vel_weights, index3D(shifted_cell_idx.x, shifted_cell_idx.y, shifted_cell_idx.z, ny, nz));
    contribute(om_w0 * om_w1 * w2, particle_velocity, grid_vels, grid_vel_weights, index3D(shifted_cell_idx.x, shifted_cell_idx.y, shifted_cell_idx.z + 1, ny, nz));
    contribute(om_w0 * w1 * om_w2, particle_velocity, grid_vels, grid_vel_weights, index3D(shifted_cell_idx.x, shifted_cell_idx.y + 1, shifted_cell_idx.z, ny, nz));
    contribute(om_w0 * w1 * w2, particle_velocity, grid_vels, grid_vel_weights, index3D(shifted_cell_idx.x, shifted_cell_idx.y + 1, shifted_cell_idx.z + 1, ny, nz));
    contribute(w0 * om_w1 * om_w2, particle_velocity, grid_vels, grid_vel_weights, index3D(shifted_cell_idx.x + 1, shifted_cell_idx.y, shifted_cell_idx.z, ny, nz));
    contribute(w0 * om_w1 * w2, particle_velocity, grid_vels, grid_vel_weights, index3D(shifted_cell_idx.x + 1, shifted_cell_idx.y, shifted_cell_idx.z + 1, ny, nz));
    contribute(w0 * w1 * om_w2, particle_velocity, grid_vels, grid_vel_weights, index3D(shifted_cell_idx.x + 1, shifted_cell_idx.y + 1, shifted_cell_idx.z, ny, nz));
    contribute(w0 * w1 * w2, particle_velocity, grid_vels, grid_vel_weights, index3D(shifted_cell_idx.x + 1, shifted_cell_idx.y + 1, shifted_cell_idx.z + 1, ny, nz));
}

__device__ float InterpolateGridVelocities(float3 pos, float* grid_vels, float dx, StaggeredGrid* grid, int ny, int nz) {
    int3 cell_idx = grid->getCellIndex(pos);

    float fx = pos.x / dx - cell_idx.x;
    float fy = pos.y / dx - cell_idx.y;
    float fz = pos.z / dx - cell_idx.z;

    float w0 = fx;
    float om_w0 = 1.0 - w0;
    float w1 = fy;
    float om_w1 = 1.0 - w1;
    float w2 = fz;
    float om_w2 = 1.0 - w2;

    return om_w0 * om_w1 * om_w2 * grid_vels[index3D(cell_idx.x, cell_idx.y, cell_idx.z, ny, nz)] +
           om_w0 * om_w1 * w2 * grid_vels[index3D(cell_idx.x, cell_idx.y, cell_idx.z + 1, ny, nz)] +
           om_w0 * w1 * om_w2 * grid_vels[index3D(cell_idx.x, cell_idx.y + 1, cell_idx.z, ny, nz)] +
           om_w0 * w1 * w2 * grid_vels[index3D(cell_idx.x, cell_idx.y + 1, cell_idx.z + 1, ny, nz)] +
           w0 * om_w1 * om_w2 * grid_vels[index3D(cell_idx.x + 1, cell_idx.y, cell_idx.z, ny, nz)] +
           w0 * om_w1 * w2 * grid_vels[index3D(cell_idx.x + 1, cell_idx.y, cell_idx.z + 1, ny, nz)] +
           w0 * w1 * om_w2 * grid_vels[index3D(cell_idx.x + 1, cell_idx.y + 1, cell_idx.z, ny, nz)] +
           w0 * w1 * w2 * grid_vels[index3D(cell_idx.x + 1, cell_idx.y + 1, cell_idx.z + 1, ny, nz)];

}

__device__ float3 interpolateCurrentGridVelocities(float3 pos, StaggeredGrid* grid) {
    float3 p_lc = pos - grid->lc();

    float u_p = InterpolateGridVelocities(p_lc - grid->half_shift_yz(), grid->u(), grid->dx(), grid, grid->ny(), grid->nz());
    float v_p = InterpolateGridVelocities(p_lc - grid->half_shift_xz(), grid->v(), grid->dx(), grid, grid->ny() + 1, grid->nz());
    float w_p = InterpolateGridVelocities(p_lc - grid->half_shift_xy(), grid->w(), grid->dx(), grid, grid->ny(), grid->nz() + 1);

    return make_float3(u_p, v_p, w_p);
}

__device__ float3 interpolateOldGridVelocities(float3 pos, StaggeredGrid* grid) {
    float3 p_lc = pos - grid->lc();

    float u_p = InterpolateGridVelocities(p_lc - grid->half_shift_yz(), grid->fu(), grid->dx(), grid, grid->ny(), grid->nz());
    float v_p = InterpolateGridVelocities(p_lc - grid->half_shift_xz(), grid->fv(), grid->dx(), grid, grid->ny() + 1, grid->nz());
    float w_p = InterpolateGridVelocities(p_lc - grid->half_shift_xy(), grid->fw(), grid->dx(), grid, grid->ny(), grid->nz() + 1);

    return make_float3(u_p, v_p, w_p);
}

__device__ float3 ClampToNonSolidCells(float3 pos, StaggeredGrid* grid) {
    float3 clamped_pos = pos;
    float cell_plus_cushion = grid->dx() + kClampCushion;

    float min_x = grid->lc().x + cell_plus_cushion;
    if (pos.x <= min_x) {
        clamped_pos.x = min_x;
    }

    float max_x = grid->uc().x - cell_plus_cushion;
    if (pos.x >= max_x) {
        clamped_pos.x = max_x;
    }

    float min_y = grid->lc().y + cell_plus_cushion;
    if (pos.y <= min_y) {
        clamped_pos.y = min_y;
    }
    
    float max_y = grid->uc().y - cell_plus_cushion;
    if (pos.y >= max_y) {
        clamped_pos.y = max_y;
    }   

    float min_z = grid->lc().z + cell_plus_cushion;
    if (pos.z <= min_z) {
        clamped_pos.z = min_z;
    }   

    float max_z = grid->uc().z - cell_plus_cushion;
    if (pos.z >= max_z) {
        clamped_pos.z = max_z;
    }

    return clamped_pos;

}


// CUDA kernel for particle advection
__global__ void advectKernel(Particles* particles, int num_particles, float dt, StaggeredGrid* grid) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= num_particles) return;

    // get particle position
    float3 particle_pos = particles[idx].pos;

    // get particle velocity
    float3 vel = interpolateCurrentGridVelocities(particle_pos, grid);

    float3 new_pos = make_float3(
        particle_pos.x + dt * vel.x,
        particle_pos.y + dt * vel.y,
        particle_pos.z + dt * vel.z
    );

    new_pos = ClampToNonSolidCells(new_pos, grid);

    particles[idx].pos = new_pos;
}

// CUDA kernel for transferring particles to grid
__global__ void particlesToGridKernel(Particles* particles, int num_particles, StaggeredGrid* grid) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= num_particles) return;

    // get particle position    
    float3 particle_pos = particles[idx].pos;

    float3 p_lc = particle_pos - grid->lc();

    // get grid cell index
    int3 cell_idx = grid->getCellIndex(p_lc);

    // set cell label to FLUID
    grid->setCellLabel(cell_idx, MaterialType::FLUID);

    int ny = grid->ny();
    int nz = grid->nz();

    splat(p_lc - grid->half_shift_yz(), grid->dx(), particles[idx].vel.x, grid->u(), grid->fu(), grid, ny, nz);
    splat(p_lc - grid->half_shift_xz(), grid->dx(), particles[idx].vel.y, grid->v(), grid->fv(), grid, ny + 1, nz);
    splat(p_lc - grid->half_shift_xy(), grid->dx(), particles[idx].vel.z, grid->w(), grid->fw(), grid, ny, nz + 1);
}

// CUDA kernel for transferring grid to particles
__global__ void gridToParticleKernel(StaggeredGrid* grid, Particles* particles, int num_particles, float flip_ratio) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= num_particles) return;

    float3 old_velocity = interpolateOldGridVelocities(particles[idx].pos, grid);
    float3 new_velocity = interpolateCurrentGridVelocities(particles[idx].pos, grid);
    
    particles[idx].vel = flip_ratio * (particles[idx].vel - old_velocity) + new_velocity;
}



PIC_FLIP::PIC_FLIP(Particles* d_particles, int num_particles, StaggeredGrid* d_grid, SimulationParameters& params)
 : particles(d_particles), 
 num_particles(num_particles), 
 grid(d_grid), 
 pressure_solver(params.nx(), params.ny(), params.nz()) {
    cudaThreadsParticle = min(num_particles, 512);
    cudaBlocksParticle = int((num_particles + cudaThreadsParticle - 1) / cudaThreadsParticle);

    cudaBlockSize = dim3(8,8,8);
    cudaGridSize = dim3(
        (params.nx() + cudaBlockSize.x - 1) / cudaBlockSize.x,
        (params.ny() + cudaBlockSize.y - 1) / cudaBlockSize.y,
        (params.nz() + cudaBlockSize.z - 1) / cudaBlockSize.z
    );
}


void PIC_FLIP::advect(Particles* d_particles, int num_particles, float dt, StaggeredGrid* d_grid) {
    advectKernel<<<cudaBlocksParticle, cudaThreadsParticle>>>(d_particles, num_particles, dt, d_grid);
    cudaDeviceSynchronize();
}


void PIC_FLIP::particlesToGrid(Particles* d_particles, int num_particles, StaggeredGrid* d_grid) {
    ZeroOutVelocitiesKernel_u<<<cudaGridSize, cudaBlockSize>>>(d_grid);
    ZeroOutVelocitiesKernel_v<<<cudaGridSize, cudaBlockSize>>>(d_grid);
    ZeroOutVelocitiesKernel_w<<<cudaGridSize, cudaBlockSize>>>(d_grid);

    ClearCellLabelsKernel<<<cudaGridSize, cudaBlockSize>>>(d_grid);
    cudaDeviceSynchronize();

    particlesToGridKernel<<<cudaBlocksParticle, cudaThreadsParticle>>>(d_particles, num_particles, d_grid);
    cudaDeviceSynchronize();

    NormalizeAndStoreVelocitiesKernel_u<<<cudaGridSize, cudaBlockSize>>>(d_grid);
    NormalizeAndStoreVelocitiesKernel_v<<<cudaGridSize, cudaBlockSize>>>(d_grid);
    NormalizeAndStoreVelocitiesKernel_w<<<cudaGridSize, cudaBlockSize>>>(d_grid);
    cudaDeviceSynchronize();

    SetLRBoundaryVelocitiesKernel<<<cudaGridSize, cudaBlockSize>>>(d_grid);
    cudaDeviceSynchronize();

    SetABBoundaryVelocitiesKernel<<<cudaGridSize, cudaBlockSize>>>(d_grid);
    cudaDeviceSynchronize();

    SetFBBoundaryVelocitiesKernel<<<cudaGridSize, cudaBlockSize>>>(d_grid);
    cudaDeviceSynchronize();
}


void PIC_FLIP::applyGravity(StaggeredGrid* d_grid, float dt) {
    ApplyGravityKernel<<<cudaGridSize, cudaBlockSize>>>(d_grid, dt);
    cudaDeviceSynchronize();

    SetLRBoundaryVelocitiesKernel<<<cudaGridSize, cudaBlockSize>>>(d_grid);
    cudaDeviceSynchronize();

    SetABBoundaryVelocitiesKernel<<<cudaGridSize, cudaBlockSize>>>(d_grid);
    cudaDeviceSynchronize();

    SetFBBoundaryVelocitiesKernel<<<cudaGridSize, cudaBlockSize>>>(d_grid);
    cudaDeviceSynchronize();

}

void PIC_FLIP::projectPressure(StaggeredGrid* d_grid) {
    MakeNeighborMaterialInfoKernel<<<cudaGridSize, cudaBlockSize>>>(d_grid);
    cudaDeviceSynchronize();
    
    pressure_solver.ProjectPressure(d_grid);
    cudaDeviceSynchronize();

    SubtractPressureGradientFromVelocityKernel<<<cudaGridSize, cudaBlockSize>>>(d_grid);
    cudaDeviceSynchronize();

}

void PIC_FLIP::gridToParticle(StaggeredGrid* d_grid, Particles* d_particles, int num_particles, float flip_ratio) {
    gridToParticleKernel<<<cudaBlocksParticle, cudaThreadsParticle>>>(d_grid, d_particles, num_particles, flip_ratio);
    cudaDeviceSynchronize();
}

PIC_FLIP::~PIC_FLIP() {}