#ifndef PIC_FLIP_CUH_
#define PIC_FLIP_CUH_

#include "Particles.cuh"
#include "StaggeredGrid.cuh"
#include "PressureSolver.cuh"
#include "../SimulationParameters.h"


class PIC_FLIP {
public:
    PIC_FLIP(Particles* d_particles, int num_particles, StaggeredGrid* d_grid, SimulationParameters& params);
    ~PIC_FLIP();

    void advect(Particles* d_particles, int num_particles, float dt, StaggeredGrid* d_grid);
    void particlesToGrid(Particles* d_particles, int num_particles, StaggeredGrid* d_grid);
    void applyGravity(StaggeredGrid* d_grid, float dt);
    void projectPressure(StaggeredGrid* d_grid);
    void gridToParticle(StaggeredGrid* d_grid, Particles* d_particles, int num_particles, float flip_ratio);

private:
    Particles* particles;
    int num_particles;
    StaggeredGrid* grid;
    PressureSolver pressure_solver;

    int cudaThreadsParticle;
    int cudaBlocksParticle;

    dim3 cudaBlockSize;
    dim3 cudaGridSize;

    float* d_u_p = nullptr;
    float* d_v_p = nullptr;
    float* d_w_p = nullptr;
};


#endif  // PIC_FLIP_CUH_
