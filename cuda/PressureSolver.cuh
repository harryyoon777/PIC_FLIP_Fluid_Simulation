#ifndef PRESSURE_SOLVER_CUH_
#define PRESSURE_SOLVER_CUH_

# include "../MaterialType.h"
# include "StaggeredGrid.cuh"


class PressureSolver {
    public:
        PressureSolver(std::size_t nx, std::size_t ny, std::size_t nz);
        ~PressureSolver();

        void ProjectPressure(StaggeredGrid* grid);
        void PlusEquals(float* p, float alpha, float* d);

    private:
        std::size_t nx_;
        std::size_t ny_;
        std::size_t nz_;
        float* r_;
        float* d_;
        float* q_;

        dim3 cudaBlockSize = dim3(8,8,8);
        dim3 cudaGridSize;
};

// __global__ void ZeroOutPressureKernel(StaggeredGrid* grid);

#endif
