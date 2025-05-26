#ifndef PARTICLES_CUH_
#define PARTICLES_CUH_

#include <cuda_runtime.h>
#include <vector>
#include <string>

struct alignas(16) Particles {
	float3 pos;
	float3 vel;
};

// Copy particles from CPU memory to GPU memory
Particles* ReadParticlesToGPU(const std::string& input_file, int* num_particles);

// Copy particles from GPU memory to CPU memory
std::vector<Particles> ReadParticlesFromGPU(Particles* d_particles, int num_particles);

Particles* GenerateParticlesToGPU(
	int num_particles_to_generate, 
    float3 generation_min_bounds, 
    float3 generation_max_bounds, 
    int* generated_count, 
    float3 initial_velocity = make_float3(0.0f, 0.0f, 0.0f)
);

// Free GPU memory
void FreeParticlesGPU(Particles* d_particles);

#endif  // PARTICLES_CUH_
