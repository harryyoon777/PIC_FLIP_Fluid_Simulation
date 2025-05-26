#include "Particles.cuh"
#include <fstream>
#include <iostream>
#include <sstream>
#include <random>

namespace {

std::string ReadLine(std::ifstream* in) {
    std::string line;
    std::getline(*in, line);
    return line;
}

int ReadNumParticles(std::ifstream* in) {
    int num_particles = -1;
    std::string line = ReadLine(in);
    std::istringstream ss(line);
    if (!(ss >> num_particles)) {
        return -1;
    }
    return num_particles;
}

float3 ReadVector3(std::istringstream* line_ss) {
    float x, y, z;
    (*line_ss) >> x >> y >> z;
    return make_float3(x, y, z);
}

Particles ReadParticles(const std::string& line) {
    std::istringstream ss(line);
    Particles p;
    p.pos = ReadVector3(&ss);
    p.vel = ReadVector3(&ss);
    return p;
}

}  // namespace

// Copy particles from CPU memory to GPU memory
Particles* ReadParticlesToGPU(const std::string& input_file, int* num_particles) {
    std::ifstream in(input_file.c_str(), std::ios::in);
    int alleged_num_particles = ReadNumParticles(&in);

    std::vector<Particles> cpu_particles;
    std::string line;
    while (std::getline(in, line)) {
        cpu_particles.push_back(ReadParticles(line));
    }
    
    if (cpu_particles.size() != alleged_num_particles) {
        std::cout << "Warning: Number of particles in file (" << cpu_particles.size() 
                  << ") does not match the specified number (" << alleged_num_particles 
                  << ")" << std::endl;
    }
    
    std::cout << "Read " << cpu_particles.size() << " particles." << std::endl;

    // Allocate GPU memory
    Particles* d_particles;
    cudaMalloc(&d_particles, cpu_particles.size() * sizeof(Particles));
    
    // Copy data from CPU to GPU
    cudaMemcpy(d_particles, cpu_particles.data(), 
               cpu_particles.size() * sizeof(Particles), 
               cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    *num_particles = cpu_particles.size();
    in.close();

    return d_particles;
}

// Copy particles from GPU memory to CPU memory
std::vector<Particles> ReadParticlesFromGPU(Particles* d_particles, int num_particles) {
    std::vector<Particles> cpu_particles(num_particles);
    cudaDeviceSynchronize();
    cudaMemcpy(cpu_particles.data(), d_particles, 
               num_particles * sizeof(Particles), 
               cudaMemcpyDeviceToHost);
    return cpu_particles;
}

Particles* GenerateParticlesToGPU(
	int num_particles_to_generate, 
    float3 generation_min_bounds, 
    float3 generation_max_bounds, 
    int* generated_count, 
    float3 initial_velocity) {

    std::vector<Particles> cpu_particles;
    cpu_particles.reserve(num_particles_to_generate);

    std::random_device rd;
    std::mt19937 gen(rd()); 
    std::uniform_real_distribution<float> distrib_x(generation_min_bounds.x, generation_max_bounds.x);
    std::uniform_real_distribution<float> distrib_y(generation_min_bounds.y, generation_max_bounds.y);
    std::uniform_real_distribution<float> distrib_z(generation_min_bounds.z, generation_max_bounds.z);

    std::cout << "Generating " << num_particles_to_generate << " particles within bounding box: " << std::endl;
    std::cout << "  Min: (" << generation_min_bounds.x << ", " << generation_min_bounds.y << ", " << generation_min_bounds.z << ")" << std::endl;
    std::cout << "  Max: (" << generation_max_bounds.x << ", " << generation_max_bounds.y << ", " << generation_max_bounds.z << ")" << std::endl;

    for (int i = 0; i < num_particles_to_generate; ++i) {
        Particles p;
        p.pos.x = distrib_x(gen);
        p.pos.y = distrib_y(gen);
        p.pos.z = distrib_z(gen);
        p.vel = initial_velocity; 
        cpu_particles.push_back(p);
    }

    Particles* d_particles_ptr;
    cudaError_t err_malloc = cudaMalloc(&d_particles_ptr, cpu_particles.size() * sizeof(Particles));
    if (err_malloc != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device memory for generated particles: %s\n", cudaGetErrorString(err_malloc));
        *generated_count = 0;
        return nullptr;
    }
    
    cudaError_t err_memcpy = cudaMemcpy(d_particles_ptr, cpu_particles.data(), 
                                       cpu_particles.size() * sizeof(Particles), 
                                       cudaMemcpyHostToDevice);
    if (err_memcpy != cudaSuccess) {
        fprintf(stderr, "Failed to copy generated particles to device: %s\n", cudaGetErrorString(err_memcpy));
        cudaFree(d_particles_ptr);
        *generated_count = 0;
        return nullptr;
    }
    
    *generated_count = cpu_particles.size();
    std::cout << "Successfully generated and transferred " << *generated_count << " particles to GPU." << std::endl;
    
    return d_particles_ptr;
}

// Free GPU memory
void FreeParticlesGPU(Particles* d_particles) {
    cudaFree(d_particles);
} 