#include <fstream>
#include <iostream>
#include <vector>
#include <sys/stat.h>
#include <iomanip>
#include <map>
#include <string>
#include <chrono>

#include <cuda_runtime.h> 

#include "SimulationParameters.h"
#include "cuda/Particles.cuh"
#include "cuda/StaggeredGrid.cuh"
#include "cuda/PIC_FLIP.cuh"


namespace {

  // Create directory if it doesn't exist
  void CreateDirectoryIfNotExists(const char* dir_path) {
    struct stat st = {0};
    if (stat(dir_path, &st) == -1) {
      mkdir(dir_path, 0700);
      std::cout << "Created directory: " << dir_path << std::endl;
    }
  }

  void Write3d(const float3& vec, std::ofstream* out) {
    (*out) << vec.x << " " << vec.y << " " << vec.z;
  }

  // Writes the position and velocity of each particle to the file with the
  // specified |output_file_name|.
  void WriteParticles(const char* output_file_name,
                      const std::vector<Particles>& particles) {
    std::ofstream out(output_file_name, std::ios::out);
    out << particles.size() << std::endl;
    for (std::vector<Particles>::const_iterator p = particles.begin();
        p != particles.end(); p++) {
      Write3d(p->pos, &out);
      out << " ";
      Write3d(p->vel, &out);
      out << std::endl;
    }
    out.close();
    std::cout << "Output file " << output_file_name << " saved." << std::endl;
  }
}  // namespace


// Helper function to measure GPU time for a given function call
float measure_gpu_time(std::function<void()> func_to_measure) {
    // Create CUDA events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // === Start Measurement ===
    cudaEventRecord(start, 0);

    func_to_measure();
    cudaDeviceSynchronize(); 

    // === End Measurement ===
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    // Calculate execution time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Destroy events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return milliseconds;
}


// Run a physics-based fluid simulation and print the resulting fluid particle
// positions at each time step to files.
int main(int argc, char** argv) {
  SimulationParameters params = ReadSimulationParameters(argc, argv);

  // Create outputs directory if it doesn't exist
  CreateDirectoryIfNotExists("outputs_cuda");

  StaggeredGrid h_grid(params.nx(), params.ny(), params.nz(), 
                    make_float3(params.lc().x(), params.lc().y(), params.lc().z()),
                    params.dx());

  // Read particles and transfer to GPU
  int num_particles;
  Particles* d_particles = nullptr;

  const std::string& particle_source_type = params.particle_source_type();
  if (particle_source_type == "file") {
    const std::string& particle_file = params.particle_file_path();
    if (particle_file.empty()) {
        std::cerr << "Error: Particle source type is 'file' but no file_path is specified in JSON." << std::endl;
        return EXIT_FAILURE;
    }
    std::cout << "Loading particles from file: " << particle_file << std::endl;
    d_particles = ReadParticlesToGPU(particle_file, &num_particles);
    if (!d_particles) {
        std::cerr << "Error: Failed to read particles from file: " << particle_file << std::endl;
        return EXIT_FAILURE;
    }
  } else if (particle_source_type == "generate_block") {
      const ParticleBlockGenerationSettings& bs = params.block_generation_settings();
      if (!bs.loaded) {
          std::cerr << "Error: Particle source type is 'generate_block' but settings are not loaded/specified correctly in JSON." << std::endl;
          return EXIT_FAILURE;
      }
      std::cout << "Generating particles in a block." << std::endl;
      // Convert Eigen::Vector3d from settings to float3 for GenerateParticlesToGPU
      float3 min_b = make_float3(static_cast<float>(bs.min_bounds[0]), static_cast<float>(bs.min_bounds[1]), static_cast<float>(bs.min_bounds[2]));
      float3 max_b = make_float3(static_cast<float>(bs.max_bounds[0]), static_cast<float>(bs.max_bounds[1]), static_cast<float>(bs.max_bounds[2]));
      float3 init_v = make_float3(static_cast<float>(bs.initial_velocity[0]), static_cast<float>(bs.initial_velocity[1]), static_cast<float>(bs.initial_velocity[2]));
      d_particles = GenerateParticlesToGPU(bs.num_particles, min_b, max_b, &num_particles, init_v);
      if (!d_particles) {
          std::cerr << "Error: Failed to generate particles in block." << std::endl;
          return EXIT_FAILURE;
      }
  }

  // Create CPU vector for output
  std::vector<Particles> h_particles(num_particles);
  
  // Transfer grid to GPU
  StaggeredGrid* d_grid;
  cudaMalloc(&d_grid, sizeof(StaggeredGrid));
  cudaMemcpy(d_grid, &h_grid, sizeof(StaggeredGrid), cudaMemcpyHostToDevice);

  // Transfer particles to grid
  PIC_FLIP pic_flip(d_particles, num_particles, d_grid, params);
  pic_flip.particlesToGrid(d_particles, num_particles, d_grid);

  char output_file_name[100];
  int frame = 0; 
  int step = 0;  

  cudaEvent_t total_simulation_gpu_start, total_simulation_gpu_stop;
  cudaEventCreate(&total_simulation_gpu_start);
  cudaEventCreate(&total_simulation_gpu_stop);

  std::map<std::string, double> accumulated_gpu_step_times_total; 
  std::map<std::string, double> accumulated_gpu_step_times_current_frame; 
  
  long long total_steps_processed = 0; 
  int total_frames_outputted = 0;
  std::cout << "Starting GPU simulation..." << std::endl;
  cudaEventRecord(total_simulation_gpu_start, 0);  



  const double kFirstPositiveFrameTime = 1.0 / 30.0 - 0.0001;
  for (double time = 0.0, frame_time = -1.0; time < params.duration_seconds();
       time += params.dt_seconds(), frame_time -= params.dt_seconds()) {

    step++; 
    total_steps_processed++;

    float advect_time_ms = 0.0f;
    float p2g_time_ms = 0.0f;
    float gravity_time_ms = 0.0f;
    float pressure_time_ms = 0.0f;
    float g2p_time_ms = 0.0f;
    double read_particles_cpu_time_ms = 0.0;

    if (frame_time < 0.0) {
      // Copy particles from GPU to CPU for output
      h_particles = ReadParticlesFromGPU(d_particles, num_particles);
      sprintf(output_file_name, "outputs_cuda/fluid.%03d.part", frame);
      WriteParticles(output_file_name, h_particles);

      // // Show GPU compute times for the current frame
      // if (total_frames_outputted > 0) { 
      //     std::cout << "\n--- Frame " << (frame -1) << " GPU Compute Times (ms) ---" << std::endl;
      //     for (const auto& pair : accumulated_gpu_step_times_current_frame) {
      //         std::cout << "  " << pair.first << ": " << std::fixed << std::setprecision(3) << pair.second << std::endl;
      //     }
      // }

      accumulated_gpu_step_times_current_frame.clear(); 

      frame_time = kFirstPositiveFrameTime;
      frame++;
      total_frames_outputted++;
    }
    
    advect_time_ms = measure_gpu_time([&]() {
        pic_flip.advect(d_particles, num_particles, params.dt_seconds(), d_grid);
    });

    p2g_time_ms = measure_gpu_time([&]() {
        pic_flip.particlesToGrid(d_particles, num_particles, d_grid);
    });

    gravity_time_ms = measure_gpu_time([&]() {
        pic_flip.applyGravity(d_grid, params.dt_seconds()); 
    });

    pressure_time_ms = measure_gpu_time([&]() {
        pic_flip.projectPressure(d_grid);
    });

    g2p_time_ms = measure_gpu_time([&]() {
        pic_flip.gridToParticle(d_grid, d_particles, num_particles, params.flip_ratio());
    });
    
    // Accumulate GPU time
    accumulated_gpu_step_times_total["Advect"] += advect_time_ms;
    accumulated_gpu_step_times_total["P2G"] += p2g_time_ms;
    accumulated_gpu_step_times_total["Gravity"] += gravity_time_ms;
    accumulated_gpu_step_times_total["Pressure"] += pressure_time_ms;
    accumulated_gpu_step_times_total["G2P"] += g2p_time_ms;

    accumulated_gpu_step_times_current_frame["Advect"] += advect_time_ms;
    accumulated_gpu_step_times_current_frame["P2G"] += p2g_time_ms;
    accumulated_gpu_step_times_current_frame["Gravity"] += gravity_time_ms;
    accumulated_gpu_step_times_current_frame["Pressure"] += pressure_time_ms;
    accumulated_gpu_step_times_current_frame["G2P"] += g2p_time_ms;

  } // End Simulation

  // // Show GPU compute times for the current frame
  // if (!accumulated_gpu_step_times_current_frame.empty() && total_frames_outputted > 0) {
  //     std::cout << "\n--- Frame " << (frame -1) << " GPU Compute Times (ms) ---" << std::endl;
  //     for (const auto& pair : accumulated_gpu_step_times_current_frame) {
  //         std::cout << "  " << pair.first << ": " << std::fixed << std::setprecision(3) << pair.second << std::endl;
  //     }
  // }

  cudaDeviceSynchronize(); 
  cudaEventRecord(total_simulation_gpu_stop, 0); 
  cudaEventSynchronize(total_simulation_gpu_stop);

  float total_simulation_gpu_milliseconds = 0;
  cudaEventElapsedTime(&total_simulation_gpu_milliseconds, total_simulation_gpu_start, total_simulation_gpu_stop);

  std::cout << "\n\n=====================================================" << std::endl;
  std::cout << "         GPU Compute Performance Summary" << std::endl;
  std::cout << "=====================================================" << std::endl;
  std::cout << std::fixed << std::setprecision(3);

  std::cout << "Total Simulation GPU Compute Time: " << total_simulation_gpu_milliseconds << " ms" << std::endl;
  std::cout << "Total Timesteps (Steps in code): " << total_steps_processed << std::endl;
  std::cout << "Total Frames Output: " << total_frames_outputted << std::endl;

  if (total_steps_processed > 0) {
    std::cout << "\n--- Average GPU Compute Time Per Timestep (ms) ---" << std::endl;
    for (const auto& pair : accumulated_gpu_step_times_total) {
        std::cout << "  " << pair.first << ": " << pair.second / total_steps_processed << std::endl;
    }
  }
  
  if (total_simulation_gpu_milliseconds > 0 && total_frames_outputted > 0) {
    double fps_gpu_based = static_cast<double>(total_frames_outputted) / (total_simulation_gpu_milliseconds / 1000.0);
    std::cout << "\nAverage FPS (based on GPU compute time and output frames): " << std::fixed << std::setprecision(2) << fps_gpu_based << std::endl;
  } else {
    std::cout << "\nAverage FPS: N/A (not enough data)" << std::endl;
  }
  std::cout << "=====================================================" << std::endl;

  cudaEventDestroy(total_simulation_gpu_start);
  cudaEventDestroy(total_simulation_gpu_stop);


  // Free GPU memory
  FreeParticlesGPU(d_particles);
  cudaFree(d_grid);
  return EXIT_SUCCESS;
} 