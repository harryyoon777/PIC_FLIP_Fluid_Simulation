#include <fstream>
#include <iostream>
#include <vector>
#include <sys/stat.h>
#include <iomanip>
#include <map>
#include <string>
#include <chrono>
#include <functional>

#include "cpu/Particle.h"
#include "SimulationParameters.h"
#include "cpu/StaggeredGrid.h"

namespace {

// Create directory if it doesn't exist
void CreateDirectoryIfNotExists(const char* dir_path) {
  struct stat st = {0};
  if (stat(dir_path, &st) == -1) {
    mkdir(dir_path, 0700);
    std::cout << "Created directory: " << dir_path << std::endl;
  }
}

void Write3d(const Eigen::Vector3d& vec, std::ofstream* out) {
  (*out) << vec[0] << " " << vec[1] << " " << vec[2];
}

// Writes the position and velocity of each particle to the file with the
// specified |output_file_name|.
void WriteParticles(const char* output_file_name,
                    const std::vector<Particle>& particles) {
  std::ofstream out(output_file_name, std::ios::out);
  out << particles.size() << std::endl;
  for (std::vector<Particle>::const_iterator p = particles.begin();
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


// Helper function to measure CPU time for a given function call
double measure_cpu_time_ms(std::function<void()> func_to_measure) {
    auto start_time = std::chrono::high_resolution_clock::now();
    func_to_measure();
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration_ms = end_time - start_time;
    return duration_ms.count();
}


// Run a physics-based fluid simulation and print the resulting fluid particle
// positions at each time step to files.
int main(int argc, char** argv) {
  SimulationParameters params = ReadSimulationParameters(argc, argv);

  // Create outputs directory if it doesn't exist
  CreateDirectoryIfNotExists("outputs_cpu");

  StaggeredGrid grid(params.nx(), params.ny(), params.nz(), params.lc(),
                     params.dx());

  std::vector<Particle> particles;

  const std::string& particle_source_type = params.particle_source_type();
  if (particle_source_type == "file") {
    const std::string& particle_file = params.particle_file_path();
    if (particle_file.empty()) {
        std::cerr << "Error: Particle source type is 'file' but no file_path is specified in JSON." << std::endl;
        return EXIT_FAILURE;
    }
    std::cout << "Loading particles from file: " << particle_file << std::endl;
    particles = ReadParticles(params.particle_file_path());
    if (particles.empty()) {
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
      particles = GenerateParticles(
          bs.num_particles,
          bs.min_bounds,      // This should be Eigen::Vector3d
          bs.max_bounds,      // This should be Eigen::Vector3d
          bs.initial_velocity // This should be Eigen::Vector3d
      );

      // Check if particles were generated successfully (e.g., if 0 were requested, it's fine)
      // GenerateParticles itself prints errors if bounds are invalid.
      // If num_particles requested was > 0 but particles.empty() after call, it might indicate an issue
      // (though GenerateParticles should return empty for invalid bounds rather than nullptr)
      if (particles.empty()) {
          std::cerr << "Error: Failed to generate particles in block." << std::endl;
          return EXIT_FAILURE;
      }
  }

  grid.ParticlesToGrid(particles);

  char output_file_name[100];
  int frame = 0;
  int step = 0;

  std::map<std::string, double> accumulated_step_times_total_cpu; 
  std::map<std::string, double> accumulated_step_times_current_frame_cpu; 
  
  long long total_steps_processed = 0; 
  int total_frames_outputted = 0;
  
  auto simulation_total_cpu_start_time = std::chrono::high_resolution_clock::now(); 

  std::cout << "Starting CPU simulation..." << std::endl;



  const double kFirstPositiveFrameTime = 1.0 / 30.0 - 0.0001;
  for (double time = 0.0, frame_time = -1.0; time < params.duration_seconds();
       time += params.dt_seconds(), frame_time -= params.dt_seconds()) {

    step++;
    total_steps_processed++;

    double advect_cpu_time_ms = 0.0;
    double p2g_cpu_time_ms = 0.0;
    double gravity_cpu_time_ms = 0.0;
    double pressure_cpu_time_ms = 0.0;
    double g2p_cpu_time_ms = 0.0;

    if (frame_time < 0.0) {
      sprintf(output_file_name, "outputs_cpu/fluid.%03d.part", frame);
      WriteParticles(output_file_name, particles);

      // // Show CPU compute times for the current frame
      // if (total_frames_outputted > 0) { 
      //     std::cout << "\n--- Frame " << (frame -1) << " CPU Compute Times (ms) ---" << std::endl;
      //     for (const auto& pair_cpu : accumulated_step_times_current_frame_cpu) {
      //         std::cout << "  " << pair_cpu.first << ": " << std::fixed << std::setprecision(3) << pair_cpu.second << std::endl;
      //     }
      // }

      accumulated_step_times_current_frame_cpu.clear();
      
      frame_time = kFirstPositiveFrameTime;
      frame++; 
      total_frames_outputted++;
    }

    advect_cpu_time_ms = measure_cpu_time_ms([&]() {
        for (std::vector<Particle>::iterator p = particles.begin();
             p != particles.end(); p++) {
          p->pos = grid.Advect(p->pos, params.dt_seconds());
        }
    });

    p2g_cpu_time_ms = measure_cpu_time_ms([&]() {
        grid.ParticlesToGrid(particles);
    });

    gravity_cpu_time_ms = measure_cpu_time_ms([&]() {
        grid.ApplyGravity(params.dt_seconds(), frame, step);
    });

    pressure_cpu_time_ms = measure_cpu_time_ms([&]() {
        grid.ProjectPressure(frame, step);
    });

    g2p_cpu_time_ms = measure_cpu_time_ms([&]() {
        for (std::vector<Particle>::iterator p = particles.begin();
             p != particles.end(); p++) {
          p->vel = grid.GridToParticle(params.flip_ratio(), *p);
        }
    });
    
    accumulated_step_times_total_cpu["Advect"] += advect_cpu_time_ms;
    accumulated_step_times_total_cpu["P2G"] += p2g_cpu_time_ms;
    accumulated_step_times_total_cpu["Gravity"] += gravity_cpu_time_ms;
    accumulated_step_times_total_cpu["Pressure"] += pressure_cpu_time_ms;
    accumulated_step_times_total_cpu["G2P"] += g2p_cpu_time_ms;

    accumulated_step_times_current_frame_cpu["Advect"] += advect_cpu_time_ms;
    accumulated_step_times_current_frame_cpu["P2G"] += p2g_cpu_time_ms;
    accumulated_step_times_current_frame_cpu["Gravity"] += gravity_cpu_time_ms;
    accumulated_step_times_current_frame_cpu["Pressure"] += pressure_cpu_time_ms;
    accumulated_step_times_current_frame_cpu["G2P"] += g2p_cpu_time_ms;
  } // End Simulation

  // // Show CPU compute times for the current frame
  // if (!accumulated_step_times_current_frame_cpu.empty() && total_frames_outputted > 0) {
  //       std::cout << "\n--- Frame " << (frame -1) << " CPU Compute Times (ms) ---" << std::endl;
  //       for (const auto& pair_cpu : accumulated_step_times_current_frame_cpu) {
  //           std::cout << "  " << pair_cpu.first << ": " << std::fixed << std::setprecision(3) << pair_cpu.second << std::endl;
  //       }
  // }

  auto simulation_total_cpu_end_time = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> simulation_total_cpu_duration_ms = simulation_total_cpu_end_time - simulation_total_cpu_start_time;

  std::cout << "\n\n=====================================================" << std::endl;
  std::cout << "         CPU Compute Performance Summary" << std::endl;
  std::cout << "=====================================================" << std::endl;
  std::cout << std::fixed << std::setprecision(3);

  std::cout << "Total Simulation CPU Compute Time: " << simulation_total_cpu_duration_ms.count() << " ms" << std::endl;
  std::cout << "Total Timesteps (Steps in code): " << total_steps_processed << std::endl;
  std::cout << "Total Frames Output: " << total_frames_outputted << std::endl;

  if (total_steps_processed > 0) {
    std::cout << "\n--- Average CPU Compute Time Per Timestep (ms) ---" << std::endl;
    for (const auto& pair : accumulated_step_times_total_cpu) {
        std::cout << "  " << pair.first << ": " << pair.second / total_steps_processed << std::endl;
    }
  }
  
  if (simulation_total_cpu_duration_ms.count() > 0 && total_frames_outputted > 0) {
    double fps = static_cast<double>(total_frames_outputted) / (simulation_total_cpu_duration_ms.count() / 1000.0);
    std::cout << "\nAverage FPS (based on total CPU compute time and output frames): " << std::fixed << std::setprecision(2) << fps << std::endl;
  } else {
    std::cout << "\nAverage FPS: N/A (not enough data)" << std::endl;
  }
  std::cout << "=====================================================" << std::endl;

  return EXIT_SUCCESS;
}
