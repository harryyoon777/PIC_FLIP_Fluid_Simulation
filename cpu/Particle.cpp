#include "Particle.h"

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

Eigen::Vector3d ReadVector3d(std::istringstream* line_ss) {
  Eigen::Vector3d vec;
  (*line_ss) >> vec[0] >> vec[1] >> vec[2];
  return vec;
}

Particle ReadParticle(const std::string& line) {
  std::istringstream ss(line);
  return Particle{.pos = ReadVector3d(&ss), .vel = ReadVector3d(&ss)};
}

}  // namespace

std::vector<Particle> ReadParticles(const std::string& input_file) {
  std::ifstream in(input_file.c_str(), std::ios::in);

  int alleged_num_particles = ReadNumParticles(&in);

  std::vector<Particle> particles;

  std::string line;
  while (std::getline(in, line)) {
    particles.push_back(ReadParticle(line));
  }
  
  if (particles.size() != static_cast<size_t>(alleged_num_particles)) {
    std::cout << "Warning: Number of particles in file (" << particles.size() 
              << ") does not match the specified number (" << alleged_num_particles 
              << ")" << std::endl;
  }
  
  std::cout << "Read " << particles.size() << " particles." << std::endl;

  in.close();

  return particles;
}

// Generate particles within a specified bounding box with initial velocity
std::vector<Particle> GenerateParticles(
    int num_particles_to_generate,
    const Eigen::Vector3d& generation_min_bounds,
    const Eigen::Vector3d& generation_max_bounds,
    const Eigen::Vector3d& initial_velocity) {
    
    std::vector<Particle> generated_particles;
    generated_particles.reserve(num_particles_to_generate);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> distrib_x(generation_min_bounds.x(), generation_max_bounds.x());
    std::uniform_real_distribution<double> distrib_y(generation_min_bounds.y(), generation_max_bounds.y());
    std::uniform_real_distribution<double> distrib_z(generation_min_bounds.z(), generation_max_bounds.z());

    std::cout << "Generating " << num_particles_to_generate << " particles within bounding box: " << std::endl;
    std::cout << "  Min: [" << generation_min_bounds.transpose() << "]" << std::endl;
    std::cout << "  Max: [" << generation_max_bounds.transpose() << "]" << std::endl;
    std::cout << "  Initial Velocity: [" << initial_velocity.transpose() << "]" << std::endl;


    for (int i = 0; i < num_particles_to_generate; ++i) {
        Particle p;
        p.pos.x() = distrib_x(gen);
        p.pos.y() = distrib_y(gen);
        p.pos.z() = distrib_z(gen);
        p.vel = initial_velocity;
        generated_particles.push_back(p);
    }

    std::cout << "Successfully generated " << generated_particles.size() << " particles (CPU)." << std::endl;
    
    return generated_particles;
}
