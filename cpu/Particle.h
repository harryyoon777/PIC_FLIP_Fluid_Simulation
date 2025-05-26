#ifndef PARTICLE_H_
#define PARTICLE_H_

#include <Eigen/Dense>
#include <vector>

struct Particle {
  Eigen::Vector3d pos;
  Eigen::Vector3d vel;
};

// Returns an array of Particles read from the file with relative path
// |input_file|.
std::vector<Particle> ReadParticles(const std::string& input_file);

// Generates particles within a specified bounding box with initial velocity
std::vector<Particle> GenerateParticles(
    int num_particles_to_generate,
    const Eigen::Vector3d& generation_min_bounds,
    const Eigen::Vector3d& generation_max_bounds,
    const Eigen::Vector3d& initial_velocity);

#endif  // PARTICLE_H_
