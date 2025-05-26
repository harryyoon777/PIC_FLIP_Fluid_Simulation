#include "SimulationParameters.h"

#include <cassert>
#include <fstream>
#include <iostream>

#include "json/json.h"


SimulationParameters::SimulationParameters(
    double dt_seconds, double duration_seconds, double density,
    const Eigen::Matrix<std::size_t, 3, 1>& dimensions, double dx,
    const Eigen::Vector3d& lc, double flip_ratio, const std::string& output_file_name_pattern,
    const std::string& particle_source_type, const std::string& particle_file_path,
    const ParticleBlockGenerationSettings& block_settings)
    : dt_seconds_(dt_seconds),
      duration_seconds_(duration_seconds),
      density_(density),
      dimensions_(dimensions),
      dx_(dx),
      lc_(lc),
      flip_ratio_(flip_ratio),
      // input_file_(input_file), // Replaced
      output_file_name_pattern_(output_file_name_pattern),
      particle_source_type_(particle_source_type),
      particle_file_path_(particle_file_path),
      block_generation_settings_(block_settings) {}

SimulationParameters::SimulationParameters(const SimulationParameters& other)
    : dt_seconds_(other.dt_seconds_),
      duration_seconds_(other.duration_seconds_),
      density_(other.density_),
      dimensions_(other.dimensions_),
      dx_(other.dx_),
      lc_(other.lc_),
      flip_ratio_(other.flip_ratio_),
      // input_file_(other.input_file_), // Replaced
      output_file_name_pattern_(other.output_file_name_pattern_),
      particle_source_type_(other.particle_source_type_),
      particle_file_path_(other.particle_file_path_),
      block_generation_settings_(other.block_generation_settings_) {
  assert(false);
}

SimulationParameters SimulationParameters::CreateFromJsonFile(
    const std::string& input_file_path) {
  std::ifstream in(input_file_path, std::ios::in);

  Json::Reader json_reader;
  Json::Value json_root;

  bool read_succeeded = json_reader.parse(in, json_root);
  assert(read_succeeded);

  double dt_seconds = json_root.get("dt", 1.0 / 300.0).asDouble();
  double duration_seconds = json_root.get("total_time", 1.0).asDouble();
  double density = json_root.get("density", 1.0 / 300.0).asDouble();
  double flip_ratio = json_root.get("flipRatio", 0.95).asDouble();

  int nx = json_root["res"][0].asInt();
  int ny = json_root["res"][1].asInt();
  int nz = json_root["res"][2].asInt();
  Eigen::Matrix<std::size_t, 3, 1> dimensions;
  dimensions << nx, ny, nz;

  double dx = json_root["h"].asDouble();

  double lc_x = json_root["lc"][0].asDouble();
  double lc_y = json_root["lc"][1].asDouble();
  double lc_z = json_root["lc"][2].asDouble();
  Eigen::Vector3d lc;
  lc << lc_x, lc_y, lc_z;

  // std::string input_file = json_root["particles"].asString();
  std::string output_file_name_pattern =
      json_root.get("output_fname", std::string("output.%04d.txt")).asString();
  
  // --- Calculate grid upper corner for validation ---
  Eigen::Vector3d uc;
  uc[0] = lc[0] + static_cast<double>(dimensions[0]) * dx;
  uc[1] = lc[1] + static_cast<double>(dimensions[1]) * dx;
  uc[2] = lc[2] + static_cast<double>(dimensions[2]) * dx;


  // --- New Particle Settings Parsing ---
  std::string particle_source_type = "file"; // Default
  std::string particle_file_path_val = ""; // Default
  ParticleBlockGenerationSettings block_settings;

  if (json_root.isMember("particle_settings")) {
    const Json::Value& particle_json = json_root["particle_settings"];
    particle_source_type = particle_json.get("source_type", "file").asString();

    if (particle_source_type == "file") {
      particle_file_path_val = particle_json.get("file_path", "").asString();
      if (particle_file_path_val.empty() && json_root.isMember("particles")) {
          std::cout << "Warning: 'particle_settings.file_path' not found or empty, "
                    << "falling back to root 'particles' key for input file." << std::endl;
          particle_file_path_val = json_root.get("particles", "").asString();
      }
      if (particle_file_path_val.empty()) {
          std::cerr << "Error: 'particle_source_type' is 'file' but 'particle_settings.file_path' (or root 'particles') is not specified." << std::endl;
          assert(false && "Particle file path missing for source_type 'file'");
      }
    } else if (particle_source_type == "generate_block") {
      if (particle_json.isMember("generation_block")) {
        const Json::Value& block_json = particle_json["generation_block"];
        block_settings.num_particles = block_json.get("num_particles", 0).asInt();
        if (block_json.isMember("min_bounds") && block_json["min_bounds"].isObject()) {
            block_settings.min_bounds[0] = block_json["min_bounds"].get("x", 0.0).asDouble();
            block_settings.min_bounds[1] = block_json["min_bounds"].get("y", 0.0).asDouble();
            block_settings.min_bounds[2] = block_json["min_bounds"].get("z", 0.0).asDouble();
        }
        if (block_json.isMember("max_bounds") && block_json["max_bounds"].isObject()) {
            block_settings.max_bounds[0] = block_json["max_bounds"].get("x", 1.0).asDouble();
            block_settings.max_bounds[1] = block_json["max_bounds"].get("y", 1.0).asDouble();
            block_settings.max_bounds[2] = block_json["max_bounds"].get("z", 1.0).asDouble();
        }
        if (block_json.isMember("initial_velocity") && block_json["initial_velocity"].isObject()) {
            block_settings.initial_velocity[0] = block_json["initial_velocity"].get("x", 0.0).asDouble();
            block_settings.initial_velocity[1] = block_json["initial_velocity"].get("y", 0.0).asDouble();
            block_settings.initial_velocity[2] = block_json["initial_velocity"].get("z", 0.0).asDouble();
        }
        block_settings.loaded = true;

        // --- Validate block bounds against grid boundaries ---
        if (block_settings.loaded) {
            bool bounds_valid = true;
            // Check min_bounds < max_bounds
            if (block_settings.min_bounds[0] >= block_settings.max_bounds[0] ||
                block_settings.min_bounds[1] >= block_settings.max_bounds[1] ||
                block_settings.min_bounds[2] >= block_settings.max_bounds[2]) {
                std::cerr << "Error: Generation block min_bounds must be strictly less than max_bounds for all axes." << std::endl;
                std::cerr << "  Block Min: [" << block_settings.min_bounds.transpose() << "]" << std::endl;
                std::cerr << "  Block Max: [" << block_settings.max_bounds.transpose() << "]" << std::endl;
                bounds_valid = false;
            }

            // Check block is within grid
            if (bounds_valid) { // Only check if min < max
                for (int i = 0; i < 3; ++i) {
                    if (block_settings.min_bounds[i] < lc[i] || 
                        block_settings.max_bounds[i] > uc[i]) {
                        std::cerr << "Error: Generation block at axis " << i 
                                  << " [" << block_settings.min_bounds[i] << ", " << block_settings.max_bounds[i] << "]"
                                  << " exceeds grid boundaries [" << lc[i] << ", " << uc[i] << "]." 
                                  << std::endl;
                        bounds_valid = false;
                        break; 
                    }
                }
            }
            if (!bounds_valid) {
                std::cerr << "Grid Lower Corner (lc): [" << lc.transpose() << "]" << std::endl;
                std::cerr << "Grid Upper Corner (uc): [" << uc.transpose() << "]" << std::endl;
                assert(false && "Generation block boundaries are invalid or exceed grid boundaries.");
            }
        }
      } else {
          std::cerr << "Warning: 'particle_source_type' is 'generate_block' but 'generation_block' settings are missing. Using defaults." << std::endl;
      }
    }
  }


  std::cout << "Simulation parameters:" << std::endl;
  std::cout << "  dt_seconds: " << dt_seconds << std::endl;
  std::cout << "  duration_seconds: " << duration_seconds << std::endl;
  std::cout << "  density: " << density << std::endl;
  std::cout << "  dimensions: [" << nx << ", " << ny << ", " << nz << "]" << std::endl;
  std::cout << "  dx: " << dx << std::endl;
  std::cout << "  lc: [" << lc_x << ", " << lc_y << ", " << lc_z << "]" << std::endl;
  std::cout << "  flip_ratio: " << flip_ratio << std::endl;
  // std::cout << "  input_file: " << input_file << std::endl;
  std::cout << "  output_file_name_pattern: " << output_file_name_pattern << std::endl;
  std::cout << "  Particle Settings:" << std::endl;
  std::cout << "  source_type: " << particle_source_type << std::endl;
  if (particle_source_type == "file") {
    std::cout << "    file_path: " << particle_file_path_val << std::endl;
  } else if (particle_source_type == "generate_block" && block_settings.loaded) {
    std::cout << "    generation_block:" << std::endl;
    std::cout << "      num_particles: " << block_settings.num_particles << std::endl;
    std::cout << "      min_bounds: [" << block_settings.min_bounds[0] << ", " << block_settings.min_bounds[1] << ", " << block_settings.min_bounds[2] << "]" << std::endl;
    std::cout << "      max_bounds: [" << block_settings.max_bounds[0] << ", " << block_settings.max_bounds[1] << ", " << block_settings.max_bounds[2] << "]" << std::endl;
    std::cout << "      initial_velocity: [" << block_settings.initial_velocity[0] << ", " << block_settings.initial_velocity[1] << ", " << block_settings.initial_velocity[2] << "]" << std::endl;
  }
  std::cout << "----------------------------------------" << std::endl;


  return SimulationParameters(dt_seconds, duration_seconds, density, dimensions,
                              dx, lc, flip_ratio,
                              output_file_name_pattern,
                              particle_source_type,
                              particle_file_path_val,
                              block_settings);
}

SimulationParameters::~SimulationParameters() {}

SimulationParameters ReadSimulationParameters(int argc, char** argv) {
  if (argc < 2) {
    std::cout << "ERROR: .json file argument not found!" << std::endl;
    std::cout << "Usage: ./FluidSimulator [.json file path]" << std::endl;
    assert(false);  // crash the program
  }

  return SimulationParameters::CreateFromJsonFile(argv[1]);
}
