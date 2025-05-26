# PIC/FLIP Fluid Simulation

A C++/CUDA implementation of the PIC/FLIP method for fluid simulation, with both CPU and CUDA versions.

## Demo Video
![simulation result](./demo/output_file.gif)
![simulation result](./demo/output_block_500000.gif)

## Requirements

- C++14 compatible compiler
- CUDA toolkit (for CUDA version)
- CMake 3.10 or higher
- Python 3.x (for visualization)

## Building

### CPU Version
1. Create folder "CpuBuild"
2. Rename CMakeLists_cpu.txt to CMakeLists.txt

```bash
cd CpuBuild
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=g++ ..
make
```

### CUDA Version
1. Create folder "CudaBuild"
2. Rename CMakeLists_cuda.txt to CMakeLists.txt
3. Set your CUDA path in CMakeLists.txt:
   ```cmake
   set(CUDA_TOOLKIT_ROOT_DIR "/path/to/your/cuda")  # Replace with your CUDA installation path
   ```

```bash
cd CudaBuild
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=g++ -DCMAKE_CUDA_COMPILER=nvcc ..
make
```

## Configurations
- `fluid_file.json`: Uses `inputs/particles.in` to initialize particles for simulation
- `fluid_block.json`: Generates a block-shaped fluid for simulation

## Running

### CPU Version
```bash
cd ../
./CpuBuild/bin/FluidSimulator_cpu inputs/fluid_file.json
```

### CUDA Version
```bash
cd ../
./CudaBuild/bin/FluidSimulator_cuda inputs/fluid_file.json
```

## Visualization

Use the provided Python script to visualize the simulation results:
```bash
# For file-based simulation (particles.in)
python ParticleViewer.py outputs_cuda/fluid.%03d.part file

# For block-based simulation
python ParticleViewer.py outputs_cuda/fluid.%03d.part block
```

Generate video from frames:
```bash
ffmpeg -framerate 30 -i output/frame_%03d.png -c:v libx264 -pix_fmt yuv420p output.mp4
```

The visualization adjusts the camera view (gluLookAt) based on the simulation type: file, block

## References

This project was implemented with reference to the following materials:

- [Fluid Simulation for Computer Graphics by Robert Bridson](https://www.cs.ubc.ca/~rbridson/fluidsimulation/fluids_notes.pdf)
- [Physics-Based Animation Course Notes (UMBC)](https://cal.cs.umbc.edu/Courses/PhysicsBasedAnimation/PhysicsBasedAnimationCourseNotes2019.pdf)
- [Fluid Tutorial (unusualnights)](https://unusualinsights.github.io/fluid_tutorial/)  
  → The CPU version used the code from this tutorial.

## License

This project is open source and available under the MIT License.
