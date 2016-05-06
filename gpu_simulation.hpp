#ifndef GPU_SIMULATION_H
#define GPU_SIMULATION_H

#include "abstractsimulation.hpp"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

class GPU_Simulation : public AbstractSimulation
{
public:
    GPU_Simulation();
    ~GPU_Simulation();

    virtual void init(int dimensions);
    virtual void render();

protected:
    struct cudaGraphicsResource *cuda_pos_resource;

protected:
    float4* dev_bodies;
    float4* dev_tmp_bodies;
    float3* dev_velocities;
    float3* dev_acceleration;
};

#endif
