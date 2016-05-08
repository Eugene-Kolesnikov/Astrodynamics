#ifndef GPU_COMPUTATIONS_H
#define GPU_COMPUTATIONS_H

#include "abstractsimulation.hpp"

extern "C" void cu_loadInitParameters(float4* dev_bodies, glm::vec4* bodies, size_t memorySize);
extern "C" void cu_initVelocities(float4* dev_bodies, float3* dev_velocities, float3* dev_acceleration, size_t N);
extern "C" void cu_integrateSystem(float4* dev_bodies, float3* dev_velocities, float3* dev_acceleration, size_t N);
extern "C" void cu_computeCenterOfMass(float4* dev_bodies, float4* dev_tmp_bodies, size_t N);

#endif
