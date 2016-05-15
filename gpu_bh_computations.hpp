#ifndef GPU_BH_COMPUTATIONS_H
#define GPU_BH_COMPUTATIONS_H

#include "abstractsimulation.hpp"

extern "C" void cu_BHloadInitParameters(float4* dev_bodies, glm::vec4* bodies, size_t memorySize);
extern "C" void cu_BHinitVelocities(float4* dev_bodies, float3* dev_velocities, float3* dev_acceleration, size_t N);
extern "C" void cu_BHintegrateSystem(float4* dev_bodies_cells, // coordinates + mass (bodies + cells)
                                     float4* dev_tmp_bodies, // for the caclation of a pivot point (bodies)
                                     float3* dev_velocities, // (bodies)
                                     float3* dev_acceleration, // (bodies)
                                     float* dev_l, // the width of a space block (cells)
                                     float* dev_delta, // the distance between the center of mass and the geometric center (cells)
                                     float3* dev_pivots, // bottom left points which represent the begining of subspaces (cells)
                                     int4* dev_child, // children pointers (cells)
                                     int* dev_nextCell,
                                     size_t N);

#endif
