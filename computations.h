#ifndef COMPUTATIONS_H
#define COMPUTATIONS_H

#include "Registry.hpp"

extern "C" void cu_loadInitParameters(float4 *dev_bodies, glm::vec4* N_Bodies, size_t memorySize);
extern "C" void cu_shiftParameters(float4 *dev_bodies, size_t N);

#endif
