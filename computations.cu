#include "computations.h"
#include <glm/gtc/type_ptr.hpp>

__global__ void shiftParameters(float4 *dev_bodies, size_t N)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid >= N)
        return;
    float4 body = dev_bodies[tid];
    body.y += 0.001;
    dev_bodies[tid] = body;
}

extern "C" void cu_loadInitParameters(float4 *dev_bodies, glm::vec4* N_Bodies, size_t memorySize)
{
    checkCudaErrors( cudaMemcpy( dev_bodies, N_Bodies, memorySize, cudaMemcpyHostToDevice ) );
}

extern "C" void cu_shiftParameters(float4 *dev_bodies, size_t N)
{
    dim3 threads(32, 1, 1);
    dim3 blocks((N - 1) / threads.x + 1, 1, 1);
    shiftParameters <<< blocks, threads >>> (dev_bodies, N);
}
