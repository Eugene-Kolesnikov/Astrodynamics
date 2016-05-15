#include "gpu_bh_computations.hpp"
#include "gpu_bh_simulation.hpp"
#include "floatComputations.hpp"
#include <glm/gtc/type_ptr.hpp>
#include <thrust/reduce.h>
#include <thrust/device_ptr.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#define _GAMMA_ 6.67408
#define _TAU_ 5e-3
#define _EPS2_ 70

__global__ void
BHinitVelocities(void* devV, void* devA, size_t N)
{
    int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    if(gtid >= N)
        return;
    float3* globalV = (float3*)devV;
    float3* globalA = (float3*)devA;
    float3* curVel = &globalV[gtid];
    float3 curAcc = globalA[gtid];
    *curVel = f3Pf3( *curVel, fTf3(_TAU_ / 2.0f, curAcc) );
}

extern "C" void cu_BHloadInitParameters(float4* dev_bodies, glm::vec4* bodies, size_t memorySize)
{
    checkCudaErrors( cudaMemcpy( dev_bodies, bodies, memorySize, cudaMemcpyHostToDevice ) );
}

extern "C" void cu_BHinitVelocities(float4* dev_bodies, float3* dev_velocities, float3* dev_acceleration, size_t N)
{
    size_t threadsPerBlock = 256;
    size_t blocks = floor((N - 1.0f) / threadsPerBlock) + 1;
    //calculateForces <<< blocks, threadsPerBlock, threadsPerBlock * sizeof(float4) >>> (dev_bodies, dev_acceleration, N);
    BHinitVelocities <<< blocks, threadsPerBlock >>> (dev_velocities, dev_acceleration, N);
}

struct min_max_float4 {
    __device__ float4 operator()(const float4& f1, const float4& f2) const {
        float4 r;
        r.x = min(f1.y, f2.y);
        r.y = max(f1.y, f2.y);
        r.z = min(f1.z, f2.z);
        r.w = max(f1.z, f2.z);
        return r;
    }
 };

void computeBoundingBox(float4* dev_bodies, float* dev_l, float3* dev_pivots, size_t N)
{
    float4 zeros = {1000000.0f, -1000000.0f, 1000000.0f, -1000000.0f};
    thrust::device_ptr<float4> dp = thrust::device_pointer_cast(dev_bodies);
    float4 res = thrust::reduce(dp, dp + AbstractSimulation::N, zeros, min_max_float4());
    GPU_BarnesHut_Simulation::pivot = glm::vec3(0.0f, res.x, res.z);
    // load data to the GPU
    float l = max(fabs(res.x - res.y), fabs(res.z - res.w));
    checkCudaErrors( cudaMemcpy( dev_l + (N-1), &l, sizeof(float), cudaMemcpyHostToDevice ) );
    checkCudaErrors( cudaMemcpy( dev_pivots + (N-1), &GPU_BarnesHut_Simulation::pivot, sizeof(float3), cudaMemcpyHostToDevice ) );
}

__global__ void
BH_buildOctotree(float4* dev_bodies_cells, float* dev_l, float3* dev_pivots, int4* dev_child, size_t N)
{
    int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    if(gtid >= N)
        return;

}

extern "C" void cu_BHintegrateSystem(float4* dev_bodies_cells, // coordinates + mass (bodies + cells)
                                     float4* dev_tmp_bodies, // for the caclation of a pivot point (bodies)
                                     float3* dev_velocities, // (bodies)
                                     float3* dev_acceleration, // (bodies)
                                     float* dev_l, // the width of a space block (cells)
                                     float* dev_delta, // the distance between the center of mass and the geometric center (cells)
                                     float3* dev_pivots, // bottom left points which represent the begining of subspaces (cells)
                                     int4* dev_child, // children pointers (cells)
                                     size_t N)
{
    int blocks, threadsPerBlock;
    computeBoundingBox(dev_bodies_cells, dev_l, dev_pivots, N);
    blocks = 2; // current GPU has 2 SMs
    threadsPerBlock = 128; // 4 warps per block
    BH_buildOctotree <<< blocks, threadsPerBlock >>> (dev_bodies_cells, dev_l, dev_pivots, dev_child, N);
}
