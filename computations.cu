#include "computations.h"
#include <glm/gtc/type_ptr.hpp>
#include <thrust/reduce.h>
#include <thrust/device_ptr.h>

#define _GAMMA_ 6.67408
#define _TAU_ 5e-3
#define _EPS2_ 70

__device__ float4 fTf4(float f, float4 f4)
{
    f4.x *= f;
    f4.y *= f;
    f4.z *= f;
    f4.w *= f;
    return f4;
}

__device__ float3 fTf3(float f, float3 f3)
{
    f3.x *= f;
    f3.y *= f;
    f3.z *= f;
    return f3;
}

__device__ float4 f4Pf4(float4 f1, float4 f2)
{
    f1.x += f2.x;
    f1.y += f2.y;
    f1.z += f2.z;
    f1.w += f2.w;
    return f1;
}

__device__ float3 f3Pf3(float3 f1, float3 f2)
{
    f1.x += f2.x;
    f1.y += f2.y;
    f1.z += f2.z;
    return f1;
}

__device__ float3
bodyBodyInteraction(float4 bi, float4 bj, float3 ai) {
    float3 r;
    // r_ij [3 FLOPS]
    r.x = bj.x - bi.x; r.y = bj.y - bi.y; r.z = bj.z - bi.z;
    // distSqr = dot(r_ij, r_ij) + EPS^2 [6 FLOPS]
    float distSqr = sqrtf(r.x * r.x + r.y * r.y + r.z * r.z);
    float distSqr_eps = r.x * r.x + r.y * r.y + r.z * r.z + _EPS2_;
    // invDistCube =1/distSqr^(3/2) [4 FLOPS (2 mul, 1 sqrt, 1 inv)]
    float distSixth_eps = distSqr_eps * distSqr_eps * distSqr_eps;
    float invDistCube = 1.0f/sqrtf(distSixth_eps);
    // s = m_j * invDistCube [1 FLOP]
    float s = bj.w * _GAMMA_ * invDistCube;
    //a_i= a_i+s*r_ij[6FLOPS]
    ai.x += r.x * s; ai.y += r.y * s; ai.z += r.z * s;
    return ai;
}

__device__ float3
tileComputation(float4 myPosition, float3 accel)
{
    extern __shared__ float4 shPosition[];
    for (int i = 0; i < blockDim.x; ++i)
        accel = bodyBodyInteraction(myPosition, shPosition[i], accel);
    return accel;
}

__global__ void
calculateForces(void* devX, void* devA, size_t N)
{
    extern __shared__ float4 shPosition[];
    int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    if(gtid >= N)
        return;
    float4* globalX = (float4*)devX;
    float3* globalA = (float3*)devA;
    float3 acc = {0.0f, 0.0f, 0.0f};
    float4 myPosition = globalX[gtid];
    for (int i = 0, tile = 0; i < N; i += blockDim.x, tile++) {
        int idx = tile * blockDim.x + threadIdx.x;
        shPosition[threadIdx.x] = globalX[idx];
        __syncthreads();
        acc = tileComputation(myPosition, acc);
        __syncthreads();
    }
    // Save the result in global memory for the integration step.
    globalA[gtid] = acc;
}

__global__ void
initVelocities(void* devV, void* devA, size_t N)
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

__global__ void
updatePositions(void* devX, void* devV, size_t N)
{
    int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    if(gtid >= N)
        return;
    float4* globalX = (float4*)devX;
    float3* globalV = (float3*)devV;
    float4* curPos = &globalX[gtid];
    float3 curVel_ = globalV[gtid];
    float4 curVel = {curVel_.x, curVel_.y, curVel_.z, 0};
    *curPos = f4Pf4( *curPos, fTf4(_TAU_, curVel) );
}

__global__ void
updateVelocities(void* devV, void* devA, size_t N)
{
    int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    if(gtid >= N)
        return;
    float3* globalV = (float3*)devV;
    float3* globalA = (float3*)devA;
    float3* curVel = &globalV[gtid];
    float3 curAcc = globalA[gtid];
    *curVel = f3Pf3( *curVel, fTf3(_TAU_, curAcc) );
}

void LeapFrog_integrator(float4* dev_bodies, float3* dev_velocities, float3* dev_acceleration, size_t N)
{
    size_t threadsPerBlock = 256;
    size_t blocks = floor((N - 1.0f) / threadsPerBlock) + 1;
    updatePositions <<< blocks, threadsPerBlock >>> (dev_bodies, dev_velocities, N);
    calculateForces <<< blocks, threadsPerBlock, threadsPerBlock * sizeof(float4) >>> (dev_bodies, dev_acceleration, N);
    updateVelocities <<< blocks, threadsPerBlock >>> (dev_velocities, dev_acceleration, N);
}

extern "C" void cu_loadInitParameters(float4 *dev_bodies, glm::vec4* N_Bodies, size_t memorySize)
{
    checkCudaErrors( cudaMemcpy( dev_bodies, N_Bodies, memorySize, cudaMemcpyHostToDevice ) );
}

extern "C" void cu_initVelocities(float4* dev_bodies, float3* dev_velocities, float3* dev_acceleration, size_t N)
{
    size_t threadsPerBlock = 256;
    size_t blocks = floor((N - 1.0f) / threadsPerBlock) + 1;
    calculateForces <<< blocks, threadsPerBlock, threadsPerBlock * sizeof(float4) >>> (dev_bodies, dev_acceleration, N);
    initVelocities <<< blocks, threadsPerBlock >>> (dev_velocities, dev_acceleration, N);
}

extern "C" void cu_integrateSystem(float4* dev_bodies, float3* dev_velocities, float3* dev_acceleration, size_t N)
{
    size_t threadsPerBlock = 256;
    size_t blocks = floor((N - 1.0f) / threadsPerBlock) + 1;
    LeapFrog_integrator(dev_bodies, dev_velocities, dev_acceleration, N);
}

__global__ void
prepareCenterOfMass(float4* bodies, size_t N)
{
    int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    if(gtid >= N)
        return;
    float4 body = bodies[gtid];
    body.x *= body.w;
    body.y *= body.w;
    body.z *= body.w;
    bodies[gtid] = body;
}

struct add_float4 {
    __device__ float4 operator()(const float4& f1, const float4& f2) const {
        float4 r;
        r.x = f1.x + f2.x;
        r.y = f1.y + f2.y;
        r.z = f1.z + f2.z;
        r.w = f1.w + f2.w;
        return r;
    }
 };

extern "C" void cu_computeCenterOfMass(float4* dev_bodies, float4* dev_tmp_bodies, size_t N)
{
    checkCudaErrors( cudaMemcpy( dev_tmp_bodies, dev_bodies, N * sizeof(float4), cudaMemcpyDeviceToDevice ) );
    size_t threadsPerBlock = 256;
    size_t blocks = floor((N - 1.0f) / threadsPerBlock) + 1;
    prepareCenterOfMass <<< blocks, threadsPerBlock >>> (dev_tmp_bodies, N);
    float4 zeros = {0.0f, 0.0f, 0.0f, 0.0f};
    thrust::device_ptr<float4> dp = thrust::device_pointer_cast(dev_tmp_bodies);
    float4 res = thrust::reduce(dp, dp + N, zeros, add_float4());
    Registry::centerOfMass = glm::vec3(res.x / res.w, res.y / res.w, res.z / res.w);
}
