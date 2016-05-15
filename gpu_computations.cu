#include "gpu_computations.hpp"
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

extern "C" void cu_loadInitParameters(float4 *dev_bodies, glm::vec4* bodies, size_t memorySize)
{
    checkCudaErrors( cudaMemcpy( dev_bodies, bodies, memorySize, cudaMemcpyHostToDevice ) );
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
    updatePositions <<< blocks, threadsPerBlock >>> (dev_bodies, dev_velocities, N);
    calculateForces <<< blocks, threadsPerBlock, threadsPerBlock * sizeof(float4) >>> (dev_bodies, dev_acceleration, N);
    updateVelocities <<< blocks, threadsPerBlock >>> (dev_velocities, dev_acceleration, N);
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
    checkCudaErrors( cudaMemcpy( dev_tmp_bodies, dev_bodies, AbstractSimulation::N * sizeof(float4), cudaMemcpyDeviceToDevice ) );
    size_t threadsPerBlock = 256;
    size_t blocks = floor((AbstractSimulation::N - 1.0f) / threadsPerBlock) + 1;
    prepareCenterOfMass <<< blocks, threadsPerBlock >>> (dev_tmp_bodies, AbstractSimulation::N);
    float4 zeros = {0.0f, 0.0f, 0.0f, 0.0f};
    thrust::device_ptr<float4> dp = thrust::device_pointer_cast(dev_tmp_bodies);
    float4 res = thrust::reduce(dp, dp + AbstractSimulation::N, zeros, add_float4());
    AbstractSimulation::centerOfMass = glm::vec3(res.x / res.w, res.y / res.w, res.z / res.w);
}

__global__ void
calculatePotential(float4* dev_potentialPositions, float4* dev_bodies, float* dev_potentialColorGray, size_t potentialN, size_t bodiesN)
{
    extern __shared__ float4 shPosition[];
    int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    if(gtid >= potentialN)
        return;
    float3 acc = {0.0f, 0.0f, 0.0f};
    float4 myPosition = dev_potentialPositions[gtid];
    for (int i = 0, tile = 0; i < bodiesN; i += blockDim.x, tile++) {
        int idx = tile * blockDim.x + threadIdx.x;
        shPosition[threadIdx.x] = dev_bodies[idx];
        __syncthreads();
        acc = tileComputation(myPosition, acc);
        __syncthreads();
    }
    // Save the result in global memory for the integration step.
    float potential = sqrtf(acc.x*acc.x+acc.y*acc.y+acc.z*acc.z);
    dev_potentialColorGray[gtid] = potential;
}

struct max_float {
    __device__ float operator()(const float& f1, const float& f2) const {
        float r;
        r = max(f1, f2);
        return r;
    }
};

 struct min_float {
     __device__ float operator()(const float& f1, const float& f2) const {
         float r;
         r = min(f1, f2);
         return r;
     }
};

__global__ void
normalizePotential(float* dev_potentialColorGray, size_t N, float maxV, float minV)
{
    int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    if(gtid >= N)
        return;
    float n_potential = (dev_potentialColorGray[gtid]/* - minV*/) / (maxV - minV);
    dev_potentialColorGray[gtid] = n_potential;
}

extern "C" void cu_initPotentialField(float4* dev_potentialPos, float4* dev_bodies, float* dev_potentialColorGray, size_t num_points)
{
    float4* dev_potentialPositions = dev_potentialPos;
    size_t threadsPerBlock = 1024;
    size_t blocks = floor((num_points - 1.0f) / threadsPerBlock) + 1;
    calculatePotential <<< blocks, threadsPerBlock, threadsPerBlock * sizeof(float4) >>> (dev_potentialPositions, dev_bodies, dev_potentialColorGray, num_points, AbstractSimulation::N);
    float max_zeros = 0.0f;
    float min_zeros = 10000.0f;
    thrust::device_ptr<float> dp = thrust::device_pointer_cast(dev_potentialColorGray);
    float max_res = thrust::reduce(dp+1, dp + num_points, max_zeros, max_float());
    float min_res = thrust::reduce(dp+1, dp + num_points, min_zeros, min_float());
    normalizePotential <<< blocks, threadsPerBlock >>> (dev_potentialColorGray, num_points, max_res, min_res);
}

__global__ void
grayToRGB_kernel(float* dev_potentialColorGray, float3* dev_potentialColorRGB, size_t num_points)
{
    int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    if(gtid >= num_points)
        return;
    float n_potential = dev_potentialColorGray[gtid];
    //float3 rgb = {0.3 * n_potential, 0.59 * n_potential, 0.11 * n_potential};
    //float3 rgb = {n_potential, 0.5*n_potential, 0.5*n_potential};
    float3 rgb = {10*n_potential, 30*n_potential, 10*n_potential};
    dev_potentialColorRGB[gtid] = rgb;
}

extern "C" void cu_grayToRGB(float* dev_potentialColorGray, float3* dev_potentialColorRGB, size_t num_points)
{
    size_t threadsPerBlock = 1024;
    size_t blocks = floor((num_points - 1.0f) / threadsPerBlock) + 1;
    grayToRGB_kernel <<< blocks, threadsPerBlock >>> (dev_potentialColorGray, dev_potentialColorRGB, num_points);
}
