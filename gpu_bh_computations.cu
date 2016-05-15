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

#define _LOCKED_ -2
#define _UNUSED_ -1

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

__global__ void setDefaultValues(float4* dev_bodies, int4* dev_child, int* dev_nextCell, size_t N)
{
    int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    if(gtid >= N)
        return;
    int4 nullLink = {_UNUSED_,_UNUSED_,_UNUSED_,_UNUSED_};
    dev_child[gtid] = nullLink;
    if(gtid == 0)
        *dev_nextCell = 2*N-2; // penultimate
}

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

#define _AVAILABLE_SMs_ 2

__device__ int int4Index(const int4& f, int id)
{
    switch(id) {
        case 0: return f.x;
        case 1: return f.y;
        case 2: return f.z;
        case 3: return f.w;
    }
    return -1;
}

__device__ int* p_int4Index(int4& f, int id)
{
    switch(id) {
        case 0: return &(f.x);
        case 1: return &(f.y);
        case 2: return &(f.z);
        case 3: return &(f.w);
    }
    return NULL;
}

__device__ int find_insertion_index(float4 body, float3 pivot, float l)
{
    float L = l / 2.0f;
    float3 geometricCenter = { 0.0f, pivot.y + L, pivot.z + L };
    float diffY = body.y - geometricCenter.y;
    float diffZ = body.z - geometricCenter.z;
    return (diffY > 0 ? (diffZ > 0 ? 2 : 1) : (diffZ > 0 ? 4 : 3));
}

__device__ void find_insertion_point_index(float4 body, int4* dev_child, float* dev_l, float3* dev_pivots, int& cell, int& child, int& childVal, size_t N)
{
    int m = 2*N-1; // root of the octotree
    int childId, tmp;
    do {
        childId = find_insertion_index(body, dev_pivots[m-N], dev_l[m-N]);
        tmp = int4Index(dev_child[m-N], childId);
        if(tmp < N) {
            childVal = tmp;
            break;
        } else {
            m = tmp;
        }
    } while(1);
    cell = m;
    child = childId;
}

__global__ void
BH_buildOctotree(float4* dev_bodies_cells, float* dev_l, float3* dev_pivots, int4* dev_child, int* dev_nextCell, size_t N)
{
    int steps = ceil((float)N / (float)(_AVAILABLE_SMs_ * blockDim.x));
    int body, cell, child, childVal, newCell;
    int body1_newID, body2_newID;
    for(int i = 0; i < steps; ++i)
    {
        int gtid = (blockIdx.x + i * _AVAILABLE_SMs_) * blockDim.x + threadIdx.x;
        if(gtid >= N) // gtid exceeded the number of bodies
            return;
        body = gtid;
        bool success = false;
        do {
            find_insertion_point_index(dev_bodies_cells[body], dev_child, dev_l, dev_pivots, cell, child, childVal, N);
            if(childVal != _LOCKED_)
            {
                if(child == atomicCAS(p_int4Index(dev_child[cell], child), childVal, _LOCKED_))
                {
                    if(childVal == _UNUSED_) {
                        // insert a body and release the lock
                        // using atomic operation is necessary for all threads to see the correct value
                        atomicCAS(p_int4Index(dev_child[cell], child), _LOCKED_, body);
                        // flag which indicates that the insertion successfully ended
                        success = true;
                    } else {
                        newCell = atomicDec((unsigned*)dev_nextCell, 2*N); // atomically get the next unused cell
                        // insert the existing and new body into newCell
                        // compute new pivot point and 'l'
                        float prevL = dev_l[cell-N];
                        float3 prevPivot = dev_pivots[cell-N];
                        float3 newPivot;
                        float newL = prevL / 2.0f;
                        switch(child) {
                            case 1:
                                newPivot.x = 0.0f;
                                newPivot.y = prevPivot.y + prevL;
                                newPivot.z = prevPivot.z;
                                break;
                            case 2:
                                newPivot.x = 0.0f;
                                newPivot.y = prevPivot.y + prevL;
                                newPivot.z = prevPivot.z + prevL;
                                break;
                            case 3:
                                newPivot = prevPivot; break;
                            case 4:
                                newPivot.x = 0.0f;
                                newPivot.y = prevPivot.y;
                                newPivot.z = prevPivot.z + prevL;
                                break;
                        }
                        dev_l[newCell] = newL;
                        dev_pivots[newCell-N] = newPivot;
                        // calculate indexes
                        body1_newID = find_insertion_index(dev_bodies_cells[body], dev_pivots[newCell-N], dev_l[newCell-N]);
                        body2_newID = find_insertion_index(dev_bodies_cells[childVal], dev_pivots[newCell-N], dev_l[newCell-N]);

                        if(body1_newID != body2_newID) {
                            *p_int4Index(dev_child[newCell], body1_newID) = body;
                            *p_int4Index(dev_child[newCell], body2_newID) = childVal;
                            // flag which indicates that the insertion successfully ended
                            success = true;
                        } else {
                            *p_int4Index(dev_child[newCell], body2_newID) = childVal;
                        }
                        __threadfence();
                        // using atomic operation is necessary for all threads to see the correct value
                        atomicCAS(p_int4Index(dev_child[cell], child), _LOCKED_, newCell);
                    }
                }
            }
        } while(success == false);

        __syncthreads();
    }
}

extern "C" void cu_BHintegrateSystem(float4* dev_bodies_cells, // coordinates + mass (bodies + cells)
                                     float4* dev_tmp_bodies, // for the caclation of a pivot point (bodies)
                                     float3* dev_velocities, // (bodies)
                                     float3* dev_acceleration, // (bodies)
                                     float* dev_l, // the width of a space block (cells)
                                     float* dev_delta, // the distance between the center of mass and the geometric center (cells)
                                     float3* dev_pivots, // bottom left points which represent the begining of subspaces (cells)
                                     int4* dev_child, // children pointers (cells)
                                     int* dev_nextCell,
                                     size_t N)
{
    int blocks, threadsPerBlock;

    threadsPerBlock = 1024;
    blocks = floor((N-1) / threadsPerBlock) + 1;
    setDefaultValues <<< blocks, threadsPerBlock >>> (dev_bodies_cells, dev_child, dev_nextCell, N);

    computeBoundingBox(dev_bodies_cells, dev_l, dev_pivots, N);

    blocks = _AVAILABLE_SMs_; // current GPU has 2 SMs
    threadsPerBlock = 1024;
    BH_buildOctotree <<< blocks, threadsPerBlock >>> (dev_bodies_cells, dev_l, dev_pivots, dev_child, dev_nextCell, N);
}
