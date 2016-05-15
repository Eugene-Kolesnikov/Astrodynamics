#include "floatComputations.hpp"
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
